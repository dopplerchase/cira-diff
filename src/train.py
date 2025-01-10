import torch
import zarr
from diffusers import UNet2DModel
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse 
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch.nn.functional as F
import gc
import os 
import torch.distributed as dist
from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
import os
import math
from typing import List, Optional, Tuple, Union
from diffusers.utils.torch_utils import randn_tensor
from torch.utils.tensorboard import SummaryWriter
import importlib.util
import sys

def train_loop(config, model, optimizer, dataset, lr_scheduler):
    """ 
    This is the main show! the training loop. 
    
    This is an amalgamation of several scripts but started with this one
    - https://huggingface.co/docs/diffusers/tutorials/basic_training 
    
    """
    
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    
    #if you are on the headnode/gpu (i.e., do only once rather than for every GPU)
    if accelerator.is_main_process:
        #i haven't used this, this pushes to huggingface hubs i think. 
        if config.push_to_hub:
            repo_name = get_full_repo_name(Path(config.output_dir).name)
            repo = Repository(config.output_dir, clone_from=repo_name)
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        
        #not sure what this does 
        accelerator.init_trackers("train_example")
        
        if config.plot_images:
            
            #image writer for the tensorboard 
            writer = SummaryWriter(config.output_dir + "logs/images")
            #need a random seed for the edm process that we run after every epoch.         
            rnd = StackedRandomGenerator('cuda',np.arange(0,config.train_batch_size,1).astype(int).tolist())
            latents = rnd.randn([config.train_batch_size, 1, 256, 256],device='cuda')

            #setup the dataset now WITHOUT shuffling, for consistent eval images 
            train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=False)

            #grab the first batch of data for those test images (should probably make a seperate file here)
            for step, batch in enumerate(train_dataloader):
                        # these are the images we want to make 
                        clean_images_eval = batch[0].to('cuda')

                        #these are the condition 
                        condition_images_eval = batch[1].to('cuda')
                        break 

            del train_dataloader 
            
            #undo the scaling before adding to the original image 
            data_mean = torch.tensor(-0.0009)
            data_std = torch.tensor(0.0807)

            clean_images_eval = clean_images_eval*data_std + data_mean
            
            for i in np.arange(0,len(config.images_idx)):
                #reshape image for adding a color image to the tensorboard 
                image = condition_images_eval[config.images_idx[i],0:1].squeeze(0).unsqueeze(-1).cpu()
                #add color and convert shapes back. 
                color_image = torch.tensor(colorize(image,vmin=-6,vmax=4,cmap='Spectral_r')).permute(2, 0, 1)

                #write the image to the tensorboard 
                writer.add_image("Example {}".format(i), color_image, 0)

                #do the same thing for the second image. 
                image = condition_images_eval[config.images_idx[i],1:2].squeeze(0).unsqueeze(-1).cpu()
                color_image = torch.tensor(colorize(image,vmin=-6,vmax=4,cmap='Spectral_r')).permute(2, 0, 1)
                writer.add_image("Example {}".format(i), color_image, 1)

                #do the same thing for 'truth'
                image = clean_images_eval[config.images_idx[i],0:1].squeeze(0).unsqueeze(-1).cpu() + condition_images_eval[config.images_idx[i],2:3].squeeze(0).unsqueeze(-1).cpu()
                color_image = torch.tensor(colorize(image,vmin=-6,vmax=4,cmap='Spectral_r')).permute(2, 0, 1)
                writer.add_image("Example {}".format(i), color_image, 2)
        
        #properly setup the dataset now with shuffling 
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True,num_workers=8,
                                                       pin_memory=True,worker_init_fn = worker_init_fn)
        
        if config.restart:
            start_epoch, global_step = load_checkpoint(config.restart_path + 'checkpoint.pth', model, optimizer, lr_scheduler, accelerator)
            epochs = np.arange(start_epoch,config.num_epochs)
        else:
            #iterator to see how many gradient steps have been done
            global_step = 0
            epochs = range(config.num_epochs)
            

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    #initalize early stopping stuff
    best_loss = float('inf') 
    no_improvement_count = 0
    loss_history = [] 
    
    
    #define loss, you can change the sigma vals here (i.e., hyperparameters), change them in the config class
    loss_fn = EDMLoss(P_mean=config.P_mean,P_std=config.P_std,sigma_data=config.sigma_data)
    
    # Now you train the model
    for epoch in epochs:
        #this is for the cmd line
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        #initalize loss to keep track of the mean loss across all batches in this epoch 
        epoch_loss = torch.tensor(0.0, device=accelerator.device)
        
        for step, batch in enumerate(train_dataloader):
            
            #my data loader returns [TARGET, CONDITION],I seperate them here just to be clear 
            
            #TARGET
            clean_images = batch[0]

            #CONDITION
            condition_images = batch[1]
            
            #this is the autograd steps within the .accumulate bit (this is important for multi-GPU training)
            with accelerator.accumulate(model):

                #send data into loss func and get the loss (the model call is in here)
                per_sample_loss = loss_fn(model,clean_images, condition_images)

                #in the loss_fn, it returns the squarred error (per pixel basis), we need the mean (across the batch) for the loss  
                loss = per_sample_loss.mean()
                
                #calc backprop 
                accelerator.backward(loss)
                
                #this is needed to enable gradient accumulation for some reason 
                if accelerator.sync_gradients:
                    #clip gradients (leftover from Butterflies example)
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    
                #step things now 
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            
            # Accumulate epoch loss on each GPU seperately 
            epoch_loss += loss.detach()
            
            #log things to the cmd line 
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            #push the same logs to the tensorboard
            accelerator.log(logs, step=global_step)
            #step the global steps, go onto the next batch 
            global_step += 1
            
            
            
        #now that we saw the dataset once, lets do some book keeping. 
        
        # Synchronize epoch loss across devices, this will just concat the two 
        epoch_loss = accelerator.gather(epoch_loss)

        # Sum up the losses across all GPUs
        total_epoch_loss = epoch_loss.sum()

        # the batches are split from the train_dataloader to each GPU
        total_samples_processed = len(train_dataloader) * accelerator.num_processes

        # Calculate mean epoch loss by dividing by the total number of batches proccessed 
        mean_epoch_loss = total_epoch_loss / total_samples_processed
        
        # Print or log the average epoch loss, need to convert to scalar to get tensorboard to work (using .item())
        logs = {"epoch_loss": mean_epoch_loss.item(), "epoch": epoch}
        accelerator.log(logs, step=epoch)

        #accumulate rolling mean for early stopping 
        loss_history.append(mean_epoch_loss.item())
        
        # Calculate the moving average if enough epochs have passed
        if len(loss_history) >= config.window_size:
            moving_average = sum(loss_history[-config.window_size:]) / config.window_size
            logs = {"moving_epoch_loss": moving_average, "epoch": epoch}
            accelerator.log(logs, step=epoch)

            # Check for improvement in the moving_average
            if moving_average < (best_loss - config.min_delta):
                best_loss = moving_average
                no_improvement_count = 0
            else:
                no_improvement_count += 1
        
        # This is the eval and saving step 
        if accelerator.is_main_process:
                
            #this is to save the model 
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    
                    #need to grab the unwrapped diffusers model from EDMPrecond (old way)
                    #accelerator.unwrap_model(model).model.save_pretrained(config.output_dir)
                    
                    #save out the checkpoint for restarts 
                    save_checkpoint(model, optimizer, lr_scheduler, epoch, global_step, accelerator, config.output_dir + 'checkpoint.pth')
                    
                    if config.plot_images:
                        #run a batch of images through for tensorboard (takes < 1 min)
                        images_batch = edm_sampler(model,latents,condition_images_eval,num_steps=18)
                        
                        #undo the scaling before adding to the original image 
                        data_mean = torch.tensor(-0.0009)
                        data_std = torch.tensor(0.0807)
                        
                        images_batch = images_batch*data_std + data_mean

                        for i in np.arange(0,len(config.images_idx)):

                            #reshape the image so we can add color to the tensorboard
                            image = images_batch[config.images_idx[i]].squeeze(0).unsqueeze(-1).cpu().numpy()+ condition_images_eval[config.images_idx[i],2:3].squeeze(0).unsqueeze(-1).cpu().numpy()
                            #colorize the image 
                            color_image = torch.tensor(colorize(image,vmin=-6,vmax=4,cmap='Spectral_r')).permute(2, 0, 1)

                            #add to board 
                            writer.add_image("Output Example {}".format(i), color_image, epoch)


                    


        # Check if training should be stopped due to lack of improvement (e.g., early stopping)
        if no_improvement_count >= config.patience:
            print(f"Early stopping triggered after {config.patience} epochs without improvement.")
                # Check if multi-GPU is being used
            if accelerator.num_processes > 1:
                print(f"Killing multi-GPU processes using dist.barrier() and dist.destroy_process_group()")
                # Signal all processes to stop
                dist.barrier()  # Ensure all processes are synchronized
                dist.destroy_process_group() 
            break
        #run some cleanup, because leaks             
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Run the model with a Python config file using a dataclass.")
    parser.add_argument('--config', type=str, required=True, 
                        help="Path to the Python configuration file containing TrainingConfig.")
    parser.add_argument('--verbose', type=int, default=1, 
                        help="Set to 1 to print config details, 0 for silent mode.")
    
    args = parser.parse_args()
    
    # Load the dataclass from the config
    config = load_config(args.config)
    
    # Print the config values only if verbose is enabled
    if args.verbose:
        print(f"\nUsing config file: {args.config}")
        print("Loaded Configuration:")
        for key, value in vars(config).items():
            print(f"{key}: {value}")

    if args.verbose:
        print('Loading dataset')
    
    # Initialize the dataset
    dataset = ZarrDataset(config.dataset_path)

    if args.verbose:
        print('Loaded')

    if args.verbose:
        print('Building model based on config and wrapping')

    #Initialize the model
    model = build_model(config)

    #wrap diffusers/pytorch model with the scalings from Karras et al. (2022)
    model_wrapped = EDMPrecond(1,model)

    if args.verbose:
        print('Built')


    if args.verbose:
        print('Initializing optimizer and scheduler')

    #initalize the optimizer
    optimizer = torch.optim.AdamW(model_wrapped.model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(dataset.length * config.num_epochs),
    )

    if args.verbose:
        print('Done')

    if args.verbose:
        print('Starting Training')

    #main method here! 
    train_loop(config, model_wrapped, optimizer, dataset, lr_scheduler)

    if args.verbose:
        print('DoneDoneDone!')

if __name__ == "__main__":
    main()