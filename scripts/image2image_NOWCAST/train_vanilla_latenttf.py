# ################### Imports ########################

import torch
import zarr
from torch.utils.data import Dataset
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import DiTTransformer2DModel
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
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
from diffusers import AutoencoderKL


# ################### \Imports ########################


# ################### Classes ########################

@dataclass
class TrainingConfig:
    """ This should be probably in some sort of config file, but for now its here... """
    #data things 
    image_size = 256  #this assumes square images
    train_batch_size = 90 #this is as big as I can fit on the GH200 [45,3,256,256]
    val_batch_size = 45
    
    #training things 
    num_epochs = 1000 #this should be similar to NVIDIA's StormCast 
    gradient_accumulation_steps = 1 #this helped with stability, i think... 
    learning_rate = 1e-5 #default value from butterflies example
    lr_warmup_steps = 500 #default value from butterflies example
    save_model_epochs = 1 #i like to save alot, doesnt cost much 
    mixed_precision = "fp16"
    output_dir = "/mnt/data1/rchas1/latenttf_10_two_inputs_radames/"  # the local path to store the model 
    push_to_hub = False 
    hub_private_repo = False
    overwrite_output_dir = True  
    seed = 0 
    restart = False #do you want to start from a previous training?
    restart_path = None
    
    dataset_path = "/mnt/data1/rchas1/diffusion_10_4_2inputs_v2_gh200_latent_radames.zarr"
    
    #tensorboard things 
    plot_images = True 
    images_idx = [3,5,10,15] #these need to be smaller than train_batch_size 
    
    #loss params (defaults to the edm paper)
    P_mean=-1.2
    P_std=1.2
    sigma_data=0.5
    
    #early stopping things 
    patience = 10  # Number of epochs to wait for improvement
    min_delta = 1e-6  # Minimum change in loss to be considered as improvement
    window_size = 5  # Define the window size for the moving average
    
    #vae path on huggingface
    vae_path = "radames/stable-diffusion-x4-upscaler-img2img"

class ZarrDataset(Dataset):
    """This is a new zarr instance of the dataset chunked at the desired batchsize to ensure the GPU is fed. """
    def __init__(self, zarr_store):
        self.store = zarr_store
        self.data = zarr.open(self.store, mode='r')
        self.length = self.data['input_images'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Load data lazily
        input_image = self.data['input_images'][idx]
        output_image = self.data['output_images'][idx]
        return torch.tensor(output_image, dtype=torch.float16),torch.tensor(input_image, dtype=torch.float16)

# ################### \Classes ########################

# ################### Funcs ########################

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
            
            #need to decode the images
            #load vae 
            vae =  AutoencoderKL.from_pretrained(config.vae_path, subfolder="vae").to('cuda')
#             vae =  AutoencoderKL.from_pretrained(config.vae_path).to('cuda')
            
            
            #image writer for the tensorboard 
            writer = SummaryWriter(config.output_dir + "logs/images")
            
            #setup the dataset now WITHOUT shuffling, for consistent eval images 
            train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=False)

            #grab the first batch of data for those test images (should probably make a seperate file here)
            for step, batch in enumerate(train_dataloader):
                        # these are the images we want to make 
                        clean_images_eval = batch[0].to(torch.float32).to('cuda')

                        #these are the condition 
                        condition_images_eval = batch[1].to(torch.float32).to('cuda')
                        break 

            del train_dataloader 
            

            #reshape images for the decoding 
            condition_images_eval_reshaped = torch.reshape(condition_images_eval,
                                                           [condition_images_eval.shape[0]*2,
                                                            vae.config.latent_channels,
                                                            condition_images_eval.shape[2],
                                                            condition_images_eval.shape[3]])
            
            clean_images_eval_reshaped =  torch.reshape(clean_images_eval,
                                                           [clean_images_eval.shape[0]*1,
                                                            vae.config.latent_channels,
                                                            clean_images_eval.shape[2],
                                                            clean_images_eval.shape[3]])
            #check recons. just in case, trust issues 
            with torch.no_grad():
                reconstructed_condition = vae.decode(condition_images_eval_reshaped.to(torch.float32)) #input should be [batch*channel,4,64,64]
                #take the mean across RGB 
                reconstructed_condition = reconstructed_condition.sample.mean(axis=1) #reconstructed should be [batch*channel,3,256,256]
                
                reconstructed_label = vae.decode(clean_images_eval_reshaped.to(torch.float32)) #input should be [batch*channel,4,64,64]
                #take the mean across RGB 
                reconstructed_label = reconstructed_label.sample.mean(axis=1) #reconstructed should be [batch*channel,3,256,256]
                
            #put it back to the normal shape 
            reconstructed_condition_unshaped = torch.reshape(reconstructed_condition,[condition_images_eval.shape[0],2,256,256])
            reconstructed_label_unshaped = torch.reshape(reconstructed_label,[condition_images_eval.shape[0],1,256,256])
            
            for i in np.arange(0,len(config.images_idx)):
                #reshape image for adding a color image to the tensorboard 
                image = reconstructed_condition_unshaped[config.images_idx[i],0:1].squeeze(0).unsqueeze(-1).cpu()
                #add color and convert shapes back. 
                color_image = torch.tensor(colorize(image,vmin=-1,vmax=1,cmap='Spectral_r')).permute(2, 0, 1)

                #write the image to the tensorboard 
                writer.add_image("Example {}".format(i), color_image, 0)

                #do the same thing for the second image. 
                image = reconstructed_condition_unshaped[config.images_idx[i],1:2].squeeze(0).unsqueeze(-1).cpu()
                color_image = torch.tensor(colorize(image,vmin=-1,vmax=1,cmap='Spectral_r')).permute(2, 0, 1)
                writer.add_image("Example {}".format(i), color_image, 1)

                #do the same thing for 'truth'
                image = reconstructed_label_unshaped[config.images_idx[i],0:1].squeeze(0).unsqueeze(-1).cpu()
                color_image = torch.tensor(colorize(image,vmin=-1,vmax=1,cmap='Spectral_r')).permute(2, 0, 1)
                writer.add_image("Example {}".format(i), color_image, 2)
        
        #split it 
        splits = torch.utils.data.random_split(dataset, [0.8,0.2]) 

        ds_train = splits[0]
        ds_val = splits[1]

        #throw it in a dataloader for fast CPU handoffs. 
        #Note, you could add preprocessing steps with image permuations here i think 
        train_dataloader = torch.utils.data.DataLoader(ds_train, batch_size=config.train_batch_size, shuffle=True,num_workers=8)

        val_dataloader = torch.utils.data.DataLoader(ds_val, batch_size=config.val_batch_size, shuffle=False)

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
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    #initalize early stopping stuff
    best_loss = float('inf') 
    best_model_state_dict = None
    no_improvement_count = 0
    loss_history = [] 
    
    
    #define loss, you can change the sigma vals here (i.e., hyperparameters), change them in the config class
    #loss_fn = EDMLoss(P_mean=config.P_mean,P_std=config.P_std,sigma_data=config.sigma_data)
    
    loss_fn = torch.nn.MSELoss(reduction='none')
    loss_fn_val = torch.nn.MSELoss(reduction='none') 
    
    # Now you train the model
    for epoch in range(config.num_epochs):
        #put model in training mode
        model.train()
        
        #this is for the cmd line
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        #initalize loss to keep track of the mean loss across all batches in this epoch 
        epoch_loss = torch.tensor(0.0, device=accelerator.device)
        
        for step, batch in enumerate(train_dataloader):
            
            #my data loader returns [clean_images,condition_images],I seperate them here just to be clear 
            # Sep. label 
            clean_images = batch[0].to(torch.float32)*vae.config.scaling_factor

            #Sep. conditions
            condition_images = batch[1].to(torch.float32)*vae.config.scaling_factor
            
            #this is the autograd steps within the .accumulate bit (this is important for multi-GPU training)
            with accelerator.accumulate(model):
                
                yhat = model(condition_images,torch.zeros(condition_images.shape[0]).to(condition_images.device),
                             class_labels=torch.zeros(condition_images.shape[0]).to(torch.int).to(condition_images.device),
                             return_dict=False)[0]
                
                #send data into loss func and get the loss (the model call is in here)
                loss = loss_fn(clean_images.to(torch.float64),yhat.to(torch.float64)).mean()
                
                #calc backprop 
                accelerator.backward(loss)
                
                #this is needed to enable gradient accumulation for some reason 
                if accelerator.sync_gradients:
                    #clip gradients (leftover from Butterflies example)
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                #step 
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            
            # Accumulate epoch loss on each GPU seperately 
            epoch_loss += loss.detach() #this is the mean squarred error across the batch 
            
            #log things on tensorboard 
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
            

        # Synchronize epoch loss across devices, this will just concat the two 
        epoch_loss = accelerator.gather(epoch_loss)

        # Sum up the losses across all GPUs
        total_epoch_loss = epoch_loss.sum() #sum of all the means

        # the batches are split from the train_dataloader to each GPU
        total_samples_processed = len(train_dataloader) * accelerator.num_processes 

        # Calculate mean epoch loss by dividing by the total number of batches proccessed 
        mean_epoch_loss = total_epoch_loss / total_samples_processed #mean of all the batches 
        
        #put model in eval mode
        model.eval()
        
        #this is for the cmd line
        progress_bar2 = tqdm(total=len(val_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar2.set_description(f"Validation Steps")
        
        #No need to track gradients here, ideally this should be just on one GPU... 
        with torch.no_grad():
            #initalize validation total mean loss 
            val_loss = torch.tensor(0.0, device='cuda:0')
            for step, batch in enumerate(val_dataloader):
                #my data loader returns [clean_images,condition_images],I seperate them here just to be clear 
                # Sep. label 
                clean_images = batch[0].to(torch.float32).to('cuda:0')*vae.config.scaling_factor

                #Sep. conditions
                condition_images = batch[1].to(torch.float32).to('cuda:0')*vae.config.scaling_factor
            
                yhat = model(condition_images,torch.zeros(condition_images.shape[0]).to(condition_images.device),
                             class_labels=torch.zeros(condition_images.shape[0]).to(torch.int).to(condition_images.device),
                             return_dict=False)[0]
                
                #send data into loss func and get the loss (the model call is in here)
                loss = loss_fn_val(clean_images,yhat).mean()
                
                val_loss += loss.detach()
                
                #log things on tensorboard 
                progress_bar2.update(1)
                logs = {"loss": loss.detach().item()}
                progress_bar2.set_postfix(**logs)

        # Sum up the losses across all GPUs
        mean_val_loss = val_loss/len(val_dataloader) #mean of the mses 
        
        # Print or log the average epoch loss, need to convert to scalar to get tensorboard to work (using .item())
        logs = {"epoch_loss": mean_epoch_loss.item(), "epoch": epoch,"val_loss":mean_val_loss.item()}
        
        accelerator.log(logs, step=epoch)

        #accumulate rolling mean 
        loss_history.append(mean_val_loss.item())
        
        # Calculate the moving average if enough epochs have passed
        if len(loss_history) >= config.window_size:
            moving_average = sum(loss_history[-config.window_size:]) / config.window_size
            logs = {"moving_epoch_loss": moving_average, "epoch": epoch}
            accelerator.log(logs, step=epoch)

            # Check for improvement in the moving_average
            if moving_average < (best_loss - config.min_delta):
                best_loss = moving_average
                no_improvement_count = 0
                # Store the best model's state dict
                best_model_state_dict = accelerator.unwrap_model(model).state_dict()
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
#                         images_batch = edm_sampler(model,latents,condition_images_eval*vae.config.scaling_factor,num_steps=18)

                        images_batch = model(condition_images_eval*vae.config.scaling_factor,
                                             torch.zeros(condition_images_eval.shape[0]).to(condition_images_eval.device),
                                             class_labels=torch.zeros(condition_images_eval.shape[0]).to(torch.int).to(condition_images_eval.device),
                                             return_dict=False)[0]
                        
                        images_batch = images_batch*(1/vae.config.scaling_factor)
                        #reshape images for the decoding 
                        images_batch_reshaped = torch.reshape(images_batch,
                                                                       [images_batch.shape[0]*1,
                                                                        vae.config.latent_channels,
                                                                        images_batch.shape[2],
                                                                        images_batch.shape[3]])
            
                        #check recons. just in case, trust issues 
                        with torch.no_grad():
                            reconstructed_images_batch = vae.decode(images_batch_reshaped.to(torch.float)) #input should be [batch*channel,4,64,64]
                            #take the mean across RGB 
                            reconstructed_images_batch = reconstructed_images_batch.sample.mean(axis=1) #reconstructed should be [batch*channel,3,256,256]

                        #put it back to the normal shape 
                        reconstructed_images_batch_unshaped = torch.reshape(reconstructed_images_batch,[images_batch.shape[0],1,256,256])
            
                        for i in np.arange(0,len(config.images_idx)):

                            #reshape the image so we can add color to the tensorboard
                            image = reconstructed_images_batch_unshaped[config.images_idx[i]].squeeze(0).unsqueeze(-1).cpu().numpy()
                            #colorize the image 
                            color_image = torch.tensor(colorize(image,vmin=-1,vmax=1,cmap='Spectral_r')).permute(2, 0, 1)

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
                
            #save the best model
            if best_model_state_dict is not None:
                torch.save(best_model_state_dict, config.output_dir + "best_model.pth")
                
            break
        #run some cleanup, because leaks             
        gc.collect()
        torch.cuda.empty_cache()

import matplotlib
import matplotlib.cm

def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    from here: https://gist.github.com/jimfleming/c1adfdb0f526465c99409cc143dea97b 
    
    A utility function for Torch/Numpy that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: Matplotlib default colormap)
    
    Returns a 4D uint8 tensor of shape [height, width, 4].
    """

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    value = value.squeeze()

    cmapper = matplotlib.colormaps.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)
    return value

def save_checkpoint(model, optimizer, lr_scheduler, epoch, step, accelerator, checkpoint_path):
    """ A function from chatGPT to help checkpoint out things for training restarts """
    checkpoint = {
        'model_state_dict': accelerator.unwrap_model(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'epoch': epoch,
        'step': step,
    }
    if accelerator.is_main_process:
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}, step {step}.")

def load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler, accelerator):
    """ A function from chatGPT to help load checkpoints training restarts """ 
    checkpoint = torch.load(checkpoint_path, map_location=accelerator.device)
    accelerator.unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    print(f"Checkpoint loaded. Resuming from epoch {epoch}, step {step}.")
    return epoch, step

# ################### \Funcs ########################


#initalize config 
config = TrainingConfig()

# Initialize the dataset
dataset = ZarrDataset(config.dataset_path)

# go ahead and build a UNET, this was the exact same as the butterfly example, but different channels. This is a big model.. 
model = DiTTransformer2DModel(in_channels=8,  out_channels=4,num_layers=6,patch_size=2)

#left this the same as the butterfly example 
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(dataset.length * config.num_epochs),
)

#main method here! 
train_loop(config, model, optimizer, dataset, lr_scheduler)
