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


# ################### \Imports ########################


# ################### Classes ########################

@dataclass
class TrainingConfig:
    """ This should be probably in some sort of config file, but for now its here... """
    #data things 
    image_size = 256  #this assumes square images
    train_batch_size = 45 #this is as big as I can fit on the GH200 [45,3,256,256]
    
    #training things 
    num_epochs = 1000 #this should be similar to NVIDIA's StormCast 
    gradient_accumulation_steps = 2 #this helped with stability, i think... 
    learning_rate = 1e-4 #default value from butterflies example
    lr_warmup_steps = 500 #default value from butterflies example
    save_model_epochs = 1 #i like to save alot, doesnt cost much 
    mixed_precision = "fp16"
    output_dir = "/mnt/data1/rchas1/edm_10_two_inputs_DiT/"  # the local path to store the model 
    push_to_hub = False 
    hub_private_repo = False
    overwrite_output_dir = True  
    seed = 0 
    restart = False #do you want to start from a previous training?
    restart_path = "/mnt/data1/rchas1/edm_10_two_inputs_DiT/"
    dataset_path = "/mnt/data1/rchas1/diffusion_10_4_2inputs_v2_gh200.zarr"
    
    #tensorboard things 
    plot_images = True 
    images_idx = [1,3,5,7] #these need to be smaller than train_batch_size 
    
    #loss params (defaults to the edm paper)
    P_mean=-1.2
    P_std=1.2
    sigma_data=0.5
    
    #early stopping things 
    patience = 100  # Number of epochs to wait for improvement
    min_delta = 1e-6  # Minimum change in loss to be considered as improvement
    window_size = 5  # Define the window size for the moving average

    


class EDMPrecond(torch.nn.Module):
    """ Original Func:: https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/networks.py#L519
    
    This is a wrapper for your diffusers model. It's purpose is to apply the preconditioning that is talked about in Karras et al. (2022)'s EDM paper. 
    
    I've made some changes for the sake of conditional-EDM (the original paper is unconditional).
    
    """
    def __init__(self,
        generation_channels,                # number of channels you want to generate
        model,                              # pytorch model from diffusers 
        use_fp16        = True,             # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data. this was the default from above
    ):
        super().__init__()
        self.generation_channels = generation_channels
        self.model = model
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        
    def forward(self, x, sigma, force_fp32=False, **model_kwargs):
        
        """ 
        
        This method is to 'call' the neural net. But this is the preconditioning from the Karras EDM paper. 
        
        note for conditional, it expects x to have the condition in the channel dim (dim=1). and the images you want to generate should already have noise.
        
        x: input stacked image with the generation images stacked with the condition images [batch,generation_channels + condition_channels,nx,ny]
        sigma: the noise level of the images in batch [??]
        force_fp32: this is forcing calculations to be a certain percision. 
        
        """
        
        #for the calculations, use float 32
        x = x.to(torch.float32)
        #reshape sigma from _ to _ 
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        
        #forcing dtype matching
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        
        #get weights from EDM 
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        # split out the images you want to generate and the condition, because the scaling will depend on this. 
        x_noisy = torch.clone(x[:,0:self.generation_channels])
        
        #the condition
        x_condition = torch.clone(x[:,self.generation_channels:])

        
        #concatinate back with the scaling applied to only the the generation dimension (x_noisy)
        model_input_images = torch.cat([x_noisy*c_in, x_condition], dim=1)
        
        #denoise the image (e.g., run it through your diffusers model) 
        F_x = self.model((model_input_images).to(dtype), c_noise.flatten(),
                         class_labels=torch.zeros(model_input_images.shape[0]).to(torch.int).to(model_input_images.device),
                         return_dict=False)[0]
        
        #force dtype
        assert F_x.dtype == dtype
        
        #apply additional scalings: make sure you apply skip just to the generation dim (x[:,0:generation_channel]) and NOT applied to (x*c_in)
        D_x = c_skip * x_noisy + c_out * F_x.to(torch.float32)
        
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

class EDMLoss:
    
    """Original Func:: https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/loss.py
    
    This is the loss function class from Karras et al. (2022)'s EDM paper. Only thing changed here is that the __call__ takes the clean_images and the condition_images seperately. It expects your model to be wrapped with that EDMPrecond class. 
    
    """
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        """ These describe the distribution of sigmas we should sample during training """
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, clean_images, condition_images, labels=None, augment_pipe=None):
        
        """ 
        
        net: is a pytorch model wrapped with EDMPrecond
        clean_images: the images you want to generate, [batch,generation_channels,nx,ny]
        condition_images:images you want to condition with [batch,condition_channels,nx,ny]
        
        """
        
        #get random seeds, one for each image in the batch 
        rnd_normal = torch.randn([clean_images.shape[0], 1, 1, 1], device=clean_images.device)
        
        #get random noise levels (sigmas)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        
        #get the loss weight for those sigmas 
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        #make the noise scalars images so we can add them to our images
        n = torch.randn_like(clean_images) * sigma
    
        #add noise to the clean images 
        noisy_images = torch.clone(clean_images + n)
        
        #cat the images for the wrapped model call 
        model_input_images = torch.cat([noisy_images, condition_images], dim=1)
        
        #call the EDMPrecond model 
        denoised_images = net(model_input_images, sigma)
        
        #calc the weighted loss at each pixel, the mean across all GPUs and pixels is in the main train_loop 
        loss = weight * ((denoised_images - clean_images) ** 2)
        
        return loss

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
            
            for i in np.arange(0,len(config.images_idx)):
                #reshape image for adding a color image to the tensorboard 
                image = condition_images_eval[config.images_idx[i],0:1].squeeze(0).unsqueeze(-1).cpu()
                #add color and convert shapes back. 
                color_image = torch.tensor(colorize(image,vmin=-1,vmax=1,cmap='Spectral_r')).permute(2, 0, 1)

                #write the image to the tensorboard 
                writer.add_image("Example {}".format(i), color_image, 0)

                #do the same thing for the second image. 
                image = condition_images_eval[config.images_idx[i],1:2].squeeze(0).unsqueeze(-1).cpu()
                color_image = torch.tensor(colorize(image,vmin=-1,vmax=1,cmap='Spectral_r')).permute(2, 0, 1)
                writer.add_image("Example {}".format(i), color_image, 1)

                #do the same thing for 'truth'
                image = clean_images_eval[config.images_idx[i],0:1].squeeze(0).unsqueeze(-1).cpu()
                color_image = torch.tensor(colorize(image,vmin=-1,vmax=1,cmap='Spectral_r')).permute(2, 0, 1)
                writer.add_image("Example {}".format(i), color_image, 2)
        
        #properly setup the dataset now with shuffling 
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True,num_workers=8)
        
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

                        for i in np.arange(0,len(config.images_idx)):

                            #reshape the image so we can add color to the tensorboard
                            image = images_batch[config.images_idx[i]].squeeze(0).unsqueeze(-1).cpu().numpy()
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
            break
        #run some cleanup, because leaks             
        gc.collect()

def edm_sampler(net, latents, condition_images, randn_like=torch.randn_like,num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    """ adapted from: https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/generate.py 
    
    only thing i had to change was provide a condition as input to this func, then take that input and concat with generated image for the model call. 
    
    net: expects a wrapped diffusers model with the EDMPrecond
    latents: a noise seed with the same shape as condition_images
    condition_images: the condition, [batch or ens_size,condition_channels,nx,ny]
    randn_like: how to generate randomness
    num_steps: the number of generation steps you want to take (note model calls are ~2x this because the second order correction)
    sigma_min: smallest amount of noise 
    sigma_max: largest amount of noise 
    rho: related to the step size with time ??? 
    S_churn: how much stocasisty you want to add to the process 
    S_min: min sigma step of when to add the stocastic bit 
    S_max: max sigma step of when to add the stocastic bit  
    S_noise: scale to the noise we add in the stocastic bit 
    
    """
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        #need to concat the condition here 
        model_input_images = torch.cat([x_hat, condition_images], dim=1)
        # Euler step.
        with torch.no_grad():
            denoised = net(model_input_images, t_hat).to(torch.float64)

        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            model_input_images = torch.cat([x_next, condition_images], dim=1)
            with torch.no_grad():
                denoised = net(model_input_images, t_next).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

class StackedRandomGenerator:  # pragma: no cover
    """
    adapted from: https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/generate.py 
    
    Wrapper for torch.Generator that allows specifying a different random seed
    for each sample in a minibatch.
    """

    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        if size[0] != len(self.generators):
            raise ValueError(
                f"Expected first dimension of size {len(self.generators)}, got {size[0]}"
            )
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )

    def randn_like(self, input):
        return self.randn(
            input.shape, dtype=input.dtype, layout=input.layout, device=input.device
        )

    def randint(self, *args, size, **kwargs):
        if size[0] != len(self.generators):
            raise ValueError(
                f"Expected first dimension of size {len(self.generators)}, got {size[0]}"
            )
        return torch.stack(
            [
                torch.randint(*args, size=size[1:], generator=gen, **kwargs)
                for gen in self.generators
            ]
        )

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
        'model_state_dict': accelerator.unwrap_model(model).model.state_dict(),
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
    accelerator.unwrap_model(model).model.load_state_dict(checkpoint['model_state_dict'])
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

model = DiTTransformer2DModel(in_channels=3,  out_channels=1,num_layers=6,patch_size=8)

#wrap diffusers/pytorch model 
model_wrapped = EDMPrecond(1,model)

#left this the same as the butterfly example 
optimizer = torch.optim.AdamW(model_wrapped.model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(dataset.length * config.num_epochs),
)

#main method here! 
train_loop(config, model_wrapped, optimizer, dataset, lr_scheduler)
