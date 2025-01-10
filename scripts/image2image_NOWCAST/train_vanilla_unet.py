import torch
import zarr
from torch.utils.data import Dataset
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DModel
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
# from huggingface_hub import HfFolder, Repository, whoami #maybe i can delete this 
from tqdm.auto import tqdm
from pathlib import Path
import os
import math
from typing import List, Optional, Tuple, Union
from diffusers.utils.torch_utils import randn_tensor
from torch.utils.tensorboard import SummaryWriter

class ZarrDataset(Dataset):
    """
    This is a new zarr instance of the dataset that loads all data into CPU memory to minimize I/O overhead.
    """
    def __init__(self, zarr_store):
        self.store = zarr_store
        self.data = zarr.open(self.store, mode='r')
        
        # Load data into CPU memory
        self.input_images = torch.tensor(self.data['input_images'][:], dtype=torch.float16, device='cpu')
        self.output_images = torch.tensor(self.data['output_images'][:], dtype=torch.float16, device='cpu')
        self.length = self.input_images.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Access data from memory (still on the CPU)
        return self.output_images[idx], self.input_images[idx]

# Initialize the dataset
zarr_store = '/home/rchas1/diffusion_10_4_2inputs_v3_gh200.zarr'
dataset = ZarrDataset(zarr_store)

# ################### \Imports ########################


# ################### Classes ########################

@dataclass
class TrainingConfig:
    """ This should be probably in some sort of config file, but for now its here... """
    image_size = 256  
    train_batch_size = 45 #need to check if this can fit...
    val_batch_size = 45
    num_epochs = 210 # 
    gradient_accumulation_steps = 1 #shouldnt need this if i am not adding noise 
    learning_rate = 1e-4 
    lr_warmup_steps = 500
    save_model_epochs = 1 
    mixed_precision = "fp16" 
    output_dir = "/mnt/data1/rchas1/vanilla_unet_10_two_inputs_v2/"  # the local path to store the model 
    push_to_hub = False
    hub_private_repo = False
    overwrite_output_dir = True  
    seed = 0 

# ################### \Classes ########################

# ################### Funcs ########################

def train_loop(config, model, optimizer, train_dataloader, lr_scheduler, val_dataloader):
    """ 
    This is the main show! the training loop 
    """
    
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_name = get_full_repo_name(Path(config.output_dir).name)
            repo = Repository(config.output_dir, clone_from=repo_name)
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

        #Stuff to have an image show up 
        writer = SummaryWriter("/mnt/data1/rchas1/vanilla_unet_10_two_inputs_v2/logs/images")
                #define random noise/seed vectors, here is enough seeds to run one batch of data through (i.e., one image per batch)

        #grab a batch of data 
        for step, batch in enumerate(train_dataloader):
                    # Sep. label 
                    clean_images_eval = batch[0].to('cuda')

                    #Sep. conditions
                    condition_images_eval = batch[1].to('cuda')
                    break 
                    
        #colorize an image 
        image = condition_images_eval[0,0:1].squeeze(0).unsqueeze(-1).cpu()
        color_image = torch.tensor(colorize(image,vmin=-4,vmax=2,cmap='Spectral_r')).permute(2, 0, 1)
        
        #add in the input images
        writer.add_image("Training Data", color_image, 0)
        
        #colorize an image 
        image = condition_images_eval[0,1:2].squeeze(0).unsqueeze(-1).cpu()
        color_image = torch.tensor(colorize(image,vmin=-4,vmax=2,cmap='Spectral_r')).permute(2, 0, 1)
        writer.add_image("Training Data", color_image, 1)


        #colorize an image 
        image = clean_images_eval[0,0:1].squeeze(0).unsqueeze(-1).cpu()
        color_image = torch.tensor(colorize(image,vmin=-4,vmax=2,cmap='Spectral_r')).permute(2, 0, 1)
        writer.add_image("Training Data", color_image, 2)
        writer.add_image("Slider", color_image, 0)
        
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    #iterator to see how many gradient steps have been done
    global_step = 0
    
    # Define parameters for early stopping, TODO: this needs to be in the config step 
    patience = 10  # Number of epochs to wait for improvement
    min_delta = 1e-6  # Minimum change in loss to be considered as improvement
    best_loss = float('inf') #fill with inf to start 
    no_improvement_count = 0
    window_size = 5  # Define the window size for the moving average
    loss_history = [] # Initialize a list to store the recent losses
    
    #define loss 
    loss_fn = torch.nn.MSELoss(reduction='none')
    
     
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
            clean_images = batch[0]

            #Sep. conditions
            condition_images = batch[1]
            
            #this is the autograd steps within the .accumulate bit (this is important for multi-GPU training)
            with accelerator.accumulate(model):
                
                yhat = model(condition_images,torch.zeros(condition_images.shape[0]).to(condition_images.device), return_dict=False)[0]
                
                #send data into loss func and get the loss (the model call is in here)
                loss = loss_fn(clean_images.to(torch.float),yhat.to(torch.float)).mean()
                
                #calc backprop 
                accelerator.backward(loss)
                
                #step 
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            
            # Accumulate epoch loss on each GPU seperately 
            epoch_loss += loss.detach()
            
            #log things on tensorboard 
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
            

        # Synchronize epoch loss across devices, this will just concat the two 
        epoch_loss = accelerator.gather(epoch_loss)

        # Sum up the losses across all GPUs
        total_epoch_loss = epoch_loss.sum()

        # the batches are split from the train_dataloader to each GPU
        total_samples_processed = len(train_dataloader) * accelerator.num_processes

        # Calculate mean epoch loss by dividing by the total number of batches proccessed 
        mean_epoch_loss = total_epoch_loss / total_samples_processed
        
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
                clean_images = batch[0].to('cuda:0')

                #Sep. conditions
                condition_images = batch[1].to('cuda:0')
            
                yhat = model(condition_images,torch.zeros(condition_images.shape[0]).to(condition_images.device), return_dict=False)[0]
                
                #send data into loss func and get the loss (the model call is in here)
                loss = loss_fn(clean_images,yhat).mean()
                
                val_loss += loss.detach()
                
                #log things on tensorboard 
                progress_bar2.update(1)
                logs = {"loss": loss.detach().item()}
                progress_bar2.set_postfix(**logs)

        # Sum up the losses across all GPUs
        mean_val_loss = val_loss.mean()
        
        # Print or log the average epoch loss, need to convert to scalar to get tensorboard to work (using .item())
        logs = {"epoch_loss": mean_epoch_loss.item(), "epoch": epoch,"val_loss":mean_val_loss.item()}
        
        accelerator.log(logs, step=epoch)

        #accumulate rolling mean 
        loss_history.append(mean_val_loss.item())
        
        # Calculate the moving average if enough epochs have passed
        if len(loss_history) >= window_size:
            moving_average = sum(loss_history[-window_size:]) / window_size
            logs = {"moving_epoch_loss": moving_average, "epoch": epoch}
            accelerator.log(logs, step=epoch)

            # Check for improvement in the moving_average
            if moving_average < (best_loss - min_delta):
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
                    #need to grab the unwrapped diffusers model from EDMPrecond 
                    accelerator.unwrap_model(model).save_pretrained(config.output_dir)
                    
                    with torch.no_grad():
                        #run a batch of images through
                        images_batch = model(condition_images,torch.zeros(condition_images.shape[0]).to(condition_images.device), return_dict=False)[0]
                    
                    #colorize the image 
                    image = images_batch[0].squeeze(0).unsqueeze(-1).cpu().numpy()
                    color_image = torch.tensor(colorize(image,vmin=-4,vmax=2,cmap='Spectral_r')).permute(2, 0, 1)
                    
                    #add static example here, generate one image, add to board
                    writer.add_image("Output Image", color_image, epoch)

                    #write the latest to the board so we can slide between the truth and this image 
                    writer.add_image("Slider", color_image, 1)
                    
                    #add the original image again so the slider updates?
                    image = clean_images_eval[0,0:1].squeeze(0).unsqueeze(-1).cpu()
                    color_image = torch.tensor(colorize(image,vmin=-4,vmax=2,cmap='Spectral_r')).permute(2, 0, 1)
                    writer.add_image("Slider", color_image, 0)

        # Check if training should be stopped due to lack of improvement
        if no_improvement_count >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
                # Check if multi-GPU is being used
            if accelerator.num_processes > 1:
                print(f"Killing multi-GPU processes using dist.barrier() and dist.destroy_process_group()")
                # Signal all processes to stop
                dist.barrier()  # Ensure all processes are synchronized
                dist.destroy_process_group() 
                
            break
                    
        gc.collect()
        
import matplotlib
import matplotlib.cm

def colorize(value, vmin=None, vmax=None, cmap=None):
    """
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

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)
    return value

# ################### \Funcs ########################

# ################### CODE ########################

#initalize config 
config = TrainingConfig()

#split it 
splits = torch.utils.data.random_split(dataset, [0.8,0.2]) 

ds_train = splits[0]
ds_val = splits[1]

#throw it in a dataloader for fast CPU handoffs. 
#Note, you could add preprocessing steps with image permuations here i think 
train_dataloader = torch.utils.data.DataLoader(ds_train, batch_size=config.train_batch_size, shuffle=True)

val_dataloader = torch.utils.data.DataLoader(ds_val, batch_size=config.val_batch_size, shuffle=False)

# go ahead and build a UNET, this was the exact same as the butterfly example, but different channels. This is a big model.. 
model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=2,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

#left this the same as the butterfly example 
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)


#main method here! 
train_loop(config, model, optimizer, train_dataloader, lr_scheduler,val_dataloader)
