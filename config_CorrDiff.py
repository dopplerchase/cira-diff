from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class TrainingConfig:
    """ This is the config file used for training a CorrDiff style diffusion model """

    #data things 
    image_size: int = 256  #this assumes square images
    train_batch_size: int = 45 #this is as big as I can fit on the GH200 [45,3,256,256]
    
    #training things 
    num_epochs: int = 1000 #this should be similar to NVIDIA's StormCast , about 30M steps (well 15M because gradient_accumulation_steps=2)
    gradient_accumulation_steps: int = 2 #this helped with stability, i think... 
    learning_rate: float = 1e-4 #default value from butterflies example 
    llr_warmup_steps: int = 500 #default value from butterflies example
    save_model_epochs: int = 1 #i like to save alot, doesnt cost much time. But this will overwrite, not write a new chkp for every file.
    mixed_precision: str = "fp16"
    output_dir: str = "/mnt/data1/rchas1/CODE_REFACTOR_EDM/"  # the local path to store the model 
    push_to_hub: bool = False 
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True  
    seed: int = 0 
    restart: bool = False #do you want to start from a previous training?
    restart_path: str = '/PATH/'
    dataset_path: str = "/mnt/data1/rchas1/diffusion_10_4_2inputs_v3_gh200_CorrDiff.zarr"
    
    #tensorboard things 
    plot_images: bool = True 
    images_idx: List[int] = field(default_factory=lambda: [3, 5, 10, 15]) #these need to be smaller than train_batch_size 
    use_tensorboard: bool = True
    
    # Loss parameters
    P_mean: float = -1.2
    P_std: float = 1.2
    sigma_data: float = 0.5
    
    #early stopping things 
    early_stopping: bool = True 
    patience: int = 100  # Number of epochs to wait for improvement
    min_delta: float = 1e-6  # Minimum change in loss to be considered as improvement
    window_size: int = 5  # Define the window size for the moving average

    #model things 
    # check out https://huggingface.co/docs/diffusers/main/en/api/models/overview, for specific models and kwargs. 
    #btw, this is the default model from the butterflies example with the channels changed to our forecasting task
    #Note: ONLY UNET2DModel is supported. Check back for more in additional releases.
    model_type: str = "UNet2DModel"  # Default model type
    in_channels: int = 4  # Noisy channels + condition channels, for CorrDiff is 1 noisy, 2 past frames, 1 unet prediction. 
    out_channels: int = 1   #how many things you want to generate, for ours is the residual btw the unet and the real 10 min forecast 
    layers_per_block: int = 2 
    block_out_channels: Tuple[int, ...] = (128, 128, 256, 256, 512, 512)
    down_block_types: Tuple[str, ...] = (
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    )
    up_block_types: Tuple[str, ...] = (
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )

    #hardware things NEED TO CHANGE THIS TO WORK WITH MORE THAN 1 GPU
    gpu_id_selection: List[int] = field(default_factory=lambda: [1])


    