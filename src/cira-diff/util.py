import matplotlib
import matplotlib.cm
import os 
import torch 
from diffusers import UNet2DModel

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

def worker_init_fn(worker_id):
    os.sched_setaffinity(0, range(os.cpu_count())) 

def load_config(config_path):
    """Dynamically load a Python file containing a dataclass."""
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Check for the TrainingConfig dataclass
    if not hasattr(config_module, "TrainingConfig"):
        raise AttributeError("The config file must define a dataclass named 'TrainingConfig'")
    
    # Instantiate the dataclass and return it
    return config_module.TrainingConfig()

def build_model(config: TrainingConfig):
    model_class = MODEL_CLASSES.get(config.model_type)
    if model_class is None:
        raise ValueError(f"Unsupported model type: {config.model_type}")

    model = model_class(
        sample_size=config.image_size,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        layers_per_block=config.layers_per_block,
        block_out_channels=config.block_out_channels,
        down_block_types=config.down_block_types,
        up_block_types=config.up_block_types
        # Add other parameters as required by the specific model
    )
    return model


MODEL_CLASSES = {
    "UNet2DModel": UNet2DModel,
}