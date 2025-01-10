from dataclasses import dataclass

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
    output_dir = "/mnt/data1/rchas1/edm_10_two_inputs/"  # the local path to store the model 
    push_to_hub = False 
    hub_private_repo = False
    overwrite_output_dir = True  
    seed = 0 
    
    #loss params (defaults to the edm paper)
    P_mean=-1.2
    P_std=1.2
    sigma_data=0.5
    
    #early stopping things 
    patience = 100  # Number of epochs to wait for improvement
    min_delta = 1e-6  # Minimum change in loss to be considered as improvement
    window_size = 5  # Define the window size for the moving average