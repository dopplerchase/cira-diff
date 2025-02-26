""" 
Python file that houses all the NVIDIA EDM code with some small adaptations to run on conditional images

Most of it is from here: https://github.com/NVlabs/edm/

""" 

import torch 

class EDMPrecond(torch.nn.Module):
    """ Original Func:: https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/networks.py#L519
    
    This is a wrapper for your pytorch model. It's purpose is to apply the preconditioning that is talked about in Karras et al. (2022)'s EDM paper. 
    
    I've made some changes for the sake of conditional-EDM (the original paper is unconditional).

    Example use: 

    model_wrapper = EDMPrecond(generation_channels=3, model=model, use_fp16=True, sigma_min=0.002, sigma_max=80, sigma_data=0.5)

    """
    def __init__(self,
        generation_channels,                # number of channels you want to GENERATE, this is the number of channels that will be denoised
        model,                              # pytorch model
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
        
        note for conditional, it expects x to have the condition in the channel dim (axis=1). and the images you want to generate should already have noise.
        
        x: input stacked image with the generation images stacked (axis=0) with the condition images [batch,generation_channels + condition_channels,nx,ny]
        sigma: the noise level of the images in batch [??]
        force_fp32: this is forcing calculations to be a certain percision. 
        
        """
        
        #for the calculations, use float 32
        x = x.to(torch.float32)
        #reshape sigma from _ to _ 
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        
        #forcing dtype matching
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        
        #get weights from Karras et al. 2022 EDM 
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        # split out the images you want to generate and the condition, because the scaling will depend on this. 
        x_noisy = torch.clone(x[:,0:self.generation_channels])
        
        #the condition
        x_condition = torch.clone(x[:,self.generation_channels:])

        #concatinate back with the scaling applied to ONLY the the generation dimension (x_noisy)
        model_input_images = torch.cat([x_noisy*c_in, x_condition], dim=1)
        
        #denoise the image (e.g., run it through your pytorch model), the model here expects 2 inputs, the images and the noise. This is following Diffusers
        F_x = self.model((model_input_images).to(dtype), c_noise.flatten(), return_dict=False)[0]
        
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
        """ These describe the distribution of sigmas we should sample during training. The default values are from Karras et al. 2022 and worked for us."""
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, clean_images, condition_images, labels=None, augment_pipe=None):
        
        """ 
        
        net: is a pytorch model wrapped with EDMPrecond (see above)
        clean_images: tensor of images you want to generate, [batch,generation_channels,nx,ny] 
        condition_images: tensor of images you want to condition with [batch,condition_channels,nx,ny]
        
        """
        
        #get random seeds, one for each image in the batch, make sure its on the device. 
        rnd_normal = torch.randn([clean_images.shape[0], 1, 1, 1], device=clean_images.device)
        
        #get random noise levels (sigmas) based on the prescribed distribution.
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
        
        #calc the weighted loss (MSE) at each pixel, the mean across all GPUs and pixels is in the main train_loop
        loss = weight * ((denoised_images - clean_images) ** 2)
        
        return loss
    
def edm_sampler(net, latents, condition_images, randn_like=torch.randn_like,num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    """ adapted from: https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/generate.py 
    
    only thing I had to change was provide a condition as input to this func, then take that input and concat with generated image for the model call. 
    
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