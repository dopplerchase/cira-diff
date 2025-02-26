# CIRA-Diffusion
main contact: Randy Chase 
email: randy 'dot' chase 'at' colostate.edu

## Introduction 

![Alt Text](./aux/movies/output.gif)
This repository is to hold the code for the diffusion model efforts at CIRA-CSU. The first couple projects for us are to do are conditional diffusion models to do image2image translation using satellite data. Specifcally, we began by forecasting GOES IR brightness temperatures out the 3 hours and the conversion of GOES data to passive microwave observations. The paper is titled: [Score based diffusion nowcasting of GOES imagery](), and an example forecast is included in the gif above. More projects to come.

## Background and our journey
When we first stared on learning how to encorperate diffusion models into our workflow, we started with [this](https://huggingface.co/docs/diffusers/en/tutorials/basic_training) example from HuggingFace that trains an <i> unconditional </i> diffusion model that generates pictures of butterflies using Denoising Diffusion Probabilistic Models ([DDPM](https://arxiv.org/abs/2006.11239)). This was a useful place to start, but with most meteorology/atmos tasks, conditional modeling we find is much more useful. To include a condition we found it useful to concatenate your condition alongside the noisy dimension (see our paper for discussion). This worked following DDPM, but with the build in DDPM sampler, it was requiring something like 1000 neural network calls to get decent data. This was just too computationally expensive to get into any operational environment. 

We then moved on to following the work out of Google and NVIDIA, where they both closely follow the [Karras et al. (2022)](https://arxiv.org/abs/2206.00364) titled: Elucidating the Design Space of Diffusion-Based Generative Models (hereafter EDM). The key advantages we found of following the EDM approach over DDPM:

1) Calls to the network are less than 100 for <i> good </i> performance 
2) training was relatively easy and stable (just long....)
3) more advancement coming out of the NVIDIA group ([Karras et al. 2024a](https://arxiv.org/abs/2312.02696),[Karras et al. 2024b](https://arxiv.org/abs/2406.02507)). 
4) NVIDIA had the code already implemented in MODULUS and used it for CorrDiff/StormCast 

Our implementation of the code comes directly out of the [original repo](https://github.com/NVlabs/edm), not MODULUS because of the bloat with modulus (i.e., we don't need all their functionality). Turns out though that MODULUS also took the main code from EDM and wrapped it with the rest of their repo. We only grab the train/generate code out of the EDM repo, and then we leverage HuggingFace's [diffusers]() as our architecture hub so we could play around with various <i> drop in </i> architectures. 

Eventually, this repo will be more generalized, but I have run out of time to do so. So for now, you will have to dig into the scripts I used to train the models for the papers located in ``./scripts/Chase2025/``.

## Getting Started (installing things)
1. Setup a Python installation on the machine you are using. If you already have conda/mamba move one to 2.  

   I recommend installing [miniforge](https://github.com/conda-forge/miniforge). Inside miniforge is Mamba, which tends to solve environments more quickly than conda and miniconda. Also anaconda has a new license out there that charges for things. Mamba is more of an open-source and free version of conda. 

2. Install a env

   We are including an environment.yml file here, but given the variety of GPUs out there, folks will probably have a challenge here installing the right torch version for their GPUs. What I am going to suggest is to do the install in steps. First make a new env: 

   ``mamba create -n cira-diff``

   activate it 

   ``mamba activate cira-diff``

   Then install pytorch first: 
   
   ``mamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia``

   Note this is how I installed pytorch for the CUDA my GPUs have. If you dont have CUDA 12, change this to one of the 11.8s or something. You can see which CUDA is compiled by running `nvidia-smi` on a node where GPUs are connected. Alternatively, if you are using GH200s, you will need to use docker and the precompiled pytorch they give you. Example [here](https://dopplerchase-ai2es-schooner-hpc.readthedocs.io/en/latest/cira.html#gh200-how-to).

   Next up, install diffusers, transformers and accelerate. If you don't want to use diffuser models, you could skip this, but know the code will break because we import it later. We also install some other common packages here too: 

   ``pip install diffusers["torch"] transformers accelerate matplotlib tensorboard py3nvml build zarr`` 

5. Install local repo 

   `` pip install . `` 
   
## Data Prep

If you want to make this repo work for your dataset, the easiest implementation would be to adapt your training dataset to fit with the code. Our training dataset was of the shape ``(generation_images, condition images)`` where ``generation images`` was ``[n_samples,generation_channels,nx,ny]`` and condition images was ``[n_samples,condition_channels,nx,ny]``. We made this dataset by: 

1. loading a bunch of files 
2. slicing them down to a reasonable size (256 by 256, and the literature suggests smaller is better). 
3. find mean and std of the data
4. normalize all data to have mean=0 and std=1 
5. save out a zarr file that will return a the tuple: ``(generation_images, condition images, nx, ny)``

If you want to use our data for now, please check out our online data repo on [dryad]()

## Training

Now that you have a dataset ready, go ahead and train one of the models. Be sure to update the file paths in the top of the training scripts to what you need. Example call 

`` accelerate launch train_edm_Chase2025.py `` 

## Generation 

An example notebook of how to run a bunch of forecasts, say on the validation or test set is also included in the scripts folder. The name is `` Run_Forecasts_Chase2025.ipynb``. Feel free to grab everything out of the notebook and put it into a script. 