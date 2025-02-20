import inspect
import os
import argparse, torch

### MOVING ARGPARSE HERE TO HANDLE FLUX devices
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id",
    type=str,
    default="black-forest-labs/FLUX.1-schnell",
    help="Base model to use for training"
)    
parser.add_argument(
    "--device",
    type=str,
    default='cuda:0',
    help="Device to run training on"
)
parser.add_argument(
    "--dtype",
    type=torch.dtype,
    default=torch.bfloat16,
    help="Data type for model precision"
)

# LoRA Configuration
parser.add_argument(
    "--slider_rank",
    type=int,
    default=1,
    help="Rank of LoRA layers"
)
parser.add_argument(
    "--slider_alpha",
    type=int,
    default=1,
    help="Alpha scaling for LoRA layers"
)
parser.add_argument(
    "--num_sliders",
    type=int,
    default=16,
    help="Number of sliders to train"
)
parser.add_argument(
    "--train_method",
    type=str,
    default='flux-attn',
    choices=['flux-attn'],
    help="Type of layers to train"
)

# Training Configuration
parser.add_argument(
    "--lr",
    type=float,
    default=2e-3,
    help="Learning rate"
)
parser.add_argument(
    "--batchsize",
    type=int,
    default=1,
    help="Batch size for training"
)
parser.add_argument(
    "--iterations",
    type=int,
    default=3000,
    help="Number of training iterations"
)
parser.add_argument(
    "--max_denoising_steps",
    type=int,
    default=2,
    help="Maximum number of denoising steps"
)
parser.add_argument(
    "--guidance_scale",
    type=float,
    default=0,
    help="Guidance scale for stable diffusion"
)

parser.add_argument(
    "--contrastive_scale",
    type=float,
    default=0.1,
    help="Scale factor for contrastive loss"
)

# Prompt Configuration
parser.add_argument(
    "--concept_prompts",
    type=str,
    nargs='+',
    default=['image of dog'],
    help="List of concept prompts to use for training"
)
parser.add_argument(
    "--diverse_prompt_num",
    type=int,
    default=200,
    help="Number of diverse prompts to generate per concept"
)

# CLIP Configuration
parser.add_argument(
    "--clip_total_samples",
    type=int,
    default=5000,
    help="Total number of samples for CLIP feature extraction"
)
parser.add_argument(
    "--clip_batch_size",
    type=int,
    default=5,
    help="Batch size for CLIP feature extraction"
)
parser.add_argument(
    "--encoder",
    type=str,
    default='clip',
    choices=['clip', 'dinov2-small'],
    help="Encoder to use for feature extraction"
)

# Saving and Logging
parser.add_argument(
    "--save_every",
    type=int,
    default=5000,
    help="Save checkpoint every N iterations"
)
parser.add_argument(
    "--save_path",
    type=str,
    default='trained_sliders/flux/',
    help="Path to save model checkpoints"
)
parser.add_argument(
    "--savepath_training_images",
    type=str,
    default='training_images/flux/',
    help="Path to save generated training images"
)
parser.add_argument(
    "--wandb_log",
    type=int,
    default=0,
    help="Enable wandb logging"
)
parser.add_argument(
    "--wandb_proj",
    type=str,
    default='adobe_auto',
    help="Wandb project name"
)
parser.add_argument(
    "--exp_name",
    type=str,
    default=None,
    help="Optional experiment name prefix"
)

# Additional Configuration
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility"
)
parser.add_argument(
    "--train_type",
    type=str,
    default='concept',
    choices=['concept', 'art'],
    help="if you want to explore all art styles, use 'art'; else leave it at default"
)
parser.add_argument(
    "--save_training_images",
    type=str,
    default='false',
    choices=['true', 'false'],
    help="if you want to store the training images post-training"
)
args = parser.parse_args()

######## FOR FLUX THIS IS THE WAY TO TAKE CARE OF DEVICES (IF YOU HAVE MULTIPLE GPUs) - Need to find a better way
os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(':')[-1]
print(torch.cuda.device_count())

# Standard library imports
import ast
import datetime
import gc
import glob
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Union
import argparse

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from diffusers import LCMScheduler, AutoencoderTiny, StableDiffusionXLPipeline, logging, FluxPipeline
logging.set_verbosity_error()
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from sklearn.decomposition import PCA
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    CLIPModel,
    CLIPProcessor,
    logging,
)
logging.set_verbosity_warning()

# Local imports
sys.path.append('.')
from utils import train_util
from utils.flux_utils import *
from utils.lora import (
    LoRANetwork,
    DEFAULT_TARGET_REPLACE,
    UNET_TARGET_REPLACE_MODULE_CONV,
)

from utils.model_util import load_models_flux, save_checkpoint
from utils.prompt_util import expand_prompts
from utils.clip_util import compute_clip_pca, extract_clip_features


modules = DEFAULT_TARGET_REPLACE
modules += UNET_TARGET_REPLACE_MODULE_CONV


def normalize_image(image):
    """
    Normalizes an image tensor using CLIP's normalization constants.
    
    Args:
        image (torch.Tensor): Input image tensor of shape (B, C, H, W)
        
    Returns:
        torch.Tensor: Normalized image tensor with same shape as input
        
    Note:
        Uses CLIP's mean and std values for normalization:
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    """
    # Define CLIP normalization constants
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(image.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(image.device)
    # Apply normalization
    return (image - mean) / std


def process_flux_latents(noise_scheduler, vae, latents, timestep, height, width, vae_scale_factor, denoised_latents, inference_step):
    """
    Process Flux latents through the diffusion pipeline
    
    Args:
        noise_scheduler: Flux noise scheduler
        vae: VAE model
        latents: Input latents
        timestep: Current timestep
        height: Image height
        width: Image width
        vae_scale_factor: VAE scaling factor
        
    Returns:
        Processed image tensor
    """

    # Use scheduler to get the previous noisy sample x_t -> x_t-1
    noise_scheduler._step_index = inference_step
    latents = noise_scheduler.step(latents, timestep, denoised_latents, return_dict=True)
    try:
        latents = latents['prev_sample']
    except:
        latents = latents.prev_sample

    # Unpack and process latents
    latents = _unpack_latents(latents, height, width, vae_scale_factor)
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    
    # Decode through VAE and resize
    image = vae.decode(latents, return_dict=False)[0]
    image = F.adaptive_avg_pool2d(image, (224, 224))
    
    # Normalize for CLIP
    return normalize_image(image)

def optimize_flux_slider(transformer, noise_scheduler, vae, clip, timestep, denoised_latents,
                        text_embeds, pool_embeds, text_ids, img_ids, network, pc_direction, 
                        device, weight_dtype, scale, nonslider_feats, encoder, batchsize, height, width, vae_scale_factor, inference_step):
    """
    Optimizes a single Flux slider network using contrastive learning
    
    Similar structure to SDXL optimize_slider but adapted for Flux architecture
    """
    # Generate noise prediction using the network
    with network:
        noise_pred = transformer(
            hidden_states=denoised_latents,
            timestep=timestep / 1000,  # Flux uses timesteps / 1000
            guidance=None,  # No guidance for Flux
            pooled_projections=pool_embeds.repeat(batchsize, 1),
            encoder_hidden_states=text_embeds.repeat(batchsize, 1, 1),
            txt_ids=text_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]

    # Process latents to get CLIP-ready images
    image = process_flux_latents(noise_scheduler, vae, noise_pred, timestep, height, width, vae_scale_factor, denoised_latents, inference_step)
    
    # Extract CLIP features
    slider_feats = extract_clip_features(clip, image, encoder)
    
    # Compute feature difference and normalize
    feats = slider_feats - nonslider_feats
    feats = feats / feats.norm()
    
    # Compute cosine similarity loss
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss = -cos(feats, pc_direction).sum()
    
    return scale * loss

def get_current_prompts_flux(iteration, args, diverse_prompts, tokenizers, text_encoders, device):
    """
    Get current training prompts and their embeddings for Flux
    
    Args:
        iteration: Current training iteration
        args: Command line arguments
        diverse_prompts: List of diverse prompts
        tokenizers: List containing CLIPTokenizer and T5TokenizerFast
        text_encoders: List containing CLIPTextModel and T5EncoderModel
        device: Target device
        
    Returns:
        tuple: (current_prompt, text_embeddings, pooled_embeddings, text_ids)
    """
    with torch.no_grad():
        # T5 embeddings
        text_embeds = train_util._get_t5_prompt_embeds(
            text_encoders[1],
            tokenizers[1], 
            diverse_prompts[iteration],
            max_sequence_length=256,
            device=device
        )
        
        # CLIP embeddings (pooled)
        pooled_embeds = train_util._get_clip_prompt_embeds(
            text_encoders[0],
            tokenizers[0],
            diverse_prompts[iteration],
            device=device
        )
        
        # Flux needs text_ids
        text_ids = torch.zeros(text_embeds.shape[1], 3).to(
            device=device, 
            dtype=text_embeds.dtype
        )
            
    return diverse_prompts[iteration], text_embeds, pooled_embeds, text_ids

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps



def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    """
    Pack latents into Flux's 2x2 patch format
    """
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def _unpack_latents(latents, height, width, vae_scale_factor):
    """
    Unpack latents from Flux's 2x2 patch format back to image space
    """
    batch_size, num_patches, channels = latents.shape

    # Account for VAE compression and packing
    height = int( 2 * (int(height) // (vae_scale_factor * 2)))
    width = int(2 * (int(width) // (vae_scale_factor * 2)))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents
    
def sample_timestep_flux(iteration, max_denoising_steps, models, image_paths, params):
    """
    Sample timestep and prepare initial latents for Flux training
    
    Args:
        iteration: Current iteration
        max_denoising_steps: Maximum number of denoising steps
        models: Dictionary containing models
        image_paths: List of image paths
        params: Training parameters
        
    Returns:
        dict: Dictionary containing timestep data
    """
    # Sample random timestep
    timestep = torch.randint(0, max_denoising_steps, (1,)).item()
    generator = torch.manual_seed(random.randint(0, 2**15))
    
    # Get current timestep
    timesteps = models['noise_scheduler'].timesteps
    current_timestep = timesteps[timestep:timestep+1]
    
    # Get noisy image from prompt
    with torch.no_grad():
        pil_image = Image.open(image_paths[iteration])
        denoised_latents, latent_image_ids = train_util.get_noisy_image_flux(
            image=pil_image,
            vae=models['vae'],
            transformer=models['transformer'],
            scheduler=models['noise_scheduler'],
            timesteps_to=current_timestep,
            generator=generator,
            params = params
        )
        denoised_latents = denoised_latents.to(params['weight_dtype']).to(params['device'])
        latent_image_ids = latent_image_ids.to(params['weight_dtype']).to(params['device'])
    
    return {
        'timestep': timestep,
        'current_timestep': current_timestep.to(params['weight_dtype']).to(params['device']),
        'denoised_latents': denoised_latents,
        'generator': generator,
        'latent_image_ids': latent_image_ids
    }


def compute_sliderspace_loss(networks, models, timestep_data, clip_principles, text_embeds, pool_embeds, text_ids, params):
    """
    Compute contrastive loss between different Flux sliders
    
    Args:
        networks (dict): Dictionary of LoRA networks
        models (dict): Dictionary containing Flux models (transformer, vae, etc.)
        timestep_data (dict): Data about current timestep and latents
        clip_principles (torch.Tensor): PCA directions for sliders
        text_embeds (torch.Tensor): T5 text embeddings
        pool_embeds (torch.Tensor): CLIP pooled embeddings
        text_ids (torch.Tensor): Text IDs for Flux
        params (dict): Training parameters
        
    Returns:
        float: Total sliderspace loss
    """
    with torch.no_grad():
        # Get base output without LoRA
        nonlora_latents = models['transformer'](
            hidden_states=timestep_data['denoised_latents'],
            timestep=timestep_data['current_timestep'] / 1000,  # Flux expects timesteps / 1000
            guidance=None,  # No guidance for Flux
            pooled_projections=pool_embeds.repeat(params['batchsize'], 1),
            encoder_hidden_states=text_embeds.repeat(params['batchsize'], 1, 1),
            txt_ids=text_ids,
            img_ids=timestep_data['latent_image_ids'],
            return_dict=False,
        )[0]

        # Step through scheduler
        models['noise_scheduler']._step_index = timestep_data['timestep']
        latents = models['noise_scheduler'].step(
            nonlora_latents, 
            timestep_data['current_timestep'], 
            timestep_data['denoised_latents'], 
            return_dict=True
        )
        latents = latents.prev_sample

        # Unpack and decode
        latents = _unpack_latents(
            latents, 
            params['height'], 
            params['width'], 
            params['vae_scale_factor']
        )
        
        # Scale for VAE
        latents = (latents / models['vae'].config.scaling_factor) + models['vae'].config.shift_factor
        
        # Decode through VAE
        base_image = models['vae'].decode(latents, return_dict=False)[0]
        
        # Resize for CLIP
        base_image = F.adaptive_avg_pool2d(base_image, (224, 224))
        
        # Normalize
        base_image = normalize_image(base_image)
        
        # Get CLIP features
        nonslider_feats = extract_clip_features(
            models['clip_model'],
            base_image,
            params['encoder']
        )

    # Compute contrastive loss for subset of sliders
    sliderspace_loss = 0
    max_sliders_for_contrast = min(3, params['num_sliders'])
    slider_idxs = list(range(params['num_sliders']))
    random_idxs = random.sample(slider_idxs, max_sliders_for_contrast)
    
    # Optimize each selected slider
    for net_id in random_idxs:
        loss = optimize_flux_slider(
            models['transformer'],
            models['noise_scheduler'],
            models['vae'],
            models['clip_model'],
            timestep_data['current_timestep'],
            timestep_data['denoised_latents'],
            text_embeds,
            pool_embeds,
            text_ids,
            timestep_data['latent_image_ids'],
            networks[net_id],
            clip_principles[int(net_id)],
            params['device'],
            params['weight_dtype'],
            params['contrastive_scale'],
            nonslider_feats,
            params['encoder'],
            params['batchsize'],
            params['height'],
            params['width'], 
            params['vae_scale_factor'],
            timestep_data['timestep']
        )
        
        if not torch.isnan(loss).any():
            sliderspace_loss += loss.item()
            loss.backward()
            
    return sliderspace_loss

def initialize_sliderspace(unet, params):
    """Initialize LoRA networks for each slider
    
    Args:
        unet: The UNet model
        params: Dictionary containing training parameters
        
    Returns:
        Dictionary of initialized LoRA networks
    """
    networks = {}
    modules = DEFAULT_TARGET_REPLACE + UNET_TARGET_REPLACE_MODULE_CONV
    
    for i in range(params['num_sliders']):
        networks[i] = LoRANetwork(
            unet,
            rank=params['rank'],
            multiplier=1.0,
            alpha=params['alpha'], 
            train_method=params['train_method'],
            fast_init=False,
        ).to(params['device'], dtype=params['weight_dtype'])

    # Initialize optimizer
    all_params = []
    for net in networks.values():
        all_params.extend(net.prepare_optimizer_params())
    optimizer = AdamW(all_params, lr=float(params['lr']))
    
    return networks, optimizer, all_params

if __name__ == "__main__":
    
    exp_name = args.exp_name
    if exp_name is None:
        exp_name = f'concept_{random.randint(0, 2**15)}'

    # FLUX parameters for training
    max_sequence_length = 512
    if 'schnell' in  args.model_id.lower():
        max_sequence_length = 256
    weighting_scheme = 'none' # ["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"]
    logit_mean = 0.0
    logit_std = 1.0
    mode_scale = 1.29
    bsz = 1
    training_eta = 1
    lr_warmup_steps = 200
    lr_num_cycles = 1
    lr_power = 1.0
    lr_scheduler = 'constant' #Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]

    # Set up parameters from arguments
    training_params = {
        'pretrained_model_name_or_path': args.model_id,
        'savepath_training_images': args.savepath_training_images+'/'+exp_name,
        'alpha': args.slider_alpha,
        'rank': args.slider_rank,
        'train_method': args.train_method,
        'num_sliders': args.num_sliders,
        'batchsize': args.batchsize,
        'iterations': args.iterations,
        'save_every': args.save_every,
        'contrastive_scale': args.contrastive_scale,
        'lr': args.lr,
        'height': 512,
        'width': 512,
        'weight_dtype': args.dtype,
        'device': 'cuda:0', #we take care of the device in the beginning of script. this is just a proxy. DO NOT CHANGE THIS.
        'encoder': args.encoder,
        'max_denoising_steps': args.max_denoising_steps,
        'save_path': args.save_path+'/'+exp_name,
        'max_sequence_length': max_sequence_length,
    }
    
    # Load models
    models, pipe = load_models_flux(training_params)

    # super hacky way to handle shapes
    vae_scale_factor = 2 ** (len(models['vae'].config.block_out_channels) - 1)
    training_params['vae_scale_factor'] = vae_scale_factor
    
    # 1. Expand prompts using Claude if required (this is just added bonus for diverse slider discovery) - you need to export your own API key
    diverse_prompts = expand_prompts(args.concept_prompts, args.diverse_prompt_num, args)

    start_time = time.time()
    # 2. Extract CLIP features
    clip_principles, prompts_training, image_paths = compute_clip_pca(
        diverse_prompts=diverse_prompts,
        pipe=pipe,
        clip_model=models['clip_model'],
        clip_processor=models['clip_processor'],
        device=training_params['device'],
        guidance_scale=args.guidance_scale,
        params=training_params,
        total_samples = args.clip_total_samples,
        num_pca_components = 100,
        batch_size = args.clip_batch_size,
    )
    
    # 3. Initialize SliderSpace sliders
    networks, optimizer, all_params = initialize_sliderspace(
        models['transformer'],
        training_params
    )
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=args.iterations,
        num_cycles=lr_num_cycles,
        power=lr_power,
    )

    # setup noise scheduler
    sigmas = np.linspace(1.0, 1 / training_params['max_denoising_steps'], training_params['max_denoising_steps'])
    image_seq_len = (int(training_params['height']) // training_params['vae_scale_factor'] // 2) * (int(training_params['width']) // training_params['vae_scale_factor'] // 2)
    mu = calculate_shift(
        image_seq_len,
        models['noise_scheduler'].config.get("base_image_seq_len", 256),
        models['noise_scheduler'].config.get("max_image_seq_len", 4096),
        models['noise_scheduler'].config.get("base_shift", 0.5),
        models['noise_scheduler'].config.get("max_shift", 1.16),
    )
    timesteps, num_inference_steps =  retrieve_timesteps(
        models['noise_scheduler'],
        training_params['max_denoising_steps'],
        training_params['device'],
        sigmas=sigmas,
        mu=mu,
    )

    # 4. Training loop
    history = []
    pbar = tqdm(range(args.iterations))
    for iteration in pbar:
        # Save checkpoint if needed
        if (iteration + 1) % training_params['save_every'] == 0:
            save_checkpoint(networks, training_params['save_path'], training_params['weight_dtype'])
            
        # Get current prompts and embeddings for Flux
        current_prompt, text_embeds, pool_embeds, text_ids = get_current_prompts_flux(
            iteration, 
            args, 
            prompts_training, 
            models['tokenizers'], 
            models['text_encoders'],
            training_params['device']
        )

        optimizer.zero_grad()
        
        # Sample timestep and prepare latents
        timestep_data = sample_timestep_flux(
            iteration,
            training_params['max_denoising_steps'],
            models,
            image_paths,
            training_params
        )

        # Get Flux sliderspace loss
        sliderspace_loss = compute_sliderspace_loss(
            networks,
            models,
            timestep_data,
            clip_principles,
            text_embeds,
            pool_embeds,
            text_ids,
            training_params
        )
        
        history.append(sliderspace_loss)
            
        # Update progress bar
        pbar.set_description(
            f"Timestep: {timestep_data['timestep']} | "
            f"Loss: {history[-1]:.4f} | "
            f"Grad Norm: {all_params[0]['params'][0].grad.norm():.4f}"
        )
    
        # Update parameters
        optimizer.step()
        
        # Log to wandb if enabled
        if args.wandb_log:
            wandb.log({
                "sliderspace_loss": history[-1],
                "timestep": timestep_data['timestep'],
                "gradient_norm": all_params[0]['params'][0].grad.norm().item()
            })
# save the final weights
save_checkpoint(networks, training_params['save_path'], training_params['weight_dtype'])


end_time = time.time()
total_time = end_time - start_time
minutes = int(total_time // 60)
seconds = total_time % 60
print(f"Total SliderSpace Discovery Time: {minutes} minutes and {seconds:.2f} seconds")

if args.save_training_images == 'false':
    shutil.rmtree(training_params['savepath_training_images'])