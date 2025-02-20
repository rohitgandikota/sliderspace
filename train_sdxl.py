# Standard library imports
import ast
import datetime
import gc
import glob
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional
import argparse
import shutil

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from diffusers import LCMScheduler, AutoencoderTiny, StableDiffusionXLPipeline, logging
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
from utils.lora import (
    LoRANetwork,
    DEFAULT_TARGET_REPLACE,
    UNET_TARGET_REPLACE_MODULE_CONV,
)

from utils.model_util import load_models_xl, save_checkpoint
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

def process_latents(noise_scheduler, vae, latents, timestep, denoised_latents):
    """
    Processes latent vectors through the diffusion and VAE pipeline.
    
    Args:
        noise_scheduler: Diffusion noise scheduler
        vae: VAE model for decoding latents to images
        latents (torch.Tensor): Input latent vectors
        timestep: Current diffusion timestep
        denoised_latents (torch.Tensor): Previously denoised latents
        
    Returns:
        torch.Tensor: Processed and normalized image tensor
        
    Note:
        1. Steps the diffusion process
        2. Decodes latents to image using VAE
        3. Resizes to 224x224 for CLIP
        4. Applies CLIP normalization
    """
    # Match VAE dtype
    dtype = vae.dtype
    
    # Step the diffusion process
    out = noise_scheduler.step(latents, timestep, denoised_latents, return_dict=True)
    
    # Handle different scheduler return formats
    try:
        out = out['pred_original_sample'] / vae.config.scaling_factor
    except:
        out = out['denoised'] / vae.config.scaling_factor
        
    # Convert to VAE dtype
    out = out.to(dtype)
    
    # Decode latents to image
    out = vae.decode(out, return_dict=False)[0]
    
    # Resize to CLIP input size
    out = F.adaptive_avg_pool2d(out, (224, 224))
    
    # Apply CLIP normalization
    out = normalize_image(out)
    return out

def optimize_slider(unet, noise_scheduler, vae, clip, timestep, denoised_latents,
                    text_embeds, pool_embeds, add_time_ids, network, pc_direction, 
                    device, weight_dtype, scale, nonslider_feats, encoder, batchsize):
    """
    Optimizes a single slider network using contrastive learning.
    
    Args:
        unet: U-Net model for noise prediction
        noise_scheduler: Diffusion noise scheduler
        vae: VAE model for decoding latents
        clip: CLIP model for feature extraction
        timestep: Current diffusion timestep
        denoised_latents (torch.Tensor): Previously denoised latents
        text_embeds (torch.Tensor): Text embeddings
        pool_embeds (torch.Tensor): Pooled text embeddings
        add_time_ids (torch.Tensor): Additional time IDs
        network: LoRA network to optimize
        pc_direction (torch.Tensor): Principal component direction for optimization
        device: torch device
        weight_dtype: Data type for weights
        scale (float): Loss scaling factor
        nonslider_feats (torch.Tensor): Features from non-slider output
        encoder (str): Type of encoder ('clip' or other)
        batchsize (int): Batch size
        
    Returns:
        torch.Tensor: Scaled loss value
        
    Note:
        1. Predicts noise using the network
        2. Processes latents through diffusion pipeline
        3. Extracts features using CLIP
        4. Computes contrastive loss against PC direction
    """
    # Generate latents using the network
    with network:
        contrast_lora_latents = train_util.predict_noise_xl(
            unet, noise_scheduler, timestep, denoised_latents,
            text_embeddings=text_embeds.repeat(batchsize, 1, 1),
            add_text_embeddings=pool_embeds.repeat(batchsize, 1),
            add_time_ids=add_time_ids.repeat(batchsize, 1),
            guidance_scale=0,
        )
    
    # Process latents to get CLIP-ready images
    slider_out = process_latents(noise_scheduler, vae, contrast_lora_latents, timestep, denoised_latents)
    
    # Extract CLIP features
    slider_feats = extract_clip_features(clip, slider_out, encoder)
    
    # Compute feature difference and normalize
    feats = slider_feats - nonslider_feats
    feats = feats / feats.norm()
    
    # Compute cosine similarity loss
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss = -cos(feats, pc_direction).sum()  # Maximize projection by minimizing negative
    
    return scale * loss

  
def get_current_prompts(iteration, args, diverse_prompts, tokenizers, text_encoders, device):
    """
    Get current training prompts and their embeddings based on training type
    
    Args:
        iteration: Current training iteration
        args: Command line arguments
        diverse_prompts: List of diverse prompts
        tokenizers: SDXL tokenizers
        text_encoders: SDXL text encoders
        device: Target device
        
    Returns:
        tuple: (current_prompt, text_embeddings, pooled_embeddings)
    """
    with torch.no_grad():
        # For concept training, use provided prompts
        diverse_text_embed, diverse_pool_embed = train_util.encode_prompts_xl(
            tokenizers, 
            text_encoders, 
            diverse_prompts[iteration%len(diverse_prompts)]
        )
            
    return diverse_prompts[iteration%len(diverse_prompts)], diverse_text_embed, diverse_pool_embed



def sample_timestep(iteration, max_denoising_steps, models, image_paths, params):
    """
    Sample timestep and prepare initial latents for training
    
    Args:
        max_denoising_steps: Maximum number of denoising steps
        models: Dictionary containing models
        current_prompt: Current training prompt
        params: Training parameters
        
    Returns:
        dict: Dictionary containing timestep data
    """
    # Sample random timestep
    timestep = torch.randint(0, max_denoising_steps, (1,)).item()
    random_seed = random.randint(0, 2**15)
    generator = torch.manual_seed(random_seed)
    
    # Prepare time embeddings
    add_time_ids = train_util.get_add_time_ids(
        params['height'], 
        params['width'], 
        dynamic_crops=False, 
        dtype=params['weight_dtype']
    ).to(params['device'])
    
    # Get noisy image from prompt
    with torch.no_grad():
        pil_image = Image.open(image_paths[iteration%len(image_paths)])  # Assuming current_prompt is an image path
        denoised_latents, noise = train_util.get_noisy_image(
            image=pil_image,
            vae=models['vae'],
            unet=models['unet'],
            scheduler=models['noise_scheduler'],
            timesteps_to=timestep,
            generator=generator,
        )
        denoised_latents = denoised_latents.to(params['weight_dtype']).to(params['device'])
        noise = noise.to(params['weight_dtype']).to(params['device'])
    
    # Get current timestep
    timesteps = models['noise_scheduler'].timesteps
    current_timestep = timesteps[timestep]
    
    return {
        'timestep': timestep,
        'current_timestep': current_timestep,
        'add_time_ids': add_time_ids,
        'denoised_latents': denoised_latents,
        'noise': noise,
        'generator': generator
    }

def compute_sliderspace_loss(networks, models, timestep_data, clip_principles, text_embeds, pool_embeds, params):
    """Compute the contrastive loss between different sliders"""
    with torch.no_grad():
        nonlora_latents = train_util.predict_noise_xl(
            models['unet'], 
            models['noise_scheduler'],
            timestep_data['current_timestep'],
            timestep_data['denoised_latents'],
            text_embeddings=text_embeds.repeat(params['batchsize'], 1, 1),
            add_text_embeddings=pool_embeds.repeat(params['batchsize'], 1),
            add_time_ids=timestep_data['add_time_ids'].repeat(params['batchsize'], 1),
            guidance_scale=0,
        )
        
        nonslider_out = process_latents(
            models['noise_scheduler'],
            models['vae'],
            nonlora_latents,
            timestep_data['current_timestep'],
            timestep_data['denoised_latents']
        )
        

        nonslider_feats = extract_clip_features(models['clip_model'], nonslider_out, params['encoder'])


    sliderspace_loss = 0
    max_sliders_for_contrast = 3
    slider_idxs = list(range(params['num_sliders']))
    random_idxs = random.sample(slider_idxs, max_sliders_for_contrast)
    
    for net_id in random_idxs:
        loss = optimize_slider(
            models['unet'],
            models['noise_scheduler'],
            models['vae'],
            models['clip_model'],
            timestep_data['current_timestep'],
            timestep_data['denoised_latents'],
            text_embeds,
            pool_embeds,
            timestep_data['add_time_ids'],
            networks[net_id],
            clip_principles[int(net_id)],
            params['device'],
            params['weight_dtype'],
            params['contrastive_scale'],
            nonslider_feats,
            params['encoder'],
            params['batchsize']
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
    parser = argparse.ArgumentParser()
    # Model Configuration
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base model to use for training"
    )
    parser.add_argument(
        "--distilled_ckpt",
        type=str,
        default='dmd2',
        help="Path to DMD checkpoint"
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
        default='xattn-strict',
        choices=['xattn', 'xattn-strict', 'noxattn'],
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
        default=4,
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
        default=1000,
        help="Total number of samples for CLIP feature extraction"
    )
    parser.add_argument(
        "--clip_batch_size",
        type=int,
        default=10,
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
        default='trained_sliders/sdxl/',
        help="Path to save model checkpoints"
    )
    parser.add_argument(
        "--savepath_training_images",
        type=str,
        default='training_images/sdxl/',
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
    # Parse arguments
    args = parser.parse_args()

    exp_name = args.exp_name
    if exp_name is None:
        exp_name = f'concept_{random.randint(0, 2**15)}'

    # Set up parameters from arguments
    training_params = {
        'pretrained_model_name_or_path': args.model_id,
        'distilled': args.distilled_ckpt,
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
        'height': 1024,
        'width': 1024,
        'weight_dtype': args.dtype,
        'device': args.device,
        'encoder': args.encoder,
        'max_denoising_steps': args.max_denoising_steps,
        'save_path': args.save_path+'/'+exp_name
    }


    # Load models
    models, pipe = load_models_xl(training_params)


    
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
        models['unet'],
        training_params
    )
    
    # 4. Training loop
    history = []
    pbar = tqdm(range(args.iterations))
    
    for iteration in pbar:
        # Save checkpoint if needed
        if (iteration + 1) % training_params['save_every'] == 0:
            save_checkpoint(networks, training_params['save_path'], training_params['weight_dtype'])
            
        # Get current prompts
        current_prompt, text_embeds, pool_embeds = get_current_prompts(
            iteration, args, prompts_training, models['tokenizers'], 
            models['text_encoders'],training_params['device']
        )
        
        optimizer.zero_grad()
        
        # Sample timestep
        timestep_data = sample_timestep(
            iteration,
            training_params['max_denoising_steps'],
            models, 
            image_paths,
            training_params
        )
        
            
        # Contrastive loss if enabled
        sliderspace_loss = compute_sliderspace_loss(
            networks,
            models,
            timestep_data,
            clip_principles,
            text_embeds,
            pool_embeds,
            training_params
        )
        history.append(sliderspace_loss)
            
        # Update progress bar
        pbar.set_description(f"Timestep: {timestep_data['timestep']} | {history[-1]:.4f} | Gradients: {all_params[0]['params'][0].grad.norm()}")

        # Update parameters
        optimizer.step()
        
        # Log to wandb if enabled
        if args.wandb_log:
            wandb.log({
                "sliderspace_loss": history[-1],
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