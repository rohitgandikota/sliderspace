import os , torch
import argparse
import copy
import gc
import itertools
import logging
import math

import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module


from collections import defaultdict


from typing import List, Optional
import argparse
import ast
from pathlib import Path
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
from huggingface_hub import hf_hub_download
import gc
import torch.nn.functional as F
import os
import torch
from tqdm.auto import tqdm
import time, datetime
import numpy as np
from torch.optim import AdamW
from contextlib import ExitStack
from safetensors.torch import load_file
import torch.nn as nn
import random
from transformers import CLIPModel

from transformers import logging
logging.set_verbosity_warning()

from diffusers import logging
logging.set_verbosity_error()


def flush():
    torch.cuda.empty_cache()
    gc.collect()
flush()
def unwrap_model(model):
    options = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)
    #if is_deepspeed_available():
    #    options += (DeepSpeedEngine,)
    while isinstance(model, options):
        model = model.module
    return model


# Function to log gradients
def log_gradients(named_parameters):
    grad_dict = defaultdict(lambda: defaultdict(float))
    for name, param in named_parameters:
        if param.requires_grad and param.grad is not None:
            grad_dict[name]['mean'] = param.grad.abs().mean().item()
            grad_dict[name]['std'] = param.grad.std().item()
            grad_dict[name]['max'] = param.grad.abs().max().item()
            grad_dict[name]['min'] = param.grad.abs().min().item()
    return grad_dict
    
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, subfolder: str = "text_encoder",
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder
        , device_map='cuda:0'
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")
def load_text_encoders(pretrained_model_name_or_path, class_one, class_two, weight_dtype):
    text_encoder_one = class_one.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="text_encoder", 
        torch_dtype=weight_dtype,
        device_map='cuda:0'
    )
    text_encoder_two = class_two.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="text_encoder_2", 
        torch_dtype=weight_dtype,
        device_map='cuda:0'
    )
    return text_encoder_one, text_encoder_two
import matplotlib.pyplot as plt
def plot_labeled_images(images, labels):
    # Determine the number of images
    n = len(images)
    
    # Create a new figure with a single row
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    
    # If there's only one image, axes will be a single object, not an array
    if n == 1:
        axes = [axes]
    
    # Plot each image
    for i, (img, label) in enumerate(zip(images, labels)):
        # Convert PIL image to numpy array
        img_array = np.array(img)
        
        # Display the image
        axes[i].imshow(img_array)
        axes[i].axis('off')  # Turn off axis
        
        # Set the title (label) for the image
        axes[i].set_title(label)
    
    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()
    

def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds

def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

    return prompt_embeds, pooled_prompt_embeds, text_ids
    
def compute_text_embeddings(prompt, text_encoders, tokenizers,max_sequence_length=256):
    device = text_encoders[0].device
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length=max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        text_ids = text_ids.to(device)
    return prompt_embeds, pooled_prompt_embeds, text_ids


def get_sigmas(timesteps, n_dim=4, device='cuda:0', dtype=torch.bfloat16):
    sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    ax1.plot(history['concept'])
    ax1.set_title('Concept Loss')
    ax2.plot(movingaverage(history['concept'], 10))
    ax2.set_title('Moving Average Concept Loss')
    plt.tight_layout()
    plt.show()

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')



@torch.no_grad()
def get_noisy_image_flux(
    image,
    vae,
    transformer,
    scheduler,
    timesteps_to=1000,
    generator=None,
    **kwargs,
):
    """
    Gets noisy latents for a given image using Flux pipeline approach.
    
    Args:
        image: PIL image or tensor
        vae: Flux VAE model
        transformer: Flux transformer model
        scheduler: Flux noise scheduler
        timesteps_to: Target timestep
        generator: Random generator for reproducibility
        
    Returns:
        tuple: (noisy_latents, noise)
    """
    device = vae.device
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    # Preprocess image
    if not isinstance(image, torch.Tensor):
        image = image_processor.preprocess(image)
    image = image.to(device=device, dtype=torch.float32)

    # Encode through VAE
    init_latents = vae.encode(image).latents
    init_latents = vae.config.scaling_factor * init_latents
    
    # Get shape for noise
    shape = init_latents.shape
    
    # Generate noise
    noise = randn_tensor(shape, generator=generator, device=device)

    # Pack latents using Flux's method
    init_latents = _pack_latents(
        init_latents, 
        shape[0],  # batch size
        transformer.config.in_channels // 4,
        height=shape[2],
        width=shape[3]
    )
    noise = _pack_latents(
        noise,
        shape[0],
        transformer.config.in_channels // 4,
        height=shape[2],
        width=shape[3]
    )

    # Get timestep
    timestep = scheduler.timesteps[timesteps_to:timesteps_to+1]
    
    # Add noise to latents
    noisy_latents = scheduler.add_noise(init_latents, noise, timestep)
    
    return noisy_latents, noise
