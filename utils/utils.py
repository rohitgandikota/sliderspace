import anthropic
client = anthropic.Anthropic()
from diffusers.image_processor import VaeImageProcessor
from typing import List, Optional
import argparse
import ast
import pandas as pd
from pathlib import Path
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler, AutoencoderTiny
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

import sys
import argparse
import wandb
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

sys.path.append('../')
from utils.lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV

from transformers import logging
logging.set_verbosity_warning()
import matplotlib.pyplot as plt
from diffusers import logging
logging.set_verbosity_error()
modules = DEFAULT_TARGET_REPLACE
modules += UNET_TARGET_REPLACE_MODULE_CONV
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
import random
import gc
import diffusers
from diffusers import DiffusionPipeline, FluxPipeline
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, LMSDiscreteScheduler, SchedulerMixin
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor, AttentionProcessor
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.utils.torch_utils import randn_tensor

import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.pipelines import StableDiffusionXLPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import XLA_AVAILABLE
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput

from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

import sys
sys.path.append('../.')
from utils.flux_utils import *
import random

import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


def flush():
    torch.cuda.empty_cache()
    gc.collect()

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


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
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

def claude_generate_prompts_sliders(prompt, 
                             num_prompts=20,
                             temperature=0.2, 
                             max_tokens=2000, 
                             frequency_penalty=0.0,
                             model="claude-3-5-sonnet-20240620",
                             verbose=False,
                             train_type='concept'):
    gpt_assistant_prompt =  f''' You are an expert in writing diverse image captions. When i provide a prompt, I want you to give me {num_prompts} alternative prompts that is similar to the provided prompt but produces diverse images. Be creative and make sure the original subjects in the original prompt are present in your prompts. Make sure that you end the prompts with keywords that will produce high quality images like ",detailed, 8k" or ",hyper-realistic, 4k".

Give me the expanded prompts in the style of a list. start with a [ and end with ] do not add any special characters like \n 
I need you to give me only the python list and nothing else. Do not explain yourself

example output format:
["prompt1", "prompt2", ...]
'''
    
    if train_type == 'art':
        gpt_assistant_prompt =  f'''You are an expert in writing art image captions. I want you to generate prompts that would create diverse artwork images. 
    Your role is to give me {num_prompts} diverse prompts that will make the image-generation model to output creative and interesting artwork images with unique and diverse artistic styles. A prompt could like "an <object/landscape> in the style of <an artist>" or "an <object/landscape> in the style of <an artistic style (e.g. cubism)>". make sure that you end the prompts with enhancing keywords like ",detailed, 8k" or ",hyper-realistic, 4k". 
    
   Give me the prompts in the style of a list. start with a [ and end with ] do not add any special characters like \n 
I need you to give me only the python list and nothing else. Do not explain yourself

example output format:
["prompt1", "prompt2", ...]
    '''
    # if 'dog' in prompt:
    #     gpt_assistant_prompt =  f'''You are an expert in prompting text-image generation models. I want you to generate simple prompts that would trigger the image generation model to generate a unique dog breeds. 
    # Your role is to give me {num_prompts} diverse prompts that will make the image-generation model to output diverse and interesting dog breeds with unique and diverse looks. make sure that you end the prompts with enhancing keywords like ",detailed, 8k" or ",hyper-realistic, 4k". 
    
    # Be creative and make sure to remember diversity is the key. Give me the prompts in the form of a list. start with a [ and end with ] do not add any special characters like \n 
    # I need you to give me only the python list and nothing else. Do not explain yourself

    # example output format:
    # ["prompt1", "prompt2", ...]
    # '''        

    if train_type == 'artclaudesemantics':
        gpt_assistant_prompt =  f'''You are an expert in prompting text-image generation models. I want you to generate simple prompts that would trigger the image generation model to generate a unique artistic images but DO NOT SPECIFY THE ART STYLE. 
    Your role is to give me {num_prompts} diverse prompts that will make the image-generation model to output diverse and interesting art images. Usually like "<some object or scene> in the style of " or "<some object or scene> in style of". Always end your prompts with "in the style of" so that i can manually add the style i want. make sure that you end the prompts with enhancing keywords like ",detailed, 8k" or ",hyper-realistic, 4k". 
    
    Be creative and make sure to remember diversity is the key. Give me the prompts in the form of a list. start with a [ and end with ] do not add any special characters like \n 
    I need you to give me only the python list and nothing else. Do not explain yourself

    example output format:
    ["prompt1", "prompt2", ...]
    '''
    gpt_user_prompt = prompt
    gpt_prompt = gpt_assistant_prompt, gpt_user_prompt
    message=[
        {
            "role": "user", 
            "content": [
                {
                    "type": "text",
                    "text": gpt_user_prompt
                }
            ]
        }
            ]
    
    output = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=gpt_assistant_prompt,
        messages=message
    )
    content = output.content[0].text
    return content

def normalize_image(image):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(image.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(image.device)
    return (image - mean) / std


@torch.no_grad()
def call_sdxl(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps: List[int] = None,
    sigmas: List[float] = None,
    denoising_end: Optional[float] = None,
    guidance_scale: float = 5.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    pooled_prompt_embeds: Optional[torch.Tensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    original_size: Optional[Tuple[int, int]] = None,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    target_size: Optional[Tuple[int, int]] = None,
    negative_original_size: Optional[Tuple[int, int]] = None,
    negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
    negative_target_size: Optional[Tuple[int, int]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    save_timesteps = None,
    clip=None,
    use_clip=True,
    encoder='clip',
):

    callback = None
    callback_steps = None

    if callback is not None:
        deprecate(
            "callback",
            "1.0.0",
            "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
        )
    if callback_steps is not None:
        deprecate(
            "callback_steps",
            "1.0.0",
            "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
        )

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    # 0. Default height and width to unet
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt,
        negative_prompt_2,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        ip_adapter_image,
        ip_adapter_image_embeds,
        callback_on_step_end_tensor_inputs,
    )

    self._guidance_scale = guidance_scale
    self._guidance_rescale = guidance_rescale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs
    self._denoising_end = denoising_end
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # 3. Encode input prompt
    lora_scale = (
        self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
    )

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=self.clip_skip,
    )

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler, num_inference_steps, device, timesteps, sigmas
    )

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Prepare added time ids & embeddings
    add_text_embeds = pooled_prompt_embeds
    if self.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

    add_time_ids = self._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = self._get_add_time_ids(
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    else:
        negative_add_time_ids = add_time_ids

    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = self.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            self.do_classifier_free_guidance,
        )

    # 8. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

    # 8.1 Apply denoising_end
    if (
        self.denoising_end is not None
        and isinstance(self.denoising_end, float)
        and self.denoising_end > 0
        and self.denoising_end < 1
    ):
        discrete_timestep_cutoff = int(
            round(
                self.scheduler.config.num_train_timesteps
                - (self.denoising_end * self.scheduler.config.num_train_timesteps)
            )
        )
        num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
        timesteps = timesteps[:num_inference_steps]

    # 9. Optionally get Guidance Scale Embedding
    timestep_cond = None
    if self.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = self.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    self._num_timesteps = len(timesteps)
    clip_features = []
    # with self.progress_bar(total=num_inference_steps) as progress_bar:
    for i, t in enumerate(timesteps):
        if self.interrupt:
            continue

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            added_cond_kwargs["image_embeds"] = image_embeds
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=self.cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        # perform guidance
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

        # compute the previous noisy sample x_t -> x_t-1
        latents_dtype = latents.dtype
        # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

        # compute the previous noisy sample x_t -> x_t-1
        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)
        try:
            denoised = latents['pred_original_sample'] / self.vae.config.scaling_factor
        except:
            denoised = latents['denoised'] / self.vae.config.scaling_factor
        latents = latents['prev_sample']

        
        # if latents.dtype != latents_dtype:
        #     if torch.backends.mps.is_available():
        #         # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
        latents = latents.to(self.vae.dtype)
        denoised = denoised.to(self.vae.dtype)
        
        if i in save_timesteps:
            if use_clip:
                denoised = self.vae.decode(denoised.to(self.vae.dtype), return_dict=False)[0]
                denoised = F.adaptive_avg_pool2d(denoised, (224, 224))
                denoised = normalize_image(denoised)
                if 'dino' in encoder:
                    denoised = clip(denoised)
                    denoised = denoised.pooler_output
                    denoised = denoised.cpu().view(denoised.shape[0], -1)
                else:
                    denoised = clip.get_image_features(denoised)
                    denoised = denoised.cpu().view(denoised.shape[0], -1)
                    
                # denoised = clip.get_image_features(denoised)
            clip_features.append(denoised)

        
        

        if callback_on_step_end is not None:
            callback_kwargs = {}
            for k in callback_on_step_end_tensor_inputs:
                callback_kwargs[k] = locals()[k]
            callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

            latents = callback_outputs.pop("latents", latents)
            prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
            negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
            add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
            negative_pooled_prompt_embeds = callback_outputs.pop(
                "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
            )
            add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
            negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

        # call the callback, if provided
        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
            # progress_bar.update()
            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, latents)

        if XLA_AVAILABLE:
            xm.mark_step()

    if not output_type == "latent":
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
        elif latents.dtype != self.vae.dtype:
            if torch.backends.mps.is_available():
                # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                self.vae = self.vae.to(latents.dtype)

        # unscale/denormalize the latents
        # denormalize with the mean and std if available and not None
        has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
        has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            )
            latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
        else:
            latents = latents / self.vae.config.scaling_factor

        image = self.vae.decode(latents, return_dict=False)[0]

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
    else:
        image = latents

    if not output_type == "latent":

        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    return image, clip_features

@torch.no_grad()

def call_flux(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 7.0,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    verbose=False,
    save_timesteps = None,
    clip=None,
    use_clip=True,
    encoder='clip'
):
    

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        mu=mu,
    )

    timesteps = timesteps
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # handle guidance
    if self.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None
    clip_features = []
    # 6. Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
           
            noise_pred = self.transformer(
                hidden_states=latents,
                # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=True)

 
            denoised = latents['prev_sample'] 
            latents = latents['prev_sample']

            denoised = self._unpack_latents(denoised, height, width, self.vae_scale_factor)
            denoised = (denoised / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            denoised = self.vae.decode(denoised, return_dict=False)[0]
            denoised = F.adaptive_avg_pool2d(denoised, (224, 224))
            if 'dino' in encoder:
                outputs = clip(**inputs)
                denoised = outputs.pooler_output
                denoised = denoised.cpu().view(denoised.shape[0], -1)
            else:
                denoised = clip.get_image_features(denoised)
                denoised = denoised.cpu().view(denoised.shape[0], -1)

            clip_features.append()
           
            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

    if output_type == "latent":
        image = latents
        return image

    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return image, clip_features




def get_diffusion_clip_directions(prompts, unet, tokenizers, text_encoders, vae, noise_scheduler, clip, batchsize=1, height=1024, width=1024, max_denoising_steps=4, savepath_training_images=None, use_clip=True,encoder='clip'):
    device = unet.device
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    os.makedirs(savepath_training_images, exist_ok=True)


    if len(noise_scheduler.timesteps) != max_denoising_steps:
        noise_scheduler_orig = noise_scheduler
        max_denoising_steps_orig = len(noise_scheduler.timesteps)
        noise_scheduler.set_timesteps(max_denoising_steps)
        timesteps_distilled = noise_scheduler.timesteps
        
        noise_scheduler.set_timesteps(max_denoising_steps_orig)
        timesteps_full = noise_scheduler.timesteps
        save_timesteps = []
        for timesteps_to_distilled in range(max_denoising_steps):
            # Get the value from timesteps_distilled that we want to find in timesteps_full
            value_to_find = timesteps_distilled[timesteps_to_distilled]
            timesteps_to_full = (timesteps_full == value_to_find).nonzero().item()
            save_timesteps.append(timesteps_to_full)

        guidance_scale = 7
    else:
        max_denoising_steps_orig = max_denoising_steps
        save_timesteps = [i for i in range(max_denoising_steps_orig)]
        guidance_scale = 7
        if max_denoising_steps_orig <=4:
            guidance_scale = 0
        
    noise_scheduler.set_timesteps(max_denoising_steps_orig)
    # if max_denoising_steps_orig == 1:
    #     noise_scheduler.set_timesteps(timesteps=[399],
    #                                  device=device)
    
    weight_dtype = unet.dtype
    device = unet.device 
    StableDiffusionXLPipeline.__call__ = call_sdxl
    pipe = StableDiffusionXLPipeline(vae = vae,
        text_encoder= text_encoders[0],
        text_encoder_2=text_encoders[1],
        tokenizer = tokenizers[0],
        tokenizer_2= tokenizers[1],
        unet=unet,
        scheduler=noise_scheduler)
    pipe.to(unet.device)
    # print(guidance_scale, max_denoising_steps_orig, save_timesteps)
    images, clip_features = pipe(prompts, guidance_scale=guidance_scale, num_inference_steps = max_denoising_steps_orig, clip=clip, save_timesteps =save_timesteps, use_clip=use_clip, encoder=encoder)
    
    return images, torch.stack(clip_features)



def get_flux_clip_directions(prompts, transformer, tokenizers, text_encoders, vae, noise_scheduler, clip, batchsize=1, height=1024, width=1024, max_denoising_steps=4, savepath_training_images=None, use_clip=True):
    device = transformer.device
    FluxPipeline.__call__ = call_flux
    pipe = FluxPipeline(noise_scheduler,
                    vae,
                    text_encoders[0],
                    tokenizers[0],
                    text_encoders[1],
                    tokenizers[1],
                    transformer,
                   )
    pipe.set_progress_bar_config(disable=True)

    os.makedirs(savepath_training_images, exist_ok=True)

    images, clip_features = pipe(
        prompts,
        height=height,
        width=width,
        guidance_scale=0,
        num_inference_steps=4,
        max_sequence_length=256,
        num_images_per_prompt=1,
        output_type='pil',
        clip=clip
    )
    
    return images, torch.stack(clip_features)




def get_diffusion_clip_directions(prompts, unet, tokenizers, text_encoders, vae, noise_scheduler, clip, batchsize=1, height=1024, width=1024, max_denoising_steps=4, savepath_training_images=None, use_clip=True,encoder='clip', num_images_per_prompt=1):

    
    device = unet.device
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    os.makedirs(savepath_training_images, exist_ok=True)


    if len(noise_scheduler.timesteps) != max_denoising_steps:
        noise_scheduler_orig = noise_scheduler
        max_denoising_steps_orig = len(noise_scheduler.timesteps)
        noise_scheduler.set_timesteps(max_denoising_steps)
        timesteps_distilled = noise_scheduler.timesteps
        
        noise_scheduler.set_timesteps(max_denoising_steps_orig)
        timesteps_full = noise_scheduler.timesteps
        save_timesteps = []
        for timesteps_to_distilled in range(max_denoising_steps):
            # Get the value from timesteps_distilled that we want to find in timesteps_full
            value_to_find = timesteps_distilled[timesteps_to_distilled]
            timesteps_to_full = (timesteps_full == value_to_find).nonzero().item()
            save_timesteps.append(timesteps_to_full)

        guidance_scale = 7
    else:
        max_denoising_steps_orig = max_denoising_steps
        save_timesteps = [i for i in range(max_denoising_steps_orig)]
        guidance_scale = 7
        if max_denoising_steps_orig <=4:
            guidance_scale = 0
        
    noise_scheduler.set_timesteps(max_denoising_steps_orig)
    # if max_denoising_steps_orig == 1:
    #     noise_scheduler.set_timesteps(timesteps=[399],
    #                                  device=device)
    
    weight_dtype = unet.dtype
    device = unet.device 
    StableDiffusionXLPipeline.__call__ = call_sdxl
    pipe = StableDiffusionXLPipeline(vae = vae,
        text_encoder= text_encoders[0],
        text_encoder_2=text_encoders[1],
        tokenizer = tokenizers[0],
        tokenizer_2= tokenizers[1],
        unet=unet,
        scheduler=noise_scheduler)
    pipe.to(unet.device)
    # print(guidance_scale, max_denoising_steps_orig, save_timesteps)
    images, clip_features = pipe(prompts, guidance_scale=guidance_scale, num_inference_steps = max_denoising_steps_orig, clip=clip, save_timesteps =save_timesteps, use_clip=use_clip, encoder=encoder)
    
    return images, torch.stack(clip_features)




