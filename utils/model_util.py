from typing import Literal, Union, Optional

import torch, gc, os
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection, T5TokenizerFast
from transformers import (
    AutoModel,
    CLIPModel,
    CLIPProcessor,
)
from huggingface_hub import hf_hub_download
from diffusers import (
    UNet2DConditionModel,
    SchedulerMixin,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    FluxPipeline,
    AutoencoderKL,
    FluxTransformer2DModel,
)
import copy
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    LMSDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers import LCMScheduler, AutoencoderTiny
import sys
sys.path.append('.')
from .flux_utils import *

TOKENIZER_V1_MODEL_NAME = "CompVis/stable-diffusion-v1-4"
TOKENIZER_V2_MODEL_NAME = "stabilityai/stable-diffusion-2-1"

AVAILABLE_SCHEDULERS = Literal["ddim", "ddpm", "lms", "euler_a"]

SDXL_TEXT_ENCODER_TYPE = Union[CLIPTextModel, CLIPTextModelWithProjection]

DIFFUSERS_CACHE_DIR = None  # if you want to change the cache dir, change this


def load_diffusers_model(
    pretrained_model_name_or_path: str,
    v2: bool = False,
    clip_skip: Optional[int] = None,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[CLIPTokenizer, CLIPTextModel, UNet2DConditionModel,]:
    # VAE はいらない

    if v2:
        tokenizer = CLIPTokenizer.from_pretrained(
            TOKENIZER_V2_MODEL_NAME,
            subfolder="tokenizer",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            # default is clip skip 2
            num_hidden_layers=24 - (clip_skip - 1) if clip_skip is not None else 23,
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        )
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            TOKENIZER_V1_MODEL_NAME,
            subfolder="tokenizer",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            num_hidden_layers=12 - (clip_skip - 1) if clip_skip is not None else 12,
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        )

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=weight_dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )

    return tokenizer, text_encoder, unet


def load_checkpoint_model(
    checkpoint_path: str,
    v2: bool = False,
    clip_skip: Optional[int] = None,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[CLIPTokenizer, CLIPTextModel, UNet2DConditionModel,]:
    pipe = StableDiffusionPipeline.from_ckpt(
        checkpoint_path,
        upcast_attention=True if v2 else False,
        torch_dtype=weight_dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )

    unet = pipe.unet
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    if clip_skip is not None:
        if v2:
            text_encoder.config.num_hidden_layers = 24 - (clip_skip - 1)
        else:
            text_encoder.config.num_hidden_layers = 12 - (clip_skip - 1)

    del pipe

    return tokenizer, text_encoder, unet


def load_models(
    pretrained_model_name_or_path: str,
    scheduler_name: AVAILABLE_SCHEDULERS,
    v2: bool = False,
    v_pred: bool = False,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[CLIPTokenizer, CLIPTextModel, UNet2DConditionModel, SchedulerMixin,]:
    if pretrained_model_name_or_path.endswith(
        ".ckpt"
    ) or pretrained_model_name_or_path.endswith(".safetensors"):
        tokenizer, text_encoder, unet = load_checkpoint_model(
            pretrained_model_name_or_path, v2=v2, weight_dtype=weight_dtype
        )
    else:  # diffusers
        tokenizer, text_encoder, unet = load_diffusers_model(
            pretrained_model_name_or_path, v2=v2, weight_dtype=weight_dtype
        )

    # VAE はいらない

    scheduler = create_noise_scheduler(
        scheduler_name,
        prediction_type="v_prediction" if v_pred else "epsilon",
    )

    return tokenizer, text_encoder, unet, scheduler


def load_diffusers_model_xl(
    pretrained_model_name_or_path: str,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[list[CLIPTokenizer], list[SDXL_TEXT_ENCODER_TYPE], UNet2DConditionModel,]:
    # returns tokenizer, tokenizer_2, text_encoder, text_encoder_2, unet

    tokenizers = [
        CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        ),
        CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
            pad_token_id=0,  # same as open clip
        ),
    ]

    text_encoders = [
        CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        ),
        CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        ),
    ]

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=weight_dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )

    return tokenizers, text_encoders, unet


def load_checkpoint_model_xl(
    checkpoint_path: str,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[list[CLIPTokenizer], list[SDXL_TEXT_ENCODER_TYPE], UNet2DConditionModel,]:
    pipe = StableDiffusionXLPipeline.from_single_file(
        checkpoint_path,
        torch_dtype=weight_dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )

    unet = pipe.unet
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
    if len(text_encoders) == 2:
        text_encoders[1].pad_token_id = 0

    del pipe

    return tokenizers, text_encoders, unet


def load_models_xl_(
    pretrained_model_name_or_path: str,
    scheduler_name: AVAILABLE_SCHEDULERS,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[
    list[CLIPTokenizer],
    list[SDXL_TEXT_ENCODER_TYPE],
    UNet2DConditionModel,
    SchedulerMixin,
]:
    if pretrained_model_name_or_path.endswith(
        ".ckpt"
    ) or pretrained_model_name_or_path.endswith(".safetensors"):
        (
            tokenizers,
            text_encoders,
            unet,
        ) = load_checkpoint_model_xl(pretrained_model_name_or_path, weight_dtype)
    else:  # diffusers
        (
            tokenizers,
            text_encoders,
            unet,
        ) = load_diffusers_model_xl(pretrained_model_name_or_path, weight_dtype)

    scheduler = create_noise_scheduler(scheduler_name)
        
    return tokenizers, text_encoders, unet, scheduler


def create_noise_scheduler(
    scheduler_name: AVAILABLE_SCHEDULERS = "ddpm",
    prediction_type: Literal["epsilon", "v_prediction"] = "epsilon",
) -> SchedulerMixin:
    # 正直、どれがいいのかわからない。元の実装だとDDIMとDDPMとLMSを選べたのだけど、どれがいいのかわからぬ。

    name = scheduler_name.lower().replace(" ", "_")
    if name == "ddim":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/ddim
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            prediction_type=prediction_type,  # これでいいの？
        )
    elif name == "ddpm":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/ddpm
        scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            prediction_type=prediction_type,
        )
    elif name == "lms":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/lms_discrete
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type=prediction_type,
        )
    elif name == "euler_a":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/euler_ancestral
        scheduler = EulerAncestralDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            # clip_sample=False,
            prediction_type=prediction_type,
        )
    else:
        raise ValueError(f"Unknown scheduler name: {name}")

    return scheduler


def load_models_xl(params):
    """
    Load all required models for training
    
    Args:
        params: Dictionary containing model parameters and configurations
        
    Returns:
        dict: Dictionary containing all loaded models and tokenizers
    """
    device = params['device']
    weight_dtype = params['weight_dtype']
    
    # Load SDXL components (UNet, text encoders, tokenizers)
    scheduler_name = 'ddim'
    tokenizers, text_encoders, unet, noise_scheduler = load_models_xl_(
        params['pretrained_model_name_or_path'],
        scheduler_name=scheduler_name,
    )
    
    # Move text encoders to device and set to eval mode
    for text_encoder in text_encoders:
        text_encoder.to(device, dtype=weight_dtype)
        text_encoder.requires_grad_(False)
        text_encoder.eval()
    
    # Set up UNet
    unet.to(device, dtype=weight_dtype)
    unet.requires_grad_(False)
    unet.eval()
    
    # Load tiny VAE for efficiency
    vae = AutoencoderTiny.from_pretrained(
        "madebyollin/taesdxl", 
        torch_dtype=weight_dtype
    )
    vae = vae.to(device, dtype=weight_dtype)
    vae.requires_grad_(False)
    
    # Load appropriate encoder (CLIP or DinoV2)
    if params['encoder'] == 'dinov2-small':
        clip_model = AutoModel.from_pretrained(
            'facebook/dinov2-small', 
            torch_dtype=weight_dtype
        )
        clip_processor= None
    else:
        clip_model = CLIPModel.from_pretrained(
            "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M", 
            torch_dtype=weight_dtype
        )
        clip_processor = CLIPProcessor.from_pretrained("wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M")
    clip_model = clip_model.to(device, dtype=weight_dtype)
    clip_model.requires_grad_(False)

    
    
    # If using DMD checkpoint, load it
    if params['distilled'] != 'None':
        if '.safetensors' in params['distilled']:
            unet.load_state_dict(load_file(params['distilled'], device=device))
        elif 'dmd2' in params['distilled']:
            repo_name = "tianweiy/DMD2"
            ckpt_name = "dmd2_sdxl_4step_unet_fp16.bin"
            unet.load_state_dict(torch.load(hf_hub_download(repo_name, ckpt_name)))
        else:
            unet.load_state_dict(torch.load(params['distilled']))

        
        # Set up LCM scheduler for DMD
        noise_scheduler = LCMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type="epsilon",
            original_inference_steps=1000
        )

    noise_scheduler.set_timesteps(params['max_denoising_steps'])
    pipe = StableDiffusionXLPipeline(vae = vae,
            text_encoder = text_encoders[0],
            text_encoder_2 = text_encoders[1],
            tokenizer = tokenizers[0],
            tokenizer_2 = tokenizers[1],
            unet = unet,
            scheduler = noise_scheduler)
    pipe.set_progress_bar_config(disable=True)
    return {
        'unet': unet,
        'vae': vae,
        'clip_model': clip_model,
        'clip_processor': clip_processor,
        'tokenizers': tokenizers,
        'text_encoders': text_encoders,
        'noise_scheduler': noise_scheduler
    }, pipe


def load_models_flux(params):
    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        params['pretrained_model_name_or_path'],
        subfolder="tokenizer",
        torch_dtype=params['weight_dtype'], device_map=params['device']
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        params['pretrained_model_name_or_path'],
        subfolder="tokenizer_2",
        torch_dtype=params['weight_dtype'], device_map=params['device']
    )
    
    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            params['pretrained_model_name_or_path'], 
            subfolder="scheduler",
            torch_dtype=params['weight_dtype'], device=params['device']
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    
    
    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        params['pretrained_model_name_or_path'],
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
       params['pretrained_model_name_or_path'], subfolder="text_encoder_2"
    )
    # Load the text encoders
    text_encoder_one, text_encoder_two = load_text_encoders(params['pretrained_model_name_or_path'], text_encoder_cls_one, text_encoder_cls_two, params['weight_dtype'])
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        params['pretrained_model_name_or_path'],
        subfolder="vae",
        torch_dtype=params['weight_dtype'], device_map='auto'
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        params['pretrained_model_name_or_path'], 
        subfolder="transformer", 
        torch_dtype=params['weight_dtype']
    )
    
    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    
    vae.to(params['device'])
    transformer.to(params['device'])
    text_encoder_one.to(params['device'])
    text_encoder_two.to(params['device'])

    # Load appropriate encoder (CLIP or DinoV2)
    if params['encoder'] == 'dinov2-small':
        clip_model = AutoModel.from_pretrained(
            'facebook/dinov2-small', 
            torch_dtype=params['weight_dtype']
        )
        clip_processor= None
    else:
        clip_model = CLIPModel.from_pretrained(
            "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M", 
            torch_dtype=params['weight_dtype']
        )
        clip_processor = CLIPProcessor.from_pretrained("wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M")
    clip_model = clip_model.to(params['device'], dtype=params['weight_dtype'])
    clip_model.requires_grad_(False)


    pipe = FluxPipeline(noise_scheduler,
                    vae,
                    text_encoder_one,
                    tokenizer_one,
                    text_encoder_two,
                    tokenizer_two,
                    transformer,
                   )
    pipe.set_progress_bar_config(disable=True)

    return {
        'transformer': transformer,
        'vae': vae,
        'clip_model': clip_model,
        'clip_processor': clip_processor,
        'tokenizers': [tokenizer_one, tokenizer_two],
        'text_encoders': [text_encoder_one,text_encoder_two],
        'noise_scheduler': noise_scheduler
    }, pipe

def save_checkpoint(networks, save_path, weight_dtype):
    """
    Save network weights and perform cleanup
    
    Args:
        networks: Dictionary of LoRA networks to save
        save_path: Path to save the checkpoints
        weight_dtype: Data type for the weights
    """
    print("Saving checkpoint...")
    
    try:
        # Create save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save each network's weights
        for net_idx, network in networks.items():
            save_name = f"{save_path}/slider_{net_idx}.pt"
            try:
                network.save_weights(
                    save_name,
                    dtype=weight_dtype,
                )
            except Exception as e:
                print(f"Error saving network {net_idx}: {str(e)}")
                continue
                
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        print("Checkpoint saved successfully.")
        
    except Exception as e:
        print(f"Error during checkpoint saving: {str(e)}")
        
    finally:
        # Ensure memory is cleaned up even if save fails
        torch.cuda.empty_cache()
        gc.collect()