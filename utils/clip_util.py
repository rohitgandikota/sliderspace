from typing import List, Optional
import math, random, os
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.decomposition import PCA


def extract_clip_features(clip, image, encoder):
    """
    Extracts feature embeddings from an image using either CLIP or DINOv2 models.
    
    Args:
        clip (torch.nn.Module): The feature extraction model (either CLIP or DINOv2)
        image (torch.Tensor): Input image tensor normalized according to model requirements
        encoder (str): Type of encoder to use ('dinov2-small' or 'clip')
    
    Returns:
        torch.Tensor: Feature embeddings extracted from the image
        
    Note:
        - For DINOv2 models, uses the pooled output features
        - For CLIP models, uses the image features from the vision encoder
        - The input image should already be properly resized and normalized
    """
    # Handle DINOv2 models
    if 'dino' in encoder:
        denoised = clip(image)
        denoised = denoised.pooler_output
    # Handle CLIP models
    else:
        denoised = clip.get_image_features(image)
    
    return denoised

@torch.no_grad()
def compute_clip_pca(
    diverse_prompts: List[str],
    pipe,
    clip_model,
    clip_processor,
    device,
    guidance_scale,
    params,
    total_samples = 5000,
    num_pca_components = 100,
    batch_size = 10
    
) -> torch.Tensor:
    """
    Extract CLIP features from generated images based on prompts.
    
    Args:
        diverse_prompts: List of prompts to generate images from
        model_components: Various model components needed for generation
        args: Training arguments
        
    Returns:
        Tensor of CLIP principle components
    """
    
    
    # Calculate how many total batches we need
    num_batches = math.ceil(total_samples / batch_size)
    # Randomly sample prompts (with replacement if needed)
    sampled_prompts_clip = random.choices(diverse_prompts, k=num_batches)
    
    clip_features_path = f"{params['savepath_training_images']}/clip_principle_directions.pt"
    
    if os.path.exists(clip_features_path):
        df = pd.read_csv(f"{params['savepath_training_images']}/training_data.csv")
        prompts_training = list(df.prompt)
        image_paths = list(df.image_path)
        return torch.load(clip_features_path).to(device), prompts_training, image_paths
    
    os.makedirs(params['savepath_training_images'], exist_ok=True)
    
    # Generate images and extract features
    img_idx = 0
    clip_features = []
    image_paths = []
    prompts_training = []
    print('Calculating Semantic PCA')
    
    for prompt in tqdm(sampled_prompts_clip):
        if 'max_sequence_length' in params:
            images = pipe(prompt, 
                     num_images_per_prompt = batch_size,
                     num_inference_steps = params['max_denoising_steps'],
                     guidance_scale=guidance_scale,
                     max_sequence_length = params['max_sequence_length'],
                     height = params['height'],
                     width = params['width'],
                     ).images
        else:  
            images = pipe(prompt, 
                         num_images_per_prompt = batch_size,
                         num_inference_steps = params['max_denoising_steps'],
                         guidance_scale=guidance_scale,
                         height = params['height'],
                         width = params['width'],
                         ).images

        
        # Process images
        clip_inputs = clip_processor(images=images, return_tensors="pt", padding=True)
        pixel_values = clip_inputs['pixel_values'].to(device)
        
        # Get image embeddings
        with torch.no_grad():
            image_features = clip_model.get_image_features(pixel_values)
            
        # Normalize embeddings
        clip_feats = image_features / image_features.norm(dim=1, keepdim=True)
        clip_features.append(clip_feats)

        for im in images:
            image_path = f"{params['savepath_training_images']}/{img_idx}.png"
            im.save(image_path)
            image_paths.append(image_path)
            prompts_training.append(prompt)
            img_idx += 1

    
    clip_features = torch.cat(clip_features)

    
    # Calculate principle components
    pca = PCA(n_components=num_pca_components)
    clip_embeds_np = clip_features.float().cpu().numpy()
    pca.fit(clip_embeds_np)
    clip_principles = torch.from_numpy(pca.components_).to(device, dtype=pipe.vae.dtype)
    
    # Save results
    torch.save(clip_principles, clip_features_path)
    pd.DataFrame({
        'prompt': prompts_training,
        'image_path': image_paths
    }).to_csv(f"{params['savepath_training_images']}/training_data.csv", index=False)
    
    return clip_principles, prompts_training, image_paths