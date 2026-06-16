"""
Reverse diffusion process (sampling / generation).

Starting from pure Gaussian noise, iteratively denoises over T steps
using the trained U-Net to produce a clean image.

Reference: Ho et al., "Denoising Diffusion Probabilistic Models", 2020
"""

import torch
from .forward_diffusion import extract

@torch.no_grad()
def p_sample(schedule, model, x, t):
    """
    Single reverse step: predict and remove noise at timestep t.

    Computes the posterior mean and optionally adds noise (for t > 0).
    """
    t_batch = torch.full((x.shape[0],), t, device=schedule.device, dtype=torch.long)
    predicted_noise = model(x, t_batch)

    beta = extract(schedule.betas, t_batch, x.shape)
    sqrt_recip_alpha = extract(schedule.sqrt_recip_alphas, t_batch, x.shape)
    sqrt_one_minus_alpha_cumprod = extract(
        schedule.sqrt_one_minus_alphas_cumprod, t_batch, x.shape
    )

    mean = sqrt_recip_alpha * (x - beta * predicted_noise / sqrt_one_minus_alpha_cumprod)

    if t > 0:
        posterior_var = extract(schedule.posterior_variance, t_batch, x.shape)
        noise = torch.randn_like(x)
        return mean + torch.sqrt(posterior_var) * noise
    return mean


@torch.no_grad()
def p_sample_loop(schedule, model, shape):
    """Full reverse process: generate images by denoising from pure noise."""
    model.eval()
    x = torch.randn(shape, device=schedule.device)

    for t in reversed(range(schedule.num_timesteps)):
        x = p_sample(schedule, model, x, t)

    return (x.clamp(-1, 1) + 1) / 2


@torch.no_grad()
def sample(schedule, model, num_images, image_size=64, channels=3):
    """Generate a batch of images from noise."""
    shape = (num_images, channels, image_size, image_size)
    return p_sample_loop(schedule, model, shape)