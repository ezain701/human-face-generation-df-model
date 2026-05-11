"""
Reverse diffusion process (sampling / generation).

Starting from pure Gaussian noise, iteratively denoises over T steps
using the trained U-Net to produce a clean image.

Reference: Ho et al., "Denoising Diffusion Probabilistic Models", 2020
"""

import torch
from .forward_diffusion import extract


@torch.no_grad()
def p_sample(schedule, model, x, t, text_emb=None, uncond_text_emb=None, guidance_scale=1.0):
    """
    Single reverse step: predict and remove noise at timestep t.

    Computes the posterior mean and optionally adds noise (for t > 0).
    """
    t_batch = torch.full((x.shape[0],), t, device=schedule.device, dtype=torch.long)
    predicted_noise = model(x, t_batch, text_emb)
    if uncond_text_emb is not None and guidance_scale != 1.0:
        uncond_noise = model(x, t_batch, uncond_text_emb)
        predicted_noise = uncond_noise + guidance_scale * (predicted_noise - uncond_noise)

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
def p_sample_loop(
    schedule,
    model,
    shape,
    text_emb=None,
    uncond_text_emb=None,
    guidance_scale=1.0,
    return_intermediates=False,
    num_intermediate_steps=8,
):
    """Full reverse process: generate images by denoising from pure noise."""
    x = torch.randn(shape, device=schedule.device)
    intermediates = []
    capture_steps = set()

    if return_intermediates:
        num_intermediate_steps = max(2, int(num_intermediate_steps))
        capture_steps = {
            int(t)
            for t in torch.linspace(
                schedule.num_timesteps - 1,
                0,
                num_intermediate_steps - 1,
                device="cpu",
            )
        }
        intermediates.append(((x.clamp(-1, 1) + 1) / 2).detach().cpu())

    for t in reversed(range(schedule.num_timesteps)):
        x = p_sample(schedule, model, x, t, text_emb, uncond_text_emb, guidance_scale)
        if return_intermediates and t in capture_steps:
            intermediates.append(((x.clamp(-1, 1) + 1) / 2).detach().cpu())

    samples = (x.clamp(-1, 1) + 1) / 2
    if return_intermediates:
        return samples, torch.cat(intermediates, dim=0)
    return samples


@torch.no_grad()
def sample(
    schedule,
    model,
    num_images,
    image_size=256,
    channels=3,
    text_emb=None,
    uncond_text_emb=None,
    guidance_scale=1.0,
    return_intermediates=False,
    num_intermediate_steps=8,
):
    """Generate a batch of images from noise."""
    shape = (num_images, channels, image_size, image_size)
    return p_sample_loop(
        schedule,
        model,
        shape,
        text_emb,
        uncond_text_emb,
        guidance_scale,
        return_intermediates=return_intermediates,
        num_intermediate_steps=num_intermediate_steps,
    )
