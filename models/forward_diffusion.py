"""
Forward diffusion process.

Gradually adds Gaussian noise to a clean image over T timesteps.
Also computes the training loss (MSE between predicted and actual noise).

Reference: Ho et al., "Denoising Diffusion Probabilistic Models", 2020
"""

import torch
import torch.nn.functional as F


def extract(tensor, t, shape):
    """Extract values from tensor at indices t, reshape for broadcasting."""
    out = tensor.gather(-1, t)
    return out.view(-1, *((1,) * (len(shape) - 1)))


def q_sample(schedule, x_start, t, noise=None):
    """
    Forward process: sample x_t given x_0 and timestep t.

    q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alpha = extract(schedule.sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alpha = extract(schedule.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise


def p_losses(schedule, model, x_start, t):
    """
    Compute training loss: predict the noise that was added at timestep t.

    Loss = MSE(predicted_noise, actual_noise)
    """
    noise = torch.randn_like(x_start)
    x_noisy = q_sample(schedule, x_start, t, noise=noise)
    predicted_noise = model(x_noisy, t)
    return F.mse_loss(predicted_noise, noise)
