"""
Noise schedule for the diffusion process.

Defines how much noise is added at each timestep. Two schedules are supported:

- Linear (Ho et al., 2020): beta increases linearly from beta_start to beta_end
  over T steps.

- Cosine (Nichol & Dhariwal, 2021): preserves image information more gradually 
  across timesteps.

All derived coefficients (alphas, cumulative products, etc.) are precomputed
at initialization.
"""

import torch
import torch.nn.functional as F
import math

def linear_beta_schedule(num_timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear schedule from the original DDPM paper."""
    return torch.linspace(beta_start, beta_end, num_timesteps)


def cosine_beta_schedule(num_timesteps, s=0.008):
    """Cosine schedule from Nichol & Dhariwal 'Improved DDPM' (2021).
    
    Defines alphas_cumprod as a cosine function, then derives betas from it.
    The small offset s prevents beta_t from being too small near t=0.
    """
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps) / num_timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(max=0.999)

class NoiseSchedule:

    def __init__(self, num_timesteps=1000, schedule="linear",
                 beta_start=1e-4, beta_end=0.02, cosine_s=0.008, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        self.schedule = schedule
        
        if schedule == "linear":
            self.betas = linear_beta_schedule(num_timesteps, beta_start, beta_end)
        elif schedule == "cosine":
            self.betas = cosine_beta_schedule(num_timesteps, cosine_s)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        self.betas = self.betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device):
        """Move all tensors to a new device."""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self
