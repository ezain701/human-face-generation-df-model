"""
Noise schedule for the diffusion process.

Defines how much noise is added at each timestep. Two schedules are supported:

- Linear: beta increases linearly from beta_start to beta_end
  over T steps. Simple and widely used, but destroys image information rapidly
  in later timesteps, leaving the last portion of the forward process with
  little useful signal to learn from. Based on Ho et al., 2020

- Cosine: the cumulative signal-preservation factor
  alphas_cumprod is defined directly as a shifted squared cosine, with betas
  derived from it. This preserves image information more gradually across
  timesteps and typically yields improved sample quality. Based on Nichol & Dhariwal, 2021

"""

import torch
import torch.nn.functional as F
import math

def linear_beta_schedule(num_timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear schedule from the original Ho et al. DDPM paper."""
    return torch.linspace(beta_start, beta_end, num_timesteps)

def cosine_beta_schedule(num_timesteps, s=0.008):
    """Cosine schedule from Nichol & Dhariwal's Improved DDPM paper.
    """
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps) / num_timesteps
    # Squared cosine that smoothly goes from 1 at t=0 to near 0 at t=1
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    # Normalize so that ᾱ_0 = 1
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    # Defensive measure to prevent β_t from ever reaching 1.0, which would cause α_t = 0
    return betas.clamp(max=0.999)

class NoiseSchedule:

    # All derived coefficients are precomputed at initialization.
    def __init__(self, num_timesteps=1000, schedule="linear", 
                 beta_start=1e-4, beta_end=0.02, cosine_s=0.008, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        self.schedule = schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.cosine_s = cosine_s
            
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
