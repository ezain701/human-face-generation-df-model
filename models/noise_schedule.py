"""
Noise schedule for the diffusion process.

Defines how much noise is added at each timestep. The linear schedule
linearly increases beta from beta_start to beta_end over T steps.
All derived coefficients (alphas, cumulative products, etc.) are
precomputed here for use by the forward and reverse processes.
"""

import torch
import torch.nn.functional as F


class NoiseSchedule:

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device

        #self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps, device=device) / num_timesteps

        alphas_cumprod = torch.cos((x + 0.008) / 1.008 * torch.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        self.alphas_cumprod = alphas_cumprod[:-1]
        self.alphas_cumprod_prev = alphas_cumprod[1:]

        self.betas = 1 - (self.alphas_cumprod / self.alphas_cumprod_prev)
        self.betas = torch.clamp(self.betas, 0.0001, 0.999)


        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)
        self.posterior_log_variance = torch.log(self.posterior_variance)
        
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
