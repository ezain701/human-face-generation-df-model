"""
Noise schedule for the diffusion process.

Supports:
- linear beta schedule
- cosine noise schedule (Nichol & Dhariwal style)

All derived coefficients (alphas, cumulative products, etc.) are
precomputed here for use by the forward and reverse processes.
"""

import math
import torch
import torch.nn.functional as F


class NoiseSchedule:
    def __init__(
        self,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        schedule_type="linear",
        cosine_s=0.008,
        device="cpu",
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        self.schedule_type = schedule_type
        self.cosine_s = cosine_s

        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)

        elif schedule_type == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps, cosine_s).to(device)

        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule from Nichol & Dhariwal:
        https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, min=1e-8, max=0.999)

        return betas.float()

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