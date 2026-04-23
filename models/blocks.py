"""Reusable building blocks for the U-Net architecture."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    """Encodes diffusion timestep as a sinusoidal embedding vector."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResidualBlock(nn.Module):
    """Residual block with timestep conditioning.
    
    - Additive, based on Ho et al. 2020. Timestep embedding added to feature maps.

    - AdaGN, based on Dhariwal & Nichol (2021). Timestep embedding is injected by
    predicting per-channel scale and shift parameters that modulate group
    normalization.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8, adagn=False, 
                 zero_init_conv=False):
        super().__init__()
        self.adagn = adagn
        self.zero_init_conv = zero_init_conv 

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
                
        if adagn:
            # Norm2 has no learnable affine params — they come from the timestep
            self.norm2 = nn.GroupNorm(groups, out_channels, affine=False)

            # Time MLP now outputs 2 * out_channels: scale and shift
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, 2 * out_channels),
            )
        else:
            # Additive: standard norm2 with learned affine
            self.norm2 = nn.GroupNorm(groups, out_channels)

            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels),
            )

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if zero_init_conv:
            # Zero-initialise the final convolution
            # Based on the zero_module pattern by Dhariwal & Nichols guided-diffusion https://github.com/openai/guided-diffusion/tree/main/guided_diffusion
            nn.init.zeros_(self.conv2.weight)
            nn.init.zeros_(self.conv2.bias)
        
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
    
    def forward(self, x, t):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        time_emb = self.time_mlp(t)

        if self.adagn:
            scale, shift = time_emb.chunk(2, dim=1)
            h = self.norm2(h)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        else:
            h = self.norm2(h) + time_emb[:, :, None, None]

        h = F.silu(h)
        h = self.conv2(h)
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block for capturing long-range dependencies."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        x = x.view(b, c, h * w).permute(0, 2, 1)
        x, _ = self.attention(x, x, x)
        x = x.permute(0, 2, 1).view(b, c, h, w)
        return x + residual


class Downsample(nn.Module):
    """Halves spatial resolution via stride-2 convolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Doubles spatial resolution via nearest interpolation + convolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)
