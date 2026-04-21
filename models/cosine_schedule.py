# Cosine Schedule

import torch
import torch.nn.Functional as F
import math


class Cosine_Noise_Schedule:

  def __init__(num_timesteps, beta_start=0, beta_end=num_timesteps+1, s=0.008):
    x = torch.linspace(beta_start, num_timesteps, beta_end, endpoint=True)
    
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    return torch.clamp(betas, 1e-5, 0.02)