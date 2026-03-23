import torch
import torch.nn.functional as F
from tqdm import tqdm


class DiffusionScheduler:
    def __init__(self, timesteps: int = 300, beta_schedule: str = "linear", device: str = "cpu"):
        self.timesteps = timesteps
        self.device = device

        if beta_schedule == "linear":
            self.betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        else:
            raise ValueError(f"Unsupported beta schedule: {beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]], dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()

    def add_noise(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alphas * x_start + sqrt_one_minus * noise, noise

    def loss(self, model, x_start):
        t = self.sample_timesteps(x_start.shape[0])
        x_noisy, noise = self.add_noise(x_start, t)
        predicted_noise = model(x_noisy, t.float())
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, model, image_size: int, batch_size: int = 16, channels: int = 3):
        model.eval()
        x = torch.randn((batch_size, channels, image_size, image_size), device=self.device)
        for i in tqdm(reversed(range(self.timesteps)), total=self.timesteps, desc="Sampling"):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.float32)
            betas_t = self.betas[i]
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[i]
            sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[i])

            model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

            if i > 0:
                noise = torch.randn_like(x)
                posterior_var_t = self.posterior_variance[i]
                x = model_mean + torch.sqrt(posterior_var_t) * noise
            else:
                x = model_mean

        return x.clamp(-1, 1)
