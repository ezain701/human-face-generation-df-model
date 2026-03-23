import argparse

import torch
from torchvision.utils import save_image

from .diffusion import DiffusionScheduler
from .model_unet import SimpleUNet
from .utils import ensure_dir, load_config, set_seed, device_from_string


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = device_from_string(cfg["train"]["device"])

    model = SimpleUNet(
        in_channels=cfg["model"]["in_channels"],
        base_channels=cfg["model"]["base_channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        time_emb_dim=cfg["model"]["time_emb_dim"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    diffusion = DiffusionScheduler(
        timesteps=cfg["diffusion"]["timesteps"],
        beta_schedule=cfg["diffusion"]["beta_schedule"],
        device=device,
    )

    out_dir = ensure_dir(args.out_dir or f"results/generated_{cfg['train']['output_name']}")
    batch_size = 16
    generated = 0

    while generated < args.num_samples:
        current_bs = min(batch_size, args.num_samples - generated)
        samples = diffusion.sample(
            model,
            image_size=cfg["dataset"]["image_size"],
            batch_size=current_bs,
            channels=cfg["model"]["in_channels"],
        ).cpu()
        samples = (samples + 1.0) / 2.0
        for i in range(current_bs):
            save_image(samples[i], out_dir / f"{generated + i:04d}.png")
        generated += current_bs

    print(f"Saved {args.num_samples} samples to {out_dir}")


if __name__ == "__main__":
    main()
