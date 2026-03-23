import argparse

import torch
from torch.optim import Adam
from tqdm import tqdm

from .datasets import get_dataloaders
from .diffusion import DiffusionScheduler
from .model_unet import SimpleUNet
from .utils import ensure_dir, load_config, save_image_grid, save_loss_plot, set_seed, device_from_string


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = device_from_string(cfg["train"]["device"])
    train_loader, _ = get_dataloaders(cfg)

    model = SimpleUNet(
        in_channels=cfg["model"]["in_channels"],
        base_channels=cfg["model"]["base_channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        time_emb_dim=cfg["model"]["time_emb_dim"],
    ).to(device)

    diffusion = DiffusionScheduler(
        timesteps=cfg["diffusion"]["timesteps"],
        beta_schedule=cfg["diffusion"]["beta_schedule"],
        device=device,
    )

    optimizer = Adam(model.parameters(), lr=cfg["train"]["lr"])
    losses = []

    ckpt_dir = ensure_dir("checkpoints")
    loss_dir = ensure_dir("results/losses")
    grid_dir = ensure_dir("results/sample_grids")

    epochs = cfg["train"]["epochs"]
    output_name = cfg["train"]["output_name"]

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            batch = batch.to(device)
            loss = diffusion.loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"].get("grad_clip", 1.0))
            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        torch.save(model.state_dict(), ckpt_dir / f"{output_name}_latest.pt")
        if epoch % cfg["train"]["save_every"] == 0:
            torch.save(model.state_dict(), ckpt_dir / f"{output_name}_epoch_{epoch}.pt")

        samples = diffusion.sample(
            model,
            image_size=cfg["dataset"]["image_size"],
            batch_size=min(cfg["sample"]["num_images"], 16),
            channels=cfg["model"]["in_channels"],
        )
        save_image_grid(samples.cpu(), grid_dir / f"{output_name}_epoch_{epoch}.png", nrow=4)
        save_loss_plot(losses, loss_dir / f"{output_name}_loss.png")

    print("Training complete.")
    print(f"Checkpoint saved to checkpoints/{output_name}_latest.pt")


if __name__ == "__main__":
    main()
