"""
Main entry point for training the diffusion model.

Usage:
    python main.py --dataset celeba --data_dir data/celeba_hq_256
    python main.py --dataset butterfly --data_dir data/butterfly --image_size 128 --epochs 50
"""

import argparse
import torch

from models.unet import UNet
from models.noise_schedule import NoiseSchedule
from data.dataset import CelebAHQDataset, ButterflyDataset, get_dataloader
from training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DDPM diffusion model")
    parser.add_argument("--dataset", type=str, default="celeba", choices=["celeba", "butterfly"])
    parser.add_argument("--data_dir", type=str, default="data/img_align_celeba")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--base_channels", type=int, default=128)
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Noise schedule ---
    schedule = NoiseSchedule(num_timesteps=args.timesteps, device=device)

    # --- Model ---
    model = UNet(base_channels=args.base_channels)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Resume from checkpoint ---
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"  Resumed at epoch {start_epoch}")

    # --- Dataset ---
    if args.dataset == "celeba":
        dataset = CelebAHQDataset(args.data_dir, image_size=args.image_size, split="train")
    else:
        dataset = ButterflyDataset(args.data_dir, image_size=args.image_size)

    dataloader = get_dataloader(dataset, batch_size=args.batch_size)
    print(f"Dataset: {args.dataset} — {len(dataset)} images, {len(dataloader)} batches/epoch")

    # --- Train ---
    trainer = Trainer(
        model=model,
        schedule=schedule,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )
    trainer.train(
        num_epochs=args.epochs,
        sample_every=args.sample_every,
        image_size=args.image_size,
        start_epoch=start_epoch,
    )

    print("Training complete.")


if __name__ == "__main__":
    main()
