"""
Main entry point for training the diffusion model.

Usage:
    python main.py --dataset celeba --data_dir data/celeba_hq_256
    python main.py --dataset butterfly --data_dir data/butterfly --image_size 128 --epochs 50
"""

import argparse
import copy
import torch

from models.unet import UNet
from models.noise_schedule import NoiseSchedule
from data.dataset import CelebAHQDataset, ButterflyDataset, get_dataloader
from training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DDPM diffusion model")
    parser.add_argument("--dataset", type=str, default="celeba", choices=["celeba", "butterfly"])
    parser.add_argument("--data_dir", type=str, default="data/celeba_hq_256")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accum_steps", type=int, default=1, help="number of batches to accumulate gradients before optimizer step")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--base_channels", type=int, default=128)
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=1.0,
        help="Max gradient norm. Set <= 0 to disable gradient clipping.",
    )
    parser.add_argument("--use_scheduler", action="store_true", help="Enable cosine annealing scheduler")
    parser.add_argument("--scheduler_tmax", type=int, default=None, help="T_max for CosineAnnealingLR (defaults to total epochs)")
    parser.add_argument("--scheduler_eta_min", type=float, default=1e-6, help="Minimum learning rate for CosineAnnealingLR")


    # EMA options
    parser.add_argument("--use_ema", action="store_true", help="Enable EMA model")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay factor")

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Noise schedule ---
    schedule = NoiseSchedule(num_timesteps=args.timesteps, device=device)
    schedule = NoiseSchedule(num_timesteps=args.timesteps, schedule=args.noise_scheduler, device=device)

    # --- Model ---
    model = UNet(base_channels=args.base_channels).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- EMA model ---
    ema_model = None
    if args.use_ema:
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad = False
        print(f"EMA enabled (decay={args.ema_decay})")

    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.use_scheduler:
        t_max = args.scheduler_tmax if args.scheduler_tmax is not None else args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=args.scheduler_eta_min,
        )
        print(
            f"Scheduler enabled: CosineAnnealingLR(T_max={t_max}, eta_min={args.scheduler_eta_min})"
        )

    # --- Resume from checkpoint ---
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)

        model.load_state_dict(ckpt["model_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            print("  Loaded scheduler state from checkpoint")
        model = model.to(device)

        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Move optimizer state tensors to the same device as the model
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

        # Force current CLI learning rate after resume
        for group in optimizer.param_groups:
            group["lr"] = args.lr

        if scheduler is not None and "scheduler_state_dict" not in ckpt:
            t_max = args.scheduler_tmax if args.scheduler_tmax is not None else args.epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=args.scheduler_eta_min,
            )

        if args.use_ema and ema_model is not None:
            if "ema_model_state_dict" in ckpt:
                ema_model.load_state_dict(ckpt["ema_model_state_dict"])
                ema_model = ema_model.to(device)
                print("  Loaded EMA weights from checkpoint")
            else:
                ema_model.load_state_dict(model.state_dict())
                print("  No EMA weights found in checkpoint; initialized EMA from model weights")

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
        ema_model=ema_model,
        ema_decay=args.ema_decay,
        schedule=schedule,
        dataloader=dataloader,
        optimizer=optimizer,
        clip_grad=args.clip_grad if args.clip_grad > 0 else None,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        scheduler=scheduler,
    )
    trainer.train(
        num_epochs=args.epochs,
        sample_every=args.sample_every,
        image_size=args.image_size,
        start_epoch=start_epoch,
        accum_steps=args.accum_steps
    )

    print("Training complete.")


if __name__ == "__main__":
    main()