"""
Main entry point for training the diffusion model.

Usage:
    python main.py --dataset celeba --data_dir data/celeba_hq_256
    python main.py --dataset butterfly --data_dir data/butterfly --image_size 128 --epochs 50
"""

import argparse
import copy
import torch
import math
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
    parser.add_argument(
        "--reset_lr_on_resume",
        action="store_true",
        help="Force learning rate from CLI after resuming; otherwise keep checkpoint LR",
    )
    parser.add_argument("--use_scheduler", action="store_true", help="Enable cosine annealing scheduler")
    parser.add_argument("--scheduler_tmax", type=int, default=None, help="T_max for CosineAnnealingLR (defaults to total epochs)")
    parser.add_argument("--scheduler_eta_min", type=float, default=1e-6, help="Minimum learning rate for CosineAnnealingLR")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")

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

    # --- Dataset ---
    if args.dataset == "celeba":
        dataset = CelebAHQDataset(args.data_dir, image_size=args.image_size, split="train")
    else:
        dataset = ButterflyDataset(args.data_dir, image_size=args.image_size)

    dataloader = get_dataloader(dataset, batch_size=args.batch_size)
    print(f"Dataset: {args.dataset} — {len(dataset)} images, {len(dataloader)} batches/epoch")

    # --- Scheduler ---
    scheduler = None
    if args.use_scheduler:
        total_steps = args.epochs * len(dataloader)
        warmup_steps = args.warmup_steps

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(args.scheduler_eta_min / args.lr, cosine)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"Scheduler: warmup ({warmup_steps} steps) + cosine decay (total_steps={total_steps})")

    # --- Resume from checkpoint ---
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)

        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)

        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Move optimizer state tensors to the same device as the model
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

        # Load scheduler state if available
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            print("  Loaded scheduler state from checkpoint")
        elif scheduler is not None:
            print("  No scheduler state found in checkpoint; scheduler will start fresh")

        # Optionally reset LR from CLI after resume
        if args.reset_lr_on_resume:
            for group in optimizer.param_groups:
                group["lr"] = args.lr
            print(f"  Reset learning rate to {args.lr}")
        else:
            print(f"  Keeping resumed learning rate: {optimizer.param_groups[0]['lr']}")

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
    )

    print("Training complete.")

if __name__ == "__main__":
    main()