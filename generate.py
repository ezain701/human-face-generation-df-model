"""
Generate images from a trained checkpoint and optionally compute FID.

Usage:
    python generate.py --checkpoint checkpoints/model_epoch_100.pt --num_images 300
    python generate.py --checkpoint checkpoints/model_epoch_100.pt --num_images 300 --compute_fid --compute_kid --data_dir data/celeba_hq_256
"""

import argparse
import os
import torch
from torchvision.utils import save_image

from models.unet import UNet
from models.noise_schedule import NoiseSchedule
from models.reverse_diffusion import sample as generate_samples
from evaluation.fid import compute_fid_score
from evaluation.kid import compute_kid

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images and evaluate FID")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_images", type=int, default=300)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--base_channels", type=int, default=64) # do not commit if less than 128
    parser.add_argument("--batch_size", type=int, default=16 , help="Batch size for generation")
    parser.add_argument("--output_dir", type=str, default="generated")
    parser.add_argument("--compute_fid", action="store_true", help="Compute FID against real images")
    parser.add_argument("--compute_kid", action="store_true", help="Compute KID against real images")
    parser.add_argument("--noise_schedule", type=str, default="linear", choices=["linear", "cosine"],
                    help="Noise schedule (must match training)")
    parser.add_argument("--data_dir", type=str, default=os.path.join(_PROJECT_ROOT, "data", "celeba_hq_256"), help="Real images dir (for FID)")
    parser.add_argument("--adagn", action="store_true", help="Use Adaptive Group Normalization for timestep conditioning (Dhariwal & Nichol 2021)")
    parser.add_argument("--zero_init_conv", action="store_true", help="Zero-initialize final conv of each residual block")

    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load model ---
    print(f"Using {args.noise_schedule} noise schedule with {args.timesteps} timesteps")
    schedule = NoiseSchedule(num_timesteps=args.timesteps, schedule=args.noise_schedule, device=device)
    model = UNet(base_channels=args.base_channels, adagn=args.adagn, zero_init_conv=args.zero_init_conv).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    # --- Generate images in batches ---
    all_images = []
    remaining = args.num_images

    while remaining > 0:
        batch_n = min(args.batch_size, remaining)
        print(f"Generating batch of {batch_n} images ({args.num_images - remaining + batch_n}/{args.num_images})...")
        images = generate_samples(schedule, model, batch_n, args.image_size)
        all_images.append(images.cpu())
        remaining -= batch_n

    all_images = torch.cat(all_images, dim=0)
    print(f"Generated images range: min={all_images.min().item():.4f}, max={all_images.max().item():.4f}, mean={all_images.mean().item():.4f}")    
    print(f"Any NaN: {torch.isnan(all_images).any()}, Any Inf: {torch.isinf(all_images).any()}")

    # --- Save individual images ---
    for i, img in enumerate(all_images):
        path = os.path.join(args.output_dir, f"generated_{i:04d}.png")
        save_image(img, path)

    grid_path = os.path.join(args.output_dir, "grid_sample.png")
    save_image(all_images[:16], grid_path, nrow=4)
    print(f"Saved {len(all_images)} images to {args.output_dir}/")
    print(f"Saved grid preview to {grid_path}")

    if args.compute_fid or args.compute_kid:
        from data.dataset import CelebAHQDataset
        test_dataset = CelebAHQDataset(args.data_dir, image_size=args.image_size, split="test")
        real_images = torch.stack([test_dataset[i] for i in range(len(test_dataset))])
        real_images = (real_images + 1) / 2  # rescale from [-1,1] to [0,1]

        print(f"Real images after rescaling: min={real_images.min().item():.4f}, max={real_images.max().item():.4f}")

    if args.compute_fid:
        print("Computing FID score...")
        fid = compute_fid_score(real_images, all_images, device=device)
        print(f"FID Score: {fid:.2f}")

    if args.compute_kid:
        print("Computing KID score...")
        kid_mean, kid_std = compute_kid(real_images, all_images, device=device)
        print(f"KID Score: {kid_mean:.4f} ± {kid_std:.4f}")

if __name__ == "__main__":
    main()
