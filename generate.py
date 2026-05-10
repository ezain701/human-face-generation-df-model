"""
Generate images from a trained checkpoint and optionally compute FID.

Usage:
    python generate.py --checkpoint checkpoints/model_epoch_100.pt --num_images 300
    python generate.py --checkpoint checkpoints/model_epoch_100.pt --num_images 300 --compute_fid --data_dir data/celeba_hq_256
    python generate.py --checkpoint checkpoints/model_epoch_100.pt --num_images 300 --use_ema
"""

import argparse
import hashlib
import json
import os
import re
import torch
from torchvision.utils import save_image

from models.unet import UNet
from models.clip_text_encoder import CLIPTextEncoder, CLIPTokenizerAdapter
from models.text_encoder import PromptTextEncoder, SimpleTokenizer
from models.noise_schedule import NoiseSchedule
from models.reverse_diffusion import sample as generate_samples
from evaluation.fid import compute_fid_score


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images and evaluate FID")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_images", type=int, default=300)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--base_channels", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--output_dir", type=str, default="generated")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt for conditioned checkpoints; metadata and deterministic seed label for unconditional checkpoints.",
    )
    parser.add_argument("--text_conditioning", action="store_true", help="Use a prompt-conditioned checkpoint")
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="Classifier-free guidance scale")
    parser.add_argument("--text_embed_dim", type=int, default=256)
    parser.add_argument("--text_vocab_size", type=int, default=8192)
    parser.add_argument("--text_max_length", type=int, default=32)
    parser.add_argument("--text_encoder_type", type=str, default=None, choices=["simple", "clip"])
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation. If omitted and --prompt is set, a stable seed is derived from the prompt.",
    )
    parser.add_argument("--compute_fid", action="store_true", help="Compute FID against real images")
    parser.add_argument("--data_dir", type=str, default="data/celeba_hq_256", help="Real images dir (for FID)")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA weights for generation if available")
    return parser.parse_args()


def _stable_seed_from_prompt(prompt: str) -> int:
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**31)


def _slugify_prompt(prompt: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", prompt.strip().lower()).strip("_")
    return slug[:48] or "prompt"


def _set_generation_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_generation_manifest(
    output_dir: str,
    prompt: str | None,
    seed: int | None,
    checkpoint: str,
    text_conditioning: bool,
    guidance_scale: float,
) -> None:
    manifest_path = os.path.join(output_dir, "generation_manifest.json")
    manifest = {
        "checkpoint": checkpoint,
        "prompt": prompt,
        "seed": seed,
        "text_conditioning": text_conditioning,
        "guidance_scale": guidance_scale if text_conditioning else None,
        "note": "Prompt conditions generation only when text_conditioning is true and the checkpoint includes text encoder weights.",
    }
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    seed = args.seed
    if seed is None and args.prompt:
        seed = _stable_seed_from_prompt(args.prompt)
    if seed is not None:
        _set_generation_seed(seed)
        print(f"Using generation seed: {seed}")

    if args.prompt:
        print(f"Prompt label: {args.prompt}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    text_config = ckpt.get("text_config", {})
    if args.text_conditioning and text_config:
        args.text_encoder_type = text_config.get("text_encoder_type", args.text_encoder_type)
        args.text_embed_dim = text_config.get("text_embed_dim", args.text_embed_dim)
        args.text_vocab_size = text_config.get("text_vocab_size", args.text_vocab_size)
        args.text_max_length = text_config.get("text_max_length", args.text_max_length)
        args.clip_model_name = text_config.get("clip_model_name", args.clip_model_name)
    if args.text_conditioning and args.text_encoder_type is None:
        args.text_encoder_type = "simple"

    # --- Load model ---
    schedule = NoiseSchedule(num_timesteps=args.timesteps, device=device)
    text_emb_dim = None
    tokenizer = None
    text_encoder = None
    if args.text_conditioning:
        if args.text_encoder_type == "clip":
            tokenizer = CLIPTokenizerAdapter(
                model_name=args.clip_model_name,
                max_length=args.text_max_length,
            )
            text_encoder = CLIPTextEncoder(model_name=args.clip_model_name, freeze=True).to(device)
            text_emb_dim = text_config.get("clip_embed_dim", text_encoder.embed_dim)
        else:
            tokenizer = SimpleTokenizer(vocab_size=args.text_vocab_size, max_length=args.text_max_length)
            text_encoder = PromptTextEncoder(
                vocab_size=args.text_vocab_size,
                max_length=args.text_max_length,
                embed_dim=args.text_embed_dim,
            ).to(device)
            text_emb_dim = args.text_embed_dim
    model = UNet(base_channels=args.base_channels, text_emb_dim=text_emb_dim).to(device)
    if args.use_ema and "ema_model_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_model_state_dict"])
        print(f"Loaded EMA weights from checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        if args.use_ema:
            print("EMA requested but no EMA weights found in checkpoint; using raw model weights")
        print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    if args.text_conditioning:
        if args.text_encoder_type == "simple":
            text_key = "ema_text_encoder_state_dict" if args.use_ema and "ema_text_encoder_state_dict" in ckpt else "text_encoder_state_dict"
            if text_key not in ckpt:
                raise ValueError("Text conditioning requested, but checkpoint has no text encoder weights.")
            text_encoder.load_state_dict(ckpt[text_key])
            print(f"Loaded text encoder weights from checkpoint key: {text_key}")
        elif "text_encoder_state_dict" in ckpt or "ema_text_encoder_state_dict" in ckpt:
            text_key = "ema_text_encoder_state_dict" if args.use_ema and "ema_text_encoder_state_dict" in ckpt else "text_encoder_state_dict"
            text_encoder.load_state_dict(ckpt[text_key])
            print(f"Loaded finetuned CLIP text encoder weights from checkpoint key: {text_key}")
        text_encoder.eval()

    model.eval()

    # --- Generate images in batches ---
    all_images = []
    remaining = args.num_images

    while remaining > 0:
        batch_n = min(args.batch_size, remaining)
        print(f"Generating batch of {batch_n} images ({args.num_images - remaining + batch_n}/{args.num_images})...")
        text_emb = None
        uncond_text_emb = None
        if args.text_conditioning:
            prompts = [args.prompt or ""] * batch_n
            token_ids = tokenizer(prompts, device=device)
            uncond_ids = tokenizer([""] * batch_n, device=device)
            with torch.no_grad():
                text_emb = text_encoder(token_ids)
                uncond_text_emb = text_encoder(uncond_ids)
        images = generate_samples(
            schedule,
            model,
            batch_n,
            args.image_size,
            text_emb=text_emb,
            uncond_text_emb=uncond_text_emb,
            guidance_scale=args.guidance_scale,
        )
        all_images.append(images.cpu())
        remaining -= batch_n

    all_images = torch.cat(all_images, dim=0)
    _save_generation_manifest(
        args.output_dir,
        args.prompt,
        seed,
        args.checkpoint,
        args.text_conditioning,
        args.guidance_scale,
    )

    # --- Save individual images ---
    prefix = "generated"
    if args.prompt:
        prefix = f"{prefix}_{_slugify_prompt(args.prompt)}"

    for i, img in enumerate(all_images):
        path = os.path.join(args.output_dir, f"{prefix}_{i:04d}.png")
        save_image(img, path)

    grid_name = "grid_sample.png"
    if args.prompt:
        grid_name = f"grid_sample_{_slugify_prompt(args.prompt)}.png"
    grid_path = os.path.join(args.output_dir, grid_name)
    save_image(all_images[:16], grid_path, nrow=4)
    print(f"Saved {len(all_images)} images to {args.output_dir}/")
    print(f"Saved grid preview to {grid_path}")

    # --- Compute FID ---
    if args.compute_fid:
        from data.dataset import CelebAHQDataset

        print("Computing FID score...")
        test_dataset = CelebAHQDataset(args.data_dir, image_size=args.image_size, split="test")
        real_images = torch.stack([test_dataset[i] for i in range(len(test_dataset))])
        real_images = (real_images + 1) / 2  # rescale from [-1,1] to [0,1]

        fid = compute_fid_score(real_images, all_images, device=device)
        print(f"FID Score: {fid:.2f}")


if __name__ == "__main__":
    main()
