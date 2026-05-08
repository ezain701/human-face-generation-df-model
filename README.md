# Human Face Generation with Diffusion Models

A PyTorch implementation of a Denoising Diffusion Probabilistic Model (DDPM) for generating realistic human faces, trained on the CelebA-HQ dataset.

Based on the paper: Ho et al., *"Denoising Diffusion Probabilistic Models"*, NeurIPS 2020.

## Project Structure

```
human-face-generation-df-model/
├── main.py                  # Training entry point
├── generate.py              # Image generation & FID evaluation
├── requirements.txt         # Python dependencies
│
├── models/
│   ├── blocks.py            # Reusable layers (ResBlock, Attention, Up/Downsample)
│   ├── unet.py              # U-Net noise prediction network
│   ├── noise_schedule.py    # Linear beta schedule and derived coefficients
│   ├── forward_diffusion.py # Forward process (q_sample) and training loss
│   └── reverse_diffusion.py # Reverse process (sampling / generation)
│
├── data/
│   ├── dataset.py           # CelebA-HQ and Butterfly dataset classes
│   ├── celeba_hq_256/       # CelebA-HQ images (not included, see Setup)
│   └── butterfly/           # Butterfly images for toy model (not included)
│
├── training/
│   └── trainer.py           # Training loop with checkpointing and sample generation
│
├── evaluation/
│   ├── fid.py               # FID score computation using InceptionV3
│   └── visualize.py         # Image grids, loss curves, diffusion process visualization
│
├── checkpoints/             # Saved model checkpoints (auto-created)
├── logs/                    # Training logs and sample images (auto-created)
└── generated/               # Generated images (auto-created)
```

## How It Works

The diffusion model operates in two phases:

1. **Forward Process (Training):** Gradually adds Gaussian noise to real images over T=1000 timesteps until they become pure noise. The model (U-Net) is trained to predict the noise that was added at each step.

2. **Reverse Process (Generation):** Starting from pure Gaussian noise, the trained U-Net iteratively removes noise over T steps, producing a clean image.

### Architecture

- **U-Net** with residual blocks, self-attention, and skip connections
- **Sinusoidal timestep embeddings** to condition the network on the current diffusion step
- **Linear noise schedule** with \(\beta\) linearly increasing from 1e-4 to 0.02

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download the CelebA-HQ dataset (256x256 resolution) and place the images in `data/celeba_hq_256/`:

```
data/celeba_hq_256/
├── 00000.jpg
├── 00001.jpg
├── ...
└── 29999.jpg
```

For the toy model, place butterfly images in `data/butterfly/`.

## Usage

### Train on CelebA-HQ (Default)

```bash
python main.py --data_dir data/celeba_hq_256 --epochs 100
```

### Train on Butterfly Toy Model

```bash
python main.py --dataset butterfly --data_dir data/butterfly --image_size 128 --epochs 50
```

### Resume Training from Checkpoint

```bash
python main.py --data_dir data/celeba_hq_256 --epochs 200 --resume checkpoints/model_epoch_100.pt
```

### Generate Images

```bash
python generate.py --checkpoint checkpoints/model_epoch_100.pt --num_images 300
```

To label a generation run and make it reproducible, pass `--prompt`. For unconditional checkpoints, the prompt is metadata and can derive a stable seed:

```bash
python generate.py \
    --checkpoint checkpoints/model_epoch_100.pt \
    --num_images 12 \
    --prompt "studio portrait, soft light"
```

If you want direct control over randomness, use `--seed`:

```bash
python generate.py \
    --checkpoint checkpoints/model_epoch_100.pt \
    --num_images 12 \
    --seed 12345
```

### Train with Prompt Conditioning

Prompt conditioning uses a small trainable text encoder and classifier-free guidance dropout. Captions can be supplied as JSON, JSONL, CSV, or TXT keyed by image filename or stem. CSV files should include columns such as `filename,prompt`.

```bash
python main.py \
    --data_dir data/celeba_hq_256 \
    --text_conditioning \
    --caption_file data/celeba_captions.csv \
    --caption_dropout 0.1
```

Generate from a prompt-conditioned checkpoint with classifier-free guidance:

```bash
python generate.py \
    --checkpoint checkpoints/model_epoch_100.pt \
    --text_conditioning \
    --prompt "studio portrait, soft light, smiling" \
    --guidance_scale 3.0
```

If no caption file is provided during training, every image receives a generic dataset-level caption. That is useful for testing the code path, but it will not teach the model meaningful prompt control.

For stronger prompt conditioning, use a frozen pretrained CLIP text encoder:

```bash
python main.py \
    --data_dir /scratch/pmoney/img_align_celeba \
    --text_conditioning \
    --text_encoder_type clip \
    --caption_file data/celeba_captions.csv
```

CLIP conditioning requires `transformers` and will download the CLIP text encoder the first time it runs.

During prompt-conditioned training, fixed prompt grids are saved automatically at each checkpoint as `logs/prompt_samples_epoch_*.png`. You can override the prompts inline:

```bash
python main.py \
    --text_conditioning \
    --sample_prompts \
    "a portrait photo of a smiling person with eyeglasses" \
    "a portrait photo of a person wearing a hat"
```

or with one prompt per line in a file:

```bash
python main.py --text_conditioning --sample_prompt_file prompts.txt
```

For CelebA, download the official attribute metadata (`list_attr_celeba.txt`) or the Kaggle CSV version (`list_attr_celeba.csv`), then convert it into prompt captions:

```bash
python make_celeba_captions.py \
    --attr_file list_attr_celeba.txt \
    --output data/celeba_captions.csv
```

### Generate Images and Compute FID

```bash
python generate.py \
    --checkpoint checkpoints/model_epoch_100.pt \
    --num_images 300 \
    --compute_fid \
    --data_dir data/celeba_hq_256
```

### All Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `celeba` | Dataset to use (`celeba` or `butterfly`) |
| `--data_dir` | `data/celeba_hq_256` | Path to image directory |
| `--image_size` | `192` | Image resolution |
| `--train_size` | `27000` | Number of CelebA training images to use |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `16` | Training batch size |
| `--lr` | `2e-4` | Learning rate (Adam) |
| `--timesteps` | `1000` | Number of diffusion steps |
| `--base_channels` | `128` | Base channel count for U-Net |
| `--text_conditioning` | False | Enable prompt conditioning with the trainable text encoder |
| `--caption_file` | `None` | Optional captions file keyed by image filename or stem |
| `--default_caption` | dataset-specific | Fallback caption for images missing from the captions file |
| `--text_embed_dim` | `256` | Prompt text embedding dimension |
| `--text_vocab_size` | `8192` | Hash-token vocabulary size |
| `--text_max_length` | `32` | Maximum prompt token count |
| `--text_encoder_type` | `simple` | Prompt encoder backend: `simple` or `clip` |
| `--clip_model_name` | `openai/clip-vit-base-patch32` | Hugging Face CLIP model name |
| `--finetune_clip_text_encoder` | False | Finetune CLIP text weights instead of using frozen embeddings |
| `--caption_dropout` | `0.1` | Caption dropout probability for classifier-free guidance |
| `--augment_level` | `face_safe` | Image augmentation strength: `none`, `basic`, `face_safe`, or `strong` |
| `--use_amp` | False | Use CUDA automatic mixed precision to reduce memory use |
| `--sample_every` | `10` | Generate samples every N epochs |
| `--checkpoint_dir` | `checkpoints` | Directory for saved checkpoints |
| `--log_dir` | `logs` | Directory for training logs and samples |
| `--resume` | `None` | Path to checkpoint to resume from |
| `--use_scheduler` | False | Enable cosine annealing LR scheduler|
| `--scheduler_tmax` | `None` | T_max for CosineAnnealingLR (defaults to total epochs) |
| `--scheduler_eta_min` | `1e-6` | Minimum learning rate for cosine annealing LR |
| `--use_ema` | False | Enable EMA model |
| `--ema_decay` | 0.999 | EMA decay factor |

### Generate Arguments

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to model checkpoint |
| `--num_images` | `300` | Number of images to generate |
| `--image_size` | `256` | Image resolution |
| `--timesteps` | `1000` | Number of diffusion steps |
| `--base_channels` | `128` | Base channel count for U-Net |
| `--batch_size` | `16` | Batch size for generation |
| `--output_dir` | `generated` | Directory for generated outputs |
| `--prompt` | `None` | Prompt for conditioned checkpoints; metadata/seed label for unconditional checkpoints |
| `--seed` | `None` | Random seed for generation |
| `--text_conditioning` | False | Use prompt conditioning and load text encoder weights from the checkpoint |
| `--guidance_scale` | `3.0` | Classifier-free guidance scale for prompt-conditioned generation |
| `--text_embed_dim` | `256` | Prompt text embedding dimension |
| `--text_vocab_size` | `8192` | Hash-token vocabulary size |
| `--text_max_length` | `32` | Maximum prompt token count |
| `--compute_fid` | False | Compute FID against the real image set |
| `--data_dir` | `data/celeba_hq_256` | Path to real images for FID computation |
| `--use_ema` | False | Use EMA weights if available |



## Evaluation

### FID Score

The Frechet Inception Distance (FID) measures how similar generated images are to real images by comparing feature distributions extracted from a pre-trained InceptionV3 network. Lower scores indicate better quality.

The `generate.py` script computes FID when run with the `--compute_fid` flag against the test split (300 images).

### Visual Evaluation

Sample images are automatically saved to `logs/` every `sample_every` epochs during training. The training loss curve can be plotted using:

```python
from evaluation.visualize import plot_training_loss
plot_training_loss("logs/training_log.json", save_path="loss_curve.png")
```

## Unit Tests

To run all unit tests, cd to the human-face-generation-df-model-main directory and enter the command pytest

## Hardware Requirements

- **GPU recommended:** Training on CPU is extremely slow. Use CUDA (NVIDIA) or MPS (Apple Silicon).
- **VRAM:** ~6-8 GB for batch size 16 at 256x256 resolution. Reduce `--batch_size` if running out of memory.

## References

- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.
- Karras, T., et al. (2018). *Progressive Growing of GANs for Improved Quality, Stability, and Variation*. ICLR 2018 (CelebA-HQ dataset).
- Dhariwal, P. and Nichol, A. *Diffusion models beat GANs on image synthesis*. Advances in Neural Information Processing Systems, vol. 34, pp. 8780–8794, 2021.
- Nichol, A. and Dhariwal, P. *Improved Denoising Diffusion Probabilistic Models* Proceedings of the International Conference on Machine Learning (ICML), pp. 8162–8171, 2021.
