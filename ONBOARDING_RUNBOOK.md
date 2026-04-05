# DDPM Face Model Onboarding Runbook (One Page)

## 1) Prerequisites

- Python 3.10+ recommended
- GPU strongly recommended (CUDA or MPS)
- Dataset prepared in image folders
  - Faces: `data/celeba_hq_256`
  - Optional toy set: `data/butterfly`

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2) Verify Setup Quickly

Run tests (CLI sanity checks):

```bash
pytest -q
```

## 3) Train From Scratch

Default face training:

```bash
python main.py \
  --dataset celeba \
  --data_dir data/celeba_hq_256 \
  --image_size 256 \
  --epochs 100 \
  --batch_size 16 \
  --lr 2e-4 \
  --timesteps 1000 \
  --base_channels 128 \
  --sample_every 10 \
  --checkpoint_dir checkpoints \
  --log_dir logs
```

Expected outputs during training:

- Checkpoints: `checkpoints/model_epoch_<N>.pt`
- Sample grids: `logs/samples_epoch_<N>.png`
- Loss log JSON: `logs/training_log.json`

## 4) Resume Training

Resume from a checkpoint and continue to a later epoch:

```bash
python main.py \
  --dataset celeba \
  --data_dir data/celeba_hq_256 \
  --epochs 200 \
  --resume checkpoints/model_epoch_100.pt
```

Notes:

- `--epochs` is total target epoch, not “extra epochs”.
- Keep architecture-related flags consistent with the original run (`--base_channels`, `--timesteps`, image size).

## 5) Generate Images

Generate 300 images from a trained checkpoint:

```bash
python generate.py \
  --checkpoint checkpoints/model_epoch_100.pt \
  --num_images 300 \
  --batch_size 16 \
  --output_dir generated
```

Expected outputs:

- Individual files: `generated/generated_0000.png`, ...
- Preview grid: `generated/grid_sample.png`

## 6) Evaluate With FID

Generate + compute FID against real test split:

```bash
python generate.py \
  --checkpoint checkpoints/model_epoch_100.pt \
  --num_images 300 \
  --compute_fid \
  --data_dir data/celeba_hq_256
```

Interpretation:

- Lower FID is better.
- Compare FID only across runs with the same preprocessing and image resolution.

## 7) Optional Toy Warm-Up (Butterfly)

```bash
python main.py \
  --dataset butterfly \
  --data_dir data/butterfly \
  --image_size 128 \
  --epochs 50
```

## 8) Quick Troubleshooting

- Out-of-memory: reduce `--batch_size`.
- Very slow training: confirm GPU is being used (startup prints device).
- Resume mismatch errors: ensure model hyperparameters match checkpoint run.
- Missing images in dataset path: verify folder exists and contains `.jpg/.jpeg/.png` files.

## 9) Minimal Daily Workflow

1. Start/continue training.
2. Check latest sample grid for visual quality.
3. Track loss in `logs/training_log.json`.
4. Periodically run generation + FID and log the score with checkpoint epoch.
