# Human Faces Generation with Diffusion Models

Starter repository for the Applied Machine Learning coursework on **Human Face Generation with Diffusion Models**.

## Goal
1. Reproduce a toy butterfly diffusion baseline
2. Adapt the pipeline to **CelebA-HQ 256x256**
3. Train an unconditional face generator
4. Generate 300 samples
5. Compute **FID** against the 300-image test split

## Repository structure

```text
human-face-diffusion-repo/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
├── src/
├── scripts/
├── notebooks/
├── checkpoints/
├── results/
└── docs/
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train butterfly baseline
```bash
python -m src.train --config configs/butterfly_baseline.yaml
```

### Train CelebA-HQ model
Put the dataset into:
```text
data/celeba_hq_256/
```

Then run:
```bash
python -m src.train --config configs/celeba_baseline.yaml
```

### Generate samples
```bash
python -m src.sample --config configs/celeba_baseline.yaml --checkpoint checkpoints/celeba_latest.pt --num_samples 300
```

### Evaluate FID
```bash
python -m src.evaluate --real_dir data/celeba_hq_256_test --fake_dir results/generated_celeba
```

## Notes
- This is a **starter repo**, not a final SOTA implementation.
- It is designed to be understandable and easy to modify.
- You will still need to tune batch size, number of diffusion steps, learning rate, and model width for your available GPU/Colab runtime.
