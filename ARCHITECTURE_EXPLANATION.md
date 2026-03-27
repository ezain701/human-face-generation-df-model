# DDPM Architecture & Pipeline Explanation

A walkthrough of this repository's implementation of **Denoising Diffusion Probabilistic Models** (Ho et al., NeurIPS 2020), mapping every concept to the code.

---

## 1. What problem diffusion solves

You want a model that can **sample new face images** from a learned distribution. A DDPM does that by:

1. **Forward process:** turn a real image \(x_0\) into noise \(x_T\) in small Gaussian steps.
2. **Learned reverse process:** start from noise and **undo** those steps using a neural network that knows "how much noise to remove" at each step.

The network does **not** output a full image in one shot; it refines the same tensor over **T** steps (default **1000**).

---

## 2. End-to-end pipeline (training)

**Entry point:** `main.py` builds four pieces, then hands them to `Trainer`:

| Piece | Role |
|--------|------|
| `NoiseSchedule` | Fixed math for how noise grows/shrinks per step |
| `UNet` | Predicts noise ε given noisy image + step index |
| DataLoader | Batches of real images in **[-1, 1]** (via `Normalize([0.5],[0.5])`) |
| `Trainer` | Samples random timesteps, computes loss, backprop |

**One training step** (`training/trainer.py`):

1. Load batch \(x_0\) (clean, normalized faces).
2. Sample random \(t \in \{0, \ldots, T-1\}\) per image.
3. Call `p_losses` → forward noising + U-Net + MSE.
4. Adam step + optional gradient clip.

The **full pipeline** in one sentence: **data → random t → noisy x_t → U-Net predicts noise → MSE to true noise**.

---

## 3. Noise schedule — `models/noise_schedule.py`

At each step \(t\), a small variance \(\beta_t\) is added. This code uses a **linear schedule**: \(\beta_t\) goes from `1e-4` to `0.02` over T steps.

Important derived quantities (all precomputed as tensors):

- \(\alpha_t = 1 - \beta_t\)
- \(\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s\) — cumulative product: `alphas_cumprod`

Training uses the closed-form **one-step noising** shortcut:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

That is exactly `q_sample` in `models/forward_diffusion.py`, using `sqrt_alphas_cumprod` and `sqrt_one_minus_alphas_cumprod`.

> **Key insight:** the schedule fixes **how fast** the signal is destroyed going forward and **how big** each denoising update is allowed to be going backward.

---

## 4. Forward diffusion & training objective — `models/forward_diffusion.py`

- **`q_sample`:** given \(x_0\) and \(t\), builds \(x_t\) in **one line** — no loop over steps, thanks to the cumulative product trick.
- **`p_losses`:** sample \(\epsilon\), build \(x_t\), run `model(x_noisy, t)`, minimize **MSE(predicted\_noise, ε)**.

This is the **ε-prediction** formulation: the U-Net is trained to predict the **same Gaussian noise** that was mixed in.

> **For your explanation:** you are not regressing the pixel values of \(x_0\); you are regressing the **noise vector** that explains how \(x_t\) was produced from \(x_0\). That objective is stable and matches the reverse update used at inference.

---

## 5. The U-Net — `models/unet.py` + `models/blocks.py`

**Role:** at each step \(t\), input is a **noisy image** \(x_t \in \mathbb{R}^{B \times 3 \times H \times W}\). Output is the **same shape** (3 channels) — the predicted noise.

### 5.1 Timestep conditioning

Diffusion is different at every \(t\). The network must know which step it is at:

1. **`SinusoidalPositionEmbedding`** (`blocks.py`): maps integer \(t\) to a vector using different frequencies of sin and cos — the same idea as Transformer positional encodings.
2. **`time_mlp`** in `UNet`: two-layer MLP + SiLU turns that into a **time embedding** of size `time_emb_dim` (256).

Inside each **`ResidualBlock`**, the time embedding is projected to the channel dimension and **added** to the feature map (broadcast over spatial dimensions). So every conv block "knows" how noisy the input is supposed to be.

### 5.2 Encoder–bottleneck–decoder + skip connections

- **Encoder (`down_blocks`):** repeated `ResidualBlock`s (sometimes + `AttentionBlock`), then a **stride-2 conv** (`Downsample`) between levels. Channel width grows via `channel_mults=(1,2,2,2)` on top of `base_channels` (128).
- **Bottleneck:** two residual blocks with a **self-attention** layer in the middle (`mid_attn`).
- **Decoder (`up_blocks`):** **concatenate** skip features from the matching encoder level (hence `channels + skip_ch`), then residual (+ attention at chosen levels), then **upsample** (nearest-neighbor ×2 + conv).

**Why skip connections:** shallow layers keep **fine spatial detail** (edges, texture); deep layers encode **global structure**. Concatenating skips is how the decoder rebuilds a sharp image while denoising.

### 5.3 Self-attention — `AttentionBlock`

Reshapes \(B \times C \times H \times W\) into a sequence of \(HW\) tokens, runs **`nn.MultiheadAttention`**, then reshapes back. This lets distant face parts (eyes, mouth, hairline) interact at lower resolutions where \(HW\) is manageable.

`attention_resolutions=(2,)` means attention is only inserted at **encoder/decoder level index 2** — not at every scale, to keep cost down.

### 5.4 Output head

`GroupNorm → SiLU → 3×3 conv` to `out_channels` (3). No sigmoid: the target is unbounded Gaussian noise, and the loss is MSE.

---

## 6. Reverse diffusion (generation) — `models/reverse_diffusion.py`

- **`p_sample`:** one step from \(x_t\) to \(x_{t-1}\). It calls the U-Net to get `predicted_noise`, computes the **posterior mean** using the DDPM formula, and adds **scaled Gaussian noise** via `posterior_variance` when \(t > 0\). At \(t = 0\) it returns the mean only (no extra noise).
- **`p_sample_loop`:** start from `torch.randn(shape)`, iterate \(t = T{-}1 \rightarrow 0\), then **clamp to [-1, 1]** and map to **[0, 1]** with `(x + 1) / 2` for saving as images.

`generate.py` loads the same `NoiseSchedule` + `UNet` hyperparameters (must match training), runs `sample` in batches, saves PNGs.

> **For your explanation:** generation is **iterative denoising**. The U-Net is reused **T times** with **decreasing** t; each call refines the same latent "canvas," gradually resolving structure and detail.

---

## 7. Full pipeline diagram

```
TRAINING
─────────────────────────────────────────────────────────────────────
  Real image x0
       │
       ├── random t (sampled per image)
       ├── random noise ε ~ N(0,I)
       │
       ▼
  x_t = sqrt(ᾱ_t) * x0 + sqrt(1 - ᾱ_t) * ε       ← q_sample
       │
       ▼
  UNet(x_t, t)  →  predicted_ε̂
       │
       ▼
  Loss = MSE(ε̂, ε)  →  backprop  →  Adam update

GENERATION
─────────────────────────────────────────────────────────────────────
  x_T ~ N(0, I)   (pure noise)
       │
  for t = T-1 down to 0:
       │
       ├── UNet(x_t, t)  →  predicted_ε̂
       ├── compute posterior mean
       └── add noise if t > 0
       │
       ▼
  x_0  →  clamp  →  rescale to [0,1]  →  face image
```

---

## 8. Vocabulary cheat sheet

| Term | Meaning |
|------|---------|
| **DDPM** | Discrete-time Gaussian diffusion with learned reverse updates |
| **β_t schedule** | How much noise is injected per step; linear here (1e-4 → 0.02) |
| **ᾱ_t (alpha_bar)** | Cumulative product of (1 - β); tells you total noise at step t |
| **ε-prediction** | Network outputs noise; loss is MSE to sampled ε |
| **U-Net** | Hourglass CNN with skip connections; here time-conditioned via residual blocks |
| **Self-attention** | Lets spatially distant features interact (long-range coherence) |
| **Skip connections** | Encoder outputs concatenated into decoder; preserve fine detail |
| **Sinusoidal embedding** | Encodes integer t into a fixed-size vector using sin/cos at many frequencies |
| **Posterior variance** | Controls how much stochasticity is added each reverse step |
| **FID score** | Frechet Inception Distance — measures realism of generated images (lower = better) |

---

## 9. Key files quick reference

| File | What it does |
|------|-------------|
| `main.py` | Wires everything together, parses CLI args, starts training |
| `models/noise_schedule.py` | Precomputes β, α, ᾱ and all derived coefficients |
| `models/forward_diffusion.py` | `q_sample` (noising) + `p_losses` (training loss) |
| `models/reverse_diffusion.py` | `p_sample` (one reverse step) + `p_sample_loop` (full generation) |
| `models/unet.py` | Full U-Net model with time conditioning, encoder, bottleneck, decoder |
| `models/blocks.py` | `SinusoidalPositionEmbedding`, `ResidualBlock`, `AttentionBlock`, `Downsample`, `Upsample` |
| `data/dataset.py` | CelebA-HQ and Butterfly dataset classes + DataLoader |
| `training/trainer.py` | Training loop, checkpointing, sample generation during training |
| `generate.py` | Load checkpoint, generate images in batches, optionally compute FID |
| `evaluation/fid.py` | FID score via InceptionV3 feature comparison |

---

*Based on: Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS 2020.*
