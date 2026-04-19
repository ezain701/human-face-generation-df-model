import torch
import numpy as np
from evaluation.fid import get_inception_model, extract_features

# Based on the paper: Bińkowski et al., Demystifying MMD GANs, ICLR 2018.

def polynomial_kernel(X, Y):
    """Polynomial kernel. 
    Measures similarity between two sets of features k(x,y) = (gamma * x·y + coef0)^degree."""

    # Scale by feature dimension (2048 for Inception)
    gamma = 1.0 / X.shape[1]
    degree = 3
    dot_product = X @ Y.T
    return (gamma * dot_product + 1) ** degree     # returns grid of similarities


def _mmd2_on_subset(real, gen):
    """Compute unbiased squared MMD between two equal sized subsets of features."""
    m = real.shape[0]
    k_rr = polynomial_kernel(real, real)
    k_gg = polynomial_kernel(gen, gen)
    k_rg = polynomial_kernel(real, gen)
    num_off_diagonal_pairs = m * (m - 1)
    sum_rr = (k_rr.sum() - k_rr.diag().sum()) / num_off_diagonal_pairs
    sum_gg = (k_gg.sum() - k_gg.diag().sum()) / num_off_diagonal_pairs
    sum_rg = k_rg.mean()

    #  If distributions match, sum_rg ≈ sum_rr ≈ sum_gg, so ≈ 0
    return (sum_rr + sum_gg - 2 * sum_rg).item()


def compute_mmd_squared(real_features, generated_features, seed=None):
    """Compute unbiased squared Maximum Mean Discrepancy with polynomial kernel, 
    averaged over random subsets."""

    # This computes the difference between two distributions of images

    # Maximum Mean Discrepancy compares distrubutions by comparing their average feature similarities
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    real_features = torch.as_tensor(real_features, dtype=torch.float64)
    generated_features = torch.as_tensor(generated_features, dtype=torch.float64)
    
    scores = []

    # This allows us to do a small test run with tiny number of generated images
    # Normally we want our subset size of 50 images
    subset_size=50
    m = min(subset_size, real_features.shape[0], generated_features.shape[0])
    if m < subset_size:
        print("Warning: m is less than recommended subset size of 50 due to low num_images")

    # Computing over all images is too expensive. 
    # Instead, draw 100 random subsets of m images and average, giving a mean and a standard deviation
    num_subsets = 100

    for _ in range(num_subsets):
        r_idx = rng.choice(real_features.shape[0], size=m, replace=False)
        g_idx = rng.choice(generated_features.shape[0], size=m, replace=False)
        unbiased_squared_mmd = _mmd2_on_subset(real_features[r_idx], generated_features[g_idx])
        scores.append(unbiased_squared_mmd)
    
    return float(np.mean(scores)), float(np.std(scores))


def compute_kid(real_images, gen_images, device="cpu"):
    """Top-level function: extract features and compute KID."""
    model = get_inception_model(device=device)
    real_features = extract_features(real_images, model, device=device)
    gen_features = extract_features(gen_images, model, device=device)
    return compute_mmd_squared(real_features, gen_features)