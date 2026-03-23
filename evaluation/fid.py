import torch
import numpy as np
from scipy.linalg import sqrtm
from torchvision import transforms
from torchvision.models import inception_v3
from torch.utils.data import DataLoader


def get_inception_model(device="cpu"):
    """Load pretrained InceptionV3 for FID feature extraction."""
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  # Remove final classification layer
    model.eval()
    return model.to(device)


def preprocess_for_inception(images):
    """Resize and normalize images for InceptionV3 (expects 299x299)."""
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return torch.stack([transform(img) for img in images])


@torch.no_grad()
def extract_features(images, model, device="cpu", batch_size=32):
    """Extract InceptionV3 features from a batch of images (values in [0,1])."""
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch = preprocess_for_inception(batch).to(device)
        feat = model(batch)
        features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)


def calculate_fid(real_features, generated_features):
    """
    Compute FID score between two sets of InceptionV3 features.

    FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*sqrt(Sigma_r @ Sigma_g))
    Lower is better.
    """
    mu_r = np.mean(real_features, axis=0)
    mu_g = np.mean(generated_features, axis=0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_g = np.cov(generated_features, rowvar=False)

    diff = mu_r - mu_g
    covmean, _ = sqrtm(sigma_r @ sigma_g, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
    return float(fid)


def compute_fid_score(real_images, generated_images, device="cpu", batch_size=32):
    """
    End-to-end FID computation.

    Args:
        real_images: Tensor of real images, shape (N, 3, H, W), values in [0, 1].
        generated_images: Tensor of generated images, same format.
        device: 'cuda' or 'cpu'.
        batch_size: Batch size for feature extraction.

    Returns:
        FID score (float). Lower = better.
    """
    inception = get_inception_model(device)
    real_features = extract_features(real_images, inception, device, batch_size)
    gen_features = extract_features(generated_images, inception, device, batch_size)
    return calculate_fid(real_features, gen_features)
