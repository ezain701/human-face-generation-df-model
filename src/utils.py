import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torchvision.utils import make_grid, save_image


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_loss_plot(losses, out_path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_image_grid(images: torch.Tensor, out_path, nrow: int = 4, normalize: bool = True) -> None:
    grid = make_grid(images, nrow=nrow, normalize=normalize, value_range=(-1, 1))
    save_image(grid, out_path)


def device_from_string(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
