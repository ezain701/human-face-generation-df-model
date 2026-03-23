import matplotlib.pyplot as plt
import torch
import json
import numpy as np
from torchvision.utils import make_grid

from models.forward_diffusion import q_sample


def show_images(images, nrow=4, title=None, figsize=(12, 12)):
    """Display a grid of images (tensor with values in [0,1])."""
    grid = make_grid(images, nrow=nrow, padding=2)
    plt.figure(figsize=figsize)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_training_loss(log_path, save_path=None):
    """Plot training loss curve from a JSON log file."""
    with open(log_path, "r") as f:
        log = json.load(f)

    epochs = [entry["epoch"] for entry in log]
    losses = [entry["avg_loss"] for entry in log]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss (MSE)")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def show_diffusion_process(schedule, x_start, num_steps=10):
    """Visualize the forward diffusion process at evenly spaced timesteps."""
    timesteps = torch.linspace(0, schedule.num_timesteps - 1, num_steps).long()
    images = [x_start[0]]

    for t in timesteps:
        t_batch = t.unsqueeze(0).to(x_start.device)
        noisy = q_sample(schedule, x_start[:1], t_batch)
        images.append(noisy[0])

    images = torch.stack(images)
    images = (images.clamp(-1, 1) + 1) / 2
    show_images(images, nrow=len(images), title="Forward Diffusion Process")
