"""Training loop with checkpointing and sample generation."""

import os
import json
import torch
from tqdm import tqdm
from torchvision.utils import save_image

from models.forward_diffusion import p_losses
from models.reverse_diffusion import sample as generate_samples


class Trainer:
    """Handles the training loop, checkpointing, and sample generation."""

    def __init__(
        self,
        model,
        schedule,
        dataloader,
        optimizer,
        device="cpu",
        checkpoint_dir="checkpoints",
        log_dir="logs",
    ):
        self.model = model.to(device)
        self.schedule = schedule
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.training_log = []

    def train(self, num_epochs, sample_every=10, image_size=256, start_epoch=0, warmup_steps=5000):
        """
        Main training loop.

        Args:
            num_epochs: Total training epochs.
            sample_every: Generate sample images every N epochs.
            image_size: Resolution of generated samples.
            start_epoch: Epoch to resume from (0 = start fresh).
            warmup_steps: Number of steps for learning rate warmup.
        """
        self.model.train()
        
        # Set up linear warmup scheduler
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        global_step = start_epoch * len(self.dataloader)  # Estimate starting step if resuming

        for epoch in range(start_epoch + 1, num_epochs + 1):
            epoch_loss = 0.0
            num_batches = 0

            progress = tqdm(self.dataloader, desc=f"Epoch {epoch}/{num_epochs}")
            for batch in progress:
                batch = batch.to(self.device)
                t = torch.randint(
                    0, self.schedule.num_timesteps, (batch.shape[0],), device=self.device
                ).long()

                loss = p_losses(self.schedule, self.model, batch, t)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                progress.set_postfix(
                    loss=loss.item(), 
                    lr=self.optimizer.param_groups[0]['lr']
                )

            avg_loss = epoch_loss / num_batches
            self.training_log.append({"epoch": epoch, "avg_loss": avg_loss})
            print(f"Epoch {epoch} — Average Loss: {avg_loss:.6f}")

            if epoch % sample_every == 0:
                self._save_samples(epoch, image_size)
                self._save_checkpoint(epoch)

        self._save_log()

    def _save_samples(self, epoch, image_size, num_samples=4):
        self.model.eval()
        samples = generate_samples(self.schedule, self.model, num_samples, image_size)
        self.model.train()
        path = os.path.join(self.log_dir, f"samples_epoch_{epoch}.png")
        save_image(samples, path, nrow=2)
        print(f"  Saved samples to {path}")

    def _save_checkpoint(self, epoch):
        path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        print(f"  Saved checkpoint to {path}")

    def _save_log(self):
        path = os.path.join(self.log_dir, "training_log.json")
        with open(path, "w") as f:
            json.dump(self.training_log, f, indent=2)
        print(f"Saved training log to {path}")
