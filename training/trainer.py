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
        ema_model=None,
        ema_decay=0.999,
        clip_grad=None,
        device="cpu",
        checkpoint_dir="checkpoints",
        log_dir="logs",
        scheduler=None,
    ):
        self.model = model.to(device)
        self.schedule = schedule
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.ema_model = ema_model.to(device) if ema_model is not None else None
        self.ema_decay = ema_decay
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.clip_grad = clip_grad
        self.scheduler=scheduler

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.training_log = []

    @torch.no_grad()
    def _update_ema(self):
        if self.ema_model is None:
            return

        ema_params = dict(self.ema_model.named_parameters())
        model_params = dict(self.model.named_parameters())

        for name, param in model_params.items():
            ema_params[name].mul_(self.ema_decay).add_(param.data, alpha=1.0 - self.ema_decay)

        ema_buffers = dict(self.ema_model.named_buffers())
        model_buffers = dict(self.model.named_buffers())

        for name, buf in model_buffers.items():
            ema_buffers[name].copy_(buf)

    def train(self, num_epochs, sample_every=10, image_size=256, start_epoch=0, accum_steps=1):
        """
        Main training loop.
        Args:
        num_epochs: Total training epochs.
        sample_every: Generate sample images every N epochs.
        image_size: Resolution of generated samples.
        start_epoch: Epoch to resume from (0 = start fresh).
        accum_steps: Number of batches to accumulate gradients over
                     before each optimizer step.
        """
        self.model.train()
        for epoch in range(start_epoch + 1, num_epochs + 1):
            epoch_loss = 0.0
            num_batches = 0
            progress = tqdm(self.dataloader, desc=f"Epoch {epoch}/{num_epochs}")

            self.optimizer.zero_grad()  # zero once before the accumulation window

            for i, batch in enumerate(progress):
                batch = batch.to(self.device)
                t = torch.randint(
                    0, self.schedule.num_timesteps, (batch.shape[0],), device=self.device
                ).long()

                loss = p_losses(self.schedule, self.model, batch, t)

                # Scale loss so the accumulated gradient matches the mean
                # over the effective batch, not the sum.
                (loss / accum_steps).backward()

                # Step only every accum_steps micro-batches
                if (i + 1) % accum_steps == 0:
                    if self.clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self._update_ema()  # EMA updates with the optimizer, not every micro-batch

                epoch_loss += loss.item()
                num_batches += 1
                progress.set_postfix(loss=loss.item())

            # Handle leftover micro-batches at end of epoch (if dataset size
            # isn't divisible by accum_steps). Otherwise their gradients get
            # discarded at the next zero_grad().
            if (i + 1) % accum_steps != 0:
                if self.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self._update_ema()

            avg_loss = epoch_loss / num_batches
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.training_log.append({"epoch": epoch, "avg_loss": avg_loss})
            print(f"Epoch {epoch} — Average Loss: {avg_loss:.6f}")

            if epoch % sample_every == 0:
                self._save_samples(epoch, image_size)
                self._save_checkpoint(epoch)
            if self.scheduler is not None:
                self.scheduler.step()
            
        self._save_log()

    def _save_samples(self, epoch, image_size, num_samples=4):
        sample_model = self.ema_model if self.ema_model is not None else self.model
        sample_model.eval()
        samples = generate_samples(self.schedule, sample_model, num_samples, image_size)
        self.model.train()
        path = os.path.join(self.log_dir, f"samples_epoch_{epoch}.png")
        save_image(samples, path, nrow=2)
        print(f"  Saved samples to {path}")

    def _save_checkpoint(self, epoch):
        path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pt")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        if self.ema_model is not None:
            checkpoint["ema_model_state_dict"] = self.ema_model.state_dict()
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        print(f"  Saved checkpoint to {path}")

    def _save_log(self):
        path = os.path.join(self.log_dir, "training_log.json")
        with open(path, "w") as f:
            json.dump(self.training_log, f, indent=2)
        print(f"Saved training log to {path}")
