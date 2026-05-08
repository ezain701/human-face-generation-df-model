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
        tokenizer=None,
        text_encoder=None,
        ema_model=None,
        ema_text_encoder=None,
        ema_decay=0.999,
        caption_dropout=0.1,
        text_config=None,
        save_text_encoder=True,
        use_amp=False,
        sample_prompts=None,
        guidance_scale=3.0,
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
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder.to(device) if text_encoder is not None else None
        self.ema_model = ema_model.to(device) if ema_model is not None else None
        self.ema_text_encoder = ema_text_encoder.to(device) if ema_text_encoder is not None else None
        self.ema_decay = ema_decay
        self.caption_dropout = caption_dropout
        self.text_config = text_config
        self.save_text_encoder = save_text_encoder
        self.use_amp = use_amp and device == "cuda"
        self.sample_prompts = sample_prompts or []
        self.guidance_scale = guidance_scale
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.clip_grad = clip_grad
        self.scheduler=scheduler

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.training_log = []
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

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

        if self.text_encoder is None or self.ema_text_encoder is None:
            return

        ema_text_params = dict(self.ema_text_encoder.named_parameters())
        text_params = dict(self.text_encoder.named_parameters())

        for name, param in text_params.items():
            ema_text_params[name].mul_(self.ema_decay).add_(param.data, alpha=1.0 - self.ema_decay)

        ema_text_buffers = dict(self.ema_text_encoder.named_buffers())
        text_buffers = dict(self.text_encoder.named_buffers())

        for name, buf in text_buffers.items():
            ema_text_buffers[name].copy_(buf)

    def _encode_prompts(self, prompts):
        if self.text_encoder is None or self.tokenizer is None:
            return None
        prompts = list(prompts)
        if self.caption_dropout > 0:
            keep = torch.rand(len(prompts), device=self.device) >= self.caption_dropout
            prompts = [prompt if keep[i].item() else "" for i, prompt in enumerate(prompts)]
        token_ids = self.tokenizer(prompts, device=self.device)
        return self.text_encoder(token_ids)

    def train(self, num_epochs, sample_every=10, image_size=256, start_epoch=0):
        """
        Main training loop.

        Args:
            num_epochs: Total training epochs.
            sample_every: Generate sample images every N epochs.
            image_size: Resolution of generated samples.
            start_epoch: Epoch to resume from (0 = start fresh).
        """
        self.model.train()

        for epoch in range(start_epoch + 1, num_epochs + 1):
            epoch_loss = 0.0
            num_batches = 0

            progress = tqdm(self.dataloader, desc=f"Epoch {epoch}/{num_epochs}")
            for batch in progress:
                prompts = None
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    batch, prompts = batch
                batch = batch.to(self.device)
                t = torch.randint(
                    0, self.schedule.num_timesteps, (batch.shape[0],), device=self.device
                ).long()
                text_emb = self._encode_prompts(prompts) if prompts is not None else None

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    loss = p_losses(self.schedule, self.model, batch, t, text_emb)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                if self.clip_grad is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self._update_ema()

                epoch_loss += loss.item()
                num_batches += 1
                progress.set_postfix(loss=loss.item())

            avg_loss = epoch_loss / num_batches
            
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.training_log.append({"epoch": epoch, "avg_loss": avg_loss, "lr": current_lr})
            print(f"Epoch {epoch} — Average Loss: {avg_loss:.6f} — LR: {current_lr:.8f}")


            if epoch % sample_every == 0:
                self._save_samples(epoch, image_size)
                self._save_checkpoint(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

        self._save_log()

    def _encode_prompts_for_sampling(self, prompts):
        if self.tokenizer is None or self.text_encoder is None:
            return None
        token_ids = self.tokenizer(prompts, device=self.device)
        return self.text_encoder(token_ids)

    def _save_samples(self, epoch, image_size, num_samples=4):
        sample_model = self.ema_model if self.ema_model is not None else self.model
        sample_text_encoder = self.ema_text_encoder if self.ema_text_encoder is not None else self.text_encoder
        sample_model.eval()
        previous_text_encoder = self.text_encoder
        self.text_encoder = sample_text_encoder
        text_emb = None
        if self.tokenizer is not None and sample_text_encoder is not None:
            text_emb = self._encode_prompts_for_sampling([""] * num_samples)
        samples = generate_samples(self.schedule, sample_model, num_samples, image_size, text_emb=text_emb)
        self.text_encoder = previous_text_encoder
        self.model.train()
        if self.text_encoder is not None:
            self.text_encoder.train()
        path = os.path.join(self.log_dir, f"samples_epoch_{epoch}.png")
        save_image(samples, path, nrow=2)
        print(f"  Saved samples to {path}")

        if self.sample_prompts and self.tokenizer is not None and sample_text_encoder is not None:
            self._save_prompt_samples(epoch, image_size, sample_model, sample_text_encoder)

    def _save_prompt_samples(self, epoch, image_size, sample_model, sample_text_encoder):
        previous_text_encoder = self.text_encoder
        self.text_encoder = sample_text_encoder
        prompts = list(self.sample_prompts)

        text_emb = self._encode_prompts_for_sampling(prompts)
        uncond_text_emb = self._encode_prompts_for_sampling([""] * len(prompts))
        samples = generate_samples(
            self.schedule,
            sample_model,
            len(prompts),
            image_size,
            text_emb=text_emb,
            uncond_text_emb=uncond_text_emb,
            guidance_scale=self.guidance_scale,
        )
        self.text_encoder = previous_text_encoder

        path = os.path.join(self.log_dir, f"prompt_samples_epoch_{epoch}.png")
        save_image(samples, path, nrow=min(4, len(prompts)))

        prompt_path = os.path.join(self.log_dir, f"prompt_samples_epoch_{epoch}.txt")
        with open(prompt_path, "w") as handle:
            for idx, prompt in enumerate(prompts):
                handle.write(f"{idx}: {prompt}\n")

        print(f"  Saved prompt samples to {path}")

    def _save_checkpoint(self, epoch):
        path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pt")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        if self.ema_model is not None:
            checkpoint["ema_model_state_dict"] = self.ema_model.state_dict()
        if self.text_encoder is not None:
            checkpoint["text_config"] = self.text_config or {}
        if self.text_encoder is not None and self.save_text_encoder:
            checkpoint["text_encoder_state_dict"] = self.text_encoder.state_dict()
        if self.ema_text_encoder is not None:
            checkpoint["ema_text_encoder_state_dict"] = self.ema_text_encoder.state_dict()
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        print(f"  Saved checkpoint to {path}")

    def _save_log(self):
        path = os.path.join(self.log_dir, "training_log.json")
        with open(path, "w") as f:
            json.dump(self.training_log, f, indent=2)
        print(f"Saved training log to {path}")
