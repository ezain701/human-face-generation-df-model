"""Lightweight prompt text encoder for conditional diffusion."""

import re

import torch
import torch.nn as nn


class SimpleTokenizer:
    """Deterministic hash tokenizer that does not require a saved vocabulary."""

    def __init__(self, vocab_size=8192, max_length=32):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token_id = 0

    def _token_to_id(self, token):
        value = 2166136261
        for char in token.encode("utf-8"):
            value ^= char
            value = (value * 16777619) & 0xFFFFFFFF
        return 1 + (value % (self.vocab_size - 1))

    def encode(self, prompt):
        tokens = re.findall(r"[a-z0-9']+", prompt.lower())
        ids = [self._token_to_id(token) for token in tokens[: self.max_length]]
        ids.extend([self.pad_token_id] * (self.max_length - len(ids)))
        return ids

    def __call__(self, prompts, device=None):
        if isinstance(prompts, str):
            prompts = [prompts]
        token_ids = [self.encode(prompt or "") for prompt in prompts]
        return torch.tensor(token_ids, dtype=torch.long, device=device)


class PromptTextEncoder(nn.Module):
    """Small trainable text encoder used to condition the U-Net."""

    def __init__(self, vocab_size=8192, max_length=32, embed_dim=256, num_layers=2, num_heads=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, token_ids):
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        padding_mask = token_ids.eq(0)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        valid = (~padding_mask).float().unsqueeze(-1)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.output_norm(pooled)
