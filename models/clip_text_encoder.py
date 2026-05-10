"""CLIP text encoder wrapper for prompt-conditioned diffusion."""

import torch
import torch.nn as nn


class CLIPTokenizerAdapter:
    """Tokenizer wrapper matching the project's tokenizer call interface."""

    def __init__(self, model_name="openai/clip-vit-base-patch32", max_length=77):
        try:
            from transformers import CLIPTokenizer
        except ImportError as exc:
            raise ImportError(
                "CLIP text conditioning requires transformers. Install it with "
                "`pip install transformers`."
            ) from exc

        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

    def __call__(self, prompts, device=None):
        if isinstance(prompts, str):
            prompts = [prompts]
        encoded = self.tokenizer(
            list(prompts),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if device is not None:
            encoded = {key: value.to(device) for key, value in encoded.items()}
        return encoded


class CLIPTextEncoder(nn.Module):
    """Frozen CLIP text encoder that returns pooled text embeddings."""

    def __init__(self, model_name="openai/clip-vit-base-patch32", freeze=True):
        super().__init__()
        try:
            from transformers import CLIPTextModel
        except ImportError as exc:
            raise ImportError(
                "CLIP text conditioning requires transformers. Install it with "
                "`pip install transformers`."
            ) from exc

        self.model_name = model_name
        self.freeze = freeze
        self.text_model = CLIPTextModel.from_pretrained(model_name)
        self.embed_dim = self.text_model.config.hidden_size

        if freeze:
            self.text_model.eval()
            for param in self.text_model.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        if self.freeze:
            super().train(False)
            self.text_model.eval()
            return self
        return super().train(mode)

    def forward(self, encoded_prompts):
        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            outputs = self.text_model(**encoded_prompts)
            return outputs.pooler_output
