"""
U-Net architecture for the diffusion model.

Predicts the noise added to an image at a given timestep.
Uses residual blocks, self-attention at lower resolutions,
and skip connections between the encoder and decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    SinusoidalPositionEmbedding,
    ResidualBlock,
    AttentionBlock,
    Downsample,
    Upsample,
)


class UNet(nn.Module):

    def __init__(
        self,
        # in_channels is the number of channels in the input image.
        in_channels=3,  
        # out_channels is the same as in_channels for image generation tasks 
        # because we're predicting noise of the same shape as the input image.      
        out_channels=3,
        # base_channels controls the width of the UNet model. 
        # Higher values = more parameters = better quality but slower training/inference.        
        base_channels=64,
        # Each entry in channel_mults multiplies the base_channels for that level of the UNet.
        channel_mults=(1, 2, 2, 2),
        # num_res_blocks controls how many residual blocks are in each level of the UNet.
        num_res_blocks=2,
        # attention_resolutions specifies which levels of the UNet should have self-attention.
        attention_resolutions=(2,),
        # time_emb_dim controls the dimensionality of the time embedding used in the residual blocks.
        time_emb_dim=256,
        # text_emb_dim enables prompt conditioning when a text embedding is provided.
        text_emb_dim=None,
        # num_heads controls the number of attention heads in the self-attention layers.
        num_heads=4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.time_emb_dim = time_emb_dim
        self.text_emb_dim = text_emb_dim
        self.num_heads = num_heads

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.text_mlp = (
            nn.Sequential(
                nn.LayerNorm(text_emb_dim),
                nn.Linear(text_emb_dim, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
            if text_emb_dim is not None
            else None
        )

        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # --- Encoder ---
        self.down_blocks = nn.ModuleList()
        channels = base_channels
        skip_channels = [base_channels]

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResidualBlock(channels, out_ch, time_emb_dim)])
                if level in attention_resolutions:
                    block.append(AttentionBlock(out_ch, num_heads))
                else:
                    block.append(nn.Identity())
                self.down_blocks.append(block)
                channels = out_ch
                skip_channels.append(channels)

            if level != len(channel_mults) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(channels), nn.Identity()]))
                skip_channels.append(channels)

        # --- Bottleneck ---
        self.mid_block1 = ResidualBlock(channels, channels, time_emb_dim)
        self.mid_attn = AttentionBlock(channels, num_heads)
        self.mid_block2 = ResidualBlock(channels, channels, time_emb_dim)

        # --- Decoder ---
        self.up_blocks = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for i in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                block = nn.ModuleList([
                    ResidualBlock(channels + skip_ch, out_ch, time_emb_dim)
                ])
                if level in attention_resolutions:
                    block.append(AttentionBlock(out_ch, num_heads))
                else:
                    block.append(nn.Identity())
                self.up_blocks.append(block)
                channels = out_ch

            if level != 0:
                self.up_blocks.append(nn.ModuleList([Upsample(channels), nn.Identity()]))

        self.output_norm = nn.GroupNorm(8, channels)
        self.output_conv = nn.Conv2d(channels, out_channels, 3, padding=1)

    def forward(self, x, t, text_emb=None):
        t_emb = self.time_mlp(t)
        if self.text_mlp is not None:
            if text_emb is None:
                text_emb = torch.zeros(x.shape[0], self.text_emb_dim, device=x.device, dtype=x.dtype)
            t_emb = t_emb + self.text_mlp(text_emb)
        x = self.input_conv(x)
        skips = [x]

        for block in self.down_blocks:
            layer, extra = block
            if isinstance(layer, Downsample):
                x = layer(x)
            else:
                x = layer(x, t_emb)
                x = extra(x)
            skips.append(x)

        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        for block in self.up_blocks:
            layer, extra = block
            if isinstance(layer, Upsample):
                x = layer(x)
            else:
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = layer(x, t_emb)
                x = extra(x)

        x = self.output_norm(x)
        x = F.silu(x)
        return self.output_conv(x)
