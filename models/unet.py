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
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mults=(1, 2, 4, 4),
        num_res_blocks=2,
        attention_resolutions=(1,2),
        time_emb_dim=256,
        num_heads=8,
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
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
                #skip_channels.append(channels)

        # --- Bottleneck ---
        self.mid_block1 = ResidualBlock(channels, channels, time_emb_dim)
        self.mid_attn = AttentionBlock(channels, num_heads)
        self.mid_block2 = ResidualBlock(channels, channels, time_emb_dim)

        # --- Decoder ---
        self.up_blocks = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for i in range(num_res_blocks):
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

        self.output_norm = nn.GroupNorm(32, channels)
        self.output_conv = nn.Conv2d(channels, out_channels, 3, padding=1)

    def forward(self, x, t):
        t = t.float()
        t_emb = self.time_mlp(t)
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
