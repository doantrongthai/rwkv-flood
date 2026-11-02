"""
SegRWKV: Fast and Accurate Image Segmentation via Receptance-Weighted Key-Value Network
Adapted for real-time flood area segmentation on edge & embedded platforms.

Original Repository: https://github.com/luminescentfrr/SegRWKV
Commit Hash: 8dbd13217c4fb734a03ffdb020833e6e83152c53
Reference: "SegRWKV: Fast and Accurate Biomedical Image Segmentation via a Receptance-Weighted Key-Value Network",
Journal of Computational Design and Engineering, 2024/2025.

Key Architectural Components:
- RWKV linear attention blocks integrated into multi-resolution stages
- Bi-directional sequence scanning with spatial token shifts
- Efficient skip-connection fusion
- Native Pure PyTorch WKV fallback for CPU / Edge portability
"""

import math
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rwkv_block import bi_wkv_pure_pytorch, q_shift_2d


class SegRWKV_Block(nn.Module):
    def __init__(self, dim: int, n_layer: int = 4, layer_id: int = 0):
        super().__init__()
        self.dim = dim
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        ratio_0_to_1 = layer_id / max(1, n_layer - 1)
        ratio_1_to_almost0 = 1.0 - (layer_id / max(1, n_layer))

        decay_speed = torch.ones(dim)
        for h in range(dim):
            decay_speed[h] = -5 + 8 * (h / max(1, dim - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
        self.time_decay = nn.Parameter(decay_speed)

        zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(dim)]) * 0.5
        self.time_first = nn.Parameter(torch.ones(dim) * math.log(0.3) + zigzag)

        x = torch.ones(1, 1, dim)
        for i in range(dim):
            x[0, 0, i] = i / dim
        self.mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        self.mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
        self.mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.proj_out = nn.Linear(dim, dim, bias=False)

        # Feed forward
        self.ffn_k = nn.Linear(dim, 2 * dim, bias=False)
        self.ffn_r = nn.Linear(dim, dim, bias=False)
        self.ffn_v = nn.Linear(2 * dim, dim, bias=False)

    def forward(self, x: torch.Tensor, resolution: Tuple[int, int]) -> torch.Tensor:
        # Spatial-Mix
        x_norm = self.ln1(x)
        xx = q_shift_2d(x_norm, shift_pixel=1, gamma=0.25, resolution=resolution)
        xk = x_norm * self.mix_k + xx * (1 - self.mix_k)
        xv = x_norm * self.mix_v + xx * (1 - self.mix_v)
        xr = x_norm * self.mix_r + xx * (1 - self.mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))
        wkv = bi_wkv_pure_pytorch(self.time_decay, self.time_first, k, v)
        x = x + self.proj_out(r * wkv)

        # Channel-Mix
        x_norm2 = self.ln2(x)
        xx2 = q_shift_2d(x_norm2, shift_pixel=1, gamma=0.25, resolution=resolution)
        xk2 = x_norm2 * self.mix_k + xx2 * (1 - self.mix_k)
        xr2 = x_norm2 * self.mix_r + xx2 * (1 - self.mix_r)

        k2 = torch.square(torch.relu(self.ffn_k(xk2)))
        r2 = torch.sigmoid(self.ffn_r(xr2))
        x = x + r2 * self.ffn_v(k2)
        return x


class SegRWKV(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, channels: Tuple[int, ...] = (24, 48, 96, 192), input_size: int = 256):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        c1, c2, c3, c4 = channels

        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        # Stage 1 down
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

        # Stage 2 down
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )

        # Stage 3 down
        self.down3 = nn.Sequential(
            nn.Conv2d(c3, c4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )

        # RWKV Bottleneck
        self.rwkv_bottleneck = SegRWKV_Block(c4, n_layer=2, layer_id=0)

        # Up stages
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(c4, c3, 1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(c3 * 2, c3, 3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(c3, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(c2 * 2, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(c2, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv_in(x)    # (B, c1, H, W)
        x2 = self.down1(x1)     # (B, c2, H/2, W/2)
        x3 = self.down2(x2)     # (B, c3, H/4, W/4)
        x4 = self.down3(x3)     # (B, c4, H/8, W/8)

        B, C4, H4, W4 = x4.shape
        x4_seq = x4.flatten(2).transpose(1, 2)
        x4_seq = self.rwkv_bottleneck(x4_seq, (H4, W4))
        x4 = x4_seq.transpose(1, 2).reshape(B, C4, H4, W4)

        d3 = self.fuse3(torch.cat([self.up3(x4), x3], dim=1))
        d2 = self.fuse2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.fuse1(torch.cat([self.up1(d2), x1], dim=1))

        out = self.out_conv(d1)
        return out


def build_model(num_classes: int = 1, input_size: int = 256, pretrained: bool = False, in_channels: int = 3, **kwargs):
    return SegRWKV(
        in_channels=in_channels,
        num_classes=num_classes,
        channels=(24, 48, 96, 192),
        input_size=input_size
    )

