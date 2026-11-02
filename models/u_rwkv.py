"""
U-RWKV: Lightweight Medical & Visual Image Segmentation with Direction-Adaptive RWKV
Adapted for real-time flood area segmentation on edge & embedded platforms.

Original Repository: https://github.com/hbyecoding/U-RWKV
Commit Hash: 32f2d3c8bd52ef53b34cabcc8f879c1d39f183e0
Reference: Ye et al., "U-RWKV: Lightweight medical image segmentation with direction-adaptive RWKV", 2025.
arXiv: https://arxiv.org/abs/2507.11415

Key Architectural Components:
- Direction-Adaptive RWKV Module (DARM) with 2D spatial token shift (q_shift)
- Multi-directional linear-attention scanning (horizontal & vertical)
- Stage-Adaptive Squeeze-and-Excitation (SASE) feature recalibration
- U-Net style multi-resolution encoder-decoder with skip connections
- Native Pure PyTorch WKV linear attention fallback included for universal compatibility
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rwkv_block import bi_wkv_pure_pytorch, q_shift_2d


class RWKV_SpatialMix(nn.Module):
    """
    Direction-Adaptive Spatial-Mix Block for visual tokens.
    """
    def __init__(self, n_embd: int, n_layer: int = 4, layer_id: int = 0):
        super().__init__()
        self.n_embd = n_embd
        self.layer_id = layer_id

        # Parameter initialization following RWKV standard
        ratio_0_to_1 = layer_id / max(1, n_layer - 1)
        ratio_1_to_almost0 = 1.0 - (layer_id / max(1, n_layer))

        # Time decay and time first initialization
        decay_speed = torch.ones(n_embd)
        for h in range(n_embd):
            decay_speed[h] = -5 + 8 * (h / max(1, n_embd - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
        self.time_decay = nn.Parameter(decay_speed)

        zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(n_embd)]) * 0.5
        self.time_first = nn.Parameter(torch.ones(n_embd) * math.log(0.3) + zigzag)

        # Spatial mix interpolation parameters
        x = torch.ones(1, 1, n_embd)
        for i in range(n_embd):
            x[0, 0, i] = i / n_embd
        self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
        self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor, resolution: Tuple[int, int]) -> torch.Tensor:
        B, T, C = x.shape
        xx = q_shift_2d(x, shift_pixel=1, gamma=0.25, resolution=resolution)

        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        wkv = bi_wkv_pure_pytorch(self.time_decay, self.time_first, k, v)
        out = self.output(sr * wkv)
        return out


class RWKV_ChannelMix(nn.Module):
    """
    Channel-Mix Block (Feed-Forward Network with Receptance gating).
    """
    def __init__(self, n_embd: int, n_layer: int = 4, layer_id: int = 0):
        super().__init__()
        self.n_embd = n_embd
        self.layer_id = layer_id

        ratio_1_to_almost0 = 1.0 - (layer_id / max(1, n_layer))
        x = torch.ones(1, 1, n_embd)
        for i in range(n_embd):
            x[0, 0, i] = i / n_embd
        self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        hidden_sz = 3 * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def forward(self, x: torch.Tensor, resolution: Tuple[int, int]) -> torch.Tensor:
        xx = q_shift_2d(x, shift_pixel=1, gamma=0.25, resolution=resolution)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        r = torch.sigmoid(self.receptance(xr))
        return r * kv


class RWKVBlock(nn.Module):
    def __init__(self, n_embd: int, n_layer: int = 4, layer_id: int = 0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.spatial_mix = RWKV_SpatialMix(n_embd, n_layer=n_layer, layer_id=layer_id)
        self.ln2 = nn.LayerNorm(n_embd)
        self.channel_mix = RWKV_ChannelMix(n_embd, n_layer=n_layer, layer_id=layer_id)

    def forward(self, x: torch.Tensor, resolution: Tuple[int, int]) -> torch.Tensor:
        x = x + self.spatial_mix(self.ln1(x), resolution)
        x = x + self.channel_mix(self.ln2(x), resolution)
        return x


class SASE(nn.Module):
    """Stage-Adaptive Squeeze-and-Excitation Module."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class ConvStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class URWKV(nn.Module):
    """
    U-RWKV Architecture for image segmentation.
    Uses RWKV attention blocks at bottleneck and deep stages with CNN skip connections.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1, channels: Tuple[int, ...] = (32, 64, 128, 256), input_size: int = 256):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        c1, c2, c3, c4 = channels

        # Encoder stages
        self.stem = ConvStem(in_channels, c1)
        self.down1 = nn.Sequential(nn.MaxPool2d(2, 2), SASE(c1), nn.Conv2d(c1, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.ReLU(inplace=True))
        self.down2 = nn.Sequential(nn.MaxPool2d(2, 2), SASE(c2), nn.Conv2d(c2, c3, 1, bias=False), nn.BatchNorm2d(c3), nn.ReLU(inplace=True))
        self.down3 = nn.Sequential(nn.MaxPool2d(2, 2), SASE(c3), nn.Conv2d(c3, c4, 1, bias=False), nn.BatchNorm2d(c4), nn.ReLU(inplace=True))

        # RWKV Bottleneck Blocks
        self.bottleneck_rwkv = nn.ModuleList([
            RWKVBlock(c4, n_layer=2, layer_id=i) for i in range(2)
        ])

        # RWKV Stage 3 Blocks
        self.stage3_rwkv = nn.ModuleList([
            RWKVBlock(c3, n_layer=2, layer_id=i) for i in range(2)
        ])

        # Decoder stages
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(c4, c3, 1, bias=False), nn.BatchNorm2d(c3), nn.ReLU(inplace=True))
        self.dec_conv3 = nn.Sequential(nn.Conv2d(c3 * 2, c3, 3, padding=1, bias=False), nn.BatchNorm2d(c3), nn.ReLU(inplace=True))

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(c3, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.ReLU(inplace=True))
        self.dec_conv2 = nn.Sequential(nn.Conv2d(c2 * 2, c2, 3, padding=1, bias=False), nn.BatchNorm2d(c2), nn.ReLU(inplace=True))

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(c2, c1, 1, bias=False), nn.BatchNorm2d(c1), nn.ReLU(inplace=True))
        self.dec_conv1 = nn.Sequential(nn.Conv2d(c1 * 2, c1, 3, padding=1, bias=False), nn.BatchNorm2d(c1), nn.ReLU(inplace=True))

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape

        # Encoder
        x1 = self.stem(x)             # (B, c1, H, W)
        x2 = self.down1(x1)           # (B, c2, H/2, W/2)
        x3 = self.down2(x2)           # (B, c3, H/4, W/4)

        # Apply RWKV on stage 3 features
        B, C3, H3, W3 = x3.shape
        x3_seq = x3.flatten(2).transpose(1, 2)
        for blk in self.stage3_rwkv:
            x3_seq = blk(x3_seq, resolution=(H3, W3))
        x3 = x3_seq.transpose(1, 2).reshape(B, C3, H3, W3)

        x4 = self.down3(x3)           # (B, c4, H/8, W/8)
        B, C4, H4, W4 = x4.shape
        x4_seq = x4.flatten(2).transpose(1, 2)
        for blk in self.bottleneck_rwkv:
            x4_seq = blk(x4_seq, resolution=(H4, W4))
        x4 = x4_seq.transpose(1, 2).reshape(B, C4, H4, W4)

        # Decoder with skip connections
        d3 = self.up3(x4)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec_conv2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec_conv1(d1)

        out = self.head(d1)
        return out


def build_model(num_classes: int = 1, input_size: int = 256, pretrained: bool = False, in_channels: int = 3, **kwargs):
    return URWKV(
        in_channels=in_channels,
        num_classes=num_classes,
        channels=(32, 64, 128, 256),
        input_size=input_size
    )

