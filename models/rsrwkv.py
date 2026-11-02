"""
RSRWKV: Linear-Complexity Remote Sensing Visual Backbone Network
Adapted for real-time flood area segmentation on edge & embedded platforms.

Original Repository: https://github.com/Ling-yunchi/RSRWKV
Commit Hash: ede06a56bc3768e93444f814025fbc29d2e7919e
Reference: Li et al., "RSRWKV: A Linear Complexity Remote Sensing Visual Backbone Network",
IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2025.
arXiv: https://arxiv.org/abs/2503.20382

Primary Application in Literature:
- Remote Sensing Water Surface Segmentation (GLH-Water: 85.06% IoU, NWPU, VHR-10).

Key Architectural Components:
- MVCShift: Multi-scale dilated depthwise spatial shift capturing continuous elongated water boundaries
- 2D-WKV Linear Attention with isotropic scanning
- ECA: Efficient Channel Attention module for spectral-spatial saliency
- Multi-scale hierarchical encoder-decoder with skip connections
- Native Pure PyTorch WKV fallback for CPU / Edge portability without external CUDA compilation
"""

import math
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rwkv_block import bi_wkv_pure_pytorch


# ─────────────────────────────────────────────────────────────────────────────
# 1. Multi-scale Variable receptive-field Convolutional Shift (MVCShift)
# ─────────────────────────────────────────────────────────────────────────────

class MVCShift(nn.Module):
    """
    MVCShift: Multi-scale dilated spatial shift.
    Expands receptive field along multiple scales to capture elongated water bodies
    and flooded areas before sequence scanning.
    """
    def __init__(self, dim: int, compress_ratio: float = 1.0, kernel_sizes=(3, 3, 3), dilations=(1, 2, 3)):
        super().__init__()
        embed_dim = int(dim * compress_ratio)
        self.conv1x1 = nn.Conv2d(dim, embed_dim, kernel_size=1, bias=False)
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(
                embed_dim, embed_dim, kernel_size=ks,
                padding=(ks // 2) * d, dilation=d,
                groups=embed_dim, bias=False
            )
            for ks, d in zip(kernel_sizes, dilations)
        ])
        self.projs = nn.ModuleList([
            nn.Conv2d(embed_dim, dim, kernel_size=1, bias=False)
            for _ in range(len(kernel_sizes))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)"""
        xx = self.conv1x1(x)
        outs = [self.projs[i](self.dw_convs[i](xx)) for i in range(len(self.dw_convs))]
        return x + sum(outs)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Efficient Channel Attention (ECA Module)
# ─────────────────────────────────────────────────────────────────────────────

class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA) for remote sensing feature recalibration.
    """
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


# ─────────────────────────────────────────────────────────────────────────────
# 3. RSRWKV Spatial Mix & Channel Mix Blocks
# ─────────────────────────────────────────────────────────────────────────────

class RSRWKV_SpatialMix(nn.Module):
    """
    RSRWKV 2D Spatial Mix Block.
    Integrates MVCShift, multi-directional scanning, and WKV linear attention.
    """
    def __init__(self, dim: int, n_layer: int = 4, layer_id: int = 0):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        self.n_layer = n_layer

        self.mvc_shift = MVCShift(dim)
        self.eca = ECA(dim)

        ratio_0_to_1 = layer_id / max(1, n_layer - 1)

        # Decay speed initialization
        decay_speed = torch.ones(dim)
        for h in range(dim):
            decay_speed[h] = -5 + 8 * (h / max(1, dim - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
        self.spatial_decay = nn.Parameter(decay_speed)

        # First-step bonus
        zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(dim)]) * 0.5
        self.spatial_first = nn.Parameter(torch.ones(dim) * math.log(0.3) + zigzag)

        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)"""
        B, C, H, W = x.shape

        # Step 1: Multi-scale dilated spatial shift
        x_shifted = self.mvc_shift(x)

        # Step 2: Linear projections
        x_seq = x_shifted.permute(0, 2, 3, 1).reshape(B, H * W, C)
        k = self.key(x_seq)
        v = self.value(x_seq)
        r = torch.sigmoid(self.receptance(x_seq))

        # Step 3: Bi-directional WKV Linear Attention
        wkv_out = bi_wkv_pure_pytorch(self.spatial_decay, self.spatial_first, k, v)
        out = self.output(r * wkv_out)

        out_2d = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        out_2d = self.eca(out_2d)
        return out_2d


class RSRWKV_ChannelMix(nn.Module):
    """
    RSRWKV Channel Mix (FFN) with gated linear activation.
    """
    def __init__(self, dim: int, hidden_rate: int = 2):
        super().__init__()
        self.dim = dim
        hidden_dim = int(dim * hidden_rate)
        self.key = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.value = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)
        self.receptance = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.receptance(x))
        k = torch.square(torch.relu(self.key(x)))
        out = r * self.value(k)
        return out


class RSRWKV_Block(nn.Module):
    """
    Complete RSRWKV Block: LayerNorm -> SpatialMix -> LayerNorm -> ChannelMix.
    """
    def __init__(self, dim: int, n_layer: int = 4, layer_id: int = 0):
        super().__init__()
        self.dim = dim
        self.ln1 = nn.GroupNorm(1, dim)
        self.spatial_mix = RSRWKV_SpatialMix(dim, n_layer=n_layer, layer_id=layer_id)
        self.ln2 = nn.GroupNorm(1, dim)
        self.channel_mix = RSRWKV_ChannelMix(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.spatial_mix(self.ln1(x))
        x = x + self.channel_mix(self.ln2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hierarchical RSRWKV Segmentation Network
# ─────────────────────────────────────────────────────────────────────────────

class RSRWKV_Segmentation(nn.Module):
    """
    Hierarchical RSRWKV Network adapted for real-time flood area segmentation.
    Uses multi-resolution encoder stages (/4, /8, /16, /32) and progressive
    feature pyramid upsampling decoder.
    """
    def __init__(
        self,
        num_classes: int = 1,
        in_channels: int = 3,
        channels: Tuple[int, ...] = (32, 64, 128, 256),
        input_size: int = 256,
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        c1, c2, c3, c4 = channels

        # Stem downsampling to /4
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )

        # Stage 1 (/4)
        self.stage1 = RSRWKV_Block(dim=c1, n_layer=4, layer_id=0)

        # Stage 2 (/8)
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        self.stage2 = RSRWKV_Block(dim=c2, n_layer=4, layer_id=1)

        # Stage 3 (/16)
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True)
        )
        self.stage3 = RSRWKV_Block(dim=c3, n_layer=4, layer_id=2)

        # Stage 4 (/32)
        self.down3 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True)
        )
        self.stage4 = RSRWKV_Block(dim=c4, n_layer=4, layer_id=3)

        # Decoder Stages
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(c4, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(c3 * 2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(c3, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(c2 * 2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(c2, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )

        # 4x Upsampling Head back to full input resolution
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x0 = self.stem(x)       # (B, c1, H/4, W/4)
        e1 = self.stage1(x0)
        e2 = self.stage2(self.down1(e1))  # (B, c2, H/8, W/8)
        e3 = self.stage3(self.down2(e2))  # (B, c3, H/16, W/16)
        e4 = self.stage4(self.down3(e3))  # (B, c4, H/32, W/32)

        # Decoder with Skip Connections
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = self.head(d1)
        return out


def build_model(
    num_classes: int = 1,
    in_channels: int = 3,
    input_size: int = 256,
    channels: Tuple[int, ...] = (32, 64, 128, 256),
    pretrained: bool = False,
    **kwargs
) -> RSRWKV_Segmentation:
    """
    Factory function for RSRWKV Segmentation model.

    Args:
        num_classes: Number of target classes (1 for binary flood segmentation)
        in_channels: Input channels (default: 3)
        input_size: Resolution (default: 256)
        channels: Channel progression across stages
    """
    model = RSRWKV_Segmentation(
        num_classes=num_classes,
        in_channels=in_channels,
        channels=channels,
        input_size=input_size,
        **kwargs
    )
    return model
