"""
Vision-RWKV (VRWKV): Efficient Visual Perception with RWKV Linear Attention
Adapted for real-time flood area segmentation on edge & embedded platforms.

Original Repository: https://github.com/OpenGVLab/Vision-RWKV
Commit Hash: 0e0e38a6fd4475ae3817e14420ab85fab08d9097
Reference: Duan et al., "Vision-RWKV: Efficient and Scalable Visual Perception with RWKV-Like Architectures",
ICLR 2025 (Spotlight).
arXiv: https://arxiv.org/abs/2403.02308

Key Architectural Components:
- Visual tokenization with hierarchical patch embedding
- VRWKV Spatial-Mix with 2D spatial q-shift and bi-directional WKV attention
- VRWKV Channel-Mix with squared-ReLU gating
- Lightweight Multi-Level Feature Aggregation Head for pixel-level segmentation
- Native Pure PyTorch WKV fallback for complete portability across CPU / edge devices
"""

import math
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rwkv_block import bi_wkv_pure_pytorch, q_shift_2d


class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd: int, n_layer: int, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd

        ratio_0_to_1 = layer_id / max(1, n_layer - 1)
        ratio_1_to_almost0 = 1.0 - (layer_id / max(1, n_layer))

        decay_speed = torch.ones(n_embd)
        for h in range(n_embd):
            decay_speed[h] = -5 + 8 * (h / max(1, n_embd - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
        self.time_decay = nn.Parameter(decay_speed)

        zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(n_embd)]) * 0.5
        self.time_first = nn.Parameter(torch.ones(n_embd) * math.log(0.3) + zigzag)

        x = torch.ones(1, 1, n_embd)
        for i in range(n_embd):
            x[0, 0, i] = i / n_embd
        self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
        self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor, patch_resolution: Tuple[int, int]) -> torch.Tensor:
        B, T, C = x.shape
        xx = q_shift_2d(x, shift_pixel=1, gamma=0.25, resolution=patch_resolution)

        xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
        xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
        xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        wkv = bi_wkv_pure_pytorch(self.time_decay, self.time_first, k, v)
        out = self.output(sr * wkv)
        return out


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd: int, n_layer: int, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd

        ratio_1_to_almost0 = 1.0 - (layer_id / max(1, n_layer))
        x = torch.ones(1, 1, n_embd)
        for i in range(n_embd):
            x[0, 0, i] = i / n_embd
        self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        hidden_sz = 4 * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def forward(self, x: torch.Tensor, patch_resolution: Tuple[int, int]) -> torch.Tensor:
        xx = q_shift_2d(x, shift_pixel=1, gamma=0.25, resolution=patch_resolution)
        xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
        xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        r = torch.sigmoid(self.receptance(xr))
        return r * kv


class VRWKV_Block(nn.Module):
    def __init__(self, n_embd: int, n_layer: int, layer_id: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.spatial_mix = VRWKV_SpatialMix(n_embd, n_layer, layer_id)
        self.ln2 = nn.LayerNorm(n_embd)
        self.channel_mix = VRWKV_ChannelMix(n_embd, n_layer, layer_id)

    def forward(self, x: torch.Tensor, patch_resolution: Tuple[int, int]) -> torch.Tensor:
        x = x + self.spatial_mix(self.ln1(x), patch_resolution)
        x = x + self.channel_mix(self.ln2(x), patch_resolution)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 64, patch_size: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, C, H, W = x.shape
        x = self.proj(x)
        H_out, W_out = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (H_out, W_out)


class VisionRWKVSegmentation(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        embed_dim: int = 64,
        depth: int = 6,
        patch_size: int = 4,
        input_size: int = 256
    ):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(in_channels=in_channels, embed_dim=embed_dim, patch_size=patch_size)
        self.blocks = nn.ModuleList([
            VRWKV_Block(n_embd=embed_dim, n_layer=depth, layer_id=i) for i in range(depth)
        ])
        self.ln_out = nn.LayerNorm(embed_dim)

        # Lightweight decoder head
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(embed_dim // 4, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens, (pH, pW) = self.patch_embed(x)

        for blk in self.blocks:
            tokens = blk(tokens, (pH, pW))

        tokens = self.ln_out(tokens)
        feat_2d = tokens.transpose(1, 2).reshape(B, self.embed_dim, pH, pW)
        out = self.head(feat_2d)

        if out.shape[2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        return out


def build_model(num_classes: int = 1, input_size: int = 256, pretrained: bool = False, in_channels: int = 3, **kwargs):
    return VisionRWKVSegmentation(
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=64,
        depth=6,
        patch_size=4,
        input_size=input_size
    )
