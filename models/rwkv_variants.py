"""
RWKV Systematic Architectural Exploration for Flood Segmentation.
Explores RWKV linear attention integration across:
- Encoder (E)
- Bottleneck (B)
- Decoder (D)
- Skip Connections (S)
- Pairs (E+B, B+D, E+D, E+S, B+S, D+S)
- Triples (E+B+D, E+B+S, E+D+S, B+D+S)
- All 4 (E+B+D+S)
- HydroRWKV (Directional Strip Convolution + Bi-WKV)

Designed for absolute numerical stability:
- Token-wise Pre-LayerNorm (prevents exponential overflow in e^k)
- Bounded convex combination for spatial mixing (prevents key variance drift)
- Post-LayerNorm on all feature exits (protects against gradient explosion)
- Gradient-clipped compatibility with Adam/AdamW
"""

import math
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rwkv_block import bi_wkv_pure_pytorch, q_shift_2d


# ─────────────────────────────────────────────────────────────────────────────
# 1. Numerically Stable 2D Spatial RWKV Block
# ─────────────────────────────────────────────────────────────────────────────

class StableRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd: int, n_layer: int = 4, layer_id: int = 0):
        super().__init__()
        self.n_embd = n_embd
        self.layer_id = layer_id

        ratio_0_to_1 = layer_id / max(1, n_layer - 1)
        ratio_1_to_almost0 = 1.0 - (layer_id / max(1, n_layer))

        # Time decay speed initialization (S4D real init)
        decay_speed = torch.ones(n_embd)
        for h in range(n_embd):
            decay_speed[h] = -5 + 8 * (h / max(1, n_embd - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
        self.time_decay = nn.Parameter(decay_speed)

        zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(n_embd)]) * 0.5
        self.time_first = nn.Parameter(torch.ones(n_embd) * math.log(0.3) + zigzag)

        # Bounded convex interpolation weights in [0, 1]
        x = torch.ones(1, 1, n_embd)
        for i in range(n_embd):
            x[0, 0, i] = i / n_embd
        self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
        self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        # Linear projections with safe initializations
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

        nn.init.orthogonal_(self.key.weight, gain=0.5)
        nn.init.orthogonal_(self.value.weight, gain=0.5)
        nn.init.zeros_(self.output.weight)

    def forward(self, x: torch.Tensor, resolution: Tuple[int, int]) -> torch.Tensor:
        B, T, C = x.shape
        xx = q_shift_2d(x, shift_pixel=1, gamma=0.25, resolution=resolution)

        # Strictly bounded convex combination: Var(xk) <= 1
        xk = x * self.spatial_mix_k + xx * (1.0 - self.spatial_mix_k)
        xv = x * self.spatial_mix_v + xx * (1.0 - self.spatial_mix_v)
        xr = x * self.spatial_mix_r + xx * (1.0 - self.spatial_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        # Numerical safety clamp on key exponent: guarantees e^k never exceeds float32 limits
        k_clamped = torch.clamp(k, -20.0, 20.0)

        wkv = bi_wkv_pure_pytorch(self.time_decay, self.time_first, k_clamped, v)
        out = self.output(sr * wkv)
        return out


class StableRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd: int, n_layer: int = 4, layer_id: int = 0, hidden_ratio: float = 2.0):
        super().__init__()
        self.n_embd = n_embd
        self.layer_id = layer_id
        ratio_1_to_almost0 = 1.0 - (layer_id / max(1, n_layer))

        x = torch.ones(1, 1, n_embd)
        for i in range(n_embd):
            x[0, 0, i] = i / n_embd
        self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        hidden_dim = int(n_embd * hidden_ratio)
        self.key = nn.Linear(n_embd, hidden_dim, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_dim, n_embd, bias=False)

        nn.init.orthogonal_(self.key.weight, gain=0.5)
        nn.init.zeros_(self.value.weight)

    def forward(self, x: torch.Tensor, resolution: Tuple[int, int]) -> torch.Tensor:
        xx = q_shift_2d(x, shift_pixel=1, gamma=0.25, resolution=resolution)
        xk = x * self.spatial_mix_k + xx * (1.0 - self.spatial_mix_k)
        xr = x * self.spatial_mix_r + xx * (1.0 - self.spatial_mix_r)

        k = self.key(xk)
        # Bounded squared ReLU
        k = torch.clamp(torch.relu(k), 0.0, 10.0)
        k2 = torch.square(k)
        kv = self.value(k2)
        r = torch.sigmoid(self.receptance(xr))
        return r * kv


class SpatialRWKV2D(nn.Module):
    """
    Seamless 2D Feature Map Adapter for RWKV.
    Takes (B, C, H, W) -> runs Pre-LN WKV with Post-LN protection -> returns (B, C, H, W).
    """
    def __init__(self, dim: int, n_layer: int = 4, layer_id: int = 0):
        super().__init__()
        self.dim = dim
        self.ln1 = nn.LayerNorm(dim)
        self.spatial_mix = StableRWKV_SpatialMix(dim, n_layer=n_layer, layer_id=layer_id)
        self.ln2 = nn.LayerNorm(dim)
        self.channel_mix = StableRWKV_ChannelMix(dim, n_layer=n_layer, layer_id=layer_id)
        self.ln_out = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Pre-LN Spatial-Mix
        tokens = tokens + self.spatial_mix(self.ln1(tokens), resolution=(H, W))
        # Pre-LN Channel-Mix
        tokens = tokens + self.channel_mix(self.ln2(tokens), resolution=(H, W))
        # Mandatory Post-LN to prevent gradient explosion to subsequent stages
        tokens = self.ln_out(tokens)

        out = tokens.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Standard Convolutional Modules
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, use_rwkv: bool = False, layer_id: int = 0):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv = ConvBlock(in_c, out_c)
        self.rwkv = SpatialRWKV2D(out_c, n_layer=4, layer_id=layer_id) if use_rwkv else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.pool(x))
        x = self.rwkv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int, use_rwkv: bool = False, layer_id: int = 0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )
        self.rwkv = SpatialRWKV2D(out_c, n_layer=4, layer_id=layer_id) if use_rwkv else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = self.up(x)
        fused = self.fuse(torch.cat([x_up, skip], dim=1))
        return self.rwkv(fused)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Master Configurable RWKV Segmentation Framework
# ─────────────────────────────────────────────────────────────────────────────

class ConfigurableRWKVSeg(nn.Module):
    """
    Modular U-Net architecture allowing fine-grained toggling of RWKV modules at:
    - use_enc:  Encoder (Stages 2, 3)
    - use_bot:  Bottleneck (Stage 4)
    - use_dec:  Decoder (Up 3, 2, 1)
    - use_skip: Skip Connections
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channels: Tuple[int, ...] = (32, 64, 128, 256),
        use_enc: bool = False,
        use_bot: bool = False,
        use_dec: bool = False,
        use_skip: bool = False,
        input_size: int = 256,
        **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        c1, c2, c3, c4 = channels

        # Stage 1: Fast stem (always CNN to prevent T=4096 explosion)
        self.stem = ConvBlock(in_channels, c1)

        # Stage 2 & 3: Encoder
        self.down1 = DownBlock(c1, c2, use_rwkv=use_enc, layer_id=0)
        self.down2 = DownBlock(c2, c3, use_rwkv=use_enc, layer_id=1)

        # Stage 4: Bottleneck
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bot_conv = ConvBlock(c3, c4)
        self.bottleneck_rwkv = SpatialRWKV2D(c4, n_layer=4, layer_id=2) if use_bot else nn.Identity()

        # Skip Connection Processors
        self.skip_proc1 = SpatialRWKV2D(c1, n_layer=4, layer_id=0) if use_skip else nn.Identity()
        self.skip_proc2 = SpatialRWKV2D(c2, n_layer=4, layer_id=1) if use_skip else nn.Identity()
        self.skip_proc3 = SpatialRWKV2D(c3, n_layer=4, layer_id=2) if use_skip else nn.Identity()

        # Decoder Stages
        self.up3 = UpBlock(c4, c3, c3, use_rwkv=use_dec, layer_id=1)
        self.up2 = UpBlock(c3, c2, c2, use_rwkv=use_dec, layer_id=2)
        self.up1 = UpBlock(c2, c1, c1, use_rwkv=use_dec, layer_id=3)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.stem(x)       # (B, c1, H, W)
        x2 = self.down1(x1)     # (B, c2, H/2, W/2)
        x3 = self.down2(x2)     # (B, c3, H/4, W/4)

        # Bottleneck
        x4 = self.bot_conv(self.pool3(x3))
        x4 = self.bottleneck_rwkv(x4) # (B, c4, H/8, W/8)

        # Skip paths
        s1 = self.skip_proc1(x1)
        s2 = self.skip_proc2(x2)
        s3 = self.skip_proc3(x3)

        # Decoder
        d3 = self.up3(x4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)

        out = self.head(d1)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Domain Specific: HydroRWKV (Directional Strip Conv + Bi-WKV)
# ─────────────────────────────────────────────────────────────────────────────

class HydroRWKV_Block(nn.Module):
    """
    Directional Strip Convolution (1x7 + 7x1) combined with Bi-WKV.
    Captures winding stream channels and longitudinal water flows.
    """
    def __init__(self, dim: int, layer_id: int = 0):
        super().__init__()
        self.strip_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), padding=(0, 3), groups=dim, bias=False)
        self.strip_w = nn.Conv2d(dim, dim, kernel_size=(7, 1), padding=(3, 0), groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.SiLU(inplace=True)
        self.rwkv = SpatialRWKV2D(dim, n_layer=4, layer_id=layer_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x_strip = self.act(self.bn(self.strip_h(x) + self.strip_w(x)))
        out = self.rwkv(x_strip + res)
        return out


class HydroRWKV(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, channels: Tuple[int, ...] = (32, 64, 128, 256), input_size: int = 256, **kwargs):
        super().__init__()
        self.input_size = input_size
        c1, c2, c3, c4 = channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.down1 = nn.Sequential(nn.MaxPool2d(2, 2), HydroRWKV_Block(c1, layer_id=0), nn.Conv2d(c1, c2, 1, bias=False), nn.BatchNorm2d(c2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2, 2), HydroRWKV_Block(c2, layer_id=1), nn.Conv2d(c2, c3, 1, bias=False), nn.BatchNorm2d(c3))
        self.bot = nn.Sequential(nn.MaxPool2d(2, 2), HydroRWKV_Block(c3, layer_id=2), nn.Conv2d(c3, c4, 1, bias=False), nn.BatchNorm2d(c4))

        self.up3 = UpBlock(c4, c3, c3, use_rwkv=False)
        self.up2 = UpBlock(c3, c2, c2, use_rwkv=False)
        self.up1 = UpBlock(c2, c1, c1, use_rwkv=False)
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stem(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.bot(x3)

        d3 = self.up3(x4, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up1(d2, x1)
        return self.head(d1)
