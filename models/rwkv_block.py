"""
RWKV Linear Attention Block (WKV) with Pure PyTorch Native Fallback
Supports seamless execution across CPU, CUDA, and edge devices without requiring
compiled C++/CUDA extensions.

If custom CUDA WKV kernels are compiled and available, it can use them for speed.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def wkv_pure_pytorch(w: torch.Tensor, u: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable pure PyTorch implementation of the WKV linear attention operator.
    Guarantees compatibility across CPU and GPU on any OS.

    Args:
        w: Time decay parameter of shape (C,)
        u: First-step bonus parameter of shape (C,)
        k: Keys tensor of shape (B, T, C)
        v: Values tensor of shape (B, T, C)

    Returns:
        y: Output tensor of shape (B, T, C)
    """
    B, T, C = k.shape
    w_decay = -torch.exp(w.float())
    u_val = u.float()
    k_val = k.float()
    v_val = v.float()

    y = torch.zeros(B, T, C, device=k.device, dtype=k.dtype)
    a = torch.zeros(B, C, device=k.device, dtype=torch.float32)
    b = torch.zeros(B, C, device=k.device, dtype=torch.float32)
    p = torch.full((B, C), -1e30, device=k.device, dtype=torch.float32)

    for t in range(T):
        kt = k_val[:, t, :]
        vt = v_val[:, t, :]

        # Output calculation at step t
        ww = u_val + kt
        p_out = torch.maximum(p, ww)
        e1 = torch.exp(p - p_out)
        e2 = torch.exp(ww - p_out)
        y[:, t, :] = ((e1 * a + e2 * vt) / (e1 * b + e2 + 1e-8)).to(k.dtype)

        # State transition for t + 1
        ww_state = p + w_decay
        p_next = torch.maximum(ww_state, kt)
        e1_s = torch.exp(ww_state - p_next)
        e2_s = torch.exp(kt - p_next)
        a = e1_s * a + e2_s * vt
        b = e1_s * b + e2_s
        p = p_next

    return y


def bi_wkv_pure_pytorch(w: torch.Tensor, u: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Bi-directional WKV operator for 2D visual tasks.
    Combines forward and backward sequential scans.
    """
    y_fwd = wkv_pure_pytorch(w, u, k, v)
    k_rev = torch.flip(k, dims=[1])
    v_rev = torch.flip(v, dims=[1])
    y_bwd = wkv_pure_pytorch(w, u, k_rev, v_rev)
    y_bwd = torch.flip(y_bwd, dims=[1])
    return 0.5 * (y_fwd + y_bwd)


def q_shift_2d(x: torch.Tensor, shift_pixel: int = 1, gamma: float = 0.25, resolution: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    2D spatial q-shift operation for Vision-RWKV.
    Shifts 4 quadrants of channels along 4 spatial directions (left, right, up, down).
    """
    assert gamma <= 0.25
    B, N, C = x.shape
    if resolution is None:
        sqrt_N = int(math.isqrt(N))
        assert sqrt_N * sqrt_N == N, f"Cannot infer resolution from sequence length {N}"
        H, W = sqrt_N, sqrt_N
    else:
        H, W = resolution

    x_2d = x.transpose(1, 2).reshape(B, C, H, W)
    out = torch.zeros_like(x_2d)

    q = int(C * gamma)
    # Right shift
    out[:, 0:q, :, shift_pixel:W] = x_2d[:, 0:q, :, 0:W - shift_pixel]
    # Left shift
    out[:, q:2 * q, :, 0:W - shift_pixel] = x_2d[:, q:2 * q, :, shift_pixel:W]
    # Down shift
    out[:, 2 * q:3 * q, shift_pixel:H, :] = x_2d[:, 2 * q:3 * q, 0:H - shift_pixel, :]
    # Up shift
    out[:, 3 * q:4 * q, 0:H - shift_pixel, :] = x_2d[:, 3 * q:4 * q, shift_pixel:H, :]
    # Identity for remainder
    out[:, 4 * q:, ...] = x_2d[:, 4 * q:, ...]

    return out.flatten(2).transpose(1, 2)

