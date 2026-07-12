# models/mednext_block.py
"""MedNeXt‑style building blocks."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .utils import drop_path, LayerNorm3d, GELU

__all__ = ["MedNeXtBlock", "LayerScale", "DownsampleBlock"]


class LayerScale(nn.Module):
    """Per‑channel scaling (ConvNeXt)."""
    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma.view(1, -1, 1, 1, 1)


class MedNeXtBlock(nn.Module):
    """MedNeXt block: 3D depthwise large‑kernel conv, GELU, LayerScale.
    
    Args:
        dim: Input channels.
        kernel_size: Kernel size.
        expand_ratio: Channel expansion factor.
        drop_path: Stochastic depth rate.
    """
    def __init__(self, dim: int, kernel_size: int = 7, expand_ratio: float = 4.0,
                 drop_path: float = 0.0):
        super().__init__()
        self.norm1 = LayerNorm3d(dim)
        self.dwconv = nn.Conv3d(dim, dim, kernel_size, padding=kernel_size // 2,
                                groups=dim)
        self.act = GELU()
        self.norm2 = LayerNorm3d(dim)
        hidden_dim = int(dim * expand_ratio)
        self.pwconv1 = nn.Conv3d(dim, hidden_dim, 1)
        self.act2 = GELU()
        self.pwconv2 = nn.Conv3d(hidden_dim, dim, 1)
        self.ls1 = LayerScale(dim)
        self.ls2 = LayerScale(dim)
        self.drop_path = drop_path

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act2(x)
        x = self.pwconv2(x)
        x = self.ls2(x)
        if self.drop_path > 0.0 and self.training:
            x = drop_path(x, self.drop_path, self.training)
        return shortcut + x


class DownsampleBlock(nn.Module):
    """2× downsampling via strided convolution."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv3d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)