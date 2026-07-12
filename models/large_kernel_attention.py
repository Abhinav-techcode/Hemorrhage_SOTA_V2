# models/large_kernel_attention.py
"""3D Large Kernel Attention (VAN‑inspired)."""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor

__all__ = ["LKA3D"]


class LKA3D(nn.Module):
    """Large Kernel Attention: depthwise dilated + pointwise expansion.
    
    Args:
        dim: Input channels.
        kernel_size: Large kernel size.
        dilation: Dilation rate.
    """
    def __init__(self, dim: int, kernel_size: int = 7, dilation: int = 3):
        super().__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
        self.dw_conv = nn.Conv3d(dim, dim, kernel_size=kernel_size,
                                 padding=padding, dilation=dilation, groups=dim)
        self.pw_conv1 = nn.Conv3d(dim, dim * 4, 1)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Conv3d(dim * 4, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.dw_conv(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        return shortcut + x