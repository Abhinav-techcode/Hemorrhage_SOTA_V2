# models/boundary_refinement.py
"""Lightweight boundary refinement module."""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor

__all__ = ["BoundaryRefinement"]


class BoundaryRefinement(nn.Module):
    """Residual depthwise convolution for final boundary sharpening.
    
    Args:
        dim: Feature channels.
        kernel_size: Kernel size for depthwise convolutions.
    """
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.dwconv1 = nn.Conv3d(dim, dim, kernel_size, padding=kernel_size//2,
                                 groups=dim)
        self.dwconv2 = nn.Conv3d(dim, dim, kernel_size, padding=kernel_size//2,
                                 groups=dim)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.dwconv1(x)
        out = self.act(out)
        out = self.dwconv2(out)
        return residual + out