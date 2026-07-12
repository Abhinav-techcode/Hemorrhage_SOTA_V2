# models/boundary_context.py
"""Boundary‑Aware Context Enhancement (lightweight edge sharpening)."""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from .utils import init_weights_kaiming

__all__ = ["BoundaryContextEnhancement"]


class BoundaryContextEnhancement(nn.Module):
    """Residual depthwise convolution with spatial attention for boundary amplification.
    
    Args:
        dim: Feature channels.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.spatial_att = nn.Sequential(
            nn.Conv3d(dim, 1, 1),
            nn.Sigmoid()
        )
        self.act = nn.GELU()
        self.apply(init_weights_kaiming)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.dwconv(x)
        att = self.spatial_att(out)
        out = out * att + identity
        return self.act(out)