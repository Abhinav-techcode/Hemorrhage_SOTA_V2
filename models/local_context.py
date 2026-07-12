# models/local_context.py
"""Lightweight Local Context Refinement branch."""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor

__all__ = ["LocalContextBlock"]


class SEModule3D(nn.Module):
    """Squeeze‑and‑Excitation in 3D."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.fc(self.avg_pool(x))


class LocalContextBlock(nn.Module):
    """Local context enhancement: residual conv + depthwise conv + SE.
    
    Args:
        dim: Feature channels.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv3d(dim, dim, 3, padding=1)
        self.dwconv = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.se = SEModule3D(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.act(self.conv1(x))
        out = self.dwconv(out)
        out = self.se(out)
        return identity + out