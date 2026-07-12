# models/feature_calibration.py
"""Lightweight Feature Calibration module applied after context fusion."""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from .utils import init_weights_kaiming

__all__ = ["FeatureCalibration"]


class FeatureCalibration(nn.Module):
    """Refine and calibrate fused features with residual block and SE attention.
    
    Args:
        dim: Feature channels.
    """
    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv3d(dim, dim, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm3d(dim)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(dim, dim, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm3d(dim)
        # SE attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim // reduction, dim, 1, bias=False),
            nn.Sigmoid()
        )
        self.apply(init_weights_kaiming)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out * self.se(out)
        return identity + out