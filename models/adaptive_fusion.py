# models/adaptive_fusion.py
"""Adaptive Context Fusion with channel‑spatial attention and routing coefficients."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from .utils import init_weights_kaiming

__all__ = ["AdaptiveContextFusion"]


class ChannelAttention(nn.Module):
    """SE‑based channel attention."""
    def __init__(self, channels: int, reduction: int = 8):
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


class SpatialAttention(nn.Module):
    """3D spatial attention."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(attn))


class AdaptiveContextFusion(nn.Module):
    """Fuses three branch outputs using routing coefficients, channel & spatial attention.
    
    Args:
        dim: Feature channels of each branch.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.weight_fusion = nn.Conv3d(dim * 3, dim, 1)  # for residual connection later
        self.ca = ChannelAttention(dim)
        self.sa = SpatialAttention()
        self.refine = nn.Sequential(
            nn.Conv3d(dim, dim, 3, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim, dim, 3, padding=1),
            nn.BatchNorm3d(dim)
        )
        self.apply(init_weights_kaiming)

    def forward(self, branch_a: Tensor, branch_b: Tensor, branch_c: Tensor,
                coeffs: Tensor) -> Tensor:
        """Forward.
        
        Args:
            branch_a, branch_b, branch_c: (B, C, D, H, W) features.
            coeffs: (B, 3, 1, 1, 1) global routing weights.
        Returns:
            Fused feature map (B, C, D, H, W).
        """
        a_w = branch_a * coeffs[:, 0:1, ...]
        b_w = branch_b * coeffs[:, 1:2, ...]
        c_w = branch_c * coeffs[:, 2:3, ...]
        concat = torch.cat([a_w, b_w, c_w], dim=1)  # (B, 3C, D, H, W)
        fused = self.weight_fusion(concat)  # (B, C, D, H, W)
        fused = self.ca(fused)
        fused = fused * self.sa(fused)
        refined = self.refine(fused)
        return fused + refined