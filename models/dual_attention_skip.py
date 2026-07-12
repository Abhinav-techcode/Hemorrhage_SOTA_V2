# models/dual_attention_skip.py
"""Dual Attention Skip Fusion: channel + spatial attention on encoder skip features."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["DualAttentionSkip"]


class ChannelAttention3D(nn.Module):
    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(dim // reduction, dim, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg = self.fc(self.avg_pool(x))
        max_ = self.fc(self.max_pool(x))
        return self.sigmoid(avg + max_)


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(attn))


class DualAttentionSkip(nn.Module):
    """Enhances skip features with channel and spatial attention, then residual.
    
    Args:
        enc_dim: Encoder feature channels.
    """
    def __init__(self, enc_dim: int):
        super().__init__()
        self.channel_att = ChannelAttention3D(enc_dim)
        self.spatial_att = SpatialAttention3D()
        self.res_conv = nn.Conv3d(enc_dim, enc_dim, 3, padding=1)

    def forward(self, enc_feat: Tensor) -> Tensor:
        ca = self.channel_att(enc_feat)
        feat_ca = enc_feat * ca
        sa = self.spatial_att(feat_ca)
        feat_sa = feat_ca * sa
        out = self.res_conv(feat_sa) + enc_feat
        return out