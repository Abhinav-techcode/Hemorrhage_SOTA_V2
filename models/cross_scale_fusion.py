# models/cross_scale_fusion.py
"""BiFPN‑inspired multi‑scale feature aggregation with fast normalised fusion."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import List
from .utils import init_weights_kaiming

__all__ = ["CrossScaleFusion"]


class _BiFPNFusion(nn.Module):
    """Fast normalised fusion: (w1*x1 + w2*x2) / (w1 + w2 + eps)."""
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.ones(1))
        self.eps = 1e-4

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        w1 = torch.relu(self.w1)
        w2 = torch.relu(self.w2)
        return (w1 * x1 + w2 * x2) / (w1 + w2 + self.eps)


class CrossScaleFusion(nn.Module):
    """BiFPN‑style cross‑scale feature fusion that outputs a pyramid of features.
    
    Args:
        in_dims: Channel dimensions of encoder stages [S0, S1, S2, S3, S4].
        out_dim: Output channel dimension for all pyramid levels.
    """
    def __init__(self, in_dims: List[int], out_dim: int):
        super().__init__()
        assert len(in_dims) == 5
        # Channel projection for each input
        self.proj_convs = nn.ModuleList([
            nn.Conv3d(d, out_dim, 1) if d != out_dim else nn.Identity()
            for d in in_dims
        ])
        # Top‑down fusion weights and post‑fusion convolutions
        self.td_fusions = nn.ModuleList([_BiFPNFusion() for _ in range(4)])
        self.td_convs = nn.ModuleList([
            nn.Conv3d(out_dim, out_dim, 3, padding=1) for _ in range(4)
        ])
        # Bottom‑up fusion weights and post‑fusion convolutions
        self.bu_fusions = nn.ModuleList([_BiFPNFusion() for _ in range(4)])
        self.bu_convs = nn.ModuleList([
            nn.Conv3d(out_dim, out_dim, 3, padding=1) for _ in range(4)
        ])
        self.apply(init_weights_kaiming)

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        """Forward.
        
        Args:
            features: List of [S0, S1, S2, S3, S4] tensors.
        Returns:
            List of refined pyramid features [P0, P1, P2, P3, P4] at same 
            spatial resolutions as inputs, all with out_dim channels.
        """
        # Project all to out_dim
        proj = [conv(f) if isinstance(conv, nn.Conv3d) else f
                for f, conv in zip(features, self.proj_convs)]

        # Top‑down: P4 -> ... -> P0
        td = [proj[-1]]
        for i in range(3, -1, -1):  # i = 3,2,1,0  (P4->P3,...,P1->P0)
            prev = td[-1]
            up = nn.functional.interpolate(prev, size=proj[i].shape[2:],
                                           mode="trilinear", align_corners=False)
            fused_weight = self.td_fusions[3 - i](proj[i], up)
            fused = self.td_convs[3 - i](fused_weight)
            td.append(fused)
        td = td[::-1]  # now P0..P4

        # Bottom‑up: P0 -> ... -> P4
        bu = [td[0]]
        for i in range(1, 5):
            prev = bu[-1]
            down = nn.functional.interpolate(prev, size=td[i].shape[2:],
                                             mode="trilinear", align_corners=False)
            fused_weight = self.bu_fusions[i-1](td[i], down)
            fused = self.bu_convs[i-1](fused_weight)
            bu.append(fused)
        return bu  # [P0, P1, P2, P3, P4]