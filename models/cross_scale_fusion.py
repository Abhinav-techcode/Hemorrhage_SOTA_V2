# models/cross_scale_fusion.py
"""BiFPN‑inspired multi‑scale feature aggregation with fast normalised fusion."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import List
from .utils import init_weights_kaiming

__all__ = ["CrossScaleFusion", "CrossScaleFusion3D"]


class _BiFPNFusion(nn.Module):
    """Fast normalised fusion: (w1*x1 + w2*x2) / (w1 + w2 + eps)."""
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.ones(1))
        self.eps = 1e-4

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        # Use softmax to guarantee w1+w2=1 without eps, preventing gradient explosion when w1,w2 approach 0
        w = torch.softmax(torch.cat([self.w1, self.w2]), dim=0).view(2, 1, 1, 1, 1)
        return w[0] * x1 + w[1] * x2


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

class CrossScaleFusion3D(nn.Module):
    def __init__(self, in_dims: List[int], out_dim: int = 256):
        super().__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        
        # Project all encoder scales to a common dimension
        self.projs = nn.ModuleList([
            nn.Conv3d(dim, out_dim, 1, bias=False) for dim in in_dims
        ])
        
        self.norms = nn.ModuleList([
            nn.BatchNorm3d(out_dim) for _ in in_dims
        ])
        
        # Learnable weights for each scale during fusion
        self.fusion_weights = nn.Parameter(torch.ones(len(in_dims), dtype=torch.float32))
        self.relu = nn.ReLU()
        
        # Post-fusion convolution
        self.post_conv = nn.Sequential(
            nn.Conv3d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuses E1, E2, E3, E4 into a single representation at the highest resolution (or a target resolution).
        Here we fuse them to the resolution of E1.
        """
        assert len(features) == len(self.in_dims)
        
        target_size = features[0].shape[2:] # (D, H, W) of E1
        
        fused = 0
        weights = torch.softmax(self.fusion_weights, dim=0)
        
        for i, (feat, proj, norm, w) in enumerate(zip(features, self.projs, self.norms, weights)):
            # Project
            x = proj(feat)
            x = norm(x)
            x = self.relu(x)
            
            # Upsample if necessary
            if x.shape[2:] != target_size:
                x = nn.functional.interpolate(x, size=target_size, mode='trilinear', align_corners=False)
                
            fused = fused + w * x
            
        fused = self.post_conv(fused)
        return fused