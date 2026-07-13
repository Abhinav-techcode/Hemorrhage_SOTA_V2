# models/boundary_decoder.py
"""Attention-Guided Decoder with Boundary Refinement and Deep Supervision."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import List

from .boundary_refinement import BoundaryRefinement
from .utils import init_weights_kaiming


class AttentionGate3D(nn.Module):
    """Attention Gate for skip connections."""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: Tensor, x: Tensor) -> Tensor:
        # g: gating signal (upsampled feature)
        # x: skip connection feature
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Handle spatial mismatch if necessary (should be matched by caller)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = nn.functional.interpolate(g1, size=x1.shape[2:], mode='trilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DecoderBlock(nn.Module):
    """Upsample + Attention Gate + Concat + Conv."""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.ag = AttentionGate3D(F_g=in_channels // 2, F_l=skip_channels, F_int=in_channels // 4)
        
        # After concat: (in_channels // 2) + skip_channels
        concat_channels = (in_channels // 2) + skip_channels
        self.conv = nn.Sequential(
            nn.Conv3d(concat_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x_up = self.up(x)
        x_skip = self.ag(g=x_up, x=skip)
        x_cat = torch.cat([x_skip, x_up], dim=1)
        return self.conv(x_cat)


class BoundaryDecoder(nn.Module):
    """
    Decoder emitting deep supervision features.
    
    Args:
        bridge_dim: Channel dimension from the UMamba bottleneck.
        skip_dims: Channel dimensions of the skip connections [S0, S1, S2, S3].
        decoder_dims: Output channel dimensions for each decoder stage [D0, D1, D2, D3].
    """
    def __init__(self, bridge_dim: int, skip_dims: List[int], decoder_dims: List[int]):
        super().__init__()
        assert len(skip_dims) == 4 and len(decoder_dims) == 4
        
        self.blocks = nn.ModuleList()
        in_ch = bridge_dim
        for i in range(4):
            # Process skips in reverse order: S3, S2, S1, S0
            self.blocks.append(
                DecoderBlock(in_ch, skip_dims[3 - i], decoder_dims[i])
            )
            in_ch = decoder_dims[i]
            
        # Boundary Refinement on the final output (S0 resolution)
        self.boundary_refine = BoundaryRefinement(decoder_dims[-1])
        
        self.apply(init_weights_kaiming)

    def forward(self, x: Tensor, skips: List[Tensor]) -> List[Tensor]:
        """
        Args:
            x: Bottleneck feature (S4 resolution).
            skips: List of features [S0, S1, S2, S3].
        Returns:
            List of decoder features at each stage.
            Indices: 0 -> S3_res, 1 -> S2_res (Quarter), 2 -> S1_res (Half), 3 -> S0_res (Full).
        """
        features = []
        for i, block in enumerate(self.blocks):
            # skips[3-i] gets S3, S2, S1, S0 in that order
            x = block(x, skips[3 - i])
            
            # Apply boundary refinement only to the final full-resolution output
            if i == 3:
                x = self.boundary_refine(x)
                
            features.append(x)
            
        return features
