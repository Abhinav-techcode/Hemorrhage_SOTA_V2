# models/dynamic_router.py
"""Research‑grade Dynamic Feature Router with global pooling, linear MLP, and channel attention.

This module replaces the original BatchNorm‑based router with a publication‑quality version
that uses both average and max pooling, a lightweight linear MLP, and a channel attention
mechanism. No BatchNorm is used, making it stable for small batch sizes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["DynamicFeatureRouter"]


class DynamicFeatureRouter(nn.Module):
    """Generates per‑branch routing coefficients from the coarsest feature map.

    The router aggregates spatial information via global average and max pooling,
    processes the concatenated descriptor through a compact linear MLP with GELU
    activations and dropout, refines the result with a lightweight channel attention,
    and finally projects to branch‑wise coefficients normalised by softmax.

    Args:
        in_channels: Number of input channels (typically from pyramid level P4).
        branches: Number of parallel context branches (default 3).
        reduction: Channel reduction ratio for the intermediate attention gate.
    """

    def __init__(self, in_channels: int, branches: int = 3, reduction: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.branches = branches
        self.reduction = reduction

        # Pooling layers (spatial reduction to 1x1x1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # After flattening and concatenating GAP and GMP: 2*in_channels -> in_channels
        self.linear1 = nn.Linear(2 * in_channels, in_channels)
        self.act1 = nn.GELU()
        self.dropout = nn.Dropout(0.1)

        # Second linear layer (in_channels -> in_channels)
        self.linear2 = nn.Linear(in_channels, in_channels)
        self.act2 = nn.GELU()

        # Lightweight channel attention (operates on the C‑dim vector)
        attn_hidden = max(16, in_channels // reduction)
        self.channel_att = nn.Sequential(
            nn.Linear(in_channels, attn_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(attn_hidden, in_channels),
            nn.Sigmoid(),
        )

        # Final projection to branch coefficients
        self.proj = nn.Linear(in_channels, branches)
        self.softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        """Initialize all Linear layers with Kaiming uniform and biases to zero."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0, mode="fan_in", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Coarsest feature map of shape (B, C, D, H, W).

        Returns:
            Global routing coefficients of shape (B, branches, 1, 1, 1) with softmax
            applied over the branch dimension.
        """
        # 1. Global pooling
        avg = self.avg_pool(x)  # (B, C, 1, 1, 1)
        max_ = self.max_pool(x)  # (B, C, 1, 1, 1)

        # 2. Flatten to (B, C) and concatenate
        avg_flat = avg.view(x.size(0), -1)  # (B, C)
        max_flat = max_.view(x.size(0), -1)  # (B, C)
        combined = torch.cat([avg_flat, max_flat], dim=1)  # (B, 2C)

        # 3. Linear MLP with dropout
        feat = self.linear1(combined)  # (B, C)
        feat = self.act1(feat)
        feat = self.dropout(feat)

        feat = self.linear2(feat)  # (B, C)
        feat = self.act2(feat)

        # 4. Channel attention refinement
        attn = self.channel_att(feat)  # (B, C)
        feat = feat * attn

        # 5. Project to branch coefficients and softmax
        coeffs_1d = self.proj(feat)  # (B, branches)
        coeffs_1d = self.softmax(coeffs_1d)

        # 6. Reshape to (B, branches, 1, 1, 1) for downstream compatibility
        coeffs = coeffs_1d.view(x.size(0), self.branches, 1, 1, 1)
        return coeffs