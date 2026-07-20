# models/umamba_bridge.py
"""UMamba bottleneck for capturing long-range dependencies."""
from __future__ import annotations

import logging
import torch
import torch.nn as nn
from torch import Tensor

from .mamba_pytorch import Mamba

from .utils import init_weights_kaiming

logger = logging.getLogger(__name__)

class UMambaBridge(nn.Module):
    """
    UMamba Bridge / Bottleneck.
    Flattens the 3D bottleneck feature, applies Mamba to capture 
    global context, and reshapes back to 3D.
    """
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, num_blocks: int = 2):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_blocks):
            self.blocks.append(
                Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            )
            self.norms.append(nn.LayerNorm(dim))

        self.apply(init_weights_kaiming)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        # Flatten spatial dimensions into a sequence: (B, C, L)
        x_flat = x.view(B, C, -1)
        # Transpose for sequence modeling: (B, L, C)
        x_seq = x_flat.transpose(1, 2)
        
        for block, norm in zip(self.blocks, self.norms):
            # Pre-norm formulation with residual connection
            residual = x_seq
            x_seq = norm(x_seq)
            x_seq = block(x_seq)
            x_seq = residual + x_seq
            
        # Transpose back: (B, C, L)
        x_flat = x_seq.transpose(1, 2)
        # Reshape to 3D: (B, C, D, H, W)
        out = x_flat.view(B, C, D, H, W)
        
        return out
