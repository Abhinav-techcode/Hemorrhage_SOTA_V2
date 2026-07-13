# models/umamba_bridge.py
"""UMamba bottleneck for capturing long-range dependencies."""
from __future__ import annotations

import logging
import torch
import torch.nn as nn
from torch import Tensor

from .utils import init_weights_kaiming

logger = logging.getLogger(__name__)

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


class MambaFallback(nn.Module):
    """Transformer-based fallback if mamba_ssm is not installed."""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        # We ignore d_state and d_conv for the fallback
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * expand),
            nn.GELU(),
            nn.Linear(d_model * expand, d_model)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, D)
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm(x))
        return x


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
        
        for _ in range(num_blocks):
            if HAS_MAMBA:
                self.blocks.append(
                    Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
                )
            else:
                self.blocks.append(
                    MambaFallback(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
                )
                
        if not HAS_MAMBA:
            logger.warning("mamba_ssm not found! UMambaBridge is using Transformer fallback.")

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
        
        for block in self.blocks:
            # Both Mamba and MambaFallback take (B, L, C)
            x_seq = block(x_seq)
            
        # Transpose back: (B, C, L)
        x_flat = x_seq.transpose(1, 2)
        # Reshape to 3D: (B, C, D, H, W)
        out = x_flat.view(B, C, D, H, W)
        
        return out
