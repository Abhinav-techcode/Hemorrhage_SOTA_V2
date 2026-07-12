# models/utils.py
"""Utility layers and functions shared across the model."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "drop_path",
    "LayerNorm3d",
    "GELU",
    "init_weights_kaiming",
]


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    """Drop paths (Stochastic Depth) per sample.
    
    Args:
        x: (B, C, D, H, W).
        drop_prob: Drop probability.
        training: Training flag.
    Returns:
        Tensor of same shape.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class LayerNorm3d(nn.Module):
    """Layer Normalization for 3D tensors (C, D, H, W)."""
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x


class GELU(nn.Module):
    """GELU activation (torch.compile friendly)."""
    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.gelu(x)


def init_weights_kaiming(module: nn.Module) -> None:
    """Initialize Conv3d and BatchNorm3d with Kaiming normal."""
    if isinstance(module, nn.Conv3d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)