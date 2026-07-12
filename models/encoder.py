# models/encoder.py
"""Hierarchical MedNeXt encoder outputting 5 scales."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import List

from .mednext_block import MedNeXtBlock, DownsampleBlock
from .utils import init_weights_kaiming

__all__ = ["Encoder"]


class Encoder(nn.Module):
    """MedNeXt encoder.
    
    Args:
        in_channels: Input channels (1 for CT).
        depths: Number of blocks per stage.
        dims: Channel dims per stage.
        kernel_sizes: Kernel sizes per stage.
        drop_path_rate: Base drop path rate.
    """
    def __init__(self,
                 in_channels: int = 1,
                 depths: List[int] = [2, 2, 4, 2, 2],
                 dims: List[int] = [24, 48, 96, 192, 384],
                 kernel_sizes: List[int] = [7, 7, 7, 7, 7],
                 drop_path_rate: float = 0.1):
        super().__init__()
        self.dims = dims
        self.stem = nn.Conv3d(in_channels, dims[0], kernel_size=3, padding=1)
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(len(depths)):
            blocks = nn.ModuleList()
            for j in range(depths[i]):
                blocks.append(MedNeXtBlock(dims[i], kernel_sizes[i],
                                           drop_path=dp_rates[cur + j]))
            cur += depths[i]
            self.stages.append(blocks)
            if i < len(depths) - 1:
                self.stages.append(DownsampleBlock(dims[i], dims[i + 1]))
        self.apply(init_weights_kaiming)

    def forward(self, x: Tensor) -> List[Tensor]:
        """Forward.
        
        Args:
            x: (B, 1, D, H, W).
        Returns:
            List of feature maps [Stage0, Stage1, Stage2, Stage3, Stage4] 
            each (B, C_i, D_i, H_i, W_i).
        """
        x = self.stem(x)
        features = []
        for module in self.stages:
            if isinstance(module, nn.ModuleList):
                for block in module:
                    x = block(x)
                features.append(x)
            else:  # DownsampleBlock
                x = module(x)
        return features