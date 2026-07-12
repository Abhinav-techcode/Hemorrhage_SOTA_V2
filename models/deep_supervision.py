# models/deep_supervision.py
"""Deep supervision heads producing segmentation maps at three resolutions."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Dict

__all__ = ["DeepSupervisionHeads"]


class SegmentationHead(nn.Module):
    """1x1x1 conv to logits."""
    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class DeepSupervisionHeads(nn.Module):
    """Generate predictions at quarter, half, and full resolutions.
    
    The decoder outputs features at encoder resolutions: 
    [S4, S3, S2, S1, S0]. We attach heads to:
        quarter resolution: S2 (1/4 of full)
        half resolution:    S1 (1/2 of full)
        full resolution:    S0
    """
    def __init__(self, decoder_dims: List[int], num_classes: int = 1):
        super().__init__()
        if len(decoder_dims) < 3:
            raise ValueError("Decoder must have at least 3 stages for deep supervision")
        # For 5 stages: indices 2 (S2), 3 (S1), 4 (S0)
        if len(decoder_dims) == 5:
            self.idx_quarter = 2
            self.idx_half = 3
            self.idx_full = 4
        else:
            # fallback: equally spaced
            self.idx_quarter = max(0, len(decoder_dims) // 4)
            self.idx_half = len(decoder_dims) // 2
            self.idx_full = len(decoder_dims) - 1

        self.head_quarter = SegmentationHead(decoder_dims[self.idx_quarter], num_classes)
        self.head_half = SegmentationHead(decoder_dims[self.idx_half], num_classes)
        self.head_full = SegmentationHead(decoder_dims[self.idx_full], num_classes)

    def forward(self, decoder_outputs: List[Tensor]) -> Dict[str, Tensor]:
        """Forward.
        
        Args:
            decoder_outputs: List of features [coarse, ..., fine] (length N).
        Returns:
            Dict with keys 'quarter', 'half', 'full' containing logits at native 
            resolutions (1/4, 1/2, full).
        """
        return {
            'quarter': self.head_quarter(decoder_outputs[self.idx_quarter]),
            'half': self.head_half(decoder_outputs[self.idx_half]),
            'full': self.head_full(decoder_outputs[self.idx_full])
        }