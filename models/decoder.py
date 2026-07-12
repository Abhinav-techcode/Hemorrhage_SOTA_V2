# models/decoder.py
"""Attention‑guided decoder with MedNeXt refinement and dual‑attention skip fusion."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import List
from .dual_attention_skip import DualAttentionSkip
from .mednext_block import MedNeXtBlock
from .utils import init_weights_kaiming

__all__ = ["Decoder"]


class DecoderStage(nn.Module):
    """One decoder level: upsample, dual‑attention skip fusion, two MedNeXt blocks.
    
    Args:
        in_dim: Input channels from previous decoder stage.
        skip_dim: Channels of the corresponding encoder skip connection.
        out_dim: Output channels of this stage.
        kernel_size: Kernel size for MedNeXt blocks.
        drop_path: Stochastic depth drop probability for both MedNeXt blocks.
    """
    def __init__(self, in_dim: int, skip_dim: int, out_dim: int,
                 kernel_size: int = 7, drop_path: float = 0.0):
        super().__init__()
        self.skip_att = DualAttentionSkip(skip_dim)
        self.upsample = nn.ConvTranspose3d(in_dim, skip_dim, kernel_size=2, stride=2)
        self.concat_proj = nn.Conv3d(skip_dim * 2, out_dim, 3, padding=1)
        self.block1 = MedNeXtBlock(out_dim, kernel_size=kernel_size,
                                   drop_path=drop_path)
        self.block2 = MedNeXtBlock(out_dim, kernel_size=kernel_size,
                                   drop_path=drop_path)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        skip_enhanced = self.skip_att(skip)
        x_up = self.upsample(x)
        # Ensure exact spatial alignment
        if x_up.shape[2:] != skip_enhanced.shape[2:]:
            x_up = nn.functional.interpolate(x_up, size=skip_enhanced.shape[2:],
                                             mode='trilinear', align_corners=False)
        fused = torch.cat([x_up, skip_enhanced], dim=1)
        fused = self.concat_proj(fused)
        fused = self.block1(fused)
        fused = self.block2(fused)
        return fused


class Decoder(nn.Module):
    """Full decoder from coarsest bridge feature to full resolution.
    
    Args:
        bridge_dim: Channels of bridge input feature.
        encoder_dims: List of encoder channel dims [S0, S1, S2, S3, S4].
        decoder_dims: Output dims for each decoder stage (length 5).
        drop_path_rate: Base drop path rate applied to all decoder blocks.
        kernel_size: Kernel size for MedNeXt blocks.
    """
    def __init__(self, bridge_dim: int, encoder_dims: List[int],
                 decoder_dims: List[int], drop_path_rate: float = 0.0,
                 kernel_size: int = 7):
        super().__init__()
        assert len(encoder_dims) == len(decoder_dims)
        rev_enc_dims = encoder_dims[::-1]  # S4..S0
        self.stages = nn.ModuleList()
        # First stage: bridge (at S4 resolution) + S4 skip
        self.stages.append(DecoderStage(bridge_dim, rev_enc_dims[0], decoder_dims[0],
                                        kernel_size, drop_path=drop_path_rate))
        for i in range(1, len(rev_enc_dims)):
            self.stages.append(DecoderStage(decoder_dims[i-1], rev_enc_dims[i],
                                            decoder_dims[i], kernel_size,
                                            drop_path=drop_path_rate))
        self.apply(init_weights_kaiming)

    def forward(self, bridge_feat: Tensor, encoder_features: List[Tensor]) -> List[Tensor]:
        """Forward.
        
        Args:
            bridge_feat: (B, C_bridge, D_s4, H_s4, W_s4).
            encoder_features: List [S0, S1, S2, S3, S4].
        Returns:
            List of decoder outputs from coarsest to finest.
        """
        rev_skips = encoder_features[::-1]
        x = bridge_feat
        outputs = []
        for stage in self.stages:
            x = stage(x, rev_skips.pop(0))
            outputs.append(x)
        return outputs  # [S4_dec, S3_dec, S2_dec, S1_dec, S0_dec]