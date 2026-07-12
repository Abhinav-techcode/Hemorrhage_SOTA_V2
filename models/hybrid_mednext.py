# models/hybrid_mednext.py
"""HybridMedNeXt++: complete novel architecture for brain hemorrhage segmentation."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List

from .encoder import Encoder
from .cross_scale_fusion import CrossScaleFusion
from .dynamic_router import DynamicFeatureRouter
from .efficient_attention import EfficientTransformerBlock3D
from .large_kernel_attention import LKA3D
from .local_context import LocalContextBlock
from .adaptive_fusion import AdaptiveContextFusion
from .feature_calibration import FeatureCalibration
from .boundary_context import BoundaryContextEnhancement
from .decoder import Decoder
from .boundary_refinement import BoundaryRefinement
from .deep_supervision import DeepSupervisionHeads
from .utils import init_weights_kaiming

__all__ = ["HybridMedNeXtPlus"]


class HybridMedNeXtPlus(nn.Module):
    """HybridMedNeXt++ architecture.
    
    Encoder → Cross Scale Fusion → Dynamic Router → Three Branches → 
    Adaptive Fusion → Feature Calibration → Boundary Enhancement → 
    Decoder → Boundary Refinement → Deep Supervision.
    
    Args:
        in_channels: Input channels (1 for CT).
        num_classes: Output classes (1 for binary).
        encoder_depths, encoder_dims, kernel_sizes, drop_path_rate: encoder params.
        fusion_dim: Channel dimension for cross‑scale fusion outputs.
        bridge_dim: Output dim of the boundary enhancement (input to decoder).
        num_heads: Number of heads in efficient transformer.
        sr_ratio: Spatial reduction ratio for attention.
        decoder_dims: Output channels for each decoder stage (length 5).
    """
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 1,
                 encoder_depths: List[int] = [2, 2, 4, 2, 2],
                 encoder_dims: List[int] = [24, 48, 96, 192, 384],
                 kernel_sizes: List[int] = [7, 7, 7, 7, 7],
                 drop_path_rate: float = 0.1,
                 fusion_dim: int = 384,
                 bridge_dim: int = 384,
                 num_heads: int = 8,
                 sr_ratio: int = 2,
                 decoder_dims: List[int] = [192, 96, 48, 24, 24]):
        super().__init__()
        assert len(encoder_dims) == len(decoder_dims) == 5, "Must have 5 stages"
        self.encoder = Encoder(in_channels, encoder_depths, encoder_dims,
                               kernel_sizes, drop_path_rate)
        self.cross_scale = CrossScaleFusion(encoder_dims, fusion_dim)
        self.router = DynamicFeatureRouter(fusion_dim)
        # Three parallel context branches (operate on P4)
        self.transformer_branch = EfficientTransformerBlock3D(
            dim=fusion_dim, num_heads=num_heads, sr_ratio=sr_ratio,
            drop_path_rate=drop_path_rate)
        self.lka_branch = LKA3D(fusion_dim)
        self.local_branch = LocalContextBlock(fusion_dim)
        # Adaptive context fusion
        self.adaptive_fusion = AdaptiveContextFusion(fusion_dim)
        # Feature calibration
        self.feature_calibration = FeatureCalibration(fusion_dim)
        # Boundary context enhancement and projection to bridge_dim
        self.boundary_enhance = nn.Sequential(
            BoundaryContextEnhancement(fusion_dim),
            nn.Conv3d(fusion_dim, bridge_dim, 1),
            nn.BatchNorm3d(bridge_dim),
            nn.ReLU(inplace=True)
        )
        # Decoder (receives bridge feature at S4 resolution and encoder skips)
        self.decoder = Decoder(bridge_dim, encoder_dims, decoder_dims,
                               drop_path_rate=drop_path_rate)
        # Boundary refinement on final decoder output (full resolution, dim decoder_dims[-1])
        self.boundary_refine = BoundaryRefinement(decoder_dims[-1])
        # Deep supervision heads
        self.deep_supervision = DeepSupervisionHeads(decoder_dims, num_classes)
        self.apply(init_weights_kaiming)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass.
        
        Args:
            x: (B, 1, D, H, W) CT volume.
        Returns:
            Dict with keys 'quarter', 'half', 'full' – logits at 1/4, 1/2, and full 
            resolution.
        """
        enc_feats = self.encoder(x)                     # [S0..S4]
        pyramid = self.cross_scale(enc_feats)           # [P0..P4]
        p4 = pyramid[-1]                                # coarsest feature
        coeffs = self.router(p4)                        # (B, 3, 1, 1, 1)
        # Three parallel branches
        a = self.transformer_branch(p4)
        b = self.lka_branch(p4)
        c = self.local_branch(p4)
        # Fuse branches
        fused = self.adaptive_fusion(a, b, c, coeffs)   # (B, fusion_dim, D/16, ...)
        # Feature calibration
        calibrated = self.feature_calibration(fused)    # (B, fusion_dim, D/16, ...)
        # Boundary enhancement and projection
        bridge = self.boundary_enhance(calibrated)      # (B, bridge_dim, D/16, ...)
        # Decoder
        decoder_outs = self.decoder(bridge, enc_feats)  # list of 5 features
        # Final boundary refinement
        decoder_outs[-1] = self.boundary_refine(decoder_outs[-1])
        # Deep supervision
        return self.deep_supervision(decoder_outs)