# models/hybrid_convnext_umamba.py
"""HybridConvNeXtV2_UMamba: novel architecture for brain hemorrhage segmentation."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List

from .convnext_v2_encoder import ConvNeXtV2Encoder
from .umamba_bridge import UMambaBridge
from .boundary_decoder import BoundaryDecoder
from .deep_supervision import DeepSupervisionHeads
from .utils import init_weights_kaiming


class ConfigurableCrossScaleFusion(nn.Module):
    """
    Fuses hierarchical features from the encoder before passing them to the decoder.
    Supports 'concat', 'attention', or 'weighted' fusion strategies.
    """
    def __init__(self, in_dims: List[int], out_dim: int, strategy: str = "weighted"):
        super().__init__()
        self.strategy = strategy.lower()
        self.out_dim = out_dim
        
        self.proj = nn.ModuleList([
            nn.Conv3d(d, out_dim, 1) if d != out_dim else nn.Identity()
            for d in in_dims
        ])
        
        if self.strategy == "weighted":
            # Simple learnable weights for bottom-up / top-down
            self.w_td = nn.Parameter(torch.ones(len(in_dims) - 1, 2))
            self.w_bu = nn.Parameter(torch.ones(len(in_dims) - 1, 2))
        elif self.strategy == "concat":
            # Concat and reduce
            self.reduce_td = nn.ModuleList([nn.Conv3d(out_dim * 2, out_dim, 1) for _ in range(len(in_dims) - 1)])
            self.reduce_bu = nn.ModuleList([nn.Conv3d(out_dim * 2, out_dim, 1) for _ in range(len(in_dims) - 1)])
        elif self.strategy == "attention":
            # Spatial attention for fusion
            self.attn_td = nn.ModuleList([
                nn.Sequential(nn.Conv3d(out_dim * 2, 1, 1), nn.Sigmoid()) for _ in range(len(in_dims) - 1)
            ])
            self.attn_bu = nn.ModuleList([
                nn.Sequential(nn.Conv3d(out_dim * 2, 1, 1), nn.Sigmoid()) for _ in range(len(in_dims) - 1)
            ])
            self.reduce_td = nn.ModuleList([nn.Conv3d(out_dim * 2, out_dim, 1) for _ in range(len(in_dims) - 1)])
            self.reduce_bu = nn.ModuleList([nn.Conv3d(out_dim * 2, out_dim, 1) for _ in range(len(in_dims) - 1)])
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")
            
        self.convs_td = nn.ModuleList([nn.Conv3d(out_dim, out_dim, 3, padding=1) for _ in range(len(in_dims) - 1)])
        self.convs_bu = nn.ModuleList([nn.Conv3d(out_dim, out_dim, 3, padding=1) for _ in range(len(in_dims) - 1)])

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        # Project
        P = [p(f) if isinstance(p, nn.Conv3d) else f for p, f in zip(self.proj, features)]
        
        eps = 1e-4
        
        # Top-down (N-1 down to 0)
        td = [P[-1]]
        for i in range(len(P) - 2, -1, -1):
            up = nn.functional.interpolate(td[-1], size=P[i].shape[2:], mode='trilinear', align_corners=False)
            
            if self.strategy == "weighted":
                w = torch.relu(self.w_td[i])
                fused = (w[0] * P[i] + w[1] * up) / (w.sum() + eps)
            elif self.strategy == "concat":
                fused = self.reduce_td[i](torch.cat([P[i], up], dim=1))
            elif self.strategy == "attention":
                cat = torch.cat([P[i], up], dim=1)
                mask = self.attn_td[i](cat)
                fused = self.reduce_td[i](cat) * mask
                
            td.append(self.convs_td[i](fused))
            
        td = td[::-1]
        
        # Bottom-up (0 up to N-1)
        bu = [td[0]]
        for i in range(1, len(P)):
            down = nn.functional.interpolate(bu[-1], size=td[i].shape[2:], mode='trilinear', align_corners=False)
            
            if self.strategy == "weighted":
                w = torch.relu(self.w_bu[i-1])
                fused = (w[0] * td[i] + w[1] * down) / (w.sum() + eps)
            elif self.strategy == "concat":
                fused = self.reduce_bu[i-1](torch.cat([td[i], down], dim=1))
            elif self.strategy == "attention":
                cat = torch.cat([td[i], down], dim=1)
                mask = self.attn_bu[i-1](cat)
                fused = self.reduce_bu[i-1](cat) * mask
                
            bu.append(self.convs_bu[i-1](fused))
            
        return bu


class HybridConvNeXtV2_UMamba(nn.Module):
    """
    HybridConvNeXtV2_UMamba architecture.
    
    Encoder: 3D ConvNeXt V2
    Multi-scale Feature Pyramid: Configurable Cross Scale Fusion
    Bottleneck: UMambaBridge
    Decoder: Attention Guided Decoder with Boundary Refinement
    Deep Supervision
    
    Args:
        in_channels: Input channels (1 for CT).
        num_classes: Output classes (1 for binary).
        encoder_depths: Number of blocks per ConvNeXt stage.
        encoder_dims: Channel dimensions of encoder stages.
        drop_path_rate: Drop path rate for encoder.
        fusion_dim: Dimension for cross-scale fusion.
        fusion_strategy: Strategy for Cross Scale Fusion ('concat', 'attention', 'weighted').
        mamba_d_state: State dimension for UMamba.
        mamba_d_conv: Convolution dimension for UMamba.
        mamba_expand: Expansion factor for UMamba.
        mamba_blocks: Number of UMamba blocks in the bottleneck.
        decoder_dims: Output channels for decoder stages.
    """
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 1,
                 encoder_depths: List[int] = [2, 2, 4, 2, 2],
                 encoder_dims: List[int] = [24, 48, 96, 192, 384],
                 drop_path_rate: float = 0.1,
                 fusion_dim: int = 384,
                 fusion_strategy: str = "weighted",
                 mamba_d_state: int = 16,
                 mamba_d_conv: int = 4,
                 mamba_expand: int = 2,
                 mamba_blocks: int = 2,
                 decoder_dims: List[int] = [192, 96, 48, 24]):
        super().__init__()
        
        # 1. 3D ConvNeXt V2 Encoder
        self.encoder = ConvNeXtV2Encoder(
            in_channels=in_channels,
            depths=encoder_depths,
            dims=encoder_dims,
            drop_path_rate=drop_path_rate
        )
        
        # 2. Multi-scale Feature Pyramid -> Configurable Cross Scale Fusion
        self.cross_scale = ConfigurableCrossScaleFusion(
            in_dims=encoder_dims,
            out_dim=fusion_dim,
            strategy=fusion_strategy
        )
        
        # 3. UMamba Bottleneck
        self.umamba_bridge = UMambaBridge(
            dim=fusion_dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            num_blocks=mamba_blocks
        )
        
        # 4. Attention Guided Decoder
        # The cross_scale fusion outputs [P0, P1, P2, P3, P4], all of dim fusion_dim.
        # Skips to decoder will be [P0, P1, P2, P3], bottleneck is P4.
        # So skip_dims are [fusion_dim, fusion_dim, fusion_dim, fusion_dim].
        self.decoder = BoundaryDecoder(
            bridge_dim=fusion_dim,
            skip_dims=[fusion_dim, fusion_dim, fusion_dim, fusion_dim],
            decoder_dims=decoder_dims
        )
        
        # 5. Deep Supervision Heads
        # The decoder outputs 4 features (since there are 4 decoder blocks mapping S4 -> S0)
        # Deep supervision needs Quarter (S2_res), Half (S1_res), Full (S0_res).
        # In our BoundaryDecoder:
        # idx 0: S3_res
        # idx 1: S2_res (Quarter)
        # idx 2: S1_res (Half)
        # idx 3: S0_res (Full)
        # We manually map this to a dictionary.
        self.head_quarter = nn.Conv3d(decoder_dims[1], num_classes, 1)
        self.head_half = nn.Conv3d(decoder_dims[2], num_classes, 1)
        self.head_full = nn.Conv3d(decoder_dims[3], num_classes, 1)

        self.apply(init_weights_kaiming)

        # Focal Loss Prior Initialization
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob)).item()
        nn.init.constant_(self.head_quarter.bias, bias_value)
        nn.init.constant_(self.head_half.bias, bias_value)
        nn.init.constant_(self.head_full.bias, bias_value)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass.
        
        Args:
            x: (B, 1, D, H, W) CT volume.
        Returns:
            Dict with keys 'quarter', 'half', 'full' – logits at 1/4, 1/2, and full resolution.
        """
        # Encoder
        enc_feats = self.encoder(x)
        
        # Cross scale fusion pyramid
        pyramid = self.cross_scale(enc_feats)
        
        # Extract bottleneck and skips
        p4 = pyramid[-1]
        skips = pyramid[:-1] # [P0, P1, P2, P3]
        
        # UMamba Bottleneck
        bottleneck = self.umamba_bridge(p4)
        
        # Decoder
        decoder_outs = self.decoder(bottleneck, skips)
        
        # Deep supervision
        return {
            'full': self.head_full(decoder_outs[3]),
            'half': self.head_half(decoder_outs[2]),
            'quarter': self.head_quarter(decoder_outs[1])
        }
