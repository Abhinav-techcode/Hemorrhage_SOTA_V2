# models/model_factory.py
"""Model factory: builds HybridMedNeXt++ from configuration dictionary."""
from __future__ import annotations

import torch.nn as nn
from typing import Dict, Any
from .hybrid_mednext import HybridMedNeXtPlus
from .hybrid_convnext_umamba import HybridConvNeXtV2_UMamba
from .hybrid_segformer_umamba import HybridSegFormerUMamba
from .umamba import UMambaBot

__all__ = ["build_model"]


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Construct model based on architecture config."""
    params = config.get('params', config)
    arch = config.get('architecture', 'hybrid_mednext_plus').lower()
    
    if arch == 'convnext_v2_umamba':
        return HybridConvNeXtV2_UMamba(
            in_channels=params.get('in_channels', 1),
            num_classes=params.get('out_channels', 1),
            encoder_depths=params.get('encoder_depths', [2, 2, 4, 2, 2]),
            encoder_dims=params.get('encoder_dims', [24, 48, 96, 192, 384]),
            drop_path_rate=params.get('drop_path_rate', 0.1),
            fusion_dim=params.get('fusion_dim', 384),
            fusion_strategy=params.get('fusion_strategy', 'weighted'),
            mamba_d_state=params.get('mamba_d_state', 16),
            mamba_d_conv=params.get('mamba_d_conv', 4),
            mamba_expand=params.get('mamba_expand', 2),
            mamba_blocks=params.get('mamba_blocks', 2),
            decoder_dims=params.get('decoder_dims', [192, 96, 48, 24])
        )
    elif arch == 'hybrid_segformer_umamba':
        return HybridSegFormerUMamba(
            in_channels=params.get('in_channels', 1),
            num_classes=params.get('out_channels', 1),
            embed_dims=params.get('embed_dims', [32, 64, 160, 256]),
            fusion_dim=params.get('fusion_dim', 256),
            decoder_dims=params.get('decoder_dims', [128, 64, 32]),
            d_state=params.get('mamba_d_state', 16)
        )
    elif arch == 'umamba_bot':
        return UMambaBot(
            in_channels=params.get('in_channels', 1),
            num_classes=params.get('out_channels', 1),
            features=params.get('features', [32, 64, 128, 256, 320]),
            d_state=params.get('mamba_d_state', 16),
            deep_supervision=params.get('deep_supervision', True)
        )
    else:
        # Default to existing model
        return HybridMedNeXtPlus(
            in_channels=params.get('in_channels', 1),
            num_classes=params.get('out_channels', 1),
            encoder_depths=params.get('encoder_depths', [2, 2, 4, 2, 2]),
            encoder_dims=params.get('encoder_dims', [24, 48, 96, 192, 384]),
            kernel_sizes=params.get('kernel_sizes', [7, 7, 7, 7, 7]),
            drop_path_rate=params.get('drop_path_rate', 0.1),
            fusion_dim=params.get('fusion_dim', 384),
            bridge_dim=params.get('bridge_dim', 384),
            num_heads=params.get('num_heads', 8),
            sr_ratio=params.get('sr_ratio', 2),
            decoder_dims=params.get('decoder_dims', [192, 96, 48, 24, 24])
        )