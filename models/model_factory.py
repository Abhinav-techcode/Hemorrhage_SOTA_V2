# models/model_factory.py
"""Model factory: builds HybridMedNeXt++ from configuration dictionary."""
from __future__ import annotations

import torch.nn as nn
from typing import Dict, Any
from .hybrid_mednext import HybridMedNeXtPlus

__all__ = ["build_model"]


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Construct HybridMedNeXt++.
    
    Config keys (with defaults):
        in_channels (int): 1
        num_classes (int): 1
        encoder_depths (list): [2,2,4,2,2]
        encoder_dims (list): [24,48,96,192,384]
        kernel_sizes (list): [7,7,7,7,7]
        drop_path_rate (float): 0.1
        fusion_dim (int): 384
        bridge_dim (int): 384
        num_heads (int): 8
        sr_ratio (int): 2
        decoder_dims (list): [192,96,48,24,24]
    """
    params = config.get('params', config)
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