"""
evaluation/losses.py
Modular Loss Engine. Supports Deep Supervision, Dynamic Weighting, and Agnostic Outputs.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Union, Tuple

import torch
import torch.nn as nn
from monai.losses import (
    DiceLoss, GeneralizedDiceLoss, DiceCELoss, DiceFocalLoss, FocalLoss, TverskyLoss
)

try:
    from monai.losses import BoundaryLoss
except ImportError:
    BoundaryLoss = None

logger = logging.getLogger(__name__)

class LossRegistry:
    _registry: Dict[str, type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(wrapped_class: type[nn.Module]):
            cls._registry[name.lower()] = wrapped_class
            return wrapped_class
        return wrapper

    @classmethod
    def get(cls, name: str) -> type[nn.Module]:
        if name.lower() not in cls._registry:
            raise KeyError(f"Loss {name} not found.")
        return cls._registry[name.lower()]

# --- Base Implementations ---
@LossRegistry.register("dice")
class MonaiDice(DiceLoss): pass

@LossRegistry.register("generalized_dice")
class MonaiGenDice(GeneralizedDiceLoss): pass

@LossRegistry.register("dice_ce")
class MonaiDiceCE(DiceCELoss): pass

@LossRegistry.register("dice_focal")
class MonaiDiceFocal(DiceFocalLoss): pass

@LossRegistry.register("focal")
class MonaiFocal(FocalLoss): pass

@LossRegistry.register("tversky")
class MonaiTversky(TverskyLoss): pass

@LossRegistry.register("bce")
class BCEWithLogitsWrap(nn.BCEWithLogitsLoss): pass

@LossRegistry.register("ce")
class CrossEntropyWrap(nn.CrossEntropyLoss): pass

@LossRegistry.register("boundary")
class SafeBoundaryLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        if BoundaryLoss is not None:
            self.loss = BoundaryLoss(**kwargs)
        else:
            logger.warning("BoundaryLoss missing. Falling back to Surface approximations.")
            self.loss = DiceLoss(sigmoid=True)
            
    def forward(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.loss(p, t)

# --- Dynamic Hybrid Loss Engine ---
class DynamicHybridLoss(nn.Module):
    def __init__(self, loss_configs: List[Dict[str, Any]], strategy: str = "static"):
        super().__init__()
        self.strategy = strategy.lower()
        self.losses = nn.ModuleList([
            LossRegistry.get(cfg["name"])(**cfg.get("params", {})) for cfg in loss_configs
        ])
        self.names = [cfg["name"] for cfg in loss_configs]
        
        initial_weights = [cfg.get("weight", 1.0) for cfg in loss_configs]
        
        if self.strategy == "uncertainty":
            self.log_vars = nn.Parameter(torch.zeros(len(self.losses)))
        elif self.strategy == "learnable":
            self.weights = nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32))
        else:
            self.register_buffer("weights", torch.tensor(initial_weights, dtype=torch.float32))

    def _compute_weighted(self, loss_vals: List[torch.Tensor]) -> torch.Tensor:
        if self.strategy == "uncertainty":
            return sum(0.5 * torch.exp(-self.log_vars[i]) * loss_vals[i] + 0.5 * self.log_vars[i] for i in range(len(self.losses)))
        else:
            w = torch.softmax(self.weights, dim=0) if self.strategy == "learnable" else (self.weights / self.weights.sum())
            return sum(w[i] * loss_vals[i] for i in range(len(self.losses)))

    def forward(self, preds: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        if isinstance(preds, dict): preds = list(preds.values())
        if isinstance(preds, torch.Tensor): preds = [preds]

        # Sort predictions by spatial volume (coarsest to finest) to ensure deterministic ordering
        preds = sorted(preds, key=lambda x: x.shape[-1] * x.shape[-2] * x.shape[-3])

        # Deep Supervision Decay Weights
        # Finest resolution gets the highest weight
        ds_weights = [1.0 / (2 ** i) for i in range(len(preds))]
        ds_weights = ds_weights[::-1]
        ds_total = sum(ds_weights)
        ds_weights = [w / ds_total for w in ds_weights]

        loss_breakdown = {}
        total_loss = 0.0

        for ds_idx, pred in enumerate(preds):
            if pred.shape[2:] != target.shape[2:]:
                pred = torch.nn.functional.interpolate(
                    pred,
                    size=target.shape[2:],
                    mode="trilinear",
                    align_corners=False
                )
            scale_loss_vals = [criterion(pred, target) for criterion in self.losses]
            scale_total = self._compute_weighted(scale_loss_vals)
            
            for i, l_val in enumerate(scale_loss_vals):
                loss_breakdown[f"{self.names[i]}_ds{ds_idx}"] = l_val.detach()
            
            total_loss = total_loss + (scale_total * ds_weights[ds_idx])

        loss_breakdown["total"] = total_loss
        return loss_breakdown

class LossFactory:
    @staticmethod
    def build(config: dict) -> nn.Module:
        return DynamicHybridLoss(config.get("losses", []), config.get("weighting_strategy", "static"))