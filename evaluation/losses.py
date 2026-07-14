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
    DiceLoss, DiceFocalLoss, FocalLoss, TverskyLoss
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

@LossRegistry.register("dice_focal")
class MonaiDiceFocal(DiceFocalLoss): pass

@LossRegistry.register("focal")
class MonaiFocal(FocalLoss): pass

@LossRegistry.register("tversky")
class MonaiTversky(TverskyLoss): pass

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

# --- Stubs for Future Active Implementations ---
@LossRegistry.register("surface")
class SurfaceLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass
    def forward(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("SurfaceLoss is not yet implemented.")

@LossRegistry.register("topk")
class TopKLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass
    def forward(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TopKLoss is not yet implemented.")

@LossRegistry.register("hausdorff")
class HausdorffLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass
    def forward(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("HausdorffLoss is not yet implemented.")

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

    def forward(self, predictions: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        if isinstance(predictions, torch.Tensor):
            predictions = [predictions]

        total_loss = 0.0
        loss_dict = {}

        # Deep Supervision logic
        # For multiple outputs, typically the first is full resolution, others are lower res
        # Weight them decaying: 1.0, 0.5, 0.25...
        for i, pred in enumerate(predictions):
            # Upsample pred if necessary
            if pred.shape[2:] != targets.shape[2:]:
                pred = nn.functional.interpolate(pred, size=targets.shape[2:], mode='trilinear', align_corners=False)

            layer_loss_vals = [loss_fn(pred, targets) for loss_fn in self.losses]
            layer_total = self._compute_weighted(layer_loss_vals)
            
            # Deep supervision weight (1.0 for first, then halved)
            weight = 1.0 / (2 ** i)
            total_loss += weight * layer_total
            
            if i == 0:
                loss_dict = {name: val for name, val in zip(self.names, layer_loss_vals)}

        loss_dict["total"] = total_loss
        return loss_dict