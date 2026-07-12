"""training/validator.py"""
import torch
import math
from typing import Any

class Validator:
    @staticmethod
    def validate_batch(images: torch.Tensor, masks: torch.Tensor) -> None:
        if images.shape[2:] != masks.shape[2:]:
            raise ValueError(f"Spatial shape mismatch: Images {images.shape}, Masks {masks.shape}")
        if not torch.isfinite(images).all():
            raise ValueError("Batch images contain NaN or Inf values.")
            
    @staticmethod
    def validate_loss(loss: torch.Tensor) -> None:
        if not math.isfinite(loss.item()):
            raise ValueError(f"Loss NaN/Inf detected: {loss.item()}")