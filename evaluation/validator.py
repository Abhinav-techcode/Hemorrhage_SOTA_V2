import torch
import math
from typing import Any, Dict

class ResearchValidator:
    """
    Research Validation Module (Phase 4)
    Asserts correctness of inputs, model outputs, probabilities, and final metric inputs.
    """
    @staticmethod
    def _extract_highest_res(preds: Any) -> torch.Tensor:
        if isinstance(preds, dict):
            return preds.get("full", list(preds.values())[-1])
        if isinstance(preds, (list, tuple)):
            return preds[-1]
        return preds

    @staticmethod
    def validate_input(images: torch.Tensor, masks: torch.Tensor) -> None:
        if images.shape[2:] != masks.shape[2:]:
            raise AssertionError(f"Input spatial shape mismatch: Images {images.shape}, Masks {masks.shape}")
        if not torch.isfinite(images).all():
            raise AssertionError("Input images contain NaN or Inf values.")
        
        unique_mask_vals = torch.unique(masks)
        if not all(val in [0, 1] for val in unique_mask_vals):
            raise AssertionError(f"Mask is not binary! Unique values found: {unique_mask_vals}")

    @staticmethod
    def validate_model_output(preds: Any, masks: torch.Tensor) -> None:
        pred_tensor = ResearchValidator._extract_highest_res(preds)
        
        if pred_tensor.shape[2:] != masks.shape[2:]:
            raise AssertionError(f"Output spatial shape mismatch: Pred {pred_tensor.shape}, Mask {masks.shape}")
        if not torch.isfinite(pred_tensor).all():
            raise AssertionError("Model logits contain NaN or Inf values.")

    @staticmethod
    def validate_probabilities(probs: torch.Tensor) -> None:
        if torch.any(probs < 0.0) or torch.any(probs > 1.0):
            raise AssertionError("Probabilities out of valid range [0, 1].")
        if not torch.isfinite(probs).all():
            raise AssertionError("Probabilities contain NaN or Inf values.")

    @staticmethod
    def validate_predictions(preds_bin: torch.Tensor) -> None:
        unique_vals = torch.unique(preds_bin)
        if not all(val in [0, 1] for val in unique_vals):
            raise AssertionError(f"Predictions are not binary after thresholding! Unique values: {unique_vals}")

    @staticmethod
    def validate_loss(loss: torch.Tensor) -> None:
        if not math.isfinite(loss.item()):
            raise AssertionError(f"Loss NaN/Inf detected: {loss.item()}")
