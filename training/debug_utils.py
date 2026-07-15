"""
training/debug_utils.py

Provides debugging utilities, output verification, and prediction statistics to
ensure that the main trainer remains clean while still supporting extensive
diagnostics and validation.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    
from torch.utils.data import DataLoader


logger = logging.getLogger("HybridMedNeXt++.DebugUtils")


def verify_loss_inputs(preds: Any, targets: torch.Tensor, task_type: str = "binary", epoch: int = 1, batch_idx: int = 0) -> None:
    """
    Verify shapes, devices, and dtypes of outputs vs masks.
    Raises RuntimeError on mismatch.
    """
    if isinstance(preds, dict):
        if "full" in preds:
            preds = preds["full"]
        else:
            preds = list(preds.values())[-1]
    elif isinstance(preds, (list, tuple)):
        preds = preds[-1]

    if not isinstance(preds, torch.Tensor):
        raise RuntimeError(f"Expected predictions to be a Tensor, got {type(preds)}")

    # 1. Device check
    if preds.device != targets.device:
        raise RuntimeError(
            f"Device mismatch! Predictions on {preds.device} but targets on {targets.device}."
        )

    # 2. Dtype check
    if targets.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32, torch.float64):
        logger.warning(f"Unusual target dtype: {targets.dtype}. Ensure loss function supports it.")

    # 3. Shape check (ignoring channels)
    if preds.shape[2:] != targets.shape[2:]:
        raise RuntimeError(
            f"Spatial shape mismatch! Predictions {preds.shape} vs Targets {targets.shape}"
        )

    # Logging Phase 1 on batch 0
    if batch_idx == 0:
        logger.info("=========================")
        logger.info("PHASE 1 — VERIFY THE MODEL")
        logger.info("=========================")
        logger.info(f"Output Shape  : {preds.shape}")
        logger.info(f"Output Dtype  : {preds.dtype}")
        logger.info(f"Output Device : {preds.device}")
        logger.info(f"Target Shape  : {targets.shape}")
        logger.info(f"Target Dtype  : {targets.dtype}")
        logger.info(f"Target Device : {targets.device}")

    # 4. Target bounds verification
    if torch.isnan(preds).any() or torch.isinf(preds).any():
        raise RuntimeError("Predictions contain NaN or Inf values!")
    
    if task_type == "binary":
        t_min, t_max = targets.min().item(), targets.max().item()
        if t_min < 0 or t_max > 1:
            raise RuntimeError(f"Target values out of bounds for binary segmentation: min={t_min}, max={t_max}")


class PredictionDebugger:
    """
    Handles prediction statistics and histogram generation.
    """
    def __init__(self, output_dir: Path, debug_predictions: bool = False, debug_interval: int = 10):
        self.debug_predictions = debug_predictions
        self.debug_interval = debug_interval
        self.debug_dir = output_dir / "debug"
        if self.debug_predictions:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            
        self.batch_count = 0
        self.epoch_stats = {"mean": 0.0, "std": 0.0, "fg_pct": 0.0, "count": 0}
        self.empty_mask_count = 0
        self.fg_mask_count = 0

    def reset_epoch(self):
        self.batch_count = 0
        self.epoch_stats = {"mean": 0.0, "std": 0.0, "fg_pct": 0.0, "count": 0}
        self.empty_mask_count = 0
        self.fg_mask_count = 0

    def process_batch(self, preds: Any, targets: torch.Tensor, epoch: int, batch_idx: int, loss_components: Dict[str, torch.Tensor] = None):
        """
        Compute prediction statistics for the batch.
        If batch_idx == 0, prints them to the console and generates a histogram.
        """
        if not self.debug_predictions or batch_idx % self.debug_interval != 0:
            return

        self.batch_count += 1
        
        # Resolve prediction tensor
        if isinstance(preds, dict):
            if "full" in preds:
                preds = preds["full"]
            else:
                preds = list(preds.values())[-1]
        elif isinstance(preds, (list, tuple)):
            preds = preds[-1]
            
        preds = preds.detach().float()
        
        # Track Empty masks vs FG masks
        if targets.sum() == 0:
            self.empty_mask_count += targets.shape[0]
        else:
            self.fg_mask_count += targets.shape[0]
        
        probs = torch.sigmoid(preds)
        
        # Statistics
        p_min, p_max, p_mean = preds.min().item(), preds.max().item(), preds.mean().item()
        s_min, s_max, s_mean = probs.min().item(), probs.max().item(), probs.mean().item()
        
        # Median and Entropy
        s_median = probs.median().item()
        
        # 95th percentile
        # torch.quantile requires float32 or float64. probs is already float32.
        p95 = torch.quantile(probs.view(-1), 0.95).item()
        
        # Binary entropy: -p log(p) - (1-p) log(1-p)
        epsilon = 1e-7
        entropy = -(probs * torch.log(probs + epsilon) + (1 - probs) * torch.log(1 - probs + epsilon)).mean().item()
        
        binary_preds = (probs > 0.5).float()
        
        px_05 = binary_preds.sum().item()
        px_03 = (probs > 0.3).sum().item()
        gt_px = targets.sum().item()
        
        # TP, FP, FN, TN
        tp = (binary_preds * targets).sum().item()
        fp = (binary_preds * (1 - targets)).sum().item()
        fn = ((1 - binary_preds) * targets).sum().item()
        tn = ((1 - binary_preds) * (1 - targets)).sum().item()
        
        fg_pct = binary_preds.mean().item() * 100.0
        
        # Accumulate
        self.epoch_stats["mean"] += s_mean
        self.epoch_stats["std"] += probs.std().item()
        self.epoch_stats["fg_pct"] += fg_pct
        self.epoch_stats["count"] += 1

        # Print stats (Phase 3 requirements)
        logger.info("=========================")
        logger.info(f"PHASE 3 — RUNTIME DIAGNOSTICS (Batch {batch_idx})")
        logger.info("=========================")
        logger.info(f"Prediction Mean       : {preds.mean().item():.4f}")
        logger.info(f"Prediction Std        : {preds.std().item():.4f}")
        logger.info(f"Average Probability   : {s_mean:.4f}")
        logger.info(f"Median Probability    : {s_median:.4f}")
        logger.info(f"95th Percentile Prob  : {p95:.4f}")
        logger.info(f"Prediction Entropy    : {entropy:.4f}")
        logger.info(f"Foreground Ratio      : {fg_pct:.4f}%")
        logger.info(f"True Positives (TP)   : {tp}")
        logger.info(f"False Positives (FP)  : {fp}")
        logger.info(f"False Negatives (FN)  : {fn}")
        logger.info(f"True Negatives (TN)   : {tn}")
        logger.info("=========================")


        if loss_components:
            logger.info("=========================")
            logger.info("PHASE 4 — VERIFY LOSS")
            logger.info("=========================")
            for k, v in loss_components.items():
                val = v.item()
                logger.info(f"{k.capitalize()} Loss: {val:.4f}")
                if torch.isnan(v) or torch.isinf(v):
                    raise RuntimeError(f"{k.capitalize()} Loss is NaN or Inf!")

        self._save_histogram(probs, epoch, batch_idx, prefix="pred")
        self._save_histogram(targets, epoch, batch_idx, prefix="gt")
            
    def end_of_epoch_check(self, epoch: int):
        total_samples = self.empty_mask_count + self.fg_mask_count
        if total_samples > 0:
            logger.info(f"[Epoch {epoch}] Empty Masks Encountered : {self.empty_mask_count}")
            logger.info(f"[Epoch {epoch}] Foreground Samples      : {self.fg_mask_count}")
            if self.empty_mask_count / total_samples > 0.5:
                logger.warning("More than 50% of batches contain only empty masks. Consider adjusting sampler.")

    def _save_histogram(self, probs: torch.Tensor, epoch: int, batch: int, prefix: str = "pred"):
        if not MATPLOTLIB_AVAILABLE:
            return
            
        try:
            # Subsample to avoid memory issues (e.g., take 100k random samples)
            flat_probs = probs.float().flatten()
            if flat_probs.numel() > 100000:
                indices = torch.randint(0, flat_probs.numel(), (100000,), device=flat_probs.device)
                flat_probs = flat_probs[indices]
                
            np_probs = flat_probs.cpu().numpy()
            
            plt.figure(figsize=(8, 6))
            plt.hist(np_probs, bins=50, color='blue', alpha=0.7)
            title_prefix = "Prediction Probabilities" if prefix == "pred" else "Ground Truth"
            plt.title(f"{title_prefix} - Epoch {epoch} Batch {batch}")
            plt.xlabel("Probability" if prefix == "pred" else "Value")
            plt.ylabel("Frequency")
            plt.xlim(0.0, 1.0)
            plt.grid(True, alpha=0.3)
            
            save_path = self.debug_dir / f"{prefix}_hist_epoch{epoch:03d}_batch{batch:04d}.png"
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to generate prediction histogram: {e}")

    def get_epoch_summary(self) -> Dict[str, float]:
        """
        Returns average prediction statistics for the epoch.
        """
        count = self.epoch_stats["count"]
        if count == 0:
            return {"Pred Mean": 0.0, "Pred Std": 0.0, "Pred FG %": 0.0}
            
        return {
            "Pred Mean": self.epoch_stats["mean"] / count,
            "Pred Std": self.epoch_stats["std"] / count,
            "Pred FG %": self.epoch_stats["fg_pct"] / count
        }

def verify_dataset(loader: DataLoader):
    """
    Scans the dataset and prints statistics before training.
    """
    logger.info("=" * 60)
    logger.info("Training Dataset Statistics (Verification)")
    logger.info("=" * 60)
    
    total_volumes = 0
    with_fg = 0
    without_fg = 0
    fg_voxels = 0
    bg_voxels = 0
    max_size = 0
    min_size = float('inf')
    total_size = 0

    dataset = loader.dataset
    for i in range(len(dataset)):
        # Try to use raw _load_case or metadata to avoid transforms if possible, 
        # but for simplicity we can inspect the metadata dict directly.
        fname = dataset._case_filenames[i]
        meta = dataset._metadata_dict.get(fname, {})
        vol = meta.get("Shape_X", 0) * meta.get("Shape_Y", 0) * meta.get("Shape_Z", 0)
        fg = meta.get("Positive_Voxels", 0)
        bg = vol - fg
        
        total_volumes += 1
        if fg > 0:
            with_fg += 1
        else:
            without_fg += 1
            
        fg_voxels += fg
        bg_voxels += bg
        max_size = max(max_size, fg)
        if fg > 0:
            min_size = min(min_size, fg)
        total_size += fg

    avg_size = total_size / with_fg if with_fg > 0 else 0
    fg_pct = (fg_voxels / (fg_voxels + bg_voxels) * 100) if (fg_voxels + bg_voxels) > 0 else 0

    logger.info(f"Total Volumes             : {total_volumes}")
    logger.info(f"Volumes With Foreground   : {with_fg}")
    logger.info(f"Volumes Without Foreground: {without_fg}")
    logger.info(f"Foreground Voxels         : {fg_voxels}")
    logger.info(f"Background Voxels         : {bg_voxels}")
    logger.info(f"Foreground %              : {fg_pct:.6f}%")
    logger.info(f"Average Mask Size         : {avg_size:.2f} voxels")
    logger.info(f"Maximum Mask Size         : {max_size} voxels")
    logger.info(f"Minimum Mask Size         : {min_size if min_size != float('inf') else 0} voxels")
    logger.info("=" * 60)

def verify_gradients(model: nn.Module):
    total_norm = 0.0
    largest_layer = ""
    smallest_layer = ""
    max_grad = -float('inf')
    min_grad = float('inf')
    has_nan = False
    has_inf = False
    
    for name, p in model.named_parameters():
        if p.grad is not None:
            norm = p.grad.data.norm(2).item()
            total_norm += norm ** 2
            
            if torch.isnan(p.grad).any():
                has_nan = True
            if torch.isinf(p.grad).any():
                has_inf = True
                
            if norm > max_grad:
                max_grad = norm
                largest_layer = name
            if norm < min_grad and norm > 0:
                min_grad = norm
                smallest_layer = name

    total_norm = total_norm ** 0.5
    
    logger.info("=" * 60)
    logger.info("Gradient Verification")
    logger.info(f"Gradient Norm         : {total_norm:.4f}")
    logger.info(f"Largest Gradient Layer: {largest_layer} ({max_grad:.4f})")
    logger.info(f"Smallest Gradient Layer: {smallest_layer} ({min_grad:.4f})")
    logger.info(f"NaN Gradients         : {has_nan}")
    logger.info(f"Inf Gradients         : {has_inf}")
    logger.info("=" * 60)
