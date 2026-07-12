import torch
from typing import Dict, Any
import logging
from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
    ConfusionMatrixMetric
)
from monai.transforms import Activations, AsDiscrete, Compose
import numpy as np

logger = logging.getLogger(__name__)

class ResearchMetricEngine:
    def __init__(self, device: str = "cpu"):
        self.device = device
        # Overlap Metrics
        self.dice = DiceMetric(include_background=False, reduction="mean")
        self.iou = DiceMetric(include_background=False, reduction="mean", jaccard=True)
        # We can implement Tversky and F-beta manually or through confusion matrix.
        
        # Classification (Voxel-wise)
        self.conf_matrix = ConfusionMatrixMetric(
            include_background=False, 
            metric_name=["accuracy", "precision", "recall", "specificity", "f1_score", "sensitivity"],
            reduction="mean"
        )
        
        # Medical Segmentation
        self.hd95 = HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean")
        self.asd = SurfaceDistanceMetric(include_background=False, reduction="mean")
        
        # Post-processing
        self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.post_label = AsDiscrete(threshold=0.5)

        # Trackers
        self.val_losses = {"total": [], "dice": [], "focal": [], "boundary": []}
        self.train_losses = {"total": [], "dice": [], "focal": [], "boundary": []}
        
        self.reset()
        
    def reset(self):
        self.dice.reset()
        self.iou.reset()
        self.conf_matrix.reset()
        self.hd95.reset()
        self.asd.reset()
        
        # Custom accumulators
        self.prediction_confidences = []
        self.entropies = []
        self.positive_volumes = 0
        self.empty_masks = 0
        self.lesion_volumes = []
        self.foreground_ratios = []
        
    def _extract_highest_res(self, preds: Any) -> torch.Tensor:
        if isinstance(preds, dict):
            return preds.get("full", list(preds.values())[-1])
        if isinstance(preds, (list, tuple)):
            return preds[-1]
        return preds
        
    @torch.no_grad()
    def update(self, y_pred: Any, y: torch.Tensor, mode: str = "val"):
        y_pred = self._extract_highest_res(y_pred).detach()
        y = y.detach()
        
        # Probabilities
        probs = torch.sigmoid(y_pred)
        
        # Post-process (Binarize)
        y_pred_bin = [self.post_pred(p) for p in y_pred]
        y_bin = [self.post_label(l) for l in y]
        
        if mode == "val":
            # Update MONAI metrics
            self.dice(y_pred=y_pred_bin, y=y_bin)
            self.iou(y_pred=y_pred_bin, y=y_bin)
            self.conf_matrix(y_pred=y_pred_bin, y=y_bin)
            
            try:
                self.hd95(y_pred=y_pred_bin, y=y_bin)
                self.asd(y_pred=y_pred_bin, y=y_bin)
            except Exception as e:
                logger.debug(f"Distance metric computation skipped for batch: {e}")
                
            # Calibration & Stats
            for b in range(y.shape[0]):
                mask = y_bin[b]
                prob = probs[b]
                
                # Stats
                fg_voxels = mask.sum().item()
                if fg_voxels > 0:
                    self.positive_volumes += 1
                    self.lesion_volumes.append(fg_voxels)
                else:
                    self.empty_masks += 1
                    
                total_voxels = mask.numel()
                self.foreground_ratios.append(fg_voxels / total_voxels if total_voxels > 0 else 0)
                
                # Calibration
                self.prediction_confidences.append(prob.mean().item())
                # Entropy: -p log(p) - (1-p) log(1-p)
                p = torch.clamp(prob, 1e-6, 1 - 1e-6)
                ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
                self.entropies.append(ent.mean().item())

    def update_loss(self, loss_dict: Dict[str, torch.Tensor], mode: str = "train"):
        target = self.train_losses if mode == "train" else self.val_losses
        for k, loss_val in loss_dict.items():
            if k not in target:
                target[k] = []
            target[k].append(loss_val.item())

    def compute(self, mode: str = "val") -> Dict[str, float]:
        metrics = {}
        
        # Aggregate Losses
        target_losses = self.train_losses if mode == "train" else self.val_losses
        for k, v in target_losses.items():
            if v:
                metrics[f"{mode}_loss_{k}"] = np.mean(v)
            target_losses[k] = [] # Reset for next epoch
            
        if mode == "train":
            # For phase 2, train metrics will be calculated if we call update(mode="train")
            # For simplicity, we can rely on standard loss tracking for train.
            return metrics

        # Validation Overlap
        metrics["val_dice"] = self._safe_agg(self.dice)
        metrics["val_iou"] = self._safe_agg(self.iou)
        
        # Validation Classification
        conf_res = self._safe_agg(self.conf_matrix, is_conf=True)
        if conf_res:
            metrics["val_accuracy"] = conf_res[0]
            metrics["val_precision"] = conf_res[1]
            metrics["val_recall"] = conf_res[2]
            metrics["val_specificity"] = conf_res[3]
            metrics["val_f1_score"] = conf_res[4]
            metrics["val_sensitivity"] = conf_res[5]
            
        # Validation Distances
        metrics["val_hd95"] = self._safe_agg(self.hd95)
        metrics["val_asd"] = self._safe_agg(self.asd)
        
        # Calibration
        metrics["val_mean_confidence"] = np.mean(self.prediction_confidences) if self.prediction_confidences else 0.0
        metrics["val_mean_entropy"] = np.mean(self.entropies) if self.entropies else 0.0
        
        # Dataset Stats
        metrics["val_positive_volumes"] = self.positive_volumes
        metrics["val_empty_masks"] = self.empty_masks
        metrics["val_mean_lesion_volume"] = np.mean(self.lesion_volumes) if self.lesion_volumes else 0.0
        metrics["val_foreground_percentage"] = (np.mean(self.foreground_ratios) * 100) if self.foreground_ratios else 0.0
        
        return {k: float(v) if not np.isnan(v) else 0.0 for k, v in metrics.items()}

    def _safe_agg(self, metric_obj, is_conf=False):
        try:
            val = metric_obj.aggregate()
            if is_conf:
                return [v.item() for v in val]
            return val.item() if isinstance(val, torch.Tensor) else val
        except Exception:
            return [] if is_conf else float("nan")
