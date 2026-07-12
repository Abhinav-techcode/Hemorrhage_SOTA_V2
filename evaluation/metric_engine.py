import torch
from typing import Dict, Any, List
import logging
import numpy as np

from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
    ConfusionMatrixMetric,
    MeanIoU
)

logger = logging.getLogger(__name__)

class ResearchMetricEngine:
    """
    Research Metric Engine
    Calculates overlap, distance, and calibration metrics for 3D Segmentation.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device
        
        # Overlap Metrics
        self.dice = DiceMetric(include_background=True, reduction="mean")
        self.iou = MeanIoU(include_background=True, reduction="mean")
        
        # Classification (Voxel-wise)
        self.conf_matrix = ConfusionMatrixMetric(
            include_background=True, 
            metric_name=["accuracy", "precision", "recall", "specificity", "f1_score", "sensitivity"],
            reduction="mean"
        )
        
        # Distance Metrics
        self.hd95 = HausdorffDistanceMetric(include_background=True, percentile=95.0, reduction="mean")
        self.asd = SurfaceDistanceMetric(include_background=True, reduction="mean")
        
        # Trackers
        self.val_losses: Dict[str, List[float]] = {"total": [], "dice": [], "focal": [], "boundary": []}
        self.train_losses: Dict[str, List[float]] = {"total": [], "dice": [], "focal": [], "boundary": []}
        
        self.reset()
        
    def reset(self):
        """Reset all internal states and accumulators."""
        self.dice.reset()
        self.iou.reset()
        self.conf_matrix.reset()
        self.hd95.reset()
        self.asd.reset()
        
        # Custom accumulators
        self.prediction_confidences: List[float] = []
        self.entropies: List[float] = []
        self.positive_volumes = 0
        self.empty_masks = 0
        self.lesion_volumes: List[int] = []
        self.foreground_ratios: List[float] = []
        
        # Manual stats accumulators
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.v_pred_list: List[float] = []
        self.v_gt_list: List[float] = []
        self.ece_confidences: List[tuple] = []
        self.ece_accuracies: List[tuple] = []
        
    def _extract_highest_res(self, preds: Any) -> torch.Tensor:
        """Extract the highest resolution output from Deep Supervision or complex model outputs."""
        if isinstance(preds, dict):
            return preds.get("full", list(preds.values())[-1])
        if isinstance(preds, (list, tuple)):
            return preds[-1]
        return preds
        
    @torch.no_grad()
    def update(self, y_logits: Any, y_true: torch.Tensor, mode: str = "val"):
        """
        Canonical Prediction Pipeline:
        Logits -> Sigmoid (Probabilities) -> Threshold (Binary)
        """
        if mode != "val":
            return
            
        try:
            # 1. Canonical Pipeline
            y_logits = self._extract_highest_res(y_logits).detach()
            y_true = y_true.detach()
            
            # Probability Map
            y_probs = torch.sigmoid(y_logits)
            
            # Binary Prediction (Threshold 0.5)
            y_preds_bin = (y_probs >= 0.5).float()
            
            # Binary Ground Truth
            y_true_bin = (y_true >= 0.5).float()
            
            # Ensure shape is List[Tensor[C, D, H, W]] for MONAI
            y_preds_bin_list = [y_preds_bin[i:i+1] for i in range(y_preds_bin.shape[0])]
            y_true_bin_list = [y_true_bin[i:i+1] for i in range(y_true_bin.shape[0])]
            
            # 2. Update MONAI Metrics (Safe)
            try:
                self.dice(y_pred=y_preds_bin_list, y=y_true_bin_list)
            except Exception as e:
                logger.error(f"Dice metric failed: {e}")
                
            try:
                self.iou(y_pred=y_preds_bin_list, y=y_true_bin_list)
            except Exception as e:
                logger.error(f"IoU metric failed: {e}")
                
            try:
                self.conf_matrix(y_pred=y_preds_bin_list, y=y_true_bin_list)
            except Exception as e:
                logger.error(f"Confusion Matrix failed: {e}")
                
            try:
                self.hd95(y_pred=y_preds_bin_list, y=y_true_bin_list)
            except Exception as e:
                pass # HD95 naturally fails on empty masks
                
            try:
                self.asd(y_pred=y_preds_bin_list, y=y_true_bin_list)
            except Exception as e:
                pass # ASD naturally fails on empty masks
                
            # 3. Update Custom Metrics
            for b in range(y_true.shape[0]):
                mask = y_true_bin[b]
                prob = y_probs[b]
                pred_bin = y_preds_bin[b]
                
                # Voxel sums
                fg_voxels = mask.sum().item()
                pred_voxels = pred_bin.sum().item()
                
                if fg_voxels > 0:
                    self.positive_volumes += 1
                    self.lesion_volumes.append(int(fg_voxels))
                else:
                    self.empty_masks += 1
                    
                total_voxels = mask.numel()
                self.foreground_ratios.append(float(fg_voxels / total_voxels) if total_voxels > 0 else 0.0)
                
                # Base counts
                tp = (pred_bin * mask).sum().item()
                fp = (pred_bin * (1 - mask)).sum().item()
                fn = ((1 - pred_bin) * mask).sum().item()
                tn = ((1 - pred_bin) * (1 - mask)).sum().item()
                
                self.tp += tp
                self.fp += fp
                self.fn += fn
                self.tn += tn
                self.v_pred_list.append(float(pred_voxels))
                self.v_gt_list.append(float(fg_voxels))
                
                # Calibration (ECE & Entropy)
                mean_prob = prob.mean().item()
                bin_idx = min(int(mean_prob * 10), 9)
                batch_acc = 1.0 if (tp + tn) / total_voxels > 0.5 else 0.0
                
                self.ece_confidences.append((bin_idx, mean_prob))
                self.ece_accuracies.append((bin_idx, batch_acc))
                
                self.prediction_confidences.append(mean_prob)
                
                p = torch.clamp(prob, 1e-6, 1 - 1e-6)
                ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
                self.entropies.append(ent.mean().item())
                
        except Exception as e:
            logger.error(f"MetricEngine update() failed: {e}", exc_info=True)

    def update_loss(self, loss_dict: Dict[str, torch.Tensor], mode: str = "train"):
        """Track losses safely."""
        try:
            target = self.train_losses if mode == "train" else self.val_losses
            for k, loss_val in loss_dict.items():
                if k not in target:
                    target[k] = []
                target[k].append(loss_val.item())
        except Exception as e:
            logger.error(f"MetricEngine update_loss() failed: {e}", exc_info=True)

    def compute(self, mode: str = "val") -> Dict[str, float]:
        """Compute all metrics and return a flat dictionary of floats."""
        metrics = {}
        
        try:
            # Aggregate Losses
            target_losses = self.train_losses if mode == "train" else self.val_losses
            for k, v in target_losses.items():
                if v:
                    metrics[f"{mode}_loss_{k}"] = float(np.mean(v))
                target_losses[k] = []
                
            if mode == "train":
                return metrics
                
            # Validation Overlap
            metrics["val_dice"] = self._safe_agg(self.dice)
            metrics["val_iou"] = self._safe_agg(self.iou)
            
            # Validation Classification
            conf_res = self._safe_agg(self.conf_matrix, is_conf=True)
            if isinstance(conf_res, list) and len(conf_res) == 6:
                metrics["val_accuracy"] = conf_res[0]
                metrics["val_precision"] = conf_res[1]
                metrics["val_recall"] = conf_res[2]
                metrics["val_specificity"] = conf_res[3]
                metrics["val_f1_score"] = conf_res[4]
                metrics["val_sensitivity"] = conf_res[5]
                
            # Derived Custom Metrics
            alpha, beta = 0.3, 0.7
            denom = (self.tp + alpha * self.fp + beta * self.fn)
            metrics["val_tversky"] = float(self.tp / denom) if denom > 0 else 0.0
            
            vs_list, rvd_list = [], []
            for vp, vg in zip(self.v_pred_list, self.v_gt_list):
                if vp + vg > 0:
                    vs_list.append(1.0 - abs(vp - vg) / (vp + vg))
                if vg > 0:
                    rvd_list.append((vp - vg) / vg * 100.0)
                    
            metrics["val_volumetric_similarity"] = float(np.mean(vs_list)) if vs_list else 0.0
            metrics["val_relative_volume_difference"] = float(np.mean(rvd_list)) if rvd_list else 0.0
            
            # ECE
            ece = 0.0
            if self.ece_confidences:
                bins = {i: {"conf": [], "acc": []} for i in range(10)}
                for (b_i, conf), (_, acc) in zip(self.ece_confidences, self.ece_accuracies):
                    bins[b_i]["conf"].append(conf)
                    bins[b_i]["acc"].append(acc)
                n_total = len(self.ece_confidences)
                for i in range(10):
                    if bins[i]["conf"]:
                        bin_conf = np.mean(bins[i]["conf"])
                        bin_acc = np.mean(bins[i]["acc"])
                        bin_weight = len(bins[i]["conf"]) / n_total
                        ece += bin_weight * abs(bin_acc - bin_conf)
            metrics["val_expected_calibration_error"] = float(ece)
            
            # Distances
            metrics["val_hd95"] = self._safe_agg(self.hd95)
            metrics["val_asd"] = self._safe_agg(self.asd)
            
            # Calibration & Stats
            metrics["val_mean_confidence"] = float(np.mean(self.prediction_confidences)) if self.prediction_confidences else 0.0
            metrics["val_mean_entropy"] = float(np.mean(self.entropies)) if self.entropies else 0.0
            metrics["val_positive_volumes"] = float(self.positive_volumes)
            metrics["val_empty_masks"] = float(self.empty_masks)
            metrics["val_mean_lesion_volume"] = float(np.mean(self.lesion_volumes)) if self.lesion_volumes else 0.0
            metrics["val_foreground_percentage"] = float(np.mean(self.foreground_ratios) * 100.0) if self.foreground_ratios else 0.0
            
            # Final Safety Catch: Remove NaNs
            metrics = {k: (v if not np.isnan(v) else 0.0) for k, v in metrics.items()}
            
        except Exception as e:
            logger.error(f"MetricEngine compute() failed: {e}", exc_info=True)
            
        return metrics

    def _safe_agg(self, metric_obj, is_conf=False) -> Any:
        try:
            val = metric_obj.aggregate()
            if is_conf:
                return [float(v.item()) for v in val]
            return float(val.item()) if isinstance(val, torch.Tensor) else float(val)
        except Exception:
            return [] if is_conf else 0.0
