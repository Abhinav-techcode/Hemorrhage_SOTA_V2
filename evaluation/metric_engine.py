import torch
from typing import Dict, Any
import logging
from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
    ConfusionMatrixMetric,
    MeanIoU
)
from monai.transforms import Activations, AsDiscrete, Compose
import numpy as np

logger = logging.getLogger(__name__)

class ResearchMetricEngine:
    def __init__(self, device: str = "cpu"):
        self.device = device
        # Overlap Metrics
        self.dice = DiceMetric(include_background=True, reduction="mean")
        self.iou = MeanIoU(include_background=True, reduction="mean")
        # We can implement Tversky and F-beta manually or through confusion matrix.
        
        # Classification (Voxel-wise)
        self.conf_matrix = ConfusionMatrixMetric(
            include_background=True, 
            metric_name=["accuracy", "precision", "recall", "specificity", "f1_score", "sensitivity"],
            reduction="mean"
        )
        
        # Medical Segmentation
        self.hd95 = HausdorffDistanceMetric(include_background=True, percentile=95.0, reduction="mean")
        self.asd = SurfaceDistanceMetric(include_background=True, reduction="mean")
        
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
        
        # Phase 2 manual stats accumulators
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.v_pred_list = []
        self.v_gt_list = []
        self.ece_confidences = []
        self.ece_accuracies = []
        
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
        
        from evaluation.validator import ResearchValidator
        ResearchValidator.validate_probabilities(probs)
        
        # Post-process (Binarize)
        y_pred_bin = [self.post_pred(p) for p in y_pred]
        y_bin = [self.post_label(l) for l in y]
        
        ResearchValidator.validate_predictions(torch.stack(y_pred_bin))
        
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
                pred_voxels = pred.sum().item()
                
                if fg_voxels > 0:
                    self.positive_volumes += 1
                    self.lesion_volumes.append(fg_voxels)
                else:
                    self.empty_masks += 1
                    
                total_voxels = mask.numel()
                self.foreground_ratios.append(fg_voxels / total_voxels if total_voxels > 0 else 0)
                
                # Base counts for derived metrics
                tp = (pred * mask).sum().item()
                fp = (pred * (1 - mask)).sum().item()
                fn = ((1 - pred) * mask).sum().item()
                tn = ((1 - pred) * (1 - mask)).sum().item()
                
                self.tp += tp
                self.fp += fp
                self.fn += fn
                self.tn += tn
                self.v_pred_list.append(pred_voxels)
                self.v_gt_list.append(fg_voxels)
                
                # Expected Calibration Error (ECE) components (10 bins)
                bin_idx = min(int(prob.mean().item() * 10), 9)
                batch_acc = 1.0 if (tp + tn) / total_voxels > 0.5 else 0.0 # voxel-wise proxy
                self.ece_confidences.append((bin_idx, prob.mean().item()))
                self.ece_accuracies.append((bin_idx, batch_acc))
                
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
            
        # Custom derived classification metrics (Phase 2)
        alpha, beta = 0.3, 0.7
        metrics["val_tversky"] = self.tp / (self.tp + alpha * self.fp + beta * self.fn) if (self.tp + self.fp + self.fn) > 0 else 0.0
        
        vs_list = []
        rvd_list = []
        for vp, vg in zip(self.v_pred_list, self.v_gt_list):
            if vp + vg > 0:
                vs_list.append(1 - abs(vp - vg) / (vp + vg))
            if vg > 0:
                rvd_list.append((vp - vg) / vg * 100.0)
                
        metrics["val_volumetric_similarity"] = np.mean(vs_list) if vs_list else 0.0
        metrics["val_relative_volume_difference"] = np.mean(rvd_list) if rvd_list else 0.0
        
        # Calculate ECE
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
        metrics["val_expected_calibration_error"] = ece
            
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
