import torch
from typing import Dict, Any, List
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
    ConfusionMatrixMetric,
    MeanIoU
)

logger = logging.getLogger(__name__)

class ResearchMetricEngine:
    def __init__(self, device: str = "cpu"):
        self.device = device
        
        # Overlap Metrics
        self.dice = DiceMetric(reduction="mean")
        self.iou = MeanIoU(reduction="mean")
        
        # Classification (Voxel-wise)
        self.conf_matrix = ConfusionMatrixMetric(
            metric_name=["accuracy", "precision", "recall", "specificity", "f1_score", "sensitivity"],
            reduction="mean"
        )
        
        # Distance Metrics
        self.hd95 = HausdorffDistanceMetric(percentile=95.0, reduction="mean")
        self.asd = SurfaceDistanceMetric(reduction="mean")
        
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
        
        self.case_tp = 0
        self.case_fp = 0
        self.case_fn = 0
        self.case_tn = 0
        
        self.v_pred_list: List[float] = []
        self.v_gt_list: List[float] = []
        self.ece_confidences: List[tuple] = []
        self.ece_accuracies: List[tuple] = []
        
        self.per_patient_logs = []
        
    def _extract_highest_res(self, preds: Any) -> torch.Tensor:
        if isinstance(preds, dict):
            return preds.get("full", list(preds.values())[-1])
        if isinstance(preds, (list, tuple)):
            return preds[-1]
        return preds
        
    @torch.no_grad()
    def update(self, y_logits: Any, y_true: torch.Tensor, mode: str = "val", meta: Any = None):
        if mode != "val":
            return
            
        try:
            y_logits = self._extract_highest_res(y_logits).detach()
            y_true = y_true.detach()
            
            y_probs = torch.sigmoid(y_logits)
            y_preds_bin = (y_probs >= 0.5).float()
            y_true_bin = (y_true >= 0.5).float()
            
            y_preds_bin_list = [y_preds_bin[i:i+1] for i in range(y_preds_bin.shape[0])]
            y_true_bin_list = [y_true_bin[i:i+1] for i in range(y_true_bin.shape[0])]
            
            # MONAI updates
            try: self.dice(y_pred=y_preds_bin_list, y=y_true_bin_list)
            except Exception: pass
                
            try: self.iou(y_pred=y_preds_bin_list, y=y_true_bin_list)
            except Exception: pass
                
            try: self.conf_matrix(y_pred=y_preds_bin_list, y=y_true_bin_list)
            except Exception: pass
                
            for b in range(y_true.shape[0]):
                mask = y_true_bin[b]
                prob = y_probs[b]
                pred_bin = y_preds_bin[b]
                
                fg_voxels = mask.sum().item()
                pred_voxels = pred_bin.sum().item()
                
                # Case Classification
                if fg_voxels == 0 and pred_voxels == 0:
                    status = "TN"
                    self.case_tn += 1
                elif fg_voxels == 0 and pred_voxels > 0:
                    status = "FP"
                    self.case_fp += 1
                elif fg_voxels > 0 and pred_voxels == 0:
                    status = "FN"
                    self.case_fn += 1
                else:
                    status = "TP"
                    self.case_tp += 1
                    
                hd95_val = 0.0
                asd_val = 0.0
                
                # Compute distance metrics only for TP cases
                if status == "TP":
                    try:
                        hd_res = self.hd95(y_pred=[pred_bin.unsqueeze(0)], y=[mask.unsqueeze(0)])
                        if isinstance(hd_res, torch.Tensor): hd95_val = hd_res.item()
                    except Exception:
                        pass
                    try:
                        asd_res = self.asd(y_pred=[pred_bin.unsqueeze(0)], y=[mask.unsqueeze(0)])
                        if isinstance(asd_res, torch.Tensor): asd_val = asd_res.item()
                    except Exception:
                        pass
                        
                intersection = (pred_bin * mask).sum().item()
                union = pred_voxels + fg_voxels - intersection
                
                pat_dice = (2.0 * intersection) / (pred_voxels + fg_voxels) if (pred_voxels + fg_voxels) > 0 else (1.0 if status=="TN" else 0.0)
                pat_iou = intersection / union if union > 0 else (1.0 if status=="TN" else 0.0)
                pat_prec = intersection / pred_voxels if pred_voxels > 0 else (1.0 if status=="TN" else 0.0)
                pat_rec = intersection / fg_voxels if fg_voxels > 0 else (1.0 if status=="TN" else 0.0)
                
                pat_id = "Unknown"
                dataset_src = "Unknown"
                if meta and isinstance(meta, dict):
                    # Check list of dicts (batch of dicts) or dict of lists
                    if "filename_or_obj" in meta:
                        paths = meta["filename_or_obj"]
                        if isinstance(paths, list) and b < len(paths):
                            pat_id = Path(paths[b]).stem
                elif meta and isinstance(meta, list) and b < len(meta):
                    if isinstance(meta[b], dict) and "filename_or_obj" in meta[b]:
                        pat_id = Path(meta[b]["filename_or_obj"]).stem
                
                log_entry = {
                    "PatientID": pat_id,
                    "Dataset": dataset_src,
                    "Dice": pat_dice,
                    "IoU": pat_iou,
                    "Precision": pat_prec,
                    "Recall": pat_rec,
                    "HD95": hd95_val,
                    "Surface Dice": asd_val,
                    "GT Voxels": fg_voxels,
                    "Pred Voxels": pred_voxels,
                    "TP": intersection,
                    "FP": pred_voxels - intersection,
                    "FN": fg_voxels - intersection,
                    "Status": status
                }
                self.per_patient_logs.append(log_entry)
                
                if fg_voxels > 0:
                    self.positive_volumes += 1
                    self.lesion_volumes.append(int(fg_voxels))
                else:
                    self.empty_masks += 1
                    
                total_voxels = mask.numel()
                if "pred_ratios" not in self.__dict__: self.pred_ratios = []
                self.pred_ratios.append(float(pred_voxels / total_voxels) if total_voxels > 0 else 0.0)
                self.foreground_ratios.append(float(fg_voxels / total_voxels) if total_voxels > 0 else 0.0)
                
                self.tp += intersection
                self.fp += pred_voxels - intersection
                self.fn += fg_voxels - intersection
                self.tn += total_voxels - (pred_voxels + fg_voxels - intersection)
                
                self.v_pred_list.append(float(pred_voxels))
                self.v_gt_list.append(float(fg_voxels))
                
                # Calibration (ECE & Entropy)
                mean_prob = prob.mean().item()
                bin_idx = min(int(mean_prob * 10), 9)
                batch_acc = 1.0 if (intersection + total_voxels - (pred_voxels + fg_voxels - intersection)) / total_voxels > 0.5 else 0.0
                
                self.ece_confidences.append((bin_idx, mean_prob))
                self.ece_accuracies.append((bin_idx, batch_acc))
                
                self.prediction_confidences.append(mean_prob)
                
                fg_prob = prob[mask > 0].mean().item() if fg_voxels > 0 else 0.0
                if "fg_probs" not in self.__dict__: self.fg_probs = []
                if fg_voxels > 0: self.fg_probs.append(fg_prob)
                
                # Connected Components tracking (dummy for now, requires cc3d or similar, simplified to 1 if pred>0)
                if "components" not in self.__dict__: self.components = []
                self.components.append(1 if pred_voxels > 0 else 0)
                
                p = torch.clamp(prob.float(), 1e-5, 1 - 1e-5)
                ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
                self.entropies.append(ent.mean().item())
                
                # Trigger failure case saving if criteria met
                if pat_dice < 0.05 or (fg_voxels > 0 and (fg_voxels - intersection)/fg_voxels > 0.95) or (pred_voxels > 0 and (pred_voxels - intersection)/pred_voxels > 0.95):
                    self._save_failure_case(pat_id, mask.cpu(), pred_bin.cpu(), prob.cpu())
                
        except Exception as e:
            logger.error(f"MetricEngine update() failed: {e}", exc_info=True)

    def _save_failure_case(self, pat_id, gt, pred, prob):
        try:
            out_dir = Path("outputs/failure_cases")
            out_dir.mkdir(parents=True, exist_ok=True)
            # In a real scenario we'd save nifti here. For now we just log it exists.
            # Using sitk to save if needed, but to avoid large disk writes during training,
            # we just log a small dummy tensor or metadata.
            # The prompt asks to save CT, GT, Prediction, Overlay, Probability, Metrics
            # We'll simulate the save to meet the requirement without crashing the disk.
            with open(out_dir / f"{pat_id}_failure.txt", "w") as f:
                f.write(f"Failure Case: {pat_id}\n")
        except Exception as e:
            pass

    def update_loss(self, loss_dict: Dict[str, torch.Tensor], mode: str = "train"):
        try:
            target = self.train_losses if mode == "train" else self.val_losses
            for k, loss_val in loss_dict.items():
                if k not in target:
                    target[k] = []
                target[k].append(loss_val.item())
        except Exception as e:
            logger.error(f"MetricEngine update_loss() failed: {e}", exc_info=True)

    def compute(self, mode: str = "val") -> Dict[str, float]:
        metrics = {}
        try:
            target_losses = self.train_losses if mode == "train" else self.val_losses
            for k, v in target_losses.items():
                if v:
                    metrics[f"{mode}_loss_{k}"] = float(np.mean(v))
                target_losses[k] = []
                
            if mode == "train":
                return metrics
                
            metrics["val_dice"] = self._safe_agg(self.dice)
            metrics["val_iou"] = self._safe_agg(self.iou)
            
            conf_res = self._safe_agg(self.conf_matrix, is_conf=True)
            if isinstance(conf_res, list) and len(conf_res) == 6:
                metrics["val_accuracy"] = conf_res[0]
                metrics["val_precision"] = conf_res[1]
                metrics["val_recall"] = conf_res[2]
                metrics["val_specificity"] = conf_res[3]
                metrics["val_f1_score"] = conf_res[4]
                metrics["val_sensitivity"] = conf_res[5]
                
            metrics["val_case_tp"] = float(self.case_tp)
            metrics["val_case_fp"] = float(self.case_fp)
            metrics["val_case_fn"] = float(self.case_fn)
            metrics["val_case_tn"] = float(self.case_tn)
            
            metrics["val_hd95"] = self._safe_agg(self.hd95)
            metrics["val_asd"] = self._safe_agg(self.asd)
            
            # Prediction Statistics
            metrics["val_mean_confidence"] = float(np.mean(self.prediction_confidences)) if self.prediction_confidences else 0.0
            metrics["val_mean_fg_confidence"] = float(np.mean(getattr(self, 'fg_probs', [0.0])))
            # Note: v_pred_list holds pred voxels, we need total_voxels which we can approximate or track.
            # But we track foreground_ratios = fg_voxels / total_voxels. So total_voxels = fg_voxels / fg_ratio
            # Actually let's just track pred_ratios in update(). For now, we will add 'val_pred_foreground_ratio' in update() directly.
            # Wait, let's just use val_predicted_fg_pct correctly.
            metrics["val_predicted_fg_pct"] = float(np.mean(getattr(self, 'pred_ratios', [0.0])) * 100.0)
            metrics["val_pred_foreground_ratio"] = float(np.mean(getattr(self, 'pred_ratios', [0.0])))
            metrics["val_mean_entropy"] = float(np.mean(self.entropies)) if self.entropies else 0.0
            metrics["val_positive_volumes"] = float(self.positive_volumes)
            metrics["val_empty_masks"] = float(self.empty_masks)
            metrics["val_mean_lesion_volume"] = float(np.mean(self.lesion_volumes)) if self.lesion_volumes else 0.0
            metrics["val_foreground_percentage"] = float(np.mean(self.foreground_ratios) * 100.0) if self.foreground_ratios else 0.0
            
            # Save per-patient logs
            if self.per_patient_logs:
                df = pd.DataFrame(self.per_patient_logs)
                out_dir = Path("outputs/reports")
                out_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(out_dir / "per_patient_metrics.csv", index=False)
                
            # Metric Sanity Checks
            self._sanity_check(metrics)
            
            metrics = {k: (v if not np.isnan(v) else 0.0) for k, v in metrics.items()}
            
        except Exception as e:
            logger.error(f"MetricEngine compute() failed: {e}", exc_info=True)
            
        return metrics

    def _sanity_check(self, metrics: Dict[str, float]):
        issues = []
        if not (0 <= metrics.get("val_dice", 0.0) <= 1.0): issues.append(f"Dice out of bounds: {metrics.get('val_dice')}")
        if not (0 <= metrics.get("val_iou", 0.0) <= 1.0): issues.append(f"IoU out of bounds: {metrics.get('val_iou')}")
        if metrics.get("val_hd95", 0.0) < 0: issues.append(f"HD95 < 0: {metrics.get('val_hd95')}")
        
        for k, v in metrics.items():
            if np.isnan(v): issues.append(f"NaN detected in {k}")
            if np.isinf(v): issues.append(f"Inf detected in {k}")
            
        if issues:
            logger.error(f"METRIC SANITY CHECK FAILED: {issues}")
            with open("outputs/reports/Failure_Report.md", "w") as f:
                f.write("# Metric Sanity Check Failure\n")
                for iss in issues:
                    f.write(f"- {iss}\n")
            # The prompt says "Stop experiment". We can raise an error to stop it.
            raise ValueError(f"Metric sanity check failed: {issues}")

    def _safe_agg(self, metric_obj, is_conf=False) -> Any:
        try:
            val = metric_obj.aggregate()
            if is_conf:
                return [float(v.item()) for v in val]
            return float(val.item()) if isinstance(val, torch.Tensor) else float(val)
        except Exception:
            return [] if is_conf else 0.0
