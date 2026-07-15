import numpy as np
import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from scipy.ndimage import (
        label as _cc_label,
        generate_binary_structure,
        distance_transform_edt,
    )
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

try:
    import cc3d
    _CC3D_OK = True
except ImportError:
    _CC3D_OK = False

try:
    from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance
    _MONAI_SURFACE_OK = True
except ImportError:
    _MONAI_SURFACE_OK = False


class PredictionAnalyzer:
    """
    Research-grade prediction analysis for 3D CT hemorrhage segmentation.

    Usage:
        stats = PredictionAnalyzer.analyze(
            probs, preds_bin, masks,
            spacing=(z_mm, y_mm, x_mm),   # from nib.affine / sitk spacing
            lesion_match_iou=0.1,          # min overlap fraction to call a GT lesion "detected"
        )
    """

    _STRUCT_26 = generate_binary_structure(3, 3) if _SCIPY_OK else None

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    @staticmethod
    def analyze(
        probs: torch.Tensor,
        preds_bin: torch.Tensor,
        masks: torch.Tensor,
        spacing: Optional[Sequence[float]] = None,
        lesion_match_iou: float = 0.1,
        compute_surface_distance: bool = True,
    ) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}

        p_np = probs.detach().float().cpu().numpy()
        p_b = preds_bin.detach().bool()
        m_b = masks.detach().bool()

        stats.update(PredictionAnalyzer._probability_stats(p_np))
        stats.update(PredictionAnalyzer._calibration_stats(p_np, m_b.cpu().numpy().astype(np.float32)))

        confusion = PredictionAnalyzer._confusion_stats(p_b, m_b)
        stats.update(confusion)
        stats.update(PredictionAnalyzer._overlap_metrics(confusion))

        if spacing is not None:
            stats.update(PredictionAnalyzer._volumetric_stats(confusion, spacing))
        else:
            stats["_warning_no_spacing"] = (
                "spacing not provided — volumes reported in voxels only, "
                "not mm3/mL. Pass spacing=(z,y,x) for clinically meaningful volume."
            )

        lesion_stats = PredictionAnalyzer._lesion_level_stats(p_b, m_b, lesion_match_iou)
        stats.update(lesion_stats)

        if compute_surface_distance:
            stats.update(PredictionAnalyzer._surface_distance_stats(p_b, m_b, spacing))

        return stats

    # ------------------------------------------------------------------ #
    # 1. Probability statistics
    # ------------------------------------------------------------------ #
    @staticmethod
    def _probability_stats(p_np: np.ndarray) -> Dict[str, Any]:
        eps = 1e-7
        p_clipped = np.clip(p_np, eps, 1 - eps)
        entropy = -(p_clipped * np.log(p_clipped) + (1 - p_clipped) * np.log(1 - p_clipped))
        return {
            "prob_mean": float(np.mean(p_np)),
            "prob_std": float(np.std(p_np)),
            "prob_min": float(np.min(p_np)),
            "prob_max": float(np.max(p_np)),
            "prob_median": float(np.median(p_np)),
            "prob_95th": float(np.percentile(p_np, 95)),
            "prob_5th": float(np.percentile(p_np, 5)),
            # Fraction of voxels the model is genuinely unsure about (near 0.5).
            # High values here on a well-trained model usually flag ambiguous
            # boundary regions or genuinely low-contrast hemorrhage.
            "prob_uncertain_frac": float(np.mean((p_np > 0.4) & (p_np < 0.6))),
            "mean_entropy": float(np.mean(entropy)),
        }

    # ------------------------------------------------------------------ #
    # 2. Calibration — is p=0.9 actually ~90% correct?
    # ------------------------------------------------------------------ #
    @staticmethod
    def _calibration_stats(p_np: np.ndarray, y_np: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
        p_flat = p_np.ravel()
        y_flat = y_np.ravel()

        brier = float(np.mean((p_flat - y_flat) ** 2))

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_idx = np.digitize(p_flat, bin_edges[1:-1])
        ece = 0.0
        n_total = len(p_flat)
        for b in range(n_bins):
            mask = bin_idx == b
            if not np.any(mask):
                continue
            conf = p_flat[mask].mean()
            acc = y_flat[mask].mean()
            weight = mask.sum() / n_total
            ece += weight * abs(conf - acc)

        return {
            "brier_score": brier,          # lower is better, 0 = perfect
            "ece": float(ece),             # lower is better, 0 = perfectly calibrated
        }

    # ------------------------------------------------------------------ #
    # 3. Confusion matrix (batch-level, vectorized — no python loop)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _confusion_stats(p_b: torch.Tensor, m_b: torch.Tensor) -> Dict[str, Any]:
        tp = (p_b & m_b).sum().item()
        fp = (p_b & ~m_b).sum().item()
        fn = (~p_b & m_b).sum().item()
        tn = (~p_b & ~m_b).sum().item()
        total = tp + fp + fn + tn
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "total_voxels": total}

    @staticmethod
    def _overlap_metrics(c: Dict[str, Any]) -> Dict[str, Any]:
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        return {
            "dice": dice, "iou": iou,
            "precision": precision, "recall": recall, "specificity": specificity,
        }

    # ------------------------------------------------------------------ #
    # 4. Volumetric statistics — physical units, clinically meaningful
    # ------------------------------------------------------------------ #
    @staticmethod
    def _volumetric_stats(c: Dict[str, Any], spacing: Sequence[float]) -> Dict[str, Any]:
        voxel_vol_mm3 = float(np.prod(spacing))  # e.g. z_mm * y_mm * x_mm
        gt_voxels = c["tp"] + c["fn"]
        pred_voxels = c["tp"] + c["fp"]

        gt_vol_mm3 = gt_voxels * voxel_vol_mm3
        pred_vol_mm3 = pred_voxels * voxel_vol_mm3
        gt_vol_ml = gt_vol_mm3 / 1000.0
        pred_vol_ml = pred_vol_mm3 / 1000.0

        rvd = ((pred_voxels - gt_voxels) / gt_voxels * 100) if gt_voxels > 0 else float("nan")

        return {
            "voxel_volume_mm3": voxel_vol_mm3,
            "gt_volume_ml": gt_vol_ml,
            "pred_volume_ml": pred_vol_ml,
            "relative_volume_diff_pct": rvd,  # signed: + = over-estimate, - = under-estimate
            # Flags for standard clinical volume thresholds — adjust cutoff
            # to whatever your cohort/protocol actually uses.
            "gt_exceeds_30ml": bool(gt_vol_ml > 30.0),
            "pred_exceeds_30ml": bool(pred_vol_ml > 30.0),
            "volume_threshold_agreement": bool((gt_vol_ml > 30.0) == (pred_vol_ml > 30.0)),
        }

    # ------------------------------------------------------------------ #
    # 5. Surface distance — HD95, ASSD (per-sample, batch-averaged)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _surface_distance_stats(
        p_b: torch.Tensor, m_b: torch.Tensor, spacing: Optional[Sequence[float]]
    ) -> Dict[str, Any]:
        px = spacing if spacing is not None else (1.0, 1.0, 1.0)

        if _MONAI_SURFACE_OK:
            # MONAI expects (B, C, ...) one-hot-ish binary tensors
            pred_f = p_b.float()
            gt_f = m_b.float()
            if pred_f.ndim == 4:  # (B, D, H, W) -> add channel
                pred_f = pred_f.unsqueeze(1)
                gt_f = gt_f.unsqueeze(1)
            try:
                hd95 = compute_hausdorff_distance(
                    pred_f, gt_f, include_background=False, percentile=95, spacing=px
                )
                assd = compute_average_surface_distance(
                    pred_f, gt_f, include_background=False, spacing=px
                )
                hd95_vals = hd95[torch.isfinite(hd95)]
                assd_vals = assd[torch.isfinite(assd)]
                return {
                    "hd95_mm": float(hd95_vals.mean().item()) if hd95_vals.numel() > 0 else float("nan"),
                    "assd_mm": float(assd_vals.mean().item()) if assd_vals.numel() > 0 else float("nan"),
                    "surface_distance_backend": "monai",
                }
            except Exception:
                pass  # fall through to scipy backend below

        if not _SCIPY_OK:
            return {
                "hd95_mm": float("nan"), "assd_mm": float("nan"),
                "surface_distance_backend": "unavailable (install scipy or monai)",
            }

        # Fallback: per-sample distance-transform-based HD95/ASSD.
        B = p_b.shape[0]
        hd95_list, assd_list = [], []
        for b in range(B):
            pred_vol = p_b[b].squeeze().cpu().numpy()
            gt_vol = m_b[b].squeeze().cpu().numpy()
            if pred_vol.sum() == 0 or gt_vol.sum() == 0:
                continue  # undefined when either surface is empty
            hd95, assd = PredictionAnalyzer._surface_distance_single(pred_vol, gt_vol, px)
            hd95_list.append(hd95)
            assd_list.append(assd)

        return {
            "hd95_mm": float(np.mean(hd95_list)) if hd95_list else float("nan"),
            "assd_mm": float(np.mean(assd_list)) if assd_list else float("nan"),
            "surface_distance_backend": "scipy_edt",
        }

    @staticmethod
    def _surface_distance_single(
        pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float]
    ) -> Tuple[float, float]:
        pred_surf = pred & ~_erode(pred)
        gt_surf = gt & ~_erode(gt)

        dt_pred = distance_transform_edt(~pred_surf, sampling=spacing)
        dt_gt = distance_transform_edt(~gt_surf, sampling=spacing)

        d_gt_to_pred = dt_pred[gt_surf]
        d_pred_to_gt = dt_gt[pred_surf]

        all_d = np.concatenate([d_gt_to_pred, d_pred_to_gt])
        hd95 = float(np.percentile(all_d, 95)) if all_d.size > 0 else float("nan")
        assd = float(all_d.mean()) if all_d.size > 0 else float("nan")
        return hd95, assd

    # ------------------------------------------------------------------ #
    # 6. Lesion-level (connected-component) detection metrics
    # ------------------------------------------------------------------ #
    @staticmethod
    def _lesion_level_stats(
        p_b: torch.Tensor, m_b: torch.Tensor, match_iou: float
    ) -> Dict[str, Any]:
        B = p_b.shape[0]
        fg_percentages, pred_counts, gt_counts = [], [], []
        avg_sizes, largest_sizes, smallest_sizes = [], [], []
        lesion_tp, lesion_fp, lesion_fn = 0, 0, 0

        for b in range(B):
            pred_vol = p_b[b].squeeze().cpu().numpy()
            gt_vol = m_b[b].squeeze().cpu().numpy()

            fg_percentages.append(float(pred_vol.sum()) / pred_vol.size if pred_vol.size > 0 else 0.0)

            pred_labels, n_pred = _connected_components(pred_vol)
            gt_labels, n_gt = _connected_components(gt_vol)
            pred_counts.append(n_pred)
            gt_counts.append(n_gt)

            if n_pred > 0:
                sizes = np.array([(pred_labels == i).sum() for i in range(1, n_pred + 1)])
                avg_sizes.append(float(sizes.mean()))
                largest_sizes.append(float(sizes.max()))
                smallest_sizes.append(float(sizes.min()))
            else:
                avg_sizes.append(0.0); largest_sizes.append(0.0); smallest_sizes.append(0.0)

            # Lesion-wise matching: a GT component is "detected" if enough of
            # its voxels are covered by ANY predicted component (handles the
            # common case of one prediction blob covering two close GT bleeds,
            # or vice versa, without penalizing shape mismatch the way strict
            # 1:1 matching would).
            for gi in range(1, n_gt + 1):
                gt_comp = gt_labels == gi
                overlap = (gt_comp & (pred_labels > 0)).sum()
                iou_frac = overlap / gt_comp.sum() if gt_comp.sum() > 0 else 0.0
                if iou_frac >= match_iou:
                    lesion_tp += 1
                else:
                    lesion_fn += 1

            for pi in range(1, n_pred + 1):
                pred_comp = pred_labels == pi
                overlap = (pred_comp & (gt_labels > 0)).sum()
                iou_frac = overlap / pred_comp.sum() if pred_comp.sum() > 0 else 0.0
                if iou_frac < match_iou:
                    lesion_fp += 1  # spurious detection, no matching GT lesion

        total_gt_lesions = lesion_tp + lesion_fn
        lesion_sensitivity = lesion_tp / total_gt_lesions if total_gt_lesions > 0 else float("nan")
        total_pred_lesions = lesion_tp + lesion_fp
        lesion_precision = lesion_tp / total_pred_lesions if total_pred_lesions > 0 else float("nan")

        return {
            "pred_foreground_percentage": float(np.mean(fg_percentages) * 100),
            "pred_lesion_count": float(np.mean(pred_counts)),
            "gt_lesion_count": float(np.mean(gt_counts)),
            "pred_avg_lesion_size": float(np.mean(avg_sizes)),
            "pred_largest_lesion": float(np.mean(largest_sizes)),
            "pred_smallest_lesion": float(np.mean(smallest_sizes)),
            "lesion_wise_sensitivity": lesion_sensitivity,  # % of true bleeds actually found
            "lesion_wise_precision": lesion_precision,      # % of flagged bleeds that are real
            "lesion_false_positive_count": lesion_fp,        # spurious detections (e.g. calcification mistaken for bleed)
            "lesion_missed_count": lesion_fn,                # entirely missed bleeds — most clinically dangerous failure
            "connected_components_backend": "cc3d" if _CC3D_OK else ("scipy" if _SCIPY_OK else "none"),
        }


# ---------------------------------------------------------------------- #
# Module-level helpers
# ---------------------------------------------------------------------- #
def _connected_components(volume: np.ndarray) -> Tuple[np.ndarray, int]:
    """26-connectivity 3D connected components. Prefers cc3d (much faster on
    large volumes); falls back to scipy.ndimage.label."""
    if volume.sum() == 0:
        return np.zeros_like(volume, dtype=np.int32), 0
    if _CC3D_OK:
        labels = cc3d.connected_components(volume.astype(np.uint8), connectivity=26)
        return labels, int(labels.max())
    if _SCIPY_OK:
        labels, n = _cc_label(volume, structure=PredictionAnalyzer._STRUCT_26)
        return labels, n
    # Last-resort fallback: treat whole foreground as one lesion.
    return (volume > 0).astype(np.int32), 1


def _erode(binary_vol: np.ndarray) -> np.ndarray:
    """One-voxel binary erosion used to extract surface voxels for HD95/ASSD."""
    from scipy.ndimage import binary_erosion
    return binary_erosion(binary_vol, structure=PredictionAnalyzer._STRUCT_26)
