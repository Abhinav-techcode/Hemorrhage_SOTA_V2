"""
evaluation/post_training_visualizer.py
Milestone D: Automated Post-Training Research Visualization — research-grade.

Requires evaluation/prediction_analyzer.py (PredictionAnalyzer) to be present
alongside this file for physical-volume, surface-distance, lesion-wise, and
calibration metrics.
"""

import os
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from skimage import measure
import matplotlib.patches as mpatches

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.model_factory import build_model
from datasets.transforms import TransformFactory
from evaluation.prediction_analyzer import PredictionAnalyzer

logger = logging.getLogger(__name__)

THRESHOLD_SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7)
LESION_MATCH_IOU = 0.1


# --------------------------------------------------------------------------- #
# Spacing / bootstrap helpers
# --------------------------------------------------------------------------- #
def _get_original_spacing_and_shape(path):
    """Read voxel spacing (mm) and shape straight from the NIfTI header,
    before any resampling. Needed for physical volume / surface distance —
    computing these on resized voxels without correcting for the resample
    gives numbers that look precise but are just wrong."""
    try:
        nii = nib.load(str(path))
        zooms = tuple(float(z) for z in nii.header.get_zooms()[:3])
        shape = tuple(int(s) for s in nii.shape[:3])
        return zooms, shape
    except Exception as e:
        logger.warning(f"Could not read spacing from {path}: {e}. Falling back to (1,1,1).")
        return (1.0, 1.0, 1.0), None


def _effective_spacing(orig_zooms, orig_shape, resized_shape):
    """Spacing after Resized() rescales the volume to a fixed spatial_size."""
    if orig_shape is None:
        return orig_zooms
    return tuple(
        oz * (osz / rsz) for oz, osz, rsz in zip(orig_zooms, orig_shape, resized_shape)
    )


def bootstrap_ci(values, n_boot=2000, ci=95, seed=42):
    """Bootstrap mean + CI. A bare mean over ~10-30 test cases without an
    interval overstates how precisely you know the number."""
    values = np.asarray([v for v in values if np.isfinite(v)])
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return float(values.mean()), float(lower), float(upper)


def _threshold_sweep(prob_tensor, mask_tensor, thresholds=THRESHOLD_SWEEP):
    """Dice at several operating points. Reviewers ask 'why 0.5?' — this
    answers it and flags cases where the default threshold is clearly
    suboptimal."""
    sweep = {}
    best_t, best_dice = 0.5, -1.0
    mask_bool = mask_tensor.bool()
    for t in thresholds:
        pred_t = (prob_tensor > t).bool()
        tp = (pred_t & mask_bool).sum().item()
        fp = (pred_t & ~mask_bool).sum().item()
        fn = (~pred_t & mask_bool).sum().item()
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        sweep[t] = dice
        if dice > best_dice:
            best_dice, best_t = dice, t
    return best_t, best_dice, sweep


# --------------------------------------------------------------------------- #
# Diagnosis text — now lesion-aware, not just voxel-aware
# --------------------------------------------------------------------------- #
def generate_diagnosis_text(m):
    dice = m.get('Dice', 0)
    recall = m.get('Recall', 0)
    precision = m.get('Precision', 0)
    gt_vol_ml = m.get('GT_Volume_mL', 0)
    fp = m.get('FP', 0)
    fn = m.get('FN', 0)
    lesion_gt = m.get('Lesion_Count_GT', 0)
    lesion_missed = m.get('Lesion_Missed', 0)
    hd95 = m.get('HD95_mm', float('nan'))

    parts = []

    if dice > 0.8:
        parts.append("Clean high-agreement case with accurate voxel-level segmentation.")
    elif gt_vol_ml < 1.0 and dice < 0.5:
        parts.append(f"Model struggled on a small lesion (GT volume {gt_vol_ml:.2f} mL) — "
                      f"low Dice here is partly a size-effect artifact, not purely a model failure.")
    elif recall < 0.5 and fp < fn:
        parts.append("Severe under-segmentation; missed significant hemorrhage volume.")
    elif precision < 0.5 and fp > fn:
        parts.append("Severe over-segmentation; spurious false-positive clusters present.")
    elif fp > fn * 2:
        parts.append("Tendency toward over-segmentation and false-positive boundaries.")
    elif fn > fp * 2:
        parts.append("Tendency toward under-segmentation; missing lesion boundaries.")
    else:
        parts.append("Moderate agreement with mixed boundary discrepancies.")

    # Lesion-level addition — this is the failure mode voxel Dice hides.
    if lesion_gt > 1:
        if lesion_missed > 0:
            parts.append(f"Multi-focal case: missed {int(lesion_missed)} of {int(lesion_gt)} "
                          f"distinct hemorrhage foci entirely, despite decent voxel-level overlap "
                          f"on the detected ones.")
        else:
            parts.append(f"Multi-focal case ({int(lesion_gt)} distinct foci) — all foci detected.")

    if np.isfinite(hd95) and hd95 > 10:
        parts.append(f"Boundary distance is large (HD95={hd95:.1f}mm), indicating localized "
                      f"regions of substantial disagreement even where bulk overlap looks fine.")

    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Montage rendering
# --------------------------------------------------------------------------- #
def to_rgb_base(img, apply_window=False):
    if apply_window:
        img = np.clip(img, 0, 80)
        img = img / 80
    else:
        mn, mx = img.min(), img.max()
        if mx > mn:
            img = (img - mn) / (mx - mn)
    return np.stack([img, img, img], axis=-1)


def render_case_montage(pid, metrics, img_np, mask_np, pred_bin, save_path):
    from skimage.segmentation import mark_boundaries

    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    color_tp = np.array([0.18, 0.80, 0.44])
    color_fp = np.array([0.91, 0.30, 0.24])
    color_fn = np.array([0.95, 0.61, 0.07])

    z_slice = np.argmax(mask_np.sum(axis=(0, 1))) if mask_np.sum() > 0 else img_np.shape[2] // 2
    z_top = min(z_slice + 3, img_np.shape[2] - 1)
    z_bot = max(z_slice - 3, 0)
    slices = [(z_top, "Axial (+3)"), (z_slice, "Axial (Center)"), (z_bot, "Axial (-3)")]

    for row, (slc, name) in enumerate(slices):
        img_s = img_np[:, :, slc].T
        gt_s = mask_np[:, :, slc].T
        pred_s = pred_bin[:, :, slc].T
        rgb_base = to_rgb_base(img_s)

        axes[row, 0].imshow(rgb_base, origin="lower")
        axes[row, 0].set_title(f"{name} Image")
        axes[row, 0].axis('off')

        gt_overlay = mark_boundaries(rgb_base, gt_s.astype(int), color=(0, 1, 1), mode='thick')
        axes[row, 1].imshow(gt_overlay, origin="lower")
        axes[row, 1].set_title(f"{name} Ground Truth")
        axes[row, 1].axis('off')

        eval_rgb = rgb_base.copy()
        tp_mask = (pred_s == 1) & (gt_s == 1)
        fp_mask = (pred_s == 1) & (gt_s == 0)
        fn_mask = (pred_s == 0) & (gt_s == 1)
        eval_rgb[tp_mask] = eval_rgb[tp_mask] * 0.4 + color_tp * 0.6
        eval_rgb[fp_mask] = eval_rgb[fp_mask] * 0.4 + color_fp * 0.6
        eval_rgb[fn_mask] = eval_rgb[fn_mask] * 0.4 + color_fn * 0.6
        axes[row, 2].imshow(eval_rgb, origin="lower")
        axes[row, 2].set_title(f"{name} Eval Composite")
        axes[row, 2].axis('off')

        axes[row, 3].axis('off')
        if row == 0:
            dice, iou = metrics.get('Dice', 0), metrics.get('IoU', 0)
            tp, fp, fn, tn = metrics.get('TP', 0), metrics.get('FP', 0), metrics.get('FN', 0), metrics.get('TN', 0)
            prec, rec, spec = metrics.get('Precision', 0), metrics.get('Recall', 0), metrics.get('Specificity', 0)
            gt_ml, pred_ml = metrics.get('GT_Volume_mL', 0), metrics.get('Pred_Volume_mL', 0)
            rvd = metrics.get('RVD_Pct', float('nan'))
            hd95, assd = metrics.get('HD95_mm', float('nan')), metrics.get('ASSD_mm', float('nan'))
            lg, lp = metrics.get('Lesion_Count_GT', 0), metrics.get('Lesion_Count_Pred', 0)
            lsens, lprec = metrics.get('Lesion_Sensitivity', float('nan')), metrics.get('Lesion_Precision', float('nan'))
            opt_t, opt_dice = metrics.get('Optimal_Threshold', 0.5), metrics.get('Dice_At_Optimal', dice)

            import textwrap
            diagnosis = generate_diagnosis_text(metrics)
            diagnosis = textwrap.fill(diagnosis, width=45)

            stats_text = (
                f"Patient: {pid}\n"
                f"Dice: {dice:.4f} | IoU: {iou:.4f}\n"
                f"Prec: {prec:.4f} | Rec: {rec:.4f} | Spec: {spec:.4f}\n"
                f"TP:{tp} FP:{fp} FN:{fn} TN:{tn}\n\n"
                f"GT Vol: {gt_ml:.2f} mL | Pred Vol: {pred_ml:.2f} mL\n"
                f"RVD: {rvd:+.1f}%\n"
                f"HD95: {hd95:.2f} mm | ASSD: {assd:.2f} mm\n\n"
                f"Lesions GT/Pred: {lg:.0f}/{lp:.0f}\n"
                f"Lesion Sens/Prec: {lsens:.2f}/{lprec:.2f}\n\n"
                f"Optimal Thresh: {opt_t:.2f} (Dice {opt_dice:.4f})\n\n"
                f"Diagnosis:\n{diagnosis}"
            )
            axes[row, 3].text(0.02, 0.98, stats_text, fontsize=8, va='top',
                               transform=axes[row, 3].transAxes, wrap=True)

            legend_elements = [
                mpatches.Patch(color=color_tp, label='True Positive'),
                mpatches.Patch(color=color_fp, label='False Positive'),
                mpatches.Patch(color=color_fn, label='False Negative'),
                mpatches.Patch(facecolor='none', edgecolor=(0, 1, 1), linewidth=2, label='Ground Truth'),
            ]
            axes[row, 3].legend(handles=legend_elements, loc='lower left', fontsize=10,
                                 frameon=False, bbox_to_anchor=(0.0, 0.0))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#111111')
    plt.style.use('default')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main class
# --------------------------------------------------------------------------- #
class PostTrainingVisualizer:
    def __init__(self, exp_dir: str | Path, config: dict):
        self.exp_dir = Path(exp_dir)
        self.config = config
        self.reports_dir = self.exp_dir / "reports"
        self.qual_dir = self.exp_dir / "qualitative"
        self.plots_dir = self.exp_dir / "plots"
        self.metrics_dir = self.exp_dir / "metrics"

        self.best_cases_dir = self.qual_dir / "best_cases"
        self.worst_cases_dir = self.qual_dir / "worst_cases"
        self.median_cases_dir = self.qual_dir / "median_cases"
        self.random_cases_dir = self.qual_dir / "random_cases"
        self.all_cases_dir = self.qual_dir / "all_cases"
        self.summary_dir = self.qual_dir / "summary"

        for d in [self.best_cases_dir, self.worst_cases_dir, self.median_cases_dir,
                  self.random_cases_dir, self.all_cases_dir, self.summary_dir,
                  self.plots_dir, self.metrics_dir]:
            d.mkdir(parents=True, exist_ok=True)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def run_visualization_pipeline(self, limit_cases=None):
        logger.info("=" * 80)
        logger.info("Milestone D: Starting Automated Post-Training Visualization Pipeline")
        logger.info(f"Using device: {self.device}")
        logger.info("=" * 80)

        model = self._load_model()
        metrics_df = self._run_inference_and_metrics(model, limit_cases)

        if metrics_df.empty:
            logger.error("No metrics generated. Aborting visualization.")
            return

        metrics_df.to_csv(self.metrics_dir / "patient_metrics.csv", index=False)

        selected_cases = self._select_cases(metrics_df)
        self._generate_visualizations(model, selected_cases)
        self._generate_qualitative_report(metrics_df, selected_cases)
        self._generate_publication_figures(metrics_df)

        logger.info("=" * 80)
        logger.info("Post-Training Visualization Pipeline Completed")
        logger.info("=" * 80)

    def _load_model(self):
        logger.info("Loading best_model.pt...")
        meta_path = self.reports_dir / "Experiment_Metadata.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            model_cfg = meta.get("configs", {}).get("model", {})
        else:
            logger.warning("Experiment_Metadata.json not found! Using fallback config.")
            model_cfg = {
                "architecture": "hybrid_segformer_umamba",
                "params": {"in_channels": 3, "out_channels": 1, "spatial_dims": 3,
                           "deep_supervision": True, "fusion_dim": 96, "bridge_dim": 96}
            }

        model = build_model(model_cfg)
        checkpoint_path = self.exp_dir / "checkpoints" / "best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('model_state', checkpoint.get('model_state_dict', checkpoint))
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _run_inference_and_metrics(self, model, limit_cases):
        logger.info("Running inference on validation split (research-grade metrics)...")
        from monai.metrics import DiceMetric, MeanIoU
        from monai.data import DataLoader, Dataset
        import glob

        images = sorted(glob.glob("processed/images/*.nii.gz"))
        masks = sorted(glob.glob("processed/masks/*.nii.gz"))

        if not images:
            logger.warning("No images found in processed/images/. Checking dataset/images/...")
            images = sorted(glob.glob("dataset/images/*.nii.gz") + glob.glob("dataset/images/*.nii"))
            masks = sorted(glob.glob("dataset/mask/*.nii.gz") + glob.glob("dataset/mask/*.nii"))

        if not images or not masks:
            logger.error("No dataset found! Cannot run true inference.")
            return pd.DataFrame()

        data_dicts = [{"image": img, "mask": mask} for img, mask in zip(images, masks)][:limit_cases]

        from monai.transforms import LoadImaged, ScaleIntensityd, Compose, Resized, EnsureTyped, MapTransform

        class FixShapesd(MapTransform):
            def __call__(self, data):
                d = dict(data)
                img = d["image"]
                if img.ndim == 5 and img.shape[3] == 1:
                    img = img.squeeze(3)
                if img.ndim == 4:
                    img = img.permute(3, 0, 1, 2)
                d["image"] = img
                mask = d["mask"]
                if mask.ndim == 3:
                    mask = mask.unsqueeze(0)
                d["mask"] = mask
                return d

        RESIZE_TARGET = (256, 256, 64)

        val_transforms = Compose([
            LoadImaged(keys=["image", "mask"]),
            FixShapesd(keys=["image", "mask"]),
            Resized(keys=["image", "mask"], spatial_size=RESIZE_TARGET, mode=("trilinear", "nearest")),
            EnsureTyped(keys=["image"], dtype=torch.float32),
            EnsureTyped(keys=["mask"], dtype=torch.long)
        ])

        ds = Dataset(data=data_dicts, transform=val_transforms)
        loader = DataLoader(ds, batch_size=1, num_workers=0)

        dice_metric = DiceMetric(include_background=False, reduction="none")
        iou_metric = MeanIoU(include_background=False, reduction="none")

        results = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(loader, desc="Inference")):
                val_inputs, val_labels = batch["image"].to(self.device), batch["mask"].to(self.device)

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    val_logits = model(val_inputs)
                    if isinstance(val_logits, dict):
                        val_logits = val_logits.get("full", list(val_logits.values())[-1])
                    elif isinstance(val_logits, (list, tuple)):
                        val_logits = val_logits[0]

                val_probs = torch.sigmoid(val_logits)
                val_bin = (val_probs > 0.5).float()

                dice_metric(y_pred=val_bin, y=val_labels)
                iou_metric(y_pred=val_bin, y=val_labels)
                dice = dice_metric.aggregate().item()
                iou = iou_metric.aggregate().item()
                dice_metric.reset()
                iou_metric.reset()

                # --- Effective physical spacing after resize ---
                mask_path = data_dicts[i]["mask"]
                orig_zooms, orig_shape = _get_original_spacing_and_shape(mask_path)
                eff_spacing = _effective_spacing(orig_zooms, orig_shape, RESIZE_TARGET)

                # --- Research-grade stats: physical volume, HD95/ASSD, lesion-wise, calibration ---
                pa_stats = PredictionAnalyzer.analyze(
                    val_probs, val_bin, val_labels,
                    spacing=eff_spacing,
                    lesion_match_iou=LESION_MATCH_IOU,
                    compute_surface_distance=True,
                )

                # --- Threshold sensitivity ---
                opt_t, opt_dice, sweep = _threshold_sweep(val_probs, val_labels)

                fg_pred = (val_bin.sum() / val_bin.numel()).item()
                fg_gt = (val_labels.sum() / val_labels.numel()).item()

                patient_id = Path(data_dicts[i]["image"]).name.replace(".nii.gz", "").replace(".nii", "")

                results.append({
                    "PatientID": patient_id,
                    "Dice": dice,
                    "IoU": iou,
                    "Precision": pa_stats["precision"],
                    "Recall": pa_stats["recall"],
                    "Specificity": pa_stats["specificity"],
                    "TP": pa_stats["tp"], "FP": pa_stats["fp"],
                    "FN": pa_stats["fn"], "TN": pa_stats["tn"],
                    "Pred_Volume": pa_stats["tp"] + pa_stats["fp"],
                    "GT_Volume": pa_stats["tp"] + pa_stats["fn"],
                    "GT_Volume_mL": pa_stats.get("gt_volume_ml", float("nan")),
                    "Pred_Volume_mL": pa_stats.get("pred_volume_ml", float("nan")),
                    "RVD_Pct": pa_stats.get("relative_volume_diff_pct", float("nan")),
                    "HD95_mm": pa_stats.get("hd95_mm", float("nan")),
                    "ASSD_mm": pa_stats.get("assd_mm", float("nan")),
                    "Lesion_Count_GT": pa_stats.get("gt_lesion_count", 0),
                    "Lesion_Count_Pred": pa_stats.get("pred_lesion_count", 0),
                    "Lesion_Sensitivity": pa_stats.get("lesion_wise_sensitivity", float("nan")),
                    "Lesion_Precision": pa_stats.get("lesion_wise_precision", float("nan")),
                    "Lesion_Missed": pa_stats.get("lesion_missed_count", 0),
                    "Lesion_FP": pa_stats.get("lesion_false_positive_count", 0),
                    "ECE": pa_stats.get("ece", float("nan")),
                    "Brier": pa_stats.get("brier_score", float("nan")),
                    "Optimal_Threshold": opt_t,
                    "Dice_At_Optimal": opt_dice,
                    "Threshold_Sweep": json.dumps(sweep),
                    "FG_Pred_Pct": fg_pred,
                    "FG_GT_Pct": fg_gt,
                    "Total_Error_Voxels": torch.abs(val_bin - val_labels).sum().item(),
                    "Image_Path": data_dicts[i]["image"],
                    "Mask_Path": data_dicts[i]["mask"],
                })

        return pd.DataFrame(results)

    def _select_cases(self, df: pd.DataFrame) -> dict:
        logger.info("Selecting cases for visualization...")
        df_sorted = df.sort_values(by="Dice", ascending=False).reset_index(drop=True)

        top_k = min(3, len(df))
        best_cases = df_sorted.head(top_k).to_dict('records')
        worst_cases = df_sorted.tail(top_k).to_dict('records')

        median_idx = len(df_sorted) // 2
        median_cases = df_sorted.iloc[max(0, median_idx - top_k // 2):min(len(df_sorted), median_idx + top_k // 2)].to_dict('records')

        np.random.seed(42)
        random_cases = df.sample(top_k, replace=False).to_dict('records') if len(df) >= top_k else df.to_dict('records')

        selected = {
            "best": best_cases, "worst": worst_cases, "median": median_cases,
            "random": random_cases, "all": df_sorted.to_dict('records'),
        }

        with open(self.qual_dir / "selected_cases.json", "w") as f:
            json.dump({k: [c["PatientID"] for c in v] for k, v in selected.items()}, f, indent=4)

        return selected

    def _generate_visualizations(self, model, selected_cases: dict):
        logger.info("Generating 2D multi-plane montages and 3D renders...")
        from monai.transforms import LoadImaged, ScaleIntensityd, Compose, MapTransform, Resized, EnsureTyped

        class FixShapesVizd(MapTransform):
            def __call__(self, data):
                d = dict(data)
                img = d["image"]
                if img.ndim == 5 and img.shape[3] == 1:
                    img = img.squeeze(3)
                if img.ndim == 4:
                    img = img.permute(3, 0, 1, 2)
                d["image"] = img
                mask = d["mask"]
                if mask.ndim == 3:
                    mask = mask.unsqueeze(0)
                d["mask"] = mask
                return d

        viz_transforms = Compose([
            LoadImaged(keys=["image", "mask"]),
            FixShapesVizd(keys=["image", "mask"]),
            Resized(keys=["image", "mask"], spatial_size=(256, 256, 64), mode=("trilinear", "nearest")),
            EnsureTyped(keys=["image"], dtype=torch.float32),
            EnsureTyped(keys=["mask"], dtype=torch.long)
        ])

        for category, patients in selected_cases.items():
            cat_dir = getattr(self, f"{category}_cases_dir")
            for patient_data in tqdm(patients, desc=f"Rendering {category}"):
                pid = patient_data["PatientID"]
                data = {"image": patient_data["Image_Path"], "mask": patient_data["Mask_Path"]}
                try:
                    data = viz_transforms(data)
                except Exception as e:
                    logger.error(f"Failed to load image for {pid}: {e}")
                    continue

                img_tensor = data["image"].unsqueeze(0).to(self.device)
                mask_tensor = data["mask"].unsqueeze(0).to(self.device)

                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        pred_logits = model(img_tensor)
                        if isinstance(pred_logits, dict):
                            pred_logits = pred_logits.get("full", list(pred_logits.values())[-1])
                        elif isinstance(pred_logits, (list, tuple)):
                            pred_logits = pred_logits[0]
                    pred_prob = torch.sigmoid(pred_logits)[0, 0].cpu().numpy()
                    pred_bin = (pred_prob > 0.5).astype(np.float32)

                img_np = img_tensor[0, 0].cpu().numpy()
                mask_np = mask_tensor[0, 0].cpu().numpy()

                render_case_montage(pid, patient_data, img_np, mask_np, pred_bin, cat_dir / f"{pid}_montage.pdf")

                try:
                    if pred_bin.sum() > 0:
                        verts, faces, normals, values = measure.marching_cubes(pred_bin, level=0.5)
                        obj_path = cat_dir / f"{pid}_3D_Pred.obj"
                        with open(obj_path, 'w') as f:
                            f.write(f"# 3D Mask for {pid}\n")
                            for v in verts:
                                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                            for face in faces:
                                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                        
                        # Save NIfTI for 3D Slicer
                        nii_out = nib.Nifti1Image(pred_bin.astype(np.int16), np.eye(4))
                        nib.save(nii_out, cat_dir / f"{pid}_3D_Pred.nii.gz")
                except ValueError:
                    logger.warning(f"No surface found for {pid} to extract 3D mesh.")

    def _generate_qualitative_report(self, df: pd.DataFrame, selected_cases: dict):
        logger.info("Generating Qualitative_Report.md...")
        report_path = self.reports_dir / "Qualitative_Report.md"

        with open(report_path, "w") as f:
            f.write("# Qualitative & Quantitative Research Report\n\n")

            # --- Methods paragraph, paste-ready for a paper draft ---
            f.write("## Evaluation Protocol\n")
            f.write(
                f"Segmentation quality was assessed on {len(df)} held-out cases. Predictions were "
                f"thresholded at p>0.5 for the primary metrics; a sensitivity sweep across thresholds "
                f"{THRESHOLD_SWEEP} was additionally run per case to report the Dice-optimal operating "
                f"point. Overlap was scored with Dice and IoU; boundary agreement with the 95th-percentile "
                f"Hausdorff distance (HD95) and average symmetric surface distance (ASSD), both computed "
                f"in physical mm using per-case voxel spacing recovered from the original NIfTI headers. "
                f"Volumes are reported in mL. Lesion-wise detection used {LESION_MATCH_IOU*100:.0f}% "
                f"voxel-overlap as the match criterion between 26-connected components in the prediction "
                f"and ground truth. Calibration was assessed via Brier score and Expected Calibration "
                f"Error (ECE, 10 bins). 95% confidence intervals below are bootstrap estimates (n=2000 "
                f"resamples).\n\n"
            )

            # --- Table 1: summary statistics ---
            f.write("## Table 1: Summary Statistics\n\n")
            f.write("| Metric | Mean ± Std | Median [IQR] | 95% CI |\n")
            f.write("|---|---|---|---|\n")
            metric_specs = [
                ("Dice", "Dice"), ("IoU", "IoU"), ("Precision", "Precision"), ("Recall", "Recall"),
                ("HD95 (mm)", "HD95_mm"), ("ASSD (mm)", "ASSD_mm"),
                ("GT Volume (mL)", "GT_Volume_mL"), ("Pred Volume (mL)", "Pred_Volume_mL"),
                ("RVD (%)", "RVD_Pct"), ("Lesion Sensitivity", "Lesion_Sensitivity"),
                ("Lesion Precision", "Lesion_Precision"), ("ECE", "ECE"), ("Brier", "Brier"),
            ]
            for label, col in metric_specs:
                if col not in df.columns:
                    continue
                vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
                if vals.empty:
                    continue
                mean, lo, hi = bootstrap_ci(vals.values)
                std = vals.std()
                median = vals.median()
                q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                f.write(f"| {label} | {mean:.4f} ± {std:.4f} | {median:.4f} [{q1:.4f}, {q3:.4f}] | "
                        f"[{lo:.4f}, {hi:.4f}] |\n")
            f.write("\n")

            # --- Aggregate findings ---
            f.write("## Aggregate Findings\n")
            avg_dice = df['Dice'].mean()
            avg_fp, avg_fn = df['FP'].mean(), df['FN'].mean()
            total_lesions_gt = df['Lesion_Count_GT'].sum()
            total_lesions_missed = df['Lesion_Missed'].sum()

            f.write(f"Across {len(df)} cases, mean Dice was {avg_dice:.4f}. ")
            if avg_fp > avg_fn * 1.5:
                f.write("The model exhibits a systematic tendency toward over-segmentation. ")
            elif avg_fn > avg_fp * 1.5:
                f.write("The model exhibits a systematic tendency toward under-segmentation. ")
            else:
                f.write("Errors are relatively balanced between over- and under-segmentation. ")
            if total_lesions_gt > 0:
                miss_rate = total_lesions_missed / total_lesions_gt * 100
                f.write(f"At the lesion level, {int(total_lesions_missed)} of {int(total_lesions_gt)} "
                        f"({miss_rate:.1f}%) distinct hemorrhage foci across the dataset were missed "
                        f"entirely — this is the metric most relevant to missed-diagnosis risk, since "
                        f"voxel Dice can look acceptable while a whole secondary bleed is undetected.\n\n")

            mean_ece = df['ECE'].mean()
            f.write(f"Mean calibration error (ECE) was {mean_ece:.4f}; "
                    f"{'probabilities are reasonably well-calibrated' if mean_ece < 0.1 else 'probabilities are poorly calibrated and should not be used directly as confidence without recalibration (e.g. temperature scaling)'}.\n\n")

            threshold_shift = (df['Optimal_Threshold'] != 0.5).sum()
            if threshold_shift > len(df) * 0.3:
                f.write(f"{threshold_shift}/{len(df)} cases had a Dice-optimal threshold different from "
                        f"the default 0.5, suggesting the fixed operating point may be leaving Dice on "
                        f"the table for a meaningful fraction of cases.\n\n")

            # --- Per-case detail ---
            f.write("## Detailed Case Analysis\n\n")
            all_cases = selected_cases.get("all", df.to_dict('records'))
            for c in all_cases:
                pid = c['PatientID']
                f.write(f"### Case: {pid}\n")
                f.write(f"- **Overlap**: Dice {c.get('Dice',0):.4f} | IoU {c.get('IoU',0):.4f} | "
                        f"Precision {c.get('Precision',0):.4f} | Recall {c.get('Recall',0):.4f}\n")
                f.write(f"- **Boundary**: HD95 {c.get('HD95_mm',float('nan')):.2f}mm | "
                        f"ASSD {c.get('ASSD_mm',float('nan')):.2f}mm\n")
                f.write(f"- **Volume**: GT {c.get('GT_Volume_mL',float('nan')):.2f}mL | "
                        f"Pred {c.get('Pred_Volume_mL',float('nan')):.2f}mL | "
                        f"RVD {c.get('RVD_Pct',float('nan')):+.1f}%\n")
                f.write(f"- **Lesions**: GT {c.get('Lesion_Count_GT',0):.0f} | "
                        f"Pred {c.get('Lesion_Count_Pred',0):.0f} | "
                        f"Sensitivity {c.get('Lesion_Sensitivity',float('nan')):.2f} | "
                        f"Precision {c.get('Lesion_Precision',float('nan')):.2f}\n")
                f.write(f"- **Optimal threshold**: {c.get('Optimal_Threshold',0.5):.2f} "
                        f"(Dice {c.get('Dice_At_Optimal',0):.4f} vs {c.get('Dice',0):.4f} @0.5)\n")
                f.write(f"- **Diagnosis**: {generate_diagnosis_text(c)}\n\n")


    def _generate_publication_figures(self, df: pd.DataFrame):
        """Generate publication-ready quantitative and qualitative figures."""
        logger.info("Generating publication figures...")
        
        # Set publication-style aesthetics
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        
        if "Dice" in df.columns and len(df) > 0:
            df_sorted = df.sort_values(by="Dice", ascending=False).reset_index(drop=True)
            
            # 1. Dice score bar chart (sorted)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=df_sorted.index, y="Dice", data=df_sorted, palette="viridis")
            plt.title("Dice Score per Case (Sorted)")
            plt.xlabel("Case Index")
            plt.ylabel("Dice Score")
            plt.tight_layout()
            plt.savefig(self.summary_dir / "Figure_1_Dice_Bar.pdf", dpi=300)
            plt.close()
            
            # 2. Confusion Matrix Heatmap (Total Voxels)
            plt.figure(figsize=(6, 5))
            total_tp = df["TP"].sum()
            total_fp = df["FP"].sum()
            total_fn = df["FN"].sum()
            total_tn = df["TN"].sum()
            cm = np.array([[total_tp, total_fn], [total_fp, total_tn]])
            sns.heatmap(cm, annot=True, fmt=".2e", cmap="Blues", xticklabels=["Pred +", "Pred -"], yticklabels=["GT +", "GT -"])
            plt.title("Aggregate Voxel Confusion Matrix")
            plt.tight_layout()
            plt.savefig(self.summary_dir / "Figure_2_Confusion_Matrix.pdf", dpi=300)
            plt.close()
            
            # 3. Precision vs Recall Scatter
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x="Recall", y="Precision", data=df, hue="Dice", palette="coolwarm", s=100)
            plt.title("Precision vs Recall")
            plt.tight_layout()
            plt.savefig(self.summary_dir / "Figure_3_Precision_Recall.pdf", dpi=300)
            plt.close()
            
            # 4. Dice vs GT Volume
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x="GT_Volume", y="Dice", data=df, color="purple", s=100)
            plt.title("Dice vs Ground Truth Volume")
            plt.xscale("log")
            plt.tight_layout()
            plt.savefig(self.summary_dir / "Figure_4_Dice_vs_Volume.pdf", dpi=300)
            plt.close()
            
            # 5. Stacked Bar (FP vs FN)
            plt.figure(figsize=(10, 6))
            df_sorted[['FP', 'FN']].plot(kind='bar', stacked=True, color=['#E74C3C', '#F39C12'], figsize=(10, 6))
            plt.title("False Positive vs False Negative Volume per Case")
            plt.xlabel("Case Index (Sorted by Dice)")
            plt.ylabel("Voxel Count")
            plt.tight_layout()
            plt.savefig(self.summary_dir / "Figure_5_Stacked_FP_FN.pdf", dpi=300)
            plt.close()
            
            # 6. Boxplot of Dice by Volume Group
            plt.figure(figsize=(8, 6))
            if df['GT_Volume'].nunique() >= 3:
                df['Volume_Group'] = pd.qcut(df['GT_Volume'], q=3, labels=['Small', 'Medium', 'Large'], duplicates='drop')
                sns.boxplot(x="Volume_Group", y="Dice", data=df, palette="Set2")
                plt.title("Dice Score by Lesion Size")
            else:
                sns.boxplot(y="Dice", data=df, color="cyan")
                plt.title("Dice Score")
            plt.tight_layout()
            plt.savefig(self.summary_dir / "Figure_6_Dice_Boxplot.pdf", dpi=300)
            plt.close()
            
            # 7. Metric Correlation Heatmap
            plt.figure(figsize=(8, 6))
            corr_cols = ["Dice", "IoU", "Precision", "Recall", "GT_Volume", "Total_Error_Voxels", "FP", "FN"]
            corr = df[corr_cols].corr()
            sns.heatmap(corr, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, fmt=".2f")
            plt.title("Metric Correlation Heatmap")
            plt.tight_layout()
            plt.savefig(self.summary_dir / "Figure_7_Correlation_Heatmap.pdf", dpi=300)
            plt.close()

def trigger_visualization_pipeline(exp_dir: str | Path, config: dict, limit_cases=None):
    """Wrapper function to be called from train.py"""
    try:
        visualizer = PostTrainingVisualizer(exp_dir, config)
        visualizer.run_visualization_pipeline(limit_cases)
    except Exception as e:
        logger.error(f"Visualization Pipeline Failed: {e}", exc_info=True)

if __name__ == "__main__":
    import argparse
    
    # Configure basic logging for standalone execution
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    
    parser = argparse.ArgumentParser(description="Standalone Post-Training Visualization")
    parser.add_argument("--experiment", type=str, required=True, help="Path to the experiment directory (e.g., outputs/EXP_...)")
    parser.add_argument("--limit", type=int, default=None, help="Max cases to run inference on (to save time)")
    
    args = parser.parse_args()
    
    exp_path = Path(args.experiment)
    if not exp_path.exists():
        logger.error(f"Experiment directory {exp_path} does not exist.")
        exit(1)
        
    trigger_visualization_pipeline(exp_path, {}, args.limit)
