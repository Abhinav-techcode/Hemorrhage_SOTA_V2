"""
evaluation/post_training_visualizer.py
Milestone D: Automated Post-Training Research Visualization
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

def to_rgb_base(img, apply_window=False):
    """Convert grayscale to RGB, with optional basic windowing."""
    if apply_window:
        img = np.clip(img, 0, 80)
        img = (img - 0) / 80
    else:
        min_val, max_val = img.min(), img.max()
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
    return np.stack([img, img, img], axis=-1)

def generate_diagnosis_text(metrics):
    dice = metrics.get('Dice', 0)
    recall = metrics.get('Recall', 0)
    precision = metrics.get('Precision', 0)
    gt_vol = metrics.get('GT_Volume', 0)
    fp = metrics.get('FP', 0)
    fn = metrics.get('FN', 0)
    
    if dice > 0.8:
        return "Clean high-agreement case with accurate segmentation."
    elif gt_vol < 1000 and dice < 0.5:
        return "Model struggled to segment small lesion structure."
    elif recall < 0.5 and fp < fn:
        return "Severe under-segmentation; missed significant hemorrhage portions."
    elif precision < 0.5 and fp > fn:
        return "Severe over-segmentation; spurious false-positive clusters present."
    elif fp > fn * 2:
        return "Tendency toward over-segmentation and false-positive boundaries."
    elif fn > fp * 2:
        return "Tendency toward under-segmentation; missing lesion boundaries."
    else:
        return "Moderate agreement with mixed boundary discrepancies."

def render_case_montage(pid, metrics, img_np, mask_np, pred_bin, save_path):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from skimage.segmentation import mark_boundaries
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Colors:
    color_tp = np.array([0.18, 0.80, 0.44]) # green #2ECC71
    color_fp = np.array([0.91, 0.30, 0.24]) # red #E74C3C
    color_fn = np.array([0.95, 0.61, 0.07]) # amber #F39C12
    
    z_slice = np.argmax(mask_np.sum(axis=(0, 1))) if mask_np.sum() > 0 else img_np.shape[2] // 2
    y_slice = np.argmax(mask_np.sum(axis=(0, 2))) if mask_np.sum() > 0 else img_np.shape[1] // 2
    x_slice = np.argmax(mask_np.sum(axis=(1, 2))) if mask_np.sum() > 0 else img_np.shape[0] // 2
    
    slices = [(img_np, z_slice, 2, "Axial"), (img_np, y_slice, 1, "Coronal"), (img_np, x_slice, 0, "Sagittal")]
    
    for row, (vol, slc, axis, name) in enumerate(slices):
        if axis == 2:
            img_s, gt_s, pred_s = img_np[:, :, slc], mask_np[:, :, slc], pred_bin[:, :, slc]
        elif axis == 1:
            img_s, gt_s, pred_s = img_np[:, slc, :], mask_np[:, slc, :], pred_bin[:, slc, :]
        else:
            img_s, gt_s, pred_s = img_np[slc, :, :], mask_np[slc, :, :], pred_bin[slc, :, :]
            
        img_s = img_s.T
        gt_s = gt_s.T
        pred_s = pred_s.T
        
        rgb_base = to_rgb_base(img_s)
        
        # Panel 0: Image
        axes[row, 0].imshow(rgb_base, origin="lower")
        axes[row, 0].set_title(f"{name} Image")
        axes[row, 0].axis('off')
        
        # Panel 1: GT Outline
        gt_overlay = mark_boundaries(rgb_base, gt_s.astype(int), color=(0, 1, 1), mode='thick')
        axes[row, 1].imshow(gt_overlay, origin="lower")
        axes[row, 1].set_title(f"{name} Ground Truth")
        axes[row, 1].axis('off')
        
        # Panel 2: Eval Composite
        eval_rgb = rgb_base.copy()
        tp_mask = (pred_s == 1) & (gt_s == 1)
        fp_mask = (pred_s == 1) & (gt_s == 0)
        fn_mask = (pred_s == 0) & (gt_s == 1)
        
        # Apply colors directly
        eval_rgb[tp_mask] = eval_rgb[tp_mask] * 0.4 + color_tp * 0.6
        eval_rgb[fp_mask] = eval_rgb[fp_mask] * 0.4 + color_fp * 0.6
        eval_rgb[fn_mask] = eval_rgb[fn_mask] * 0.4 + color_fn * 0.6
        
        axes[row, 2].imshow(eval_rgb, origin="lower")
        axes[row, 2].set_title(f"{name} Eval Composite")
        axes[row, 2].axis('off')
        
        # Panel 3: Stats / Legend
        axes[row, 3].axis('off')
        if row == 0:
            dice_score = metrics.get('Dice', 0)
            iou_score = metrics.get('IoU', 0)
            tp, fp, fn, tn = metrics.get('TP', 0), metrics.get('FP', 0), metrics.get('FN', 0), metrics.get('TN', 0)
            prec, rec, spec = metrics.get('Precision', 0), metrics.get('Recall', 0), metrics.get('Specificity', 0)
            gt_vol, pred_vol = metrics.get('GT_Volume', 0), metrics.get('Pred_Volume', 0)
            
            over_seg = (fp / gt_vol * 100) if gt_vol > 0 else 0
            under_seg = (fn / gt_vol * 100) if gt_vol > 0 else 0
            
            diagnosis = generate_diagnosis_text(metrics)
            
            stats_text = (
                f"Patient: {pid}\n"
                f"Dice: {dice_score:.4f} | IoU: {iou_score:.4f}\n"
                f"Prec: {prec:.4f} | Rec: {rec:.4f} | Spec: {spec:.4f}\n"
                f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}\n"
                f"GT Vol: {gt_vol} | Pred Vol: {pred_vol}\n"
                f"Over-seg: +{over_seg:.1f}% | Under-seg: -{under_seg:.1f}%\n\n"
                f"Diagnosis:\n{diagnosis}"
            )
            
            axes[row, 3].text(0.05, 0.95, stats_text, fontsize=11, verticalalignment='top', transform=axes[row, 3].transAxes, wrap=True)
            
            legend_elements = [
                mpatches.Patch(color=color_tp, label='True Positive (TP)'),
                mpatches.Patch(color=color_fp, label='False Positive (FP)'),
                mpatches.Patch(color=color_fn, label='False Negative (FN)'),
                mpatches.Patch(facecolor='none', edgecolor=(0, 1, 1), linewidth=2, label='Ground Truth')
            ]
            axes[row, 3].legend(handles=legend_elements, loc='lower left', fontsize=11, frameon=False, bbox_to_anchor=(0.05, 0.05))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# Add project root to path so we can import models when running as standalone script
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import framework dependencies
from models.model_factory import build_model
from datasets.transforms import TransformFactory

logger = logging.getLogger(__name__)

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
        
        for d in [self.best_cases_dir, self.worst_cases_dir, self.median_cases_dir, self.random_cases_dir, self.all_cases_dir, self.summary_dir, self.plots_dir, self.metrics_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
    def run_visualization_pipeline(self, limit_cases=10):
        """Main entry point for Milestone D."""
        logger.info("=" * 80)
        logger.info("Milestone D: Starting Automated Post-Training Visualization Pipeline")
        logger.info(f"Using device: {self.device}")
        logger.info("=" * 80)
        
        # 1. Load Model
        model = self._load_model()
        
        # 2. Run Inference to generate metrics
        metrics_df = self._run_inference_and_metrics(model, limit_cases)
        
        if metrics_df.empty:
            logger.error("No metrics generated. Aborting visualization.")
            return

        # Save metrics
        metrics_df.to_csv(self.metrics_dir / "patient_metrics.csv", index=False)
        
        # 3. Case Selection
        selected_cases = self._select_cases(metrics_df)
        
        # 4. Generate Qualitative Visualizations (2D Montages & 3D Renders)
        self._generate_visualizations(model, selected_cases)
        
        # 5. Generate Qualitative Report
        self._generate_qualitative_report(metrics_df, selected_cases)
        
        # 6. Generate Publication Figures
        self._generate_publication_figures(metrics_df)
        
        logger.info("=" * 80)
        logger.info("Post-Training Visualization Pipeline Completed")
        logger.info("=" * 80)

    def _load_model(self):
        logger.info("Loading best_model.pt...")
        # Load experiment metadata to get model config
        meta_path = self.reports_dir / "Experiment_Metadata.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            model_cfg = meta.get("configs", {}).get("model", {})
        else:
            # Fallback if metadata is missing
            logger.warning("Experiment_Metadata.json not found! Using fallback config.")
            model_cfg = {
                "architecture": "hybrid_segformer_umamba",
                "params": {
                    "in_channels": 3,
                    "out_channels": 1,
                    "spatial_dims": 3,
                    "deep_supervision": True,
                    "fusion_dim": 96,
                    "bridge_dim": 96
                }
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
        logger.info("Running inference on validation split...")
        import monai
        from monai.metrics import DiceMetric, MeanIoU
        from monai.data import DataLoader, Dataset
        import glob
        
        # Prioritize preprocessed data (which has 3 channels for multi-window)
        images = sorted(glob.glob("processed/images/*.nii.gz"))
        masks = sorted(glob.glob("processed/masks/*.nii.gz"))
        
        if not images:
            logger.warning("No images found in processed/images/. Checking dataset/images/...")
            images = sorted(glob.glob("dataset/images/*.nii.gz") + glob.glob("dataset/images/*.nii"))
            masks = sorted(glob.glob("dataset/mask/*.nii.gz") + glob.glob("dataset/mask/*.nii"))
            
        if not images or not masks:
            logger.error("No dataset found! Cannot run true inference.")
            return pd.DataFrame()
            
        # Match images and masks by name
        data_dicts = [{"image": img, "mask": mask} for img, mask in zip(images, masks)][:limit_cases]
        
        from monai.transforms import LoadImaged, ScaleIntensityd, Compose, Resized, EnsureTyped, MapTransform
        
        class FixShapesd(MapTransform):
            def __call__(self, data):
                d = dict(data)
                # Image: (H, W, D, 1, C) -> (C, H, W, D)
                img = d["image"]
                if img.ndim == 5 and img.shape[3] == 1:
                    img = img.squeeze(3) # (H, W, D, C)
                if img.ndim == 4:
                    img = img.permute(3, 0, 1, 2) # (C, H, W, D)
                d["image"] = img
                
                # Mask: (H, W, D) -> (1, H, W, D)
                mask = d["mask"]
                if mask.ndim == 3:
                    mask = mask.unsqueeze(0)
                d["mask"] = mask
                return d

        val_transforms = Compose([
            LoadImaged(keys=["image", "mask"]),
            FixShapesd(keys=["image", "mask"]),
            Resized(keys=["image", "mask"], spatial_size=(256, 256, 64), mode=("trilinear", "nearest")),
            ScaleIntensityd(keys=["image"]),
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
                
                # Forward pass (handle Deep Supervision if present)
                val_outputs = model(val_inputs)
                if isinstance(val_outputs, dict):
                    val_outputs = val_outputs.get("full", list(val_outputs.values())[-1])
                elif isinstance(val_outputs, (list, tuple)):
                    val_outputs = val_outputs[0] # Take highest resolution output
                
                val_outputs = torch.sigmoid(val_outputs)
                val_outputs_bin = (val_outputs > 0.5).float()
                
                # Compute metrics
                dice_metric(y_pred=val_outputs_bin, y=val_labels)
                iou_metric(y_pred=val_outputs_bin, y=val_labels)
                
                dice = dice_metric.aggregate().item()
                iou = iou_metric.aggregate().item()
                
                dice_metric.reset()
                iou_metric.reset()
                
                # foreground percentage
                fg_pred = (val_outputs_bin.sum() / val_outputs_bin.numel()).item()
                fg_gt = (val_labels.sum() / val_labels.numel()).item()
                
                pred_bool = val_outputs_bin.bool()
                gt_bool = val_labels.bool()
                tp = (pred_bool & gt_bool).sum().item()
                fp = (pred_bool & ~gt_bool).sum().item()
                fn = (~pred_bool & gt_bool).sum().item()
                tn = (~pred_bool & ~gt_bool).sum().item()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
                pred_vol = pred_bool.sum().item()
                gt_vol = gt_bool.sum().item()
                
                patient_id = Path(data_dicts[i]["image"]).name.replace(".nii.gz", "").replace(".nii", "")
                
                results.append({
                    "PatientID": patient_id,
                    "Dice": dice,
                    "IoU": iou,
                    "Precision": precision,
                    "Recall": recall,
                    "Specificity": specificity,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                    "Pred_Volume": pred_vol,
                    "GT_Volume": gt_vol,
                    "FG_Pred_Pct": fg_pred,
                    "FG_GT_Pct": fg_gt,
                    "Total_Error_Voxels": torch.abs(val_outputs_bin - val_labels).sum().item(),
                    "Image_Path": data_dicts[i]["image"],
                    "Mask_Path": data_dicts[i]["mask"]
                })
                
        df = pd.DataFrame(results)
        return df

    def _select_cases(self, df: pd.DataFrame) -> dict:
        """Select Best, Worst, Median, and Random cases based on Dice score."""
        logger.info("Selecting cases for visualization...")
        df_sorted = df.sort_values(by="Dice", ascending=False).reset_index(drop=True)
        
        top_k = min(3, len(df))
        best_cases = df_sorted.head(top_k).to_dict('records')
        worst_cases = df_sorted.tail(top_k).to_dict('records')
        
        median_idx = len(df_sorted) // 2
        median_cases = df_sorted.iloc[max(0, median_idx - top_k//2) : min(len(df_sorted), median_idx + top_k//2)].to_dict('records')
        
        np.random.seed(42)
        random_cases = df.sample(top_k, replace=False).to_dict('records') if len(df) >= top_k else df.to_dict('records')
        
        selected = {
            "best": best_cases,
            "worst": worst_cases,
            "median": median_cases,
            "random": random_cases,
            "all": df_sorted.to_dict('records')
        }
        
        with open(self.qual_dir / "selected_cases.json", "w") as f:
            # Just save IDs for JSON
            json_friendly = {k: [c["PatientID"] for c in v] for k, v in selected.items()}
            json.dump(json_friendly, f, indent=4)
            
        return selected

    def _generate_visualizations(self, model, selected_cases: dict):
        """Generate 2D and 3D visual outputs for selected cases."""
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
            ScaleIntensityd(keys=["image"]),
            EnsureTyped(keys=["image"], dtype=torch.float32),
            EnsureTyped(keys=["mask"], dtype=torch.long)
        ])
        
        for category, patients in selected_cases.items():
            cat_dir = getattr(self, f"{category}_cases_dir")
            for patient_data in tqdm(patients, desc=f"Rendering {category}"):
                pid = patient_data["PatientID"]
                
                # Load raw images for visualization
                data = {"image": patient_data["Image_Path"], "mask": patient_data["Mask_Path"]}
                try:
                    data = viz_transforms(data)
                except Exception as e:
                    logger.error(f"Failed to load image for {pid}: {e}")
                    continue
                    
                img_tensor = data["image"].unsqueeze(0).to(self.device)
                mask_tensor = data["mask"].unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    pred_logits = model(img_tensor)
                    if isinstance(pred_logits, dict):
                        pred_logits = pred_logits.get("full", list(pred_logits.values())[-1])
                    elif isinstance(pred_logits, (list, tuple)):
                        pred_logits = pred_logits[0]
                    pred_prob = torch.sigmoid(pred_logits)[0, 0].cpu().numpy()
                    pred_bin = (pred_prob > 0.5).astype(np.float32)
                    
                img_np = img_tensor[0, 0].cpu().numpy()
                mask_np = mask_tensor[0, 0].cpu().numpy()
                
                # Find slice with max hemorrhage in Ground Truth
                z_slice = np.argmax(mask_np.sum(axis=(0, 1))) if mask_np.sum() > 0 else img_np.shape[2] // 2
                y_slice = np.argmax(mask_np.sum(axis=(0, 2))) if mask_np.sum() > 0 else img_np.shape[1] // 2
                x_slice = np.argmax(mask_np.sum(axis=(1, 2))) if mask_np.sum() > 0 else img_np.shape[0] // 2
                
                # Render 2D Montages
                render_case_montage(pid, patient_data, img_np, mask_np, pred_bin, cat_dir / f"{pid}_montage.png")
                
                # 3D Mesh Extraction via Marching Cubes
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
                except ValueError:
                    logger.warning(f"No surface found for {pid} to extract 3D mesh.")

    def _generate_qualitative_report(self, df: pd.DataFrame, selected_cases: dict):
        """Generate a markdown report summarizing the visual findings."""
        logger.info("Generating Qualitative_Report.md...")
        report_path = self.reports_dir / "Qualitative_Report.md"
        
        with open(report_path, "w") as f:
            f.write("# Qualitative Research Report\n\n")
            f.write("## Overview\nThis report analyzes the performance across all evaluated cases to identify systemic failure patterns and structural successes.\n\n")
            
            f.write("## Detailed Case Analysis\n\n")
            
            all_cases = selected_cases.get("all", df.to_dict('records'))
            
            for c in all_cases:
                pid = c['PatientID']
                dice = c.get('Dice', 0)
                iou = c.get('IoU', 0)
                prec = c.get('Precision', 0)
                rec = c.get('Recall', 0)
                tp = c.get('TP', 0)
                fp = c.get('FP', 0)
                fn = c.get('FN', 0)
                
                diagnosis = generate_diagnosis_text(c)
                
                f.write(f"### Case: {pid}\n")
                f.write(f"- **Metrics**: Dice: {dice:.4f} | IoU: {iou:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}\n")
                f.write(f"- **Voxel Stats**: TP: {tp} | FP: {fp} | FN: {fn}\n")
                f.write(f"- **Diagnosis**: {diagnosis}\n\n")

            f.write("## Aggregate Findings\n")
            
            avg_dice = df['Dice'].mean()
            avg_fp = df['FP'].mean()
            avg_fn = df['FN'].mean()
            
            f.write(f"Across the evaluated cases, the model achieved an average Dice score of {avg_dice:.4f}. ")
            if avg_fp > avg_fn * 1.5:
                f.write("Overall, the model exhibits a tendency toward over-segmentation and false-positive boundaries.")
            elif avg_fn > avg_fp * 1.5:
                f.write("Overall, the model exhibits a tendency toward under-segmentation and false-negative regions.")
            else:
                f.write("Overall, the model maintains a relatively balanced trade-off between sensitivity and specificity.")
            f.write("\n")

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
            plt.savefig(self.summary_dir / "Figure_1_Dice_Bar.png", dpi=300)
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
            plt.savefig(self.summary_dir / "Figure_2_Confusion_Matrix.png", dpi=300)
            plt.close()
            
            # 3. Precision vs Recall Scatter
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x="Recall", y="Precision", data=df, hue="Dice", palette="coolwarm", s=100)
            plt.title("Precision vs Recall")
            plt.tight_layout()
            plt.savefig(self.summary_dir / "Figure_3_Precision_Recall.png", dpi=300)
            plt.close()
            
            # 4. Dice vs GT Volume
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x="GT_Volume", y="Dice", data=df, color="purple", s=100)
            plt.title("Dice vs Ground Truth Volume")
            plt.xscale("log")
            plt.tight_layout()
            plt.savefig(self.summary_dir / "Figure_4_Dice_vs_Volume.png", dpi=300)
            plt.close()
            
            # 5. Stacked Bar (FP vs FN)
            plt.figure(figsize=(10, 6))
            df_sorted[['FP', 'FN']].plot(kind='bar', stacked=True, color=['#E74C3C', '#F39C12'], figsize=(10, 6))
            plt.title("False Positive vs False Negative Volume per Case")
            plt.xlabel("Case Index (Sorted by Dice)")
            plt.ylabel("Voxel Count")
            plt.tight_layout()
            plt.savefig(self.summary_dir / "Figure_5_Stacked_FP_FN.png", dpi=300)
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
            plt.savefig(self.summary_dir / "Figure_6_Dice_Boxplot.png", dpi=300)
            plt.close()
            
            # 7. Metric Correlation Heatmap
            plt.figure(figsize=(8, 6))
            corr_cols = ["Dice", "IoU", "Precision", "Recall", "GT_Volume", "Total_Error_Voxels", "FP", "FN"]
            corr = df[corr_cols].corr()
            sns.heatmap(corr, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, fmt=".2f")
            plt.title("Metric Correlation Heatmap")
            plt.tight_layout()
            plt.savefig(self.summary_dir / "Figure_7_Correlation_Heatmap.png", dpi=300)
            plt.close()

def trigger_visualization_pipeline(exp_dir: str | Path, config: dict, limit_cases=10):
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
    parser.add_argument("--limit", type=int, default=10, help="Max cases to run inference on (to save time)")
    
    args = parser.parse_args()
    
    exp_path = Path(args.experiment)
    if not exp_path.exists():
        logger.error(f"Experiment directory {exp_path} does not exist.")
        exit(1)
        
    trigger_visualization_pipeline(exp_path, {}, args.limit)
