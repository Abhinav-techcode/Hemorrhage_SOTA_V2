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
        self.summary_dir = self.qual_dir / "summary"
        
        for d in [self.best_cases_dir, self.worst_cases_dir, self.median_cases_dir, self.random_cases_dir, self.summary_dir, self.plots_dir, self.metrics_dir]:
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
                
                patient_id = Path(data_dicts[i]["image"]).name.replace(".nii.gz", "").replace(".nii", "")
                
                results.append({
                    "PatientID": patient_id,
                    "Dice": dice,
                    "IoU": iou,
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
            "random": random_cases
        }
        
        with open(self.qual_dir / "selected_cases.json", "w") as f:
            # Just save IDs for JSON
            json_friendly = {k: [c["PatientID"] for c in v] for k, v in selected.items()}
            json.dump(json_friendly, f, indent=4)
            
        return selected

    def _generate_visualizations(self, model, selected_cases: dict):
        """Generate 2D and 3D visual outputs for selected cases."""
        logger.info("Generating 2D multi-plane montages and 3D renders...")
        from monai.transforms import LoadImaged, ScaleIntensityd, Compose, MapTransform
        
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
            ScaleIntensityd(keys=["image"])
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
                fig, axes = plt.subplots(3, 4, figsize=(16, 12))
                
                slices = [(img_np, z_slice, 2, "Axial"), (img_np, y_slice, 1, "Coronal"), (img_np, x_slice, 0, "Sagittal")]
                
                for row, (vol, slc, axis, name) in enumerate(slices):
                    if axis == 2:
                        img_s, gt_s, pred_s = img_np[:, :, slc], mask_np[:, :, slc], pred_bin[:, :, slc]
                    elif axis == 1:
                        img_s, gt_s, pred_s = img_np[:, slc, :], mask_np[:, slc, :], pred_bin[:, slc, :]
                    else:
                        img_s, gt_s, pred_s = img_np[slc, :, :], mask_np[slc, :, :], pred_bin[slc, :, :]
                        
                    axes[row, 0].imshow(img_s.T, cmap="gray", origin="lower")
                    axes[row, 0].set_title(f"{name} Image")
                    axes[row, 0].axis('off')
                    
                    axes[row, 1].imshow(img_s.T, cmap="gray", origin="lower")
                    axes[row, 1].imshow(np.ma.masked_where(gt_s.T == 0, gt_s.T), cmap="Greens", alpha=0.5, origin="lower")
                    axes[row, 1].set_title(f"{name} Ground Truth")
                    axes[row, 1].axis('off')
                    
                    axes[row, 2].imshow(img_s.T, cmap="gray", origin="lower")
                    axes[row, 2].imshow(np.ma.masked_where(pred_s.T == 0, pred_s.T), cmap="Reds", alpha=0.5, origin="lower")
                    axes[row, 2].set_title(f"{name} Prediction")
                    axes[row, 2].axis('off')
                    
                    diff = gt_s.T - pred_s.T
                    axes[row, 3].imshow(img_s.T, cmap="gray", origin="lower")
                    # False Negatives (Green)
                    axes[row, 3].imshow(np.ma.masked_where(diff != 1, diff), cmap="Greens", alpha=0.7, origin="lower")
                    # False Positives (Red)
                    axes[row, 3].imshow(np.ma.masked_where(diff != -1, diff), cmap="Reds", alpha=0.7, origin="lower")
                    axes[row, 3].set_title(f"{name} Difference (FN=G, FP=R)")
                    axes[row, 3].axis('off')
                
                plt.suptitle(f"{category.capitalize()} Case: {pid} | Dice: {patient_data['Dice']:.4f}")
                plt.tight_layout()
                plt.savefig(cat_dir / f"{pid}_montage.png", dpi=150)
                plt.close(fig)
                
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
            f.write("## Overview\nThis report analyzes the best, worst, and median performance cases to identify systemic failure patterns and structural successes.\n\n")
            f.write("## Case Groupings\n")
            f.write(f"- **Best Cases**: {', '.join([c['PatientID'] for c in selected_cases['best']])}\n")
            f.write(f"- **Worst Cases**: {', '.join([c['PatientID'] for c in selected_cases['worst']])}\n")
            f.write(f"- **Median Cases**: {', '.join([c['PatientID'] for c in selected_cases['median']])}\n")
            f.write(f"- **Random (Fixed Seed) Cases**: {', '.join([c['PatientID'] for c in selected_cases['random']])}\n")
            f.write("\n## Analysis & Recommendations\n")
            f.write("- *Post-training visuals are saved as high-res PNG montages and 3D OBJ meshes.*\n")

    def _generate_publication_figures(self, df: pd.DataFrame):
        """Generate publication-ready quantitative and qualitative figures."""
        logger.info("Generating publication figures...")
        
        # Set publication-style aesthetics
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        
        if "Dice" in df.columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(df["Dice"], bins=10, kde=True, color="blue")
            plt.title("Distribution of Validation Dice Scores")
            plt.xlabel("Dice Score")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(self.summary_dir / "Figure_1_DiceDistribution.png", dpi=300)
            plt.close()
            
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x="Dice", y="Total_Error_Voxels", data=df, color="red")
            plt.title("Error Volume vs Dice Performance")
            plt.xlabel("Dice Score")
            plt.ylabel("Total Error Voxels (FP + FN)")
            plt.tight_layout()
            plt.savefig(self.summary_dir / "Figure_2_ErrorAnalysis.png", dpi=300)
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
