import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import yaml
from datasets.dataloader import BrainHemorrhageDataModule, DataLoaderConfig
from datasets.transforms import TransformFactory

def main():
    print("Starting Phase 4.4 - Crop Strategy Audit...")
    
    config_dir = Path("configs")
    with open(config_dir / "dataset.yaml", "r") as f:
        dataset_cfg = yaml.safe_load(f)
    transform_factory = TransformFactory(config_dir / "augmentation.yaml")
    
    loader_config = DataLoaderConfig(
        dataset_config=dataset_cfg,
        batch_size=1,
        num_workers=4,
        roi_size=(64, 256, 256)
    )
    
    data_module = BrainHemorrhageDataModule(
        config=loader_config,
        transform_factory=transform_factory
    )
    
    # train_ds = data_module.train_dataset
    
    out_dir = Path("outputs/audits")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    num_samples = 1000
    results = []
    
    total_sampled = 0
    positive_patches = 0
    background_patches = 0
    
    train_loader = data_module.build_train_loader()
    pbar = tqdm(total=num_samples)
    loader_iter = iter(train_loader)
    
    while total_sampled < num_samples:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
            
        for i in range(batch["image"].shape[0]):
            if total_sampled >= num_samples:
                break
                
            img = batch["image"][i]
            mask = batch["mask"][i]
            
            fg_voxels = int((mask > 0).sum())
            total_voxels = mask.numel()
            fg_pct = (fg_voxels / total_voxels) * 100.0
            
            is_lesion = fg_voxels > 0
            if is_lesion:
                positive_patches += 1
                indices = torch.nonzero(mask > 0)
                centroid = indices.float().mean(dim=0).cpu().numpy()
                center = np.array(mask.shape) / 2.0
                dist = np.linalg.norm(centroid - center)
                min_vals = indices.min(dim=0).values.cpu().numpy()
                max_vals = indices.max(dim=0).values.cpu().numpy()
                bbox_size = (max_vals - min_vals).prod()
            else:
                background_patches += 1
                dist = None
                bbox_size = 0
                
            meta = batch.get("image_meta_dict", {})
            file_path = "Unknown"
            if "filename_or_obj" in meta:
                if isinstance(meta["filename_or_obj"], list):
                    file_path = meta["filename_or_obj"][i]
                else:
                    file_path = meta["filename_or_obj"]
                    
            patient_id = Path(file_path).parent.name if file_path != "Unknown" else "Unknown"
            dataset_name = Path(file_path).parent.parent.name if file_path != "Unknown" else "Unknown"
            
            results.append({
                "patient_id": patient_id,
                "dataset": dataset_name,
                "is_lesion": is_lesion,
                "fg_voxels": fg_voxels,
                "fg_pct": fg_pct,
                "bbox_size": bbox_size,
                "dist_to_center": dist
            })
            
            total_sampled += 1
            pbar.update(1)

    pbar.close()
    
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "crop_statistics.csv", index=False)
    
    pos_ratio = positive_patches / total_sampled if total_sampled > 0 else 0
    
    with open(out_dir / "Crop_Sampling_Report.md", "w") as f:
        f.write("# Crop Sampling Strategy Audit\n\n")
        f.write(f"- **Total Sampled**: {total_sampled}\n")
        f.write(f"- **Positive Patches**: {positive_patches}\n")
        f.write(f"- **Background-only Patches**: {background_patches}\n")
        f.write(f"- **Positive Sampling Ratio**: {pos_ratio:.4f}\n\n")
        f.write("### Pos:Neg Configuration Check\n")
        f.write("The transform uses `pos=1, neg=1` (50% target). ")
        f.write(f"Actual achieved: {pos_ratio*100:.2f}%. ")
        if pos_ratio < 0.3:
            f.write("Sampler is starving. Lesions might be too small or preprocessing is destroying them.\n")
        else:
            f.write("Sampler is functioning close to requested ratio.\n")
            
    plt.figure(figsize=(10, 6))
    pos_df = df[df["is_lesion"] == True]
    if not pos_df.empty:
        plt.scatter(pos_df["dist_to_center"], pos_df["bbox_size"], alpha=0.5)
        plt.xlabel("Distance from Crop Center to Lesion Centroid (Voxels)")
        plt.ylabel("Bounding Box Size (Voxels)")
        plt.title("Lesion Centering vs Size in Positive Crops")
        plt.grid(True)
        plt.savefig(out_dir / "crop_distribution.png")
        
    print("Crop Audit Complete! Saved to outputs/audits/")

if __name__ == "__main__":
    main()
