import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

import yaml

def process_file(mask_path):
    # Try to load without monai transforms to see the raw state (or preprocessed state, depending on where we look)
    # The dataset uses a preprocessed directory or raw directory.
    # We will use the raw dataset splits.
    try:
        mask = nib.load(mask_path).get_fdata()
        total_voxels = mask.size
        fg_voxels = int((mask > 0).sum())
        fg_pct = (fg_voxels / total_voxels) * 100.0
        
        if fg_voxels > 0:
            indices = np.argwhere(mask > 0)
            min_vals = indices.min(axis=0)
            max_vals = indices.max(axis=0)
            bbox_size = (max_vals - min_vals).prod()
            thickness = max_vals[0] - min_vals[0] if len(mask.shape) == 3 else 0
        else:
            bbox_size = 0
            thickness = 0
            
        return {
            "has_lesion": fg_voxels > 0,
            "total_voxels": total_voxels,
            "fg_voxels": fg_voxels,
            "fg_pct": fg_pct,
            "bbox_size": bbox_size,
            "thickness": thickness
        }
    except Exception as e:
        return None

def main():
    print("Starting Lesion Size Distribution Audit...")
    
    config_dir = Path("configs")
    with open(config_dir / "dataset.yaml", "r") as f:
        dataset_cfg = yaml.safe_load(f)
    
    split_file = Path(dataset_cfg["split_file"])
    masks_dir = Path(dataset_cfg["processed_masks_dir"])
    
    with open(split_file, "r") as f:
        splits = json.load(f)
        
    train_files = splits.get("train", [])
    
    out_dir = Path("outputs/audits")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for item in tqdm(train_files, desc="Scanning Full Dataset"):
        # The split contains filenames directly
        mask_path = masks_dir / item
        res = process_file(mask_path)
        if res:
            res["patient"] = item.split(".")[0]
            results.append(res)
            
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "lesion_distribution.csv", index=False)
    
    pos_df = df[df["has_lesion"] == True]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].hist(pos_df["fg_voxels"], bins=50, color='blue', alpha=0.7)
    axes[0, 0].set_title("Lesion Volume (Voxels)")
    
    axes[0, 1].hist(pos_df["thickness"], bins=50, color='red', alpha=0.7)
    axes[0, 1].set_title("Lesion Thickness (Slices)")
    
    axes[1, 0].hist(pos_df["bbox_size"], bins=50, color='green', alpha=0.7)
    axes[1, 0].set_title("Bounding Box Size")
    
    axes[1, 1].hist(pos_df["fg_pct"], bins=50, color='purple', alpha=0.7)
    axes[1, 1].set_title("Lesion Occupancy (%) Before Cropping")
    
    plt.tight_layout()
    plt.savefig(out_dir / "lesion_distribution_histograms.png")
    
    print("Lesion Size Distribution Audit Complete! Saved to outputs/audits/")

if __name__ == "__main__":
    main()
