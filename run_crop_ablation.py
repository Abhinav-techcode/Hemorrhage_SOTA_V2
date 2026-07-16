import os
import gc
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import yaml
from datasets.dataloader import BrainHemorrhageDataModule, DataLoaderConfig
from datasets.transforms import TransformFactory

def run_ablation(roi_size):
    print(f"\n--- Testing Crop Size: {roi_size} ---")
    
    config_dir = Path("configs")
    with open(config_dir / "dataset.yaml", "r") as f:
        dataset_cfg = yaml.safe_load(f)
    transform_factory = TransformFactory(config_dir / "augmentation.yaml")
    
    loader_config = DataLoaderConfig(
        dataset_config=dataset_cfg,
        batch_size=4,
        num_workers=4,
        roi_size=roi_size
    )
    
    data_module = BrainHemorrhageDataModule(
        config=loader_config,
        transform_factory=transform_factory
    )
    
    train_loader = data_module.build_train_loader()
    
    num_samples = 200
    total_sampled = 0
    fg_pcts = []
    positive_patches = 0
    
    # We will simulate memory usage by pushing a batch to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_mem_allocated = 0
    
    loader_iter = iter(train_loader)
    
    pbar = tqdm(total=num_samples, desc=f"Evaluating {roi_size}")
    while total_sampled < num_samples:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
            
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        img = batch["image"].to(device)
        mask = batch["mask"].to(device)
        
        if torch.cuda.is_available():
            max_mem_allocated = max(max_mem_allocated, torch.cuda.max_memory_allocated() / (1024**3))
            
        for i in range(mask.shape[0]):
            if total_sampled >= num_samples:
                break
                
            m = mask[i]
            fg_voxels = int((m > 0).sum())
            total_voxels = m.numel()
            
            fg_pct = (fg_voxels / total_voxels) * 100.0
            fg_pcts.append(fg_pct)
            
            if fg_voxels > 0:
                positive_patches += 1
                
            total_sampled += 1
            pbar.update(1)
            
        del img, mask, batch
        
    pbar.close()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    mean_occ = np.mean(fg_pcts)
    median_occ = np.median(fg_pcts)
    pos_pct = (positive_patches / total_sampled) * 100.0
    
    return {
        "Crop Size": str(roi_size),
        "Mean Occupancy (%)": mean_occ,
        "Median Occupancy (%)": median_occ,
        "Positive Crops (%)": pos_pct,
        "GPU Mem (GB)": max_mem_allocated
    }

def main():
    print("Starting Crop Size Ablation...")
    
    sizes_to_test = [
        (64, 256, 256),
        (64, 192, 192),
        (64, 160, 160),
        (64, 128, 128)
    ]
    
    results = []
    
    for size in sizes_to_test:
        res = run_ablation(size)
        results.append(res)
        print(res)
        
    df = pd.DataFrame(results)
    
    out_dir = Path("outputs/audits")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "crop_ablation_results.csv", index=False)
    
    print("\nCrop Size Ablation Complete!")
    print(df.to_markdown())

if __name__ == "__main__":
    main()
