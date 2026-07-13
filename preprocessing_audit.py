import os
import torch
import numpy as np
from pathlib import Path
from rich.progress import track
from datasets.dataloader import BrainHemorrhageDataModule, DataLoaderConfig
from datasets.transforms import TransformFactory
from training.train import load_all_configs
import json

def run_preprocessing_audit():
    print("="*50)
    print("Preprocessing Audit: Phase 1")
    print("="*50)
    
    config_dir = Path("configs")
    configs = load_all_configs(config_dir)
    
    transform_factory = TransformFactory("configs/augmentation.yaml")
    
    loader_cfg = DataLoaderConfig(
        batch_size=1,
        num_workers=4,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        drop_last=False,
        seed=42,
        dataset_config=configs["dataset"],
    )
    
    data_module = BrainHemorrhageDataModule(config=loader_cfg, transform_factory=transform_factory)
    
    # We will audit the train dataloader which has the RandCropByPosNegLabeld 
    train_loader = data_module.build_train_loader()
    
    audit_dir = Path("outputs/preprocessing_audit")
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    stats = []
    
    print("Auditing 100 random samples...")
    # Iterate through the dataloader up to 100 times
    for i, batch in enumerate(track(train_loader, total=100)):
        if i >= 100:
            break
            
        images = batch["image"]
        masks = batch["mask"]
        
        # Calculate stats for the single patch
        patch_lesion_voxels = int(masks.sum().item())
        patch_total_voxels = int(masks.numel())
        
        stats.append({
            "sample_index": i,
            "patch_lesion_voxels": patch_lesion_voxels,
            "patch_total_voxels": patch_total_voxels,
        })
        
    # Aggregate statistics
    total_lesions_captured = sum(s["patch_lesion_voxels"] for s in stats)
    
    summary = {
        "total_samples_audited": len(stats),
        "total_lesion_voxels_captured": total_lesions_captured,
        "avg_lesion_voxels_per_patch": total_lesions_captured / len(stats) if stats else 0
    }
    
    with open(audit_dir / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\nAudit Complete!")
    print(f"Total Lesion Voxels Captured in patches: {total_lesions_captured}")
    print(f"Average Lesion Voxels Per Patch: {summary['avg_lesion_voxels_per_patch']:.2f}")

if __name__ == "__main__":
    run_preprocessing_audit()
