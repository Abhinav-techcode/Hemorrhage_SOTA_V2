import os
import torch
import numpy as np
from pathlib import Path
from rich.progress import track
from rich.console import Console
from rich.table import Table
from datasets.dataloader import BrainHemorrhageDataModule, DataLoaderConfig
from datasets.transforms import TransformFactory
from training.train import load_all_configs
from datasets.dataset import BrainHemorrhageDataset
import json

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped

def run_dataset_audit():
    console = Console()
    console.print("[bold cyan]Dataset & Preprocessing Audit: Phase 4.2[/]")
    
    config_dir = Path("configs")
    configs = load_all_configs(config_dir)
    transform_factory = TransformFactory("configs/augmentation.yaml")
    
    # 1. Audit Raw Dataset Volumes (Before RandCrop)
    raw_transforms = Compose([
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        EnsureTyped(keys=["image", "mask"], dtype=torch.float32)
    ])
    
    dataset = BrainHemorrhageDataset(mode="train", config=configs["dataset"], transform=raw_transforms)
    
    total_volumes = len(dataset)
    positive_volumes = 0
    empty_volumes = 0
    lesion_sizes = []
    
    console.print(f"Auditing raw volumes ({total_volumes} total)...")
    
    audit_limit = min(total_volumes, 500)
    for i in track(range(audit_limit), description="Loading raw volumes..."):
        try:
            data = dataset[i]
            mask = data["mask"]
            fg_voxels = int(mask.sum().item())
            if fg_voxels > 0:
                positive_volumes += 1
                lesion_sizes.append(fg_voxels)
            else:
                empty_volumes += 1
        except Exception as e:
            console.print(f"Failed loading index {i}: {e}")
            
    # 2. Audit Dataloader Crops (After RandCrop)
    loader_cfg = DataLoaderConfig(
        batch_size=1, num_workers=4, pin_memory=False, persistent_workers=False,
        prefetch_factor=2, drop_last=False, seed=42, dataset_config=configs["dataset"]
    )
    data_module = BrainHemorrhageDataModule(config=loader_cfg, transform_factory=transform_factory)
    train_loader = data_module.build_train_loader()
    
    positive_crops = 0
    background_crops = 0
    
    console.print("Auditing 100 random crops from train dataloader...")
    for i, batch in enumerate(track(train_loader, total=100, description="Sampling crops...")):
        if i >= 100: break
        mask = batch["mask"]
        if mask.sum().item() > 0:
            positive_crops += 1
        else:
            background_crops += 1
            
    # Report
    avg_lesion = np.mean(lesion_sizes) if lesion_sizes else 0
    med_lesion = np.median(lesion_sizes) if lesion_sizes else 0
    max_lesion = np.max(lesion_sizes) if lesion_sizes else 0
    min_lesion = np.min(lesion_sizes) if lesion_sizes else 0
    
    table = Table(title="Dataset & Preprocessing Audit Report")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Total Volumes Audited", str(audit_limit))
    table.add_row("Positive Volumes", str(positive_volumes))
    table.add_row("Empty Volumes", str(empty_volumes))
    table.add_row("Positive Crops Sampled", str(positive_crops))
    table.add_row("Background-only Crops Sampled", str(background_crops))
    table.add_row("Average Lesion Voxels", f"{avg_lesion:.2f}")
    table.add_row("Median Lesion Voxels", f"{med_lesion:.2f}")
    table.add_row("Largest Lesion Voxels", str(max_lesion))
    table.add_row("Smallest Lesion Voxels", str(min_lesion))
    
    console.print(table)

if __name__ == "__main__":
    run_dataset_audit()
