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
    
    # New Metrics
    dataset_stats = {}
    lesion_slices = []
    foreground_ratios = []
    
    console.print(f"Auditing raw volumes ({total_volumes} total)...")
    
    audit_limit = min(total_volumes, 500)
    for i in track(range(audit_limit), description="Loading raw volumes..."):
        try:
            data = dataset[i]
            mask = data["mask"]
            # Assume data dictionary has 'filename_or_obj' to extract dataset source
            source = "Unknown"
            if "filename_or_obj" in data:
                path = data["filename_or_obj"]
                if "BHSD" in path: source = "BHSD"
                elif "CQ500" in path: source = "CQ500"
                elif "PhysioNet" in path: source = "PhysioNet"
            
            if source not in dataset_stats:
                dataset_stats[source] = {"positive": 0, "empty": 0, "lesions": []}
                
            fg_voxels = int(mask.sum().item())
            if fg_voxels > 0:
                positive_volumes += 1
                lesion_sizes.append(fg_voxels)
                dataset_stats[source]["positive"] += 1
                dataset_stats[source]["lesions"].append(fg_voxels)
                # Find which slices have lesions (Z axis)
                # mask shape is usually [1, D, H, W]
                z_sums = mask[0].sum(dim=(1, 2))
                active_slices = (z_sums > 0).sum().item()
                lesion_slices.append(active_slices)
                
            else:
                empty_volumes += 1
                dataset_stats[source]["empty"] += 1
                
            foreground_ratios.append(float(fg_voxels / mask.numel()))
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
    cropped_away_lesions = 0
    total_crops_audited = min(100, len(train_loader))
    
    console.print(f"Auditing {total_crops_audited} random crops from train dataloader...")
    for i, batch in enumerate(track(train_loader, total=total_crops_audited, description="Sampling crops...")):
        if i >= total_crops_audited: break
        mask = batch["mask"]
        if mask.sum().item() > 0:
            positive_crops += 1
        else:
            background_crops += 1
            # We assume if the original volume was positive but crop is empty, it was cropped away
            # In a real check we'd correlate indices. For now we use global ratio
            pass
            
    # Estimate cropped away lesions: if dataset has P% positive but crops have C% positive
    orig_pos_ratio = positive_volumes / max(1, audit_limit)
    crop_pos_ratio = positive_crops / max(1, total_crops_audited)
    if orig_pos_ratio > crop_pos_ratio:
        cropped_away_pct = ((orig_pos_ratio - crop_pos_ratio) / orig_pos_ratio) * 100
    else:
        cropped_away_pct = 0.0

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
    table.add_row("Estimated Cropped-Away Lesions", f"{cropped_away_pct:.1f}%")
    table.add_row("Crop Success Rate", f"{(positive_crops / max(1, total_crops_audited)) * 100:.1f}%")
    
    console.print(table)
    
    # Save Output
    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "Dataset_Audit.md"
    
    with open(report_path, "w") as f:
        f.write("# Dataset & Preprocessing Audit\n\n")
        f.write("## Overall Statistics\n")
        f.write(f"- **Total Volumes Audited**: {audit_limit}\n")
        f.write(f"- **Positive Volumes**: {positive_volumes}\n")
        f.write(f"- **Empty Volumes**: {empty_volumes}\n")
        f.write(f"- **Average Lesion Voxels**: {avg_lesion:.2f}\n")
        f.write(f"- **Median Lesion Voxels**: {med_lesion:.2f}\n")
        f.write(f"- **Estimated Cropped-Away Lesions**: {cropped_away_pct:.1f}%\n")
        f.write(f"- **Crop Success Rate**: {(positive_crops / max(1, total_crops_audited)) * 100:.1f}%\n")
        
        f.write("\n## Dataset-Wise Statistics\n")
        for src, stats in dataset_stats.items():
            f.write(f"### {src}\n")
            f.write(f"- Positive: {stats['positive']}\n")
            f.write(f"- Empty: {stats['empty']}\n")
            s_lesions = stats['lesions']
            if s_lesions:
                f.write(f"- Avg Lesion Size: {np.mean(s_lesions):.2f}\n")
            
    # Attempt to plot histograms (if matplotlib available)
    try:
        import matplotlib.pyplot as plt
        plot_dir = Path("outputs/preprocessing_audit")
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure()
        plt.hist(lesion_sizes, bins=50)
        plt.title("Lesion Size Histogram")
        plt.savefig(plot_dir / "lesion_size_histogram.png")
        
        plt.figure()
        plt.hist(lesion_slices, bins=50)
        plt.title("Lesion Slice Distribution")
        plt.savefig(plot_dir / "lesion_slice_distribution.png")
        
        plt.figure()
        plt.hist(foreground_ratios, bins=50)
        plt.title("Foreground Ratio Histogram")
        plt.savefig(plot_dir / "foreground_ratio_histogram.png")
        
        console.print("[green]Histograms saved to outputs/preprocessing_audit/[/green]")
    except ImportError:
        console.print("[yellow]matplotlib not found. Skipping histograms.[/yellow]")

if __name__ == "__main__":
    run_dataset_audit()
