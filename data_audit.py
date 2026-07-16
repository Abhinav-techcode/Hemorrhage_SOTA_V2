"""
data_audit.py

Scientific audit script to evaluate the Hemorrhage_SOTA_V2 data pipeline.
Computes dataset statistics, bounding box metrics, sampling efficiency,
lesion occupancy ratios, and visualizes RandCrop behavior.
"""
import argparse
import logging
import json
from pathlib import Path
import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

from datasets.dataloader import BrainHemorrhageDataModule, DataLoaderConfig
from datasets.transforms import TransformFactory
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

def compute_bounding_box(mask):
    """Computes 3D bounding box of non-zero elements in a mask."""
    indices = np.where(mask > 0)
    if len(indices[0]) == 0:
        return None
    return tuple((np.min(idx), np.max(idx)) for idx in indices)

def run_audit(config_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading configs...")
    with open(config_dir / "dataset.yaml", "r") as f:
        dataset_cfg = yaml.safe_load(f)
    
    logger.info("Building TransformFactory...")
    transform_factory = TransformFactory(config_dir / "augmentation.yaml")
    
    logger.info("Building Dataloaders (Audit Mode)...")
    loader_config = DataLoaderConfig(
        dataset_config=dataset_cfg,
        batch_size=1,
        num_workers=4
    )
    
    data_module = BrainHemorrhageDataModule(
        config=loader_config,
        transform_factory=transform_factory
    )
    
    train_loader = data_module.build_train_loader()
    
    audit_results = {
        "dataset_stats": {
            "total_patients": 0,
            "positive_patients": 0,
            "empty_patients": 0,
            "lesion_voxels": []
        },
        "sampling_stats": {
            "total_crops": 0,
            "positive_crops": 0,
            "background_only_crops": 0,
            "lesion_occupancy_ratios": []
        }
    }
    
    # We will run 1 simulated epoch
    logger.info(f"Simulating 1 Epoch ({len(train_loader)} crops)...")
    
    saved_crops = 0
    crop_dir = output_dir / "crops"
    crop_dir.mkdir(exist_ok=True)
    
    for i, batch in enumerate(tqdm(train_loader, desc="Auditing Training Crops")):
        images = batch["image"]
        masks = batch["mask"]
        
        for b in range(images.shape[0]):
            img = images[b, 0].numpy()
            mask = masks[b, 0].numpy()
            
            crop_voxels = mask.size
            lesion_voxels = int(np.sum(mask > 0))
            
            audit_results["sampling_stats"]["total_crops"] += 1
            if lesion_voxels > 0:
                audit_results["sampling_stats"]["positive_crops"] += 1
                occupancy = lesion_voxels / crop_voxels
                audit_results["sampling_stats"]["lesion_occupancy_ratios"].append(occupancy)
            else:
                audit_results["sampling_stats"]["background_only_crops"] += 1
                
            # Save 10 random crops for visual verification
            if saved_crops < 10 and lesion_voxels > 0 and np.random.rand() < 0.2:
                img_nii = nib.Nifti1Image(img, np.eye(4))
                mask_nii = nib.Nifti1Image(mask.astype(np.uint8), np.eye(4))
                
                nib.save(img_nii, crop_dir / f"crop_{saved_crops}_img.nii.gz")
                nib.save(mask_nii, crop_dir / f"crop_{saved_crops}_mask.nii.gz")
                saved_crops += 1

    # Post-process stats
    l_voxels = audit_results["sampling_stats"]["lesion_occupancy_ratios"]
    if l_voxels:
        audit_results["sampling_stats"]["mean_occupancy"] = float(np.mean(l_voxels))
        audit_results["sampling_stats"]["median_occupancy"] = float(np.median(l_voxels))
        audit_results["sampling_stats"]["max_occupancy"] = float(np.max(l_voxels))
        audit_results["sampling_stats"]["min_occupancy"] = float(np.min(l_voxels))
    
    with open(output_dir / "Data_Audit_Report.json", "w") as f:
        json.dump(audit_results, f, indent=4)
        
    logger.info("Data Audit Complete. Results saved to " + str(output_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="configs")
    parser.add_argument("--output_dir", type=str, default="outputs/data_audit")
    args = parser.parse_args()
    
    run_audit(Path(args.config_dir), Path(args.output_dir))
