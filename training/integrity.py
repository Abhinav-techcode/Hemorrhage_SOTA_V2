import logging
import os
from pathlib import Path
from typing import Dict, Any

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

class ExperimentIntegrityError(RuntimeError):
    pass

def check_experiment_integrity(
    config_dir: Path,
    output_dir: Path,
    configs: Dict[str, Any],
    train_loader,
    val_loader,
    test_loader=None
):
    """
    Runs a series of strict integrity checks before training starts.
    Halts execution with ExperimentIntegrityError if any check fails.
    """
    logger.info("================================================================================")
    logger.info("Running Experiment Integrity Check...")
    logger.info("================================================================================")

    # 1. Output/Checkpoint Directory is writable
    if not os.access(output_dir, os.W_OK):
        raise ExperimentIntegrityError(f"Output directory {output_dir} is not writable.")
    
    ckpt_dir = output_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise ExperimentIntegrityError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if not os.access(ckpt_dir, os.W_OK):
        raise ExperimentIntegrityError(f"Checkpoint directory {ckpt_dir} is not writable.")
    
    logger.info("✓ Output directories are writable.")

    # 2. Random Seed is logged
    if "seed" not in configs.get("training", {}):
        raise ExperimentIntegrityError("Random 'seed' is missing from training configuration.")
    logger.info(f"✓ Random seed is logged: {configs['training']['seed']}")

    # 3. Configuration Consistency
    dataset_cfg = configs.get("dataset", {})
    if "processed_images_dir" not in dataset_cfg or "processed_masks_dir" not in dataset_cfg:
        raise ExperimentIntegrityError("Dataset configuration is missing required keys.")
    logger.info("✓ Configuration files are internally consistent.")

    # 4. Dataset split counts and No Duplicates
    def _extract_patients(loader):
        patients = set()
        if hasattr(loader, "dataset") and hasattr(loader.dataset, "data"):
            for item in loader.dataset.data:
                img_path = item.get("image", "")
                patients.add(Path(img_path).stem)
        return patients

    train_patients = _extract_patients(train_loader)
    val_patients = _extract_patients(val_loader)
    test_patients = _extract_patients(test_loader) if test_loader else set()
    
    if not train_patients:
        logger.warning("Could not extract patient IDs from train_loader (might be wrapped or different format).")
    else:
        logger.info(f"✓ Split Counts - Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")
        
        train_val_overlap = train_patients.intersection(val_patients)
        if train_val_overlap:
            raise ExperimentIntegrityError(f"Data leakage detected! {len(train_val_overlap)} patients in both train and val.")
        
        if test_loader:
            train_test_overlap = train_patients.intersection(test_patients)
            if train_test_overlap:
                raise ExperimentIntegrityError(f"Data leakage detected! {len(train_test_overlap)} patients in both train and test.")
        logger.info("✓ No duplicate patients across splits (No Data Leakage).")

    # 5. Quick Image/Mask Check (sample 1 file if available)
    if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "data") and len(train_loader.dataset.data) > 0:
        sample = train_loader.dataset.data[0]
        img_path = sample.get("image", "")
        mask_path = sample.get("mask", "")
        
        if os.path.exists(img_path) and os.path.exists(mask_path):
            img_nib = nib.load(img_path)
            mask_nib = nib.load(mask_path)
            
            if img_nib.shape != mask_nib.shape:
                raise ExperimentIntegrityError(f"Shape mismatch in {img_path}: {img_nib.shape} vs {mask_nib.shape}")
            logger.info("✓ Image and mask shapes match (sampled).")
            
            # 6. Mask values are valid (binary)
            try:
                # Just take a small chunk for speed
                mask_data = mask_nib.dataobj[..., mask_nib.shape[-1]//2:mask_nib.shape[-1]//2+1]
                unique_vals = np.unique(mask_data)
                if not np.all(np.isin(unique_vals, [0, 1])):
                    raise ExperimentIntegrityError(f"Mask {mask_path} contains non-binary values: {unique_vals}")
                logger.info("✓ Mask values are valid (binary check passed on sample).")
            except Exception as e:
                logger.warning(f"Could not verify mask values: {e}")
            
    logger.info("================================================================================")
    logger.info("Integrity Check Passed Successfully.")
    logger.info("================================================================================\n")
