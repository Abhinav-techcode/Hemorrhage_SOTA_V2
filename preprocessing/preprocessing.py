"""
=====================================================================
Hemorrhage_SOTA_V2
Research-Grade Medical Image Preprocessing Pipeline

Author  : Abhinav Gupta
Purpose : Preprocessing for 3D Intracranial Hemorrhage Segmentation

Datasets
---------
1. BHSD
2. CQ500
3. PhysioNet

Output
------
Processed Dataset ready for 3D Segmentation Models
(Attention U-Net, TransUNet, MedNeXt, nnUNet, etc.)

Note
----
Offline augmentation has been removed from this pipeline to maximize
storage efficiency and ensure reproducibility. Apply TorchIO or MONAI 
augmentations ONLINE inside the PyTorch Dataset/DataLoader during training.
=====================================================================
"""

# =====================================================================
# IMPORTS
# =====================================================================

import gc
import os
import sys
import json
import logging
import random
import time
import warnings
import yaml
import hashlib
import re
import traceback
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.exposure import match_histograms
from sklearn.model_selection import GroupKFold, train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =====================================================================
# PIPELINE METADATA & ENVIRONMENT
# =====================================================================

PIPELINE_VERSION = "3.0.0"

def get_git_commit() -> str:
    """Safely attempts to retrieve the current Git commit hash."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], 
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
    except Exception:
        return "Unknown_Or_Not_In_Git"

def get_gpu_name() -> str:
    """Safely attempts to retrieve the GPU name using nvidia-smi."""
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
            stderr=subprocess.DEVNULL
        )
        return output.decode('utf-8').strip().split('\n')[0]
    except Exception:
        return "No GPU Detected / nvidia-smi failed"

GIT_COMMIT = get_git_commit()
MACHINE_NODE = platform.node()
SYSTEM_OS = platform.platform()
PYTHON_VERSION = sys.version.split()[0]
SITK_VERSION = sitk.Version.VersionString()
NUMPY_VERSION = np.__version__
GPU_NAME = get_gpu_name()
CPU_COUNT = os.cpu_count() or 1

# =====================================================================
# RANDOM SEED
# =====================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# =====================================================================
# PROJECT PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
MASTER_DATASET = DATASET_ROOT / "Merged_Dataset"
IMAGE_DIR = MASTER_DATASET / "images"
MASK_DIR = MASTER_DATASET / "masks"
CONFIG_FILE = PROJECT_ROOT / "configs" / "preprocessing.yaml"

# =====================================================================
# OUTPUT
# =====================================================================

OUTPUT_ROOT = PROJECT_ROOT / "processed"
OUTPUT_IMAGE_DIR = OUTPUT_ROOT / "images"
OUTPUT_MASK_DIR = OUTPUT_ROOT / "masks"
OUTPUT_VIS_DIR = OUTPUT_ROOT / "visualization"
OUTPUT_REPORT_DIR = OUTPUT_ROOT / "reports"
OUTPUT_SPLIT_DIR = OUTPUT_ROOT / "splits"

for d in [OUTPUT_IMAGE_DIR, OUTPUT_MASK_DIR, OUTPUT_VIS_DIR, 
          OUTPUT_REPORT_DIR, OUTPUT_SPLIT_DIR, CONFIG_FILE.parent]:
    d.mkdir(parents=True, exist_ok=True)

# =====================================================================
# REPORT FILES
# =====================================================================

METADATA_CSV = OUTPUT_REPORT_DIR / "Processed_Metadata.csv"
QUALITY_CSV = OUTPUT_REPORT_DIR / "Quality_Report.csv"
PREPROCESSING_LOG = OUTPUT_REPORT_DIR / "preprocessing.log"
FAILED_CASES = OUTPUT_REPORT_DIR / "Failed_Cases.csv"
MANIFEST_JSON = OUTPUT_REPORT_DIR / "pipeline_manifest.json"

# =====================================================================
# PREPROCESSING PARAMETERS & CONFIG
# =====================================================================

DEFAULT_CONFIG = {
    "orientation": "RAS",
    "target_spacing": [0.5, 0.5, 5.0],
    "multi_window": True,
    "windows": {
        "brain": {"center": 40, "width": 80},
        "bone": {"center": 600, "width": 2000},
        "subdural": {"center": 80, "width": 200}
    },
    "normalization": "minmax",
    "remove_small_components": True,
    "min_component_size": 100,
    "fill_holes": True,
    "verify_dataset": True,
    "save_visualization": True,
    "save_metadata": True,
    "target_depth": 128,
    "apply_histogram_matching": False,
    "histogram_reference_path": None,
    "geometry_atol": 1e-4,
    "num_workers": 4,
    "overwrite_existing": True,
    "splits": {
        "train": 0.70,
        "val": 0.15,
        "test": 0.15,
        "k_folds": 5
    }
}

if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "r") as f:
        CONFIG = yaml.safe_load(f)
else:
    CONFIG = DEFAULT_CONFIG
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(CONFIG, f, default_flow_style=False)

# Compute Configuration Hash for Metadata Tracking
CONFIG_HASH = hashlib.md5(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()

# Dtypes
DTYPE_IMAGE = np.float32
DTYPE_MASK = np.uint8

# =====================================================================
# LOGGER
# =====================================================================

logger = logging.getLogger("Preprocessing")
logger.setLevel(logging.INFO)
logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

file_handler = logging.FileHandler(PREPROCESSING_LOG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info(f"Initialized Pipeline Version: {PIPELINE_VERSION}")
logger.info(f"Loaded Configuration Hash: {CONFIG_HASH}")
logger.info(f"Target Depth: {CONFIG.get('target_depth', 128)}")
logger.info(f"Multi-Window: {CONFIG.get('multi_window', True)}")
logger.info(f"Environment: Python {PYTHON_VERSION} | SITK {SITK_VERSION} | CPU {CPU_COUNT}")

# =====================================================================
# DATASET
# =====================================================================

IMAGE_FILES = sorted(IMAGE_DIR.glob("*.nii.gz"))
MASK_FILES = sorted(MASK_DIR.glob("*.nii.gz"))

assert len(IMAGE_FILES) > 0, "No images found."
assert len(IMAGE_FILES) == len(MASK_FILES), "Image-mask count mismatch."

_image_names = {p.name for p in IMAGE_FILES}
_mask_names = {p.name for p in MASK_FILES}
_missing_masks = sorted(_image_names - _mask_names)
_missing_images = sorted(_mask_names - _image_names)

if _missing_masks or _missing_images:
    raise FileNotFoundError(
        f"Image/mask filename mismatch detected.\n"
        f"Images without a matching mask ({len(_missing_masks)}): {_missing_masks[:10]}\n"
        f"Masks without a matching image ({len(_missing_images)}): {_missing_images[:10]}"
    )

TOTAL_CASES = len(IMAGE_FILES)
logger.info(f"Total verified cases ready for processing: {TOTAL_CASES}")

# =====================================================================
# MEMORY CLEANER
# =====================================================================

def clear_memory() -> None:
    """Forces garbage collection to free unused memory."""
    gc.collect()

# =====================================================================
# PART 2A-1
# MEDICAL IMAGE CORE
# =====================================================================

def _orthonormalize_direction(direction_flat: Tuple[float, ...], dim: int = 3) -> Tuple[float, ...]:
    """
    Orthonormalize a direction cosine matrix.
    """
    matrix = np.array(direction_flat, dtype=np.float64).reshape(dim, dim)
    u, _, vt = np.linalg.svd(matrix)
    orthonormal = u @ vt
    if np.linalg.det(orthonormal) * np.linalg.det(matrix) < 0:
        u[:, -1] *= -1
        orthonormal = u @ vt
    return tuple(orthonormal.flatten())

def _apply_orthonormal_direction(image: sitk.Image) -> sitk.Image:
    """Applies orthonormalized direction cosine matrix to a SimpleITK image."""
    fixed = sitk.Image(image)
    fixed.SetDirection(_orthonormalize_direction(image.GetDirection(), image.GetDimension()))
    return fixed

def load_case(image_path: Path, mask_path: Path) -> Tuple[sitk.Image, sitk.Image]:
    """Loads a CT image and its segmentation mask, applying orthonormal direction."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    try:
        image = sitk.ReadImage(str(image_path))
        mask = sitk.ReadImage(str(mask_path))
    except Exception as e:
        raise RuntimeError(f"Failed to read image/mask '{image_path}': {e}") from e

    image = _apply_orthonormal_direction(image)
    mask = _apply_orthonormal_direction(mask)
    return image, mask

def save_case(image: sitk.Image, mask: sitk.Image, image_path: Path, mask_path: Path) -> None:
    """Saves image and mask with orthonormal direction to disk."""
    image = _apply_orthonormal_direction(image)
    mask = _apply_orthonormal_direction(mask)
    sitk.WriteImage(image, str(image_path))
    sitk.WriteImage(mask, str(mask_path))

def orient_to_ras(image: sitk.Image, mask: sitk.Image) -> Tuple[sitk.Image, sitk.Image]:
    """Reorients image and mask to the configured anatomical orientation (default RAS)."""
    image = sitk.DICOMOrient(image, CONFIG["orientation"])
    mask = sitk.DICOMOrient(mask, CONFIG["orientation"])
    return image, mask

def resample(image: sitk.Image, spacing: List[float], interpolator: int) -> sitk.Image:
    """Resamples an image to the given spacing using the specified interpolator."""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        max(1, int(np.round(original_size[i] * original_spacing[i] / spacing[i])))
        for i in range(3)
    ]

    output_direction = _orthonormalize_direction(image.GetDirection(), image.GetDimension())
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(output_direction)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    
    return resampler.Execute(image)

def resample_ct(image: sitk.Image) -> sitk.Image:
    """Resamples CT image with linear interpolation."""
    return resample(image, CONFIG["target_spacing"], sitk.sitkLinear)

def resample_mask(mask: sitk.Image) -> sitk.Image:
    """Resamples segmentation mask with nearest neighbor interpolation."""
    return resample(mask, CONFIG["target_spacing"], sitk.sitkNearestNeighbor)

def sitk_to_numpy(image: sitk.Image) -> np.ndarray:
    """Converts a SimpleITK image to a NumPy array (dimensions [X, Y, Z] or [C, X, Y, Z])."""
    array = sitk.GetArrayFromImage(image)
    if array.ndim == 4:
        array = np.transpose(array, (3, 2, 1, 0))
    else:
        array = np.transpose(array, (2, 1, 0))
    return np.asarray(array)

def numpy_to_sitk(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    """
    Converts a NumPy array back to a SimpleITK image, inheriting geometry from a reference.

    Instead of using CopyInformation (which can raise a size mismatch error after padding/cropping),
    this function **always** manually sets Spacing, Origin, and Direction.
    """
    if array.ndim == 4:
        # (C, X, Y, Z) -> (Z, Y, X, C)
        transposed = np.transpose(array, (3, 2, 1, 0))
        image = sitk.GetImageFromArray(transposed, isVector=True)
    else:
        # (X, Y, Z) -> (Z, Y, X)
        transposed = np.transpose(array, (2, 1, 0))
        image = sitk.GetImageFromArray(transposed)

    # Always apply spatial metadata manually to avoid CopyInformation issues
    image.SetSpacing(reference.GetSpacing())
    image.SetOrigin(reference.GetOrigin())
    image.SetDirection(reference.GetDirection())

    image = _apply_orthonormal_direction(image)
    return image

def verify_geometry(image: sitk.Image, mask: sitk.Image, atol: Optional[float] = None) -> bool:
    """Verifies that the geometry of image and mask match (size, spacing, origin, direction)."""
    atol = atol or CONFIG.get("geometry_atol", 1e-4)
    if image.GetSize()[:3] != mask.GetSize()[:3]:
        raise ValueError(f"Size mismatch: image={image.GetSize()} mask={mask.GetSize()}")
    if not np.allclose(image.GetSpacing(), mask.GetSpacing(), atol=atol):
        raise ValueError(f"Spacing mismatch: image={image.GetSpacing()} mask={mask.GetSpacing()}")
    if not np.allclose(image.GetOrigin(), mask.GetOrigin(), atol=atol):
        raise ValueError(f"Origin mismatch: image={image.GetOrigin()} mask={mask.GetOrigin()}")
    if not np.allclose(image.GetDirection(), mask.GetDirection(), atol=atol):
        raise ValueError(f"Direction mismatch: image={image.GetDirection()} mask={mask.GetDirection()}")
    return True

# =====================================================================
# PART 2A-2
# INTENSITY PROCESSING + MASK PROCESSING
# =====================================================================

def apply_window(image: np.ndarray) -> np.ndarray:
    """Applies multi‑window or single‑window CT windowing."""
    if CONFIG.get("multi_window", False):
        channels = []
        for win_name in ["brain", "bone", "subdural"]:
            c = CONFIG["windows"][win_name]["center"]
            w = CONFIG["windows"][win_name]["width"]
            ch = np.clip(image, c - w / 2, c + w / 2)
            channels.append(ch)
        return np.stack(channels, axis=0)
    else:
        c = CONFIG["windows"]["brain"]["center"]
        w = CONFIG["windows"]["brain"]["width"]
        return np.clip(image, c - w / 2, c + w / 2)

def normalize(image: np.ndarray) -> np.ndarray:
    """Normalizes image using Z‑score or Min‑Max normalization."""
    image = image.astype(np.float32)
    if CONFIG["normalization"] == "zscore":
        axes = tuple(range(1, image.ndim)) if image.ndim == 4 else None
        mean = np.mean(image, axis=axes, keepdims=True)
        std = np.std(image, axis=axes, keepdims=True)
        return (image - mean) / (std + 1e-8)
    else:
        axes = tuple(range(1, image.ndim)) if image.ndim == 4 else None
        minimum = np.min(image, axis=axes, keepdims=True)
        maximum = np.max(image, axis=axes, keepdims=True)
        return (image - minimum) / (maximum - minimum + 1e-8)

def histogram_matching(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Matches the histogram of the image to a reference image."""
    if not CONFIG.get("apply_histogram_matching", False):
        return image
    
    if image.ndim == 4:
        for c in range(image.shape[0]):
            image[c] = match_histograms(image[c], reference[c])
        return image
    return match_histograms(image, reference)

def standardize_depth(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pads or crops the depth (Z‑axis) to the configured target_depth."""
    target_z = CONFIG.get("target_depth", None)
    if target_z is None:
        return image, mask
    
    current_z = image.shape[-1]
    if current_z == target_z:
        return image, mask
        
    if current_z < target_z:
        pad_front = (target_z - current_z) // 2
        pad_back = target_z - current_z - pad_front
        
        if image.ndim == 4:
            pad_img = ((0, 0), (0, 0), (0, 0), (pad_front, pad_back))
        else:
            pad_img = ((0, 0), (0, 0), (pad_front, pad_back))
            
        pad_msk = ((0, 0), (0, 0), (pad_front, pad_back))
        
        image = np.pad(image, pad_img, mode='constant', constant_values=0)
        mask = np.pad(mask, pad_msk, mode='constant', constant_values=0)
        
    else:
        coords = np.argwhere(mask > 0)
        if coords.size > 0:
            center_z = int(np.mean(coords[:, 2]))
        else:
            center_z = current_z // 2
            
        start_z = max(0, center_z - target_z // 2)
        end_z = start_z + target_z
        if end_z > current_z:
            end_z = current_z
            start_z = current_z - target_z
            
        if image.ndim == 4:
            image = image[..., start_z:end_z]
        else:
            image = image[:, :, start_z:end_z]
        mask = mask[:, :, start_z:end_z]
        
    return image, mask

def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Cleans the mask by removing small components and filling holes."""
    mask = (mask > 0).astype(np.uint8)
    if CONFIG["remove_small_components"]:
        sitk_mask = sitk.GetImageFromArray(np.transpose(mask, (2, 1, 0)))
        components = sitk.ConnectedComponent(sitk_mask)
        statistics = sitk.LabelShapeStatisticsImageFilter()
        statistics.Execute(components)
        cleaned = np.zeros_like(mask)
        for label in statistics.GetLabels():
            if statistics.GetNumberOfPixels(label) >= CONFIG["min_component_size"]:
                region = sitk.GetArrayFromImage(components == label)
                cleaned[np.transpose(region, (2, 1, 0)) > 0] = 1
        mask = cleaned.astype(np.uint8)
        
    if CONFIG["fill_holes"]:
        sitk_mask = sitk.GetImageFromArray(np.transpose(mask, (2, 1, 0)))
        sitk_mask = sitk.BinaryFillhole(sitk_mask)
        mask = np.transpose(sitk.GetArrayFromImage(sitk_mask), (2, 1, 0)).astype(np.uint8)
        
    return mask

# =====================================================================
# PART 2A-3
# LESION ANALYSIS + METADATA + VALIDATION
# =====================================================================

def get_bounding_box(mask: np.ndarray, coords: Optional[np.ndarray] = None) -> Tuple[int, ...]:
    """Returns (xmin, xmax, ymin, ymax, zmin, zmax) of the lesion."""
    if coords is None:
        coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return (-1, -1, -1, -1, -1, -1)
    xmin, ymin, zmin = coords.min(axis=0)
    xmax, ymax, zmax = coords.max(axis=0)
    return (int(xmin), int(xmax), int(ymin), int(ymax), int(zmin), int(zmax))

def get_centroid(mask: np.ndarray, coords: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
    """Returns centroid (x, y, z) of the lesion."""
    if coords is None:
        coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return (-1.0, -1.0, -1.0)
    center = coords.mean(axis=0)
    return (float(center[0]), float(center[1]), float(center[2]))

def compute_image_statistics(image: np.ndarray) -> Dict[str, float]:
    """Computes basic intensity statistics."""
    return {
        "Intensity_Min": float(image.min()),
        "Intensity_Max": float(image.max()),
        "Intensity_Mean": float(image.mean()),
        "Intensity_STD": float(image.std()),
        "Intensity_Median": float(np.median(image))
    }

def compute_mask_statistics(mask: np.ndarray, spacing: Tuple[float, ...]) -> Dict[str, Any]:
    """Computes lesion volume and foreground percentage."""
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    positive_voxels = int(mask.sum())
    return {
        "Positive_Voxels": positive_voxels,
        "Foreground_Percentage": (positive_voxels / mask.size) * 100,
        "Lesion_Volume_mm3": positive_voxels * voxel_volume
    }

def quality_check(image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """Performs final quality checks on the processed data."""
    return {
        "Shape_Match": image.shape[-3:] == mask.shape,
        "Binary_Mask": np.all(np.isin(np.unique(mask), [0, 1])),
        "Contains_NaN": np.isnan(image).any(),
        "Contains_Inf": np.isinf(image).any(),
        "Empty_Mask": mask.sum() == 0,
        "Image_Dtype": str(image.dtype),
        "Mask_Dtype": str(mask.dtype),
    }

def build_metadata(filename: str, dataset: str, image: np.ndarray, mask: np.ndarray, 
                   spacing: Tuple[float, ...], orig_shape: Tuple[int, ...], 
                   orig_spacing: Tuple[float, ...], orig_direction: Tuple[float, ...],
                   direction: Tuple[float, ...], orig_origin: Tuple[float, ...],
                   origin: Tuple[float, ...], interpolator: str,
                   processing_time_sec: Optional[float] = None) -> Dict[str, Any]:
    """
    Builds a comprehensive metadata dictionary for a processed case.
    """
    stats = compute_image_statistics(image)
    lesion = compute_mask_statistics(mask, spacing)
    coords = np.argwhere(mask > 0)
    bbox = get_bounding_box(mask, coords=coords)
    centroid = get_centroid(mask, coords=coords)
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    
    return {
        "Filename": filename,
        "Dataset": dataset,
        "Patient_ID": extract_patient_id(filename, dataset),
        
        # Pipeline Identity & Reproducibility Tracking
        "Pipeline_Version": PIPELINE_VERSION,
        "Config_Hash": CONFIG_HASH,
        "Git_Commit": GIT_COMMIT,
        "Random_Seed": SEED,
        "Processing_Timestamp": datetime.now().isoformat(),
        
        # Hardware & Software Dependencies
        "Machine_Node": MACHINE_NODE,
        "Python_Version": PYTHON_VERSION,
        "SimpleITK_Version": SITK_VERSION,
        "NumPy_Version": NUMPY_VERSION,
        
        # Configuration Variables used for this slice
        "Window_Mode": "Multi-Channel" if CONFIG.get("multi_window", False) else "Single-Channel",
        "Normalization": CONFIG["normalization"],
        "Target_Depth": CONFIG.get("target_depth", 128),
        "Interpolator_Used": interpolator,
        
        # Geometry Mapping (Original to Processed)
        "Original_Shape": str(orig_shape),
        "Original_Spacing": str(orig_spacing),
        "Original_Direction": str(orig_direction),
        "Original_Origin": str(orig_origin),
        "Processed_Shape": str(image.shape),
        "Processed_Spacing": str(spacing),
        "Processed_Direction": str(direction),
        "Processed_Origin": str(origin),
        
        # Split spatial parameters
        "Shape_X": image.shape[-3] if image.ndim == 4 else image.shape[0],
        "Shape_Y": image.shape[-2] if image.ndim == 4 else image.shape[1],
        "Shape_Z": image.shape[-1] if image.ndim == 4 else image.shape[2],
        "Spacing_X": spacing[0],
        "Spacing_Y": spacing[1],
        "Spacing_Z": spacing[2],
        "Voxel_Volume_mm3": voxel_volume,
        "Total_Voxels": int(image.size),
        
        **stats,
        **lesion,
        "BBox_X_Min": bbox[0], "BBox_X_Max": bbox[1],
        "BBox_Y_Min": bbox[2], "BBox_Y_Max": bbox[3],
        "BBox_Z_Min": bbox[4], "BBox_Z_Max": bbox[5],
        "Centroid_X": centroid[0], "Centroid_Y": centroid[1], "Centroid_Z": centroid[2],
        "Processing_Time_Sec": round(processing_time_sec, 4) if processing_time_sec else None
    }

def verify_case(image: np.ndarray, mask: np.ndarray) -> bool:
    """Verifies final processed arrays meet expectations (dtype, shape, no NaN/Inf)."""
    if np.isnan(image).any() or np.isinf(image).any():
        raise ValueError("Image contains NaN or Inf.")
    if image.shape[-3:] != mask.shape:
        raise ValueError(f"Shape mismatch: image={image.shape} mask={mask.shape}")
    if image.dtype != DTYPE_IMAGE:
        raise TypeError(f"Image dtype should be {DTYPE_IMAGE}, got {image.dtype}")
    if mask.dtype != DTYPE_MASK:
        raise TypeError(f"Mask dtype should be {DTYPE_MASK}, got {mask.dtype}")
    return True

# =====================================================================
# PART 2A-5
# COMPLETE PREPROCESSING PIPELINE (per case)
# =====================================================================

def preprocess_case(image_path: Path, mask_path: Path) -> Dict[str, Any]:
    """
    Runs the full preprocessing pipeline on a single case and returns
    processed image/mask arrays plus spatial metadata.

    This function is used internally by process_single_case.
    """
    image, mask = load_case(image_path, mask_path)
    
    orig_spacing = image.GetSpacing()
    orig_shape = image.GetSize()
    orig_direction = image.GetDirection()
    orig_origin = image.GetOrigin()
    
    image, mask = orient_to_ras(image, mask)
    image = resample_ct(image)
    mask = resample_mask(mask)
    verify_geometry(image, mask)
    
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    origin = image.GetOrigin()
    
    image_np = sitk_to_numpy(image)
    mask_np = sitk_to_numpy(mask)

    image_np = apply_window(image_np)
    
    if CONFIG.get("apply_histogram_matching", False):
        ref_path = CONFIG.get("histogram_reference_path", None)
        if ref_path and Path(ref_path).exists():
            ref_img = sitk_to_numpy(sitk.ReadImage(ref_path))
            ref_img = apply_window(ref_img)
            image_np = histogram_matching(image_np, ref_img)
        else:
            logger.warning("Histogram matching enabled but no valid reference_path provided.")
            
    image_np = normalize(image_np)
    image_np, mask_np = standardize_depth(image_np, mask_np)
    mask_np = clean_mask(mask_np)

    image_np = image_np.astype(DTYPE_IMAGE)
    mask_np = mask_np.astype(DTYPE_MASK)
    verify_case(image_np, mask_np)

    image_out = numpy_to_sitk(image_np, image)
    mask_out = numpy_to_sitk(mask_np, mask)

    return {
        "image": image_out,
        "mask": mask_out,
        "image_np": image_np,
        "mask_np": mask_np,
        "spacing": spacing,
        "orig_shape": orig_shape,
        "orig_spacing": orig_spacing,
        "orig_direction": orig_direction,
        "orig_origin": orig_origin,
        "origin": origin,
        "direction": direction,
        "shape": image_np.shape,
        "interpolator": "Image: Linear, Mask: NearestNeighbor",
        "quality": quality_check(image_np, mask_np)
    }

def save_qc_visualization(image: np.ndarray, mask: np.ndarray, filename: str, metadata: Dict[str, Any]) -> None:
    """
    Saves a 3‑panel QC visualization (middle slice, overlay, mask)
    for the best pathology slice and the central slice.
    """
    try:
        if mask.sum() == 0:
            best_slice = image.shape[-1] // 2
        else:
            lesion_pixels = mask.sum(axis=tuple(range(mask.ndim - 1)))
            best_slice = int(np.argmax(lesion_pixels))
        
        mid_slice = image.shape[-1] // 2
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        img_disp = image[0] if image.ndim == 4 else image
        
        fig.suptitle(f"Case QC: {filename} | Patient ID: {metadata['Patient_ID']}\n"
                     f"Volume: {metadata['Processed_Shape']} | Lesion Vol: {metadata['Lesion_Volume_mm3']:.1f} mm³", 
                     fontsize=14, fontweight='bold')
        
        axes[0].imshow(img_disp[..., mid_slice], cmap='gray')
        axes[0].set_title(f"Z-Center Slice ({mid_slice})", fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(img_disp[..., best_slice], cmap='gray')
        axes[1].imshow(mask[..., best_slice], cmap='Reds', alpha=0.3)
        axes[1].set_title(f"Lesion Overlay (Max Pathology Z: {best_slice})", fontsize=12)
        axes[1].axis('off')
        
        axes[2].imshow(mask[..., best_slice], cmap='gray')
        axes[2].set_title(f"Binary Mask (Z: {best_slice})", fontsize=12)
        axes[2].axis('off')
        
        fig.text(0.01, 0.02, f"Pipeline V{PIPELINE_VERSION} | Hash: {CONFIG_HASH[:8]} | Timestamp: {metadata['Processing_Timestamp']}", 
                 fontsize=8, color='gray')
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.92]) 
        plt.savefig(OUTPUT_VIS_DIR / f"{filename}.png", bbox_inches='tight', dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Failed to generate QC visualization for {filename}: {e}")

# =====================================================================
# PART 2B-1
# DATASET PROCESSING ENGINE
# =====================================================================

def get_dataset_name(filename: str) -> str:
    """Determines the dataset origin from the filename."""
    name = filename.upper()
    if "BHSD" in name: return "BHSD"
    elif "CQ500" in name: return "CQ500"
    elif "PHYSIONET" in name or "PHYSIO" in name: return "PHYSIONET"
    elif "INSTANCE" in name: return "INSTANCE"
    return "UNKNOWN"

import re

def extract_patient_id(filename: str, dataset: str = "") -> str:
    """
    Extract patient ID directly from merged filenames.

    Examples
    --------
    BHSD_0001.nii.gz   -> BHSD_0001
    CQ500_0015.nii.gz  -> CQ500_0015
    PHYSIO_0042.nii.gz -> PHYSIO_0042
    """

    stem = Path(filename).stem.replace(".nii", "")

    match = re.match(r"([A-Za-z0-9]+_\d+)", stem)

    if match:
        return match.group(1)

    return stem
def process_single_case(image_path: Path, mask_path: Path) -> Dict[str, Any]:
    """
    Processes a single case: preprocessing, saving outputs to disk,
    and returning only lightweight metadata (no image arrays).

    This function is designed to be called by a worker process.
    """
    case_start = time.time()
    dataset_name = get_dataset_name(image_path.name)
    try:
        # Perform all preprocessing steps (in memory)
        result = preprocess_case(image_path, mask_path)
        
        # Build metadata dictionary
        metadata = build_metadata(
            filename=image_path.name,
            dataset=dataset_name,
            image=result["image_np"],
            mask=result["mask_np"],
            spacing=result["spacing"],
            orig_shape=result["orig_shape"],
            orig_spacing=result["orig_spacing"],
            orig_direction=result["orig_direction"],
            orig_origin=result["orig_origin"],
            direction=result["direction"],
            origin=result["origin"],
            interpolator=result["interpolator"],
            processing_time_sec=time.time() - case_start
        )
        
        # Save the processed NIfTI files directly from the worker
        save_case(result["image"], result["mask"],
                  OUTPUT_IMAGE_DIR / image_path.name,
                  OUTPUT_MASK_DIR / image_path.name)
        
        if CONFIG.get("save_visualization", True):
            save_qc_visualization(result["image_np"], result["mask_np"], image_path.stem, metadata)
        
        # Explicitly delete large arrays to help memory
        del result
        
        # Return ONLY small status & metadata dict (no image/mask objects)
        return {
            "status": "success",
            "metadata": metadata,
            "quality": metadata,  # quality info already inside metadata; but we keep separate key if needed
            "Filename": image_path.name
        }
    except Exception as e:
        # Capture full exception context and return as lightweight error report
        return {
            "status": "error",
            "Filename": image_path.name,
            "Dataset": dataset_name,
            "Pipeline_Stage": "process_single_case",
            "Function_Name": "preprocess_case",
            "Exception_Type": type(e).__name__,
            "Exception_Message": str(e),
            "Full_Traceback": traceback.format_exc(),
            "Processing_Timestamp": datetime.now().isoformat()
        }

def already_processed(filename: str) -> bool:
    """Checks if the output files already exist."""
    return (OUTPUT_IMAGE_DIR / filename).exists() and (OUTPUT_MASK_DIR / filename).exists()

# =====================================================================
# PART 5B: PATIENT-WISE SPLITS
# =====================================================================

def generate_splits(metadata_records: List[Dict[str, Any]]) -> None:
    """Generates patient‑wise train/val/test and GroupKFold splits, avoiding data leakage."""
    logger.info("Generating patient-wise data splits using GroupKFold...")
    df = pd.DataFrame(metadata_records)
    
    unique_patients = df['Patient_ID'].unique()
    train_val_patients, test_patients = train_test_split(
        unique_patients, test_size=CONFIG["splits"]["test"], random_state=SEED)
    
    val_ratio = CONFIG["splits"]["val"] / (CONFIG["splits"]["train"] + CONFIG["splits"]["val"])
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=val_ratio, random_state=SEED)
    
    splits = {
        "train": df[df['Patient_ID'].isin(train_patients)]['Filename'].tolist(),
        "val": df[df['Patient_ID'].isin(val_patients)]['Filename'].tolist(),
        "test": df[df['Patient_ID'].isin(test_patients)]['Filename'].tolist()
    }
    
    with open(OUTPUT_SPLIT_DIR / "dataset_splits.json", "w") as f:
        json.dump(splits, f, indent=4)
        
    gkf = GroupKFold(n_splits=CONFIG["splits"]["k_folds"])
    kfold_splits = {}
    
    train_val_df = df[df['Patient_ID'].isin(train_val_patients)].reset_index(drop=True)
    X_dummy = np.zeros(len(train_val_df))
    
    for i, (train_idx, val_idx) in enumerate(
        gkf.split(X_dummy, groups=train_val_df['Patient_ID'])
    ):
        kfold_splits[f"fold_{i}"] = {
            "train": train_val_df.iloc[train_idx]['Filename'].tolist(),
            "val": train_val_df.iloc[val_idx]['Filename'].tolist()
        }
        
    with open(OUTPUT_SPLIT_DIR / "kfold_splits.json", "w") as f:
        json.dump(kfold_splits, f, indent=4)
    logger.info("GroupKFold splits generated successfully avoiding patient leakage.")

# =====================================================================
# PART 5C: PIPELINE MANIFEST GENERATION
# =====================================================================

def save_pipeline_manifest(elapsed_time: float, total: int, processed: int, failed: int) -> None:
    """Generates a comprehensive diagnostic research manifest file."""
    manifest = {
        "Pipeline_Version": PIPELINE_VERSION,
        "Configuration_Hash": CONFIG_HASH,
        "Dataset_Version": "Merged_V2", 
        "Input_Dataset": str(MASTER_DATASET),
        "Output_Dataset": str(OUTPUT_ROOT),
        "Processing_Date": datetime.now().isoformat(),
        "Total_Runtime_Sec": round(elapsed_time, 2),
        "Total_Cases": total,
        "Processed_Cases": processed,
        "Failed_Cases": failed,
        "Success_Rate": f"{(processed / total * 100) if total > 0 else 0:.2f}%",
        "Python_Version": PYTHON_VERSION,
        "SimpleITK_Version": SITK_VERSION,
        "Git_Commit": GIT_COMMIT,
        "Operating_System": SYSTEM_OS,
        "CPU_Count": CPU_COUNT,
        "GPU_Name": GPU_NAME
    }
    with open(MANIFEST_JSON, "w") as f:
        json.dump(manifest, f, indent=4)
    logger.info(f"Pipeline manifest saved to {MANIFEST_JSON}")

# =====================================================================
# PROCESS ENTIRE DATASET (Robust Multiprocessing)
# =====================================================================

def process_dataset() -> None:
    """
    Main dataset processing function using multiprocessing.
    Workers write results directly to disk and return only metadata.
    """
    logger.info("=" * 70)
    logger.info("PROCESSING DATASET")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    metadata_records = []
    failed_records = []
    
    # Use os.cpu_count() - 1 workers for stability
    opt_workers = max(1, min(CPU_COUNT - 1, CONFIG.get("num_workers", 4)))
    logger.info(f"Launching processing using {opt_workers} concurrent workers.")
    
    # --- Task Aggregation & Debugging ---
    overwrite_existing = CONFIG.get("overwrite_existing", True)
    tasks_to_submit = []
    skipped_count = 0
    
    for image_path in IMAGE_FILES:
        mask_path = MASK_DIR / image_path.name
        if overwrite_existing or not already_processed(image_path.name):
            tasks_to_submit.append((image_path, mask_path))
        else:
            skipped_count += 1

    logger.info(f"DEBUG - Number of image files: {len(IMAGE_FILES)}")
    logger.info(f"DEBUG - Number of mask files: {len(MASK_FILES)}")
    logger.info(f"DEBUG - Number of tasks created: {len(tasks_to_submit)}")
    logger.info(f"DEBUG - Number of skipped files: {skipped_count}")

    if not tasks_to_submit:
        logger.warning("No tasks to process. All cases were skipped. Set 'overwrite_existing': True in your config to re-run.")
        return

    # --- Execution ---
    futures = []
    
    # In Python 3.11+, we can use max_tasks_per_child to prevent C++/Matplotlib memory leaks 
    # from crashing the worker process over hundreds of iterations.
    try:
        executor = ProcessPoolExecutor(max_workers=opt_workers, max_tasks_per_child=10)
    except TypeError:
        # Fallback for Python < 3.11
        executor = ProcessPoolExecutor(max_workers=opt_workers)
        
    with executor:
        for img_path, msk_path in tasks_to_submit:
            futures.append(executor.submit(process_single_case, img_path, msk_path))
            
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Cases"):
            res = future.result()
            if res["status"] == "success":
                # Worker already saved images; just collect metadata
                metadata_records.append(res["metadata"])
            else:
                # Error report
                failed_records.append(res)
                logger.error(f"Error processing {res.get('Filename', 'Unknown')}: "
                             f"{res.get('Exception_Message', 'No message')}")
                
    elapsed = time.time() - start_time
    
    if len(metadata_records) > 0:
        df = pd.DataFrame(metadata_records)
        df.to_csv(METADATA_CSV, index=False)
        generate_splits(metadata_records)
        
        success_rate = (len(metadata_records) / TOTAL_CASES) * 100
        avg_time = df['Processing_Time_Sec'].mean()
        avg_vol_mm3 = df['Lesion_Volume_mm3'].mean()
        avg_depth = df['Target_Depth'].mean() if 'Target_Depth' in df.columns else CONFIG.get("target_depth", 128)
        avg_x = df['Shape_X'].mean() if 'Shape_X' in df.columns else 0
        avg_y = df['Shape_Y'].mean() if 'Shape_Y' in df.columns else 0
        avg_z = df['Shape_Z'].mean() if 'Shape_Z' in df.columns else 0

        logger.info("=" * 70)
        logger.info("RESEARCH LOGGING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total Cases             : {TOTAL_CASES}")
        logger.info(f"Processed Cases         : {len(metadata_records)}")
        logger.info(f"Failed Cases            : {len(failed_records)}")
        logger.info(f"Success Rate            : {success_rate:.2f}%")
        logger.info(f"Average Processing Time : {avg_time:.2f} sec")
        logger.info(f"Average Volume Shape    : ({avg_x:.1f}, {avg_y:.1f}, {avg_z:.1f})")
        logger.info(f"Average Depth           : {avg_depth:.1f}")
        logger.info(f"Average Lesion Volume   : {avg_vol_mm3:.2f} mm3")
        logger.info(f"Total Runtime           : {elapsed:.2f} sec")
        logger.info(f"Pipeline Version        : {PIPELINE_VERSION}")
        logger.info(f"Configuration Hash      : {CONFIG_HASH}")
        logger.info("=" * 70)

    if failed_records:
        pd.DataFrame(failed_records).to_csv(FAILED_CASES, index=False)
        logger.warning(f"{len(failed_records)} cases failed during processing.")

    save_pipeline_manifest(elapsed, TOTAL_CASES, len(metadata_records), len(failed_records))
    
    logger.info(f"Preprocessing FINISHED in {elapsed:.2f} seconds.")
    logger.info(f"Generated outputs saved to {OUTPUT_ROOT}")


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def run_pipeline() -> None:
    """Entry point for the preprocessing pipeline."""
    process_dataset()
    logger.info("Entire preprocessing pipeline completed.")
    logger.info("Zipping the processed directory into process2.zip for Colab upload...")
    import shutil
    shutil.make_archive("process2", 'zip', "processed")
    logger.info("Zipping complete! process2.zip is ready.")

def main() -> None:
    """Main function."""
    run_pipeline()

if __name__ == "__main__":
    main()