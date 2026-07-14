"""
transforms.py

Research-grade MONAI dictionary-based transform pipelines for 3D Brain Hemorrhage CT
segmentation. Uses a Factory pattern, lazy configuration loading, strict type safety,
and SHA256 reproducibility tracking. Optimized for PyTorch 2.6.0 and MONAI 1.6.0.

Author: Senior Medical Imaging Research Engineer
"""

from __future__ import annotations

import hashlib
import logging
import random
import sys
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import yaml
import monai
from monai.transforms import (
    Compose,
    EnsureTyped,
    ResizeWithPadOrCropd,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
    CropForegroundd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    CenterSpatialCropd,
    Rand3DElasticd,
    RandCoarseDropoutd,
)

class SinglePosNegCropd:
    """Wrapper for RandCropByPosNegLabeld that returns a dict instead of a list of length 1.
    This prevents MONAI Compose from mapping subsequent transforms over the list."""
    def __init__(self, keys, label_key, spatial_size, pos, neg, image_key, image_threshold):
        self.cropper = RandCropByPosNegLabeld(
            keys=keys, label_key=label_key, spatial_size=spatial_size,
            pos=pos, neg=neg, num_samples=1, image_key=image_key, image_threshold=image_threshold
        )
    def __call__(self, data):
        out = self.cropper(data)
        return out[0] if isinstance(out, list) else out

from monai.utils import set_determinism

logger = logging.getLogger(__name__)

# ===========================================================================
# Custom Exceptions
# ===========================================================================
class AugmentationConfigError(Exception):
    """Raised when the augmentation configuration is invalid or missing."""
    pass

class TransformValidationError(Exception):
    """Raised when the pipeline fails self-validation (shape, dtype, channels)."""
    pass

# ===========================================================================
# Enums for Type Safety
# ===========================================================================
class InterpolationMode(str, Enum):
    TRILINEAR = "trilinear"
    NEAREST = "nearest"
    BILINEAR = "bilinear"

class PaddingMode(str, Enum):
    ZEROS = "zeros"
    BORDER = "border"
    REFLECTION = "reflection"
    CONSTANT = "constant"

# ===========================================================================
# Configuration Dataclass
# ===========================================================================
@dataclass
class AugmentationConfig:
    """Immutable configuration container mapped directly from YAML."""
    
    seed: int = 42
    deterministic: bool = True
    
    # Spatial Transforms
    flip_enabled: bool = True
    flip_prob: float = 0.5
    flip_spatial_axis: Optional[int] = 2

    affine_enabled: bool = True
    affine_prob: float = 0.2
    affine_rotate_range: Sequence[float] = (0.0, 0.0, 0.1745)
    affine_shear_range: Sequence[float] = (0.0, 0.0, 0.0)
    affine_scale_range: Sequence[float] = (0.05, 0.05, 0.05)
    affine_translate_range: Sequence[float] = (5.0, 5.0, 5.0)

    zoom_enabled: bool = True
    zoom_prob: float = 0.15
    zoom_min_zoom: float = 0.95
    zoom_max_zoom: float = 1.05
    zoom_mode: InterpolationMode = InterpolationMode.TRILINEAR
    zoom_padding_mode: PaddingMode = PaddingMode.CONSTANT

    elastic_enabled: bool = True
    elastic_prob: float = 0.1
    elastic_sigma_range: Sequence[float] = (5.0, 7.0)
    elastic_magnitude_range: Sequence[float] = (50.0, 100.0)

    dropout_enabled: bool = True
    dropout_prob: float = 0.1
    dropout_holes: int = 1
    dropout_spatial_size: int = 8

    # Intensity Transforms
    gaussian_noise_enabled: bool = True
    gaussian_noise_prob: float = 0.15
    gaussian_noise_std: float = 0.02

    gaussian_smooth_enabled: bool = True
    gaussian_smooth_prob: float = 0.10
    gaussian_smooth_sigma: Sequence[float] = (0.5, 1.0)

    scale_intensity_enabled: bool = True
    scale_intensity_prob: float = 0.15
    scale_intensity_factors: float = 0.05

    shift_intensity_enabled: bool = True
    shift_intensity_prob: float = 0.15
    shift_intensity_offsets: float = 0.05

    adjust_contrast_enabled: bool = True
    adjust_contrast_prob: float = 0.15
    adjust_contrast_gamma: Sequence[float] = (0.8, 1.2)

    # Shared parameters
    spatial_image_mode: InterpolationMode = InterpolationMode.TRILINEAR
    spatial_mask_mode: InterpolationMode = InterpolationMode.NEAREST
    spatial_padding_mode: PaddingMode = PaddingMode.ZEROS

    def validate(self) -> None:
        """Validates probability bounds and critical configurations on startup."""
        probs = [
            self.flip_prob, self.affine_prob, self.zoom_prob, 
            self.gaussian_noise_prob, self.gaussian_smooth_prob,
            self.scale_intensity_prob, self.shift_intensity_prob,
            self.adjust_contrast_prob, self.elastic_prob, self.dropout_prob
        ]
        for prob in probs:
            if not isinstance(prob, (int, float)) or not (0.0 <= prob <= 1.0):
                raise AugmentationConfigError(f"Probability {prob} out of bounds [0.0, 1.0]")

# ===========================================================================
# Caching & Environment Logging
# ===========================================================================
def _log_environment() -> None:
    logger.info("="*50)
    logger.info("ENVIRONMENT VERIFICATION")
    logger.info("="*50)
    logger.info(f"Python  : {sys.version.split()[0]}")
    logger.info(f"PyTorch : {torch.__version__}")
    logger.info(f"MONAI   : {monai.__version__}")
    logger.info(f"CUDA    : {torch.version.cuda if torch.cuda.is_available() else 'CPU'}")
    logger.info("="*50)

@lru_cache(maxsize=1)
def load_and_hash_config(config_path: str) -> tuple[AugmentationConfig, str]:
    """Reads YAML, computes SHA256, validates, and caches the result."""
    path = Path(config_path)
    if not path.exists():
        raise AugmentationConfigError(f"Configuration file missing: {path}")
    
    with open(path, "rb") as f:
        file_bytes = f.read()
    
    config_hash = hashlib.sha256(file_bytes).hexdigest()
    
    try:
        raw_config = yaml.safe_load(file_bytes) or {}
        if "augmentation" in raw_config:
            aug_cfg = raw_config["augmentation"]
            
            # Map nested dictionary to flat dataclass
            flat_cfg = {}
            if "random_flip" in aug_cfg:
                flat_cfg["flip_enabled"] = aug_cfg["random_flip"].get("enabled", True)
                flat_cfg["flip_prob"] = aug_cfg["random_flip"].get("prob", 0.5)
                # handle spatial_axis array
                axis = aug_cfg["random_flip"].get("spatial_axis", 2)
                flat_cfg["flip_spatial_axis"] = axis if isinstance(axis, int) else tuple(axis)
            
            if "random_affine" in aug_cfg:
                flat_cfg["affine_enabled"] = aug_cfg["random_affine"].get("enabled", True)
                flat_cfg["affine_prob"] = aug_cfg["random_affine"].get("prob", 0.2)
                flat_cfg["affine_rotate_range"] = aug_cfg["random_affine"].get("rotate_range", (0.0, 0.0, 0.1745))
                flat_cfg["affine_scale_range"] = aug_cfg["random_affine"].get("scale_range", (0.05, 0.05, 0.05))
            
            if "elastic_deformation" in aug_cfg:
                flat_cfg["elastic_enabled"] = aug_cfg["elastic_deformation"].get("enabled", True)
                flat_cfg["elastic_prob"] = aug_cfg["elastic_deformation"].get("prob", 0.15)
                flat_cfg["elastic_sigma_range"] = aug_cfg["elastic_deformation"].get("sigma_range", (5.0, 7.0))
                flat_cfg["elastic_magnitude_range"] = aug_cfg["elastic_deformation"].get("magnitude_range", (50.0, 100.0))
            
            if "gaussian_noise" in aug_cfg:
                flat_cfg["gaussian_noise_enabled"] = aug_cfg["gaussian_noise"].get("enabled", True)
                flat_cfg["gaussian_noise_prob"] = aug_cfg["gaussian_noise"].get("prob", 0.15)
            
            if "gaussian_blur" in aug_cfg:
                flat_cfg["gaussian_smooth_enabled"] = aug_cfg["gaussian_blur"].get("enabled", True)
                flat_cfg["gaussian_smooth_prob"] = aug_cfg["gaussian_blur"].get("prob", 0.10)
                
            if "gamma" in aug_cfg:
                flat_cfg["adjust_contrast_enabled"] = aug_cfg["gamma"].get("enabled", True)
                flat_cfg["adjust_contrast_prob"] = aug_cfg["gamma"].get("prob", 0.20)
                
            if "intensity_shift" in aug_cfg:
                flat_cfg["shift_intensity_enabled"] = aug_cfg["intensity_shift"].get("enabled", True)
                flat_cfg["shift_intensity_prob"] = aug_cfg["intensity_shift"].get("prob", 0.15)
                
            if "zoom" in aug_cfg:
                flat_cfg["zoom_enabled"] = aug_cfg["zoom"].get("enabled", True)
                flat_cfg["zoom_prob"] = aug_cfg["zoom"].get("prob", 0.15)
                
            if "coarse_dropout" in aug_cfg:
                flat_cfg["dropout_enabled"] = aug_cfg["coarse_dropout"].get("enabled", True)
                flat_cfg["dropout_prob"] = aug_cfg["coarse_dropout"].get("prob", 0.10)
            
            # Map valid keys
            valid_keys = AugmentationConfig.__dataclass_fields__.keys()
            filtered = {k: v for k, v in flat_cfg.items() if k in valid_keys}
            
            # Additional root keys (like seed, deterministic)
            for k, v in raw_config.items():
                if k in valid_keys and k not in filtered:
                    filtered[k] = v
        else:
            valid_keys = AugmentationConfig.__dataclass_fields__.keys()
            filtered = {k: v for k, v in raw_config.items() if k in valid_keys}
        
        # Parse enums safely
        if "zoom_mode" in filtered: filtered["zoom_mode"] = InterpolationMode(filtered["zoom_mode"])
        if "zoom_padding_mode" in filtered: filtered["zoom_padding_mode"] = PaddingMode(filtered["zoom_padding_mode"])
        if "spatial_image_mode" in filtered: filtered["spatial_image_mode"] = InterpolationMode(filtered["spatial_image_mode"])
        if "spatial_mask_mode" in filtered: filtered["spatial_mask_mode"] = InterpolationMode(filtered["spatial_mask_mode"])
        if "spatial_padding_mode" in filtered: filtered["spatial_padding_mode"] = PaddingMode(filtered["spatial_padding_mode"])

        config = AugmentationConfig(**filtered)
        config.validate()
        
        logger.info(f"Loaded Augmentation Config | SHA256: {config_hash}")
        
        # Log to display exact configuration loaded (per User request)
        print("\nAugmentation Pipeline")
        print(f"✓ Random Flip           : {'Enabled' if config.flip_enabled else 'Disabled'} (p={config.flip_prob:.2f})")
        print(f"✓ Random Affine         : {'Enabled' if config.affine_enabled else 'Disabled'} (p={config.affine_prob:.2f})")
        print(f"✓ Elastic Deformation   : {'Enabled' if config.elastic_enabled else 'Disabled'} (p={config.elastic_prob:.2f})")
        print(f"✓ Gaussian Noise        : {'Enabled' if config.gaussian_noise_enabled else 'Disabled'} (p={config.gaussian_noise_prob:.2f})")
        print(f"✓ Gaussian Blur         : {'Enabled' if config.gaussian_smooth_enabled else 'Disabled'} (p={config.gaussian_smooth_prob:.2f})")
        print(f"✓ Gamma Correction      : {'Enabled' if config.adjust_contrast_enabled else 'Disabled'} (p={config.adjust_contrast_prob:.2f})")
        print(f"✓ Intensity Shift       : {'Enabled' if config.shift_intensity_enabled else 'Disabled'} (p={config.shift_intensity_prob:.2f})")
        print(f"✓ Zoom                  : {'Enabled' if config.zoom_enabled else 'Disabled'} (p={config.zoom_prob:.2f})")
        print(f"✓ Coarse Dropout        : {'Enabled' if getattr(config, 'dropout_enabled', False) else 'Disabled'} (p={getattr(config, 'dropout_prob', 0.1):.2f})")
        print("")
        
        return config, config_hash
    except Exception as e:
        raise AugmentationConfigError(f"Configuration parsing failed: {str(e)}") from e

# ===========================================================================
# Pipeline Builder & Factory
# ===========================================================================
class TransformFactory:
    """Factory pattern for MONAI transformation pipelines."""
    
    def __init__(self, config_path: Union[str, Path] = "configs/augmentation.yaml"):
        _log_environment()
        self.config, self.config_hash = load_and_hash_config(str(config_path))
        self._set_global_seed()

    def _set_global_seed(self) -> None:
        if self.config.deterministic:
            set_determinism(seed=self.config.seed)
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            
    def get_config_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self.config)

    def build_train_pipeline(self) -> Compose:
        cfg = self.config
        xforms: List[Any] = [

            EnsureTyped(keys=["image"], dtype=torch.float32),
            EnsureTyped(keys=["mask"], dtype=torch.long),
            CropForegroundd(keys=["image", "mask"], source_key="image"),
            SpatialPadd(keys=["image", "mask"], spatial_size=(64, 256, 256)),
            SinglePosNegCropd(
                keys=["image", "mask"],
                label_key="mask",
                spatial_size=(64, 256, 256),
                pos=1,
                neg=1,
                image_key="image",
                image_threshold=0,
            ),

        ]

        if cfg.flip_enabled:
            xforms.append(RandFlipd(keys=["image", "mask"], prob=cfg.flip_prob, spatial_axis=cfg.flip_spatial_axis))

        if cfg.affine_enabled:
            xforms.append(RandAffined(
                keys=["image", "mask"],
                prob=cfg.affine_prob,
                rotate_range=cfg.affine_rotate_range,
                shear_range=cfg.affine_shear_range,
                scale_range=cfg.affine_scale_range,
                translate_range=cfg.affine_translate_range,
                mode=(cfg.spatial_image_mode.value, cfg.spatial_mask_mode.value),
                padding_mode=cfg.spatial_padding_mode.value,
            ))

        if cfg.zoom_enabled:
            xforms.append(RandZoomd(
                keys=["image", "mask"],
                prob=cfg.zoom_prob,
                min_zoom=cfg.zoom_min_zoom,
                max_zoom=cfg.zoom_max_zoom,
                mode=(cfg.zoom_mode.value, cfg.spatial_mask_mode.value),
                padding_mode=cfg.zoom_padding_mode.value,
            ))

        if cfg.gaussian_noise_enabled:
            xforms.append(RandGaussianNoised(keys=["image"], prob=cfg.gaussian_noise_prob, std=cfg.gaussian_noise_std))

        if cfg.gaussian_smooth_enabled:
            xforms.append(RandGaussianSmoothd(keys=["image"], prob=cfg.gaussian_smooth_prob, sigma_x=cfg.gaussian_smooth_sigma))

        if cfg.scale_intensity_enabled:
            xforms.append(RandScaleIntensityd(keys=["image"], prob=cfg.scale_intensity_prob, factors=cfg.scale_intensity_factors))

        if cfg.shift_intensity_enabled:
            xforms.append(RandShiftIntensityd(keys=["image"], prob=cfg.shift_intensity_prob, offsets=cfg.shift_intensity_offsets))

        if cfg.adjust_contrast_enabled:
            xforms.append(RandAdjustContrastd(keys=["image"], prob=cfg.adjust_contrast_prob, gamma=cfg.adjust_contrast_gamma))

        if getattr(cfg, 'elastic_enabled', False):
            xforms.append(Rand3DElasticd(
                keys=["image", "mask"],
                prob=cfg.elastic_prob,
                sigma_range=cfg.elastic_sigma_range,
                magnitude_range=cfg.elastic_magnitude_range,
                mode=(cfg.spatial_image_mode.value, cfg.spatial_mask_mode.value),
                padding_mode=cfg.spatial_padding_mode.value,
            ))

        if getattr(cfg, 'dropout_enabled', False):
            xforms.append(RandCoarseDropoutd(
                keys=["image"],
                holes=cfg.dropout_holes,
                spatial_size=cfg.dropout_spatial_size,
                prob=cfg.dropout_prob,
                fill_value=0.0
            ))

        xforms.append(EnsureTyped(keys=["mask"], dtype=torch.long))
        return Compose(xforms)

    def build_eval_pipeline(self) -> Compose:
        return Compose([
        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["mask"], dtype=torch.long),
        CropForegroundd(keys=["image", "mask"], source_key="image"),
        SpatialPadd(keys=["image", "mask"], spatial_size=(64, 256, 256)),
        CenterSpatialCropd(keys=["image", "mask"], roi_size=(64, 256, 256)),
    ])
    @staticmethod
    def validate_pipeline(pipeline: Compose, sample_shape: tuple = (64, 256, 256)) -> None:
        """Runs a mock tensor through the pipeline to ensure constraints hold."""
        dummy_data = {
            "image": torch.rand(3, *sample_shape, dtype=torch.float32),
            "mask": torch.randint(0, 2, (1, *sample_shape), dtype=torch.long),
            "metadata": {"case": "test"}
        }
        
        try:
            out = pipeline(dummy_data)
            if out["image"].shape != (3, *sample_shape):
                raise TransformValidationError(f"Image shape corrupted. Expected (3, {sample_shape}), got {out['image'].shape}")
            if out["mask"].shape != (1, *sample_shape):
                raise TransformValidationError(f"Mask shape corrupted. Expected (1, {sample_shape}), got {out['mask'].shape}")
            if out["mask"].dtype != torch.long:
                raise TransformValidationError(f"Mask dtype must be torch.long. Got {out['mask'].dtype}")
            if "metadata" not in out:
                raise TransformValidationError("Metadata key dropped during transformation.")
        except Exception as e:
            raise TransformValidationError(f"Pipeline validation failed: {str(e)}") from e

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import tempfile
    
    mock_yaml = """
    seed: 42
    deterministic: true
    affine_enabled: true
    """
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(mock_yaml)
        tmp_path = f.name
        
    try:
        factory = TransformFactory(config_path=tmp_path)
        train_transforms = factory.build_train_pipeline()
        
        logger.info("Running strict pipeline validation...")
        factory.validate_pipeline(train_transforms)
        
        print("\n===============================")
        print("PASS")
        print("===============================\n")
    finally:
        Path(tmp_path).unlink()
