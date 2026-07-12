"""
dataset.py — BrainHemorrhageDataset for 3D Intracranial Hemorrhage Segmentation

Research-grade PyTorch Dataset supporting train / val / test / inference modes.
Integrates directly with the Hemorrhage_SOTA_V2 preprocessing pipeline.

Author      : Abhinav Gupta
Institution : DeepMind / Academic Medical Imaging Lab
Version     : 3.0.4  (project root resolution fixed for datasets/ subdirectory)
Created     : 2026-07-11

Compatible with:
    - UNet, Attention UNet, UNet++, UNETR, SwinUNETR, SegResNet, MedNeXt,
      TransUNet, nnUNet, and any 3D segmentation model accepting (C,D,H,W) input.

Design goals:
    • Lazy loading – never preloads the entire dataset into RAM
    • Memory efficiency – configurable per‑worker cache (LRU or custom)
    • Robust validation – fast light/full checks without loading pixel data
    • Reproducibility – exact file lists from preprocessing splits, seed control
    • Clear, actionable exceptions for every failure mode
    • Fully configurable via configs/dataset.yaml or environment variable
    • Production‑grade logging, type hints, and docstrings (PEP8)
    • Extensible for future multi‑task / classification heads
"""

from __future__ import annotations

import json
import logging
import os
import random
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# TypedDict for sample output
# ---------------------------------------------------------------------------

class BrainHemorrhageSample(TypedDict):
    """Expected dictionary format returned by __getitem__."""
    image: torch.Tensor       # (C, D, H, W)
    mask: Optional[torch.Tensor]  # (1, D, H, W) or None
    case_id: str
    patient_id: str
    dataset: str
    spacing: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    direction: Tuple[float, ...]
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class DatasetError(Exception):
    """Base class for all dataset-related errors."""

class MissingFileError(DatasetError):
    """Raised when a required file does not exist on disk."""

class CorruptedFileError(DatasetError):
    """Raised when a file cannot be opened, decoded, or contains invalid values."""

class GeometryMismatchError(DatasetError):
    """Raised when image and mask geometries do not match."""

class InvalidMaskError(DatasetError):
    """Raised when the mask is not binary or contains invalid values."""

class SplitError(DatasetError):
    """Raised when the split configuration is invalid or missing."""

class ConfigurationError(DatasetError):
    """Raised when the dataset configuration is invalid."""


# ---------------------------------------------------------------------------
# Geometry Metadata Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeometryMeta:
    """Immutable holder for spatial metadata parsed from a row."""
    spacing: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    direction: Tuple[float, ...]  # length 9
    shape: Optional[Tuple[int, int, int]] = None


# ---------------------------------------------------------------------------
# Cache Backend Abstraction
# ---------------------------------------------------------------------------

class CacheBackend(ABC):
    """Abstract cache backend for dataset samples."""
    @abstractmethod
    def get(self, key: str) -> Optional[BrainHemorrhageSample]:
        ...

    @abstractmethod
    def put(self, key: str, sample: BrainHemorrhageSample) -> None:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...


class LRUCache(CacheBackend):
    """Thread‑safe LRU cache with a maximum capacity."""
    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[str, BrainHemorrhageSample] = OrderedDict()

    def get(self, key: str) -> Optional[BrainHemorrhageSample]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, sample: BrainHemorrhageSample) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)  # evict oldest
            self._cache[key] = sample

    def clear(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# Configuration Helpers
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = (
    "processed_images_dir",
    "processed_masks_dir",
    "metadata_csv",
    "split_file",
    "kfold_split_file",
)

def _resolve_config(
    config: Union[str, Path, Dict[str, Any], None] = None,
) -> Dict[str, Any]:
    """
    Resolve dataset configuration from a YAML file, dictionary, or environment.

    Priority:
        1. User‑supplied dictionary (overrides everything)
        2. User‑supplied YAML file path
        3. Environment variable ``DATASET_CONFIG``
        4. Default ``configs/dataset.yaml`` in project root

    The project root is assumed to be two directories above this file
    (since dataset.py lives in a ``datasets/`` subdirectory of the project).
    If the file is relocated, provide an explicit ``config`` argument.
    """
    # dataset.py is assumed to reside in a subdirectory (e.g., datasets/)
    # so the project root is the parent of the parent directory.
    project_root = Path(__file__).resolve().parent.parent

    defaults = {
        "processed_images_dir": str(project_root / "processed" / "images"),
        "processed_masks_dir": str(project_root / "processed" / "masks"),
        "metadata_csv": str(project_root / "processed" / "reports" / "Processed_Metadata.csv"),
        "split_file": str(project_root / "processed" / "splits" / "dataset_splits.json"),
        "kfold_split_file": str(project_root / "processed" / "splits" / "kfold_splits.json"),
        "cache_size": 0,
        "load_mask": True,
        "geometry_tolerance": 1e-4,
        "verbose": False,
        "seed": 42,
        "image_dtype": "float32",
        "mask_dtype": "uint8",
    }

    if config is None:
        config_path = os.environ.get("DATASET_CONFIG", None)
        if config_path and Path(config_path).exists():
            return _load_yaml_config(config_path, defaults)
        default_yaml = project_root / "configs" / "dataset.yaml"
        if default_yaml.exists():
            return _load_yaml_config(str(default_yaml), defaults)
        return defaults

    if isinstance(config, dict):
        merged = defaults.copy()
        merged.update(config)
        return merged

    if isinstance(config, (str, Path)):
        return _load_yaml_config(str(config), defaults)

    raise ConfigurationError(f"Unsupported config type: {type(config)}")


def _load_yaml_config(path: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Load a YAML config file and merge with defaults."""
    with open(path, "r") as f:
        user = yaml.safe_load(f) or {}
    merged = defaults.copy()
    merged.update(user)
    _validate_config_keys(merged)
    return merged


def _validate_config_keys(cfg: Dict[str, Any]) -> None:
    """Ensure all required keys are present."""
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        raise ConfigurationError(f"Missing required configuration keys: {missing}")


# ---------------------------------------------------------------------------
# String Parsing Utilities
# ---------------------------------------------------------------------------

def _safe_parse_tuple(rep: str, expected_len: int) -> Tuple[float, ...]:
    """Parse a string representation of a numeric tuple, e.g., '(1.0, 2.0, 3.0)'."""
    try:
        cleaned = rep.strip().strip("()[]").replace(" ", "")
        if not cleaned:
            return tuple()
        parts = [float(x) for x in cleaned.split(",")]
        if len(parts) != expected_len:
            raise ValueError(f"Expected {expected_len} values, got {len(parts)}")
        return tuple(parts)
    except Exception as e:
        raise ValueError(f"Failed to parse tuple from '{rep}': {e}") from e


def _parse_geometry_from_row(row: Dict[str, Any]) -> GeometryMeta:
    """Extract spacing, origin, direction from metadata row."""
    try:
        spacing = (
            float(row["Spacing_X"]),
            float(row["Spacing_Y"]),
            float(row["Spacing_Z"]),
        )
        origin = _safe_parse_tuple(row.get("Processed_Origin", "0,0,0"), 3)
        direction = _safe_parse_tuple(row.get("Processed_Direction", "1,0,0,0,1,0,0,0,1"), 9)
        shape = (
            int(row["Shape_X"]),
            int(row["Shape_Y"]),
            int(row["Shape_Z"]),
        ) if "Shape_X" in row else None
        return GeometryMeta(spacing=spacing, origin=origin, direction=direction, shape=shape)
    except KeyError as e:
        raise DatasetError(f"Missing required metadata field: {e}") from e


def _read_image_header(filepath: Path) -> Dict[str, Any]:
    """
    Read only the image header (spacing, origin, direction, size) without loading pixel data.
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(filepath))
    reader.ReadImageInformation()
    return {
        "size": reader.GetSize(),
        "spacing": reader.GetSpacing(),
        "origin": reader.GetOrigin(),
        "direction": reader.GetDirection(),
    }


# ---------------------------------------------------------------------------
# Main Dataset Class
# ---------------------------------------------------------------------------

class BrainHemorrhageDataset(Dataset):
    """
    PyTorch Dataset for 3D brain CT intracranial hemorrhage segmentation.

    Provides preprocessed images and binary masks with full geometric and
    clinical metadata.

    Parameters
    ----------
    mode : str
        One of ``'train'``, ``'val'``, ``'test'``, ``'inference'``.
    fold : int, optional
        Cross‑validation fold index (0‑based). Only used when mode is
        ``'train'`` or ``'val'`` with k‑fold splits.
    config : dict, str, Path, optional
        Configuration dictionary or path to a YAML file. When None, falls
        back to ``configs/dataset.yaml`` or environment variable
        ``DATASET_CONFIG``.
    cache_backend : CacheBackend, optional
        Pre‑configured cache backend. If None, an LRU cache of size
        ``cache_size`` is used (when ``cache_size > 0``).
    cache_size : int, optional
        Maximum number of samples to cache per worker when using the
        default LRU cache. Ignored if ``cache_backend`` is provided.
    load_mask : bool, optional
        Whether to load and return the mask. Set to False for pure inference.
    verbose : bool, optional
        Enable detailed logging.

    Raises
    ------
    ConfigurationError, SplitError, MissingFileError, CorruptedFileError
    """

    _SUPPORTED_MODES = ("train", "val", "test", "inference")

    def __init__(
        self,
        mode: str = "train",
        fold: Optional[int] = None,
        config: Optional[Union[str, Path, Dict[str, Any]]] = None,
        cache_backend: Optional[CacheBackend] = None,
        cache_size: Optional[int] = None,
        load_mask: Optional[bool] = None,
        transform=None,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.mode = mode.lower()
        if self.mode not in self._SUPPORTED_MODES:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {self._SUPPORTED_MODES}.")
        self.fold = fold
        # MONAI transform pipeline
        self.transform = transform

        # ---- Configuration ----
        cfg = _resolve_config(config)
        self.processed_images_dir = Path(cfg["processed_images_dir"])
        self.processed_masks_dir = Path(cfg["processed_masks_dir"])
        self.metadata_csv_path = Path(cfg["metadata_csv"])
        self.split_file = Path(cfg["split_file"])
        self.kfold_split_file = Path(cfg["kfold_split_file"])
        self.geometry_tolerance = float(cfg.get("geometry_tolerance", 1e-4))
        self.load_mask_flag = bool(cfg.get("load_mask", True)) if load_mask is None else load_mask
        self.seed = int(cfg.get("seed", 42))
        self.image_dtype = getattr(np, cfg.get("image_dtype", "float32"))
        self.mask_dtype = getattr(np, cfg.get("mask_dtype", "uint8"))

        # ---- Cache ----
        if cache_backend is not None:
            self._cache: Optional[CacheBackend] = cache_backend
        elif cache_size is not None and cache_size > 0:
            self._cache = LRUCache(cache_size)
        elif cfg.get("cache_size", 0) > 0:
            self._cache = LRUCache(cfg["cache_size"])
        else:
            self._cache = None

        # ---- Logging ----
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.DEBUG if verbose else logging.WARNING)

        # ---- Metadata & Case List ----
        self._metadata_dict: Dict[str, Dict[str, Any]] = self._load_metadata()
        self._case_filenames: List[str] = self._get_case_list()
        if not self._case_filenames:
            raise SplitError(f"No cases found for mode '{self.mode}' with fold {self.fold}.")
        self._validate_files_exist()

        self.logger.info(
            "%s initialized in mode '%s' (fold=%s) with %d cases.",
            self.__class__.__name__, self.mode, self.fold, len(self._case_filenames)
        )

    # ------------------------------------------------------------------
    # Metadata Loading
    # ------------------------------------------------------------------

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load Processed_Metadata.csv and return a dict keyed by filename."""
        if not self.metadata_csv_path.exists():
            raise MissingFileError(f"Metadata CSV not found: {self.metadata_csv_path}")
        try:
            df = pd.read_csv(self.metadata_csv_path)
        except Exception as e:
            raise CorruptedFileError(f"Cannot read metadata CSV: {e}") from e
        if "Filename" not in df.columns:
            raise DatasetError("Metadata CSV must contain a 'Filename' column.")

        metadata = {}
        for _, row in df.iterrows():
            fname = row["Filename"]
            metadata[fname] = row.to_dict()
        return metadata

    # ------------------------------------------------------------------
    # Case List from Splits
    # ------------------------------------------------------------------

    def _get_case_list(self) -> List[str]:
        """Determine the list of case filenames for the current mode and fold."""
        if self.mode == "inference":
            files = sorted(p.name for p in self.processed_images_dir.glob("*.nii.gz"))
            return files

        # k‑fold split
        if self.mode in ("train", "val") and self.fold is not None:
            if not self.kfold_split_file.exists():
                raise SplitError(f"K‑fold split file not found: {self.kfold_split_file}")
            with open(self.kfold_split_file, "r") as f:
                kfold = json.load(f)
            fold_key = f"fold_{self.fold}"
            if fold_key not in kfold:
                raise SplitError(f"Fold {self.fold} not found in {self.kfold_split_file}")
            split_dict = kfold[fold_key]
            return sorted(split_dict.get(self.mode, []))

        # Standard dataset splits
        if not self.split_file.exists():
            raise SplitError(f"Split file not found: {self.split_file}")
        with open(self.split_file, "r") as f:
            splits = json.load(f)
        if self.mode not in splits:
            raise SplitError(f"Mode '{self.mode}' not found in {self.split_file}")
        return sorted(splits[self.mode])

    def _validate_files_exist(self) -> None:
        """Check that every required image (and mask, if applicable) exists on disk."""
        missing = []
        for fname in self._case_filenames:
            img_path = self.processed_images_dir / fname
            if not img_path.exists():
                missing.append(("image", str(img_path)))
            if self.load_mask_flag and self.mode != "inference":
                mask_path = self.processed_masks_dir / fname
                if not mask_path.exists():
                    missing.append(("mask", str(mask_path)))
        if missing:
            raise MissingFileError(
                f"{len(missing)} missing file(s) detected. First 5: {missing[:5]}"
            )

    # ------------------------------------------------------------------
    # Context Helpers
    # ------------------------------------------------------------------

    def _case_context(self, filename: str) -> str:
        """Return a string with case identification details for error messages."""
        row = self._metadata_dict.get(filename, {})
        patient = row.get("Patient_ID", "unknown")
        dataset = row.get("Dataset", "unknown")
        return f"file='{filename}', patient='{patient}', dataset='{dataset}', mode='{self.mode}', fold={self.fold}"

    # ------------------------------------------------------------------
    # Core Loading Hooks (extensible)
    # ------------------------------------------------------------------

    def _load_sitk_image(self, filepath: Path) -> sitk.Image:
        """Load a NIfTI file as a SimpleITK image, handle errors gracefully."""
        try:
            return sitk.ReadImage(str(filepath))
        except Exception as e:
            raise CorruptedFileError(f"{self._case_context(filepath.name)}: Failed to read {filepath}: {e}") from e

    @staticmethod
    def _sitk_to_numpy(image: sitk.Image) -> np.ndarray:
        """
        Convert a SimpleITK image to a numpy array with shape (C, Z, Y, X).

        Input layouts (SimpleITK):
            - 3D scalar: (Z, Y, X)
            - 4D vector: (Z, Y, X, C)

        Output layout: (C, Z, Y, X)
        """
        array = sitk.GetArrayFromImage(image)  # (Z, Y, X) or (Z, Y, X, C)
        if array.ndim == 3:
            # Single channel: add a leading channel dimension
            array = array[np.newaxis, ...]          # (1, Z, Y, X)
        elif array.ndim == 4:
            # Multi‑channel: move channels from last axis to first, preserving Z,Y,X order
            array = np.transpose(array, (3, 0, 1, 2))  # (Z,Y,X,C) -> (C, Z, Y, X)
        else:
            raise CorruptedFileError(f"Unexpected image dimensions: {array.shape}")

        # Sanity check: spatial shape must be the three dimensions after the channel
        assert array.ndim == 4, "Internal error: _sitk_to_numpy must return 4D array"
        return array

    def _load_image_array(self, filename: str) -> np.ndarray:
        """Load and convert an image to a (C, D, H, W) numpy array."""
        img_path = self.processed_images_dir / filename
        sitk_img = self._load_sitk_image(img_path)
        arr = self._sitk_to_numpy(sitk_img).astype(self.image_dtype)
        del sitk_img
        return arr

    def _load_mask_array(self, filename: str, image_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
        """Load and validate a mask, returning (1, D, H, W) or None."""
        if not (self.load_mask_flag and self.mode != "inference"):
            return None
        mask_path = self.processed_masks_dir / filename
        if not mask_path.exists():
            raise MissingFileError(f"{self._case_context(filename)}: mask file missing")
        sitk_mask = self._load_sitk_image(mask_path)
        mask_arr = sitk.GetArrayFromImage(sitk_mask)  # (Z, Y, X)
        del sitk_mask
        if mask_arr.ndim != 3:
            raise InvalidMaskError(f"{self._case_context(filename)}: mask should be 3D, got {mask_arr.shape}")
        unique = np.unique(mask_arr)
        if not np.all(np.isin(unique, [0, 1])):
            raise InvalidMaskError(
                f"{self._case_context(filename)}: mask contains values other than 0/1: {unique[:5]}"
            )
        mask_arr = mask_arr[np.newaxis, ...].astype(self.mask_dtype)  # (1, Z, Y, X)
        if image_shape != mask_arr.shape[1:]:
            raise GeometryMismatchError(
                f"{self._case_context(filename)}: shape mismatch image {image_shape} vs mask {mask_arr.shape[1:]}"
            )
        if mask_arr.sum() == 0:
            self.logger.debug("%s: mask is empty (no foreground).", self._case_context(filename))
        return mask_arr

    # ------------------------------------------------------------------
    # Main Case Loading
    # ------------------------------------------------------------------

    def _load_case(self, filename: str) -> BrainHemorrhageSample:
        """Load a single case: image, mask, and metadata."""
        ctx = self._case_context(filename)
        self.logger.debug("Loading %s", ctx)

        image_np = self._load_image_array(filename)
        if np.isnan(image_np).any() or np.isinf(image_np).any():
            raise CorruptedFileError(f"{ctx}: image contains NaN or Inf")

        image_shape = image_np.shape[1:]  # (Z, Y, X)
        mask_np = self._load_mask_array(filename, image_shape)

        meta_row = self._metadata_dict[filename]
        geom = _parse_geometry_from_row(meta_row)

        image_tensor = torch.from_numpy(image_np)
        mask_tensor = torch.from_numpy(mask_np) if mask_np is not None else None

        sample: BrainHemorrhageSample = {
            "image": image_tensor,
            "mask": mask_tensor,
            "case_id": filename.replace(".nii.gz", ""),
            "patient_id": meta_row.get("Patient_ID", "unknown"),
            "dataset": meta_row.get("Dataset", "unknown"),
            "spacing": geom.spacing,
            "origin": geom.origin,
            "direction": geom.direction,
            "metadata": meta_row,
        }
        return sample
        # ------------------------------------------------------------------
    # PyTorch Dataset Methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of cases in the current dataset split."""
        return len(self._case_filenames)

    def __getitem__(self, idx: int) -> BrainHemorrhageSample:
        """
        Return a single sample.

        Workflow
        --------
        Disk / Cache
            ↓
        Online MONAI Transform
            ↓
        Return Sample

        Notes
        -----
        • Cache stores ONLY raw samples.
        • Training augmentation is applied ONLINE.
        • Validation/Test transforms remain deterministic.
        """

        filename = self._case_filenames[idx]

        # ----------------------------------------------------------
        # Load raw sample from cache (if available)
        # ----------------------------------------------------------
        sample = None

        if self._cache is not None:
            sample = self._cache.get(filename)

        # ----------------------------------------------------------
        # Otherwise load from disk
        # ----------------------------------------------------------
        if sample is None:
            try:
                sample = self._load_case(filename)
            except Exception:
                self.logger.error(
                    "Failed to load case '%s'",
                    filename,
                    exc_info=True,
                )
                raise

            # Store ONLY the raw sample in cache
            if self._cache is not None:
                self._cache.put(filename, sample)

        # ----------------------------------------------------------
        # Apply ONLINE transforms
        # ----------------------------------------------------------
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __repr__(self) -> str:
        """String representation of the dataset."""
        return (
            f"{self.__class__.__name__}("
            f"mode='{self.mode}', "
            f"fold={self.fold}, "
            f"cases={len(self._case_filenames)})"
        )


    # ------------------------------------------------------------------
    # Helper: load the full list of known filenames from the standard split file
    # ------------------------------------------------------------------

    def _load_all_split_filenames(self) -> set[str]:
        """
        Return a set of all filenames present in **any** split (train/val/test)
        from ``dataset_splits.json``.

        This is used during ``verify()`` to decide whether a file in the processed
        images directory is truly an orphan or just belongs to another split.
        """
        if not self.split_file.exists():
            # If the file is missing, we cannot distinguish orphans –
            # assume all images are valid (no orphan warnings).
            return set()

        try:
            with open(self.split_file, "r") as f:
                splits = json.load(f)
        except Exception:
            self.logger.warning("Failed to parse split file %s", self.split_file)
            return set()

        all_files = set()
        for split_name in ("train", "val", "test"):
            if split_name in splits:
                all_files.update(splits[split_name])
        return all_files

    # ------------------------------------------------------------------
    # Verification & Integrity Checks
    # ------------------------------------------------------------------

    def verify(self, level: str = "full") -> Dict[str, Any]:
        """
        Perform dataset integrity checks.

        Parameters
        ----------
        level : str
            'light'  – checks file existence, metadata presence, duplicate filenames.
            'full'   – also verifies geometry (spacing, origin, direction) using only
                       image headers (no pixel data loaded).

        Returns
        -------
        dict with keys ``status`` ('pass' or 'fail'), ``errors``, ``warnings``.
        """
        errors: List[str] = []
        warnings: List[str] = []
        fnames_set = set(self._case_filenames)

        # 1. Duplicate filenames (in the current split)
        if len(fnames_set) != len(self._case_filenames):
            errors.append("Duplicate filenames found in case list.")

        # 2. File existence & metadata presence
        for fname in self._case_filenames:
            ctx = self._case_context(fname)
            img_path = self.processed_images_dir / fname
            if not img_path.exists():
                errors.append(f"{ctx}: image missing")
                continue

            if fname not in self._metadata_dict:
                errors.append(f"{ctx}: metadata missing")
            else:
                for col in ["Patient_ID", "Dataset", "Spacing_X", "Spacing_Y", "Spacing_Z"]:
                    if col not in self._metadata_dict[fname] or pd.isna(self._metadata_dict[fname][col]):
                        errors.append(f"{ctx}: missing metadata field '{col}'")

            if self.load_mask_flag and self.mode != "inference":
                mask_path = self.processed_masks_dir / fname
                if not mask_path.exists():
                    errors.append(f"{ctx}: mask missing")
                    continue
                try:
                    header = _read_image_header(mask_path)
                    # Quick mask binary check via first few voxels (optional, can't check full without loading)
                    # We'll rely on the preprocessing pipeline for mask correctness.
                except Exception as e:
                    errors.append(f"{ctx}: corrupted mask header ({e})")

        # 3. Geometry cross‑checks (header only, no pixel loading)
        if level == "full":
            for fname in self._case_filenames:
                ctx = self._case_context(fname)
                img_path = self.processed_images_dir / fname
                mask_path = self.processed_masks_dir / fname
                if not img_path.exists() or not mask_path.exists():
                    continue
                try:
                    img_header = _read_image_header(img_path)
                    mask_header = _read_image_header(mask_path)
                except Exception as e:
                    errors.append(f"{ctx}: failed to read header ({e})")
                    continue

                if img_header["size"] != mask_header["size"]:
                    errors.append(f"{ctx}: size mismatch img={img_header['size']} mask={mask_header['size']}")
                if not np.allclose(img_header["spacing"], mask_header["spacing"], atol=self.geometry_tolerance):
                    errors.append(f"{ctx}: spacing mismatch img={img_header['spacing']} mask={mask_header['spacing']}")
                if not np.allclose(img_header["origin"], mask_header["origin"], atol=self.geometry_tolerance):
                    errors.append(f"{ctx}: origin mismatch img={img_header['origin']} mask={mask_header['origin']}")
                if not np.allclose(img_header["direction"], mask_header["direction"], atol=self.geometry_tolerance):
                    errors.append(f"{ctx}: direction mismatch")

        # 4. Orphan files & duplicate patients
        if self.mode != "inference":
            # Use the complete list of filenames from the split file (all splits)
            # to avoid false orphan warnings when verify() is called from a single split.
            known_filenames = self._load_all_split_filenames()
            all_images = set(p.name for p in self.processed_images_dir.glob("*.nii.gz"))
            orphans = all_images - known_filenames
            if orphans:
                warnings.append(f"Orphan images (not in any split): {sorted(orphans)}")

        if self._metadata_dict:
            pat_counts = pd.Series([m.get("Patient_ID") for m in self._metadata_dict.values()]).value_counts()
            dupes = pat_counts[pat_counts > 1]
            if not dupes.empty:
                warnings.append(f"Duplicate Patient_IDs: {dupes.to_dict()}")

        status = "pass" if not errors else "fail"
        return {"status": status, "errors": errors, "warnings": warnings}

    # ------------------------------------------------------------------
    # Dataset Summary Statistics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """
        Compute summary statistics using metadata only (no pixel loading).

        Returns a dictionary with keys:
            total_cases, per_dataset, patient_count,
            shape_stats (min/max/mean per axis),
            spacing_stats (min/max/mean per axis),
            empty_masks, mean_lesion_volume_mm3,
            memory_estimate_mb_per_sample, cache_info.
        """
        shapes = {"x": [], "y": [], "z": []}
        spacings = {"x": [], "y": [], "z": []}
        per_dataset: Dict[str, int] = {}
        patients = set()
        empty_mask_count = 0
        lesion_volumes = []

        for fname, row in self._metadata_dict.items():
            if fname not in self._case_filenames:
                continue
            ds = row.get("Dataset", "unknown")
            per_dataset[ds] = per_dataset.get(ds, 0) + 1
            patients.add(row.get("Patient_ID", "unknown"))
            shapes["x"].append(row.get("Shape_X", 0))
            shapes["y"].append(row.get("Shape_Y", 0))
            shapes["z"].append(row.get("Shape_Z", 0))
            spacings["x"].append(row.get("Spacing_X", 0))
            spacings["y"].append(row.get("Spacing_Y", 0))
            spacings["z"].append(row.get("Spacing_Z", 0))
            if row.get("Empty_Mask", False) or row.get("Positive_Voxels", 0) == 0:
                empty_mask_count += 1
            lesion_volumes.append(row.get("Lesion_Volume_mm3", 0))

        def _stats(arr):
            if not arr:
                return {"min": None, "max": None, "mean": None}
            return {"min": float(np.min(arr)), "max": float(np.max(arr)), "mean": float(np.mean(arr))}

        if shapes["x"]:
            avg_shape = (int(np.mean(shapes["x"])), int(np.mean(shapes["y"])), int(np.mean(shapes["z"])))
            channels = 3  # typical multi‑window; could be read from config
            mem_mb = channels * avg_shape[0] * avg_shape[1] * avg_shape[2] * np.dtype(self.image_dtype).itemsize / (1024 ** 2)
        else:
            mem_mb = 0.0

        return {
            "total_cases": len(self._case_filenames),
            "per_dataset": per_dataset,
            "patient_count": len(patients),
            "shape_stats": {ax: _stats(shapes[ax]) for ax in "xyz"},
            "spacing_stats": {ax: _stats(spacings[ax]) for ax in "xyz"},
            "empty_masks": empty_mask_count,
            "mean_lesion_volume_mm3": float(np.mean(lesion_volumes)) if lesion_volumes else None,
            "memory_estimate_mb_per_sample": round(mem_mb, 2),
            "cache_info": str(self._cache) if self._cache else "disabled",
        }

    # ------------------------------------------------------------------
    # Affine & MONAI Helpers
    # ------------------------------------------------------------------

    def get_affine(self, filename: str) -> np.ndarray:
        """
        Reconstruct a 4×4 affine matrix from the metadata for a given case.

        Raises MissingFileError if filename is not in the metadata dictionary.
        """
        if filename not in self._metadata_dict:
            raise MissingFileError(f"{filename} not found in metadata")
        geom = _parse_geometry_from_row(self._metadata_dict[filename])
        affine = np.eye(4)
        affine[:3, :3] = np.array(geom.direction).reshape(3, 3) * np.diag(geom.spacing)
        affine[:3, 3] = geom.origin
        return affine

    @staticmethod
    def to_meta_tensor(sample: BrainHemorrhageSample) -> BrainHemorrhageSample:
        """
        Convert a dataset sample into MONAI MetaTensors (requires MONAI).

        Attaches affine and meta information to the image and mask tensors.
        """
        try:
            from monai.data.meta_tensor import MetaTensor
        except ImportError:
            raise RuntimeError("MONAI is required for MetaTensor conversion.")

        spacing = sample["spacing"]
        origin = sample["origin"]
        direction = np.array(sample["direction"]).reshape(3, 3)
        affine = np.eye(4)
        affine[:3, :3] = direction * np.diag(spacing)
        affine[:3, 3] = origin

        image_meta = MetaTensor(sample["image"], affine=affine)
        result: BrainHemorrhageSample = {**sample, "image": image_meta}  # type: ignore[typeddict-item]
        if sample["mask"] is not None:
            mask_meta = MetaTensor(sample["mask"], affine=affine)
            result["mask"] = mask_meta  # type: ignore[typeddict-item]
        return result

    @staticmethod
    def worker_init_fn(worker_id: int, seed: int = 42) -> None:
        """Seed Python, NumPy, and PyTorch for deterministic DataLoader workers."""
        torch.manual_seed(seed + worker_id)
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    # ------------------------------------------------------------------
    # Public Metadata Access
    # ------------------------------------------------------------------

    def get_metadata_dataframe(self) -> pd.DataFrame:
        """Return the full metadata as a pandas DataFrame (for external analysis)."""
        return pd.DataFrame.from_dict(self._metadata_dict, orient="index")

    @property
    def case_ids(self) -> List[str]:
        """Return the list of case filenames in this dataset split."""
        return self._case_filenames.copy()


# ---------------------------------------------------------------------------
# Quick test / debugging (can be removed in production)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ds = BrainHemorrhageDataset(mode="train", verbose=True)
    print(f"Dataset size: {len(ds)}")
    sample = ds[0]
    print("Sample keys:", sample.keys())
    print("Image shape:", sample["image"].shape)
    if sample["mask"] is not None:
        print("Mask shape:", sample["mask"].shape)

    print("\n--- Verification (light) ---")
    verif = ds.verify("light")
    print("Status:", verif["status"])
    print("Errors:", verif["errors"][:3])
    print("Warnings:", verif["warnings"][:3])

    print("\n--- Summary ---")
    summ = ds.summary()
    for k, v in summ.items():
        print(f"{k}: {v}")