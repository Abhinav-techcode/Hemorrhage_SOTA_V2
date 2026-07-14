"""
training/train.py

Research-Grade Training Launcher
================================

HybridMedNeXt++ Framework

Responsibilities
----------------
1. Parse command-line arguments.
2. Configure runtime environment.
3. Load and validate configuration files.
4. Build all framework components.
5. Launch SegmentationTrainer.

No training logic should be implemented here.
"""

from __future__ import annotations

import argparse
import logging
import os

# PyTorch NVIDIA MIG Workarounds MUST be set before torch is imported!
# 1. expandable_segments:True prevents MIG NVML driver assert crashes by allocating memory in smaller chunks.
# 2. PYTORCH_NVML_BASED_CUDA_CHECK=0 forces PyTorch to avoid polling NVML directly.
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"

import platform
import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*unable to generate class balanced samples.*")
from typing import Any, Dict

import numpy as np
import torch
import yaml

# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ==========================================================
# Framework Imports
# ==========================================================

from models.model_factory import build_model

from datasets.dataloader import (
    BrainHemorrhageDataModule,
    DataLoaderConfig,
)

from datasets.transforms import TransformFactory

from evaluation.losses import LossFactory
from evaluation.metric_engine import ResearchMetricEngine

from training.optimizer import OptimizerFactory
from training.scheduler import (
    SchedulerFactory,
    scheduler_step,
)

from training.trainer import SegmentationTrainer
from training.config import TrainerConfig

# ==========================================================
# Logger
# ==========================================================

LOGGER = logging.getLogger("HybridSegFormer-UMamba")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

# ==========================================================
# YAML Loader
# ==========================================================

def load_yaml(path: Path) -> Dict[str, Any]:
    """
    Safely load YAML configuration.
    """

    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg if cfg is not None else {}


# ==========================================================
# Random Seed
# ==========================================================

def seed_everything(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True


# ==========================================================
# CUDA Optimisation
# ==========================================================

def configure_cuda():

    if not torch.cuda.is_available():
        return

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.backends.cudnn.allow_tf32 = True

    torch.backends.cudnn.benchmark = False

    torch.set_float32_matmul_precision("high")


import json
import subprocess
import hashlib

def compute_fingerprint(data: Any) -> str:
    try:
        return hashlib.md5(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()
    except Exception:
        return "Unknown"

def save_experiment_metadata(configs: Dict[str, Any], args: argparse.Namespace, exp_dir: Path):
    reports_dir = exp_dir / "reports"
    
    import monai
    metadata = {
        "environment": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "monai_version": getattr(monai, "__version__", "unknown"),
            "cuda_available": torch.cuda.is_available(),
        },
        "git": {},
        "configs": configs,
        "args": vars(args),
        "versions": {
            "model": "v1.0",
            "architecture": configs.get("model", {}).get("name", "Unknown"),
            "dataset": "v1.0",
            "transform": "v1.0",
            "loss": "v1.0",
            "metric": "v1.0"
        },
        "fingerprints": {
            "dataset_config": compute_fingerprint(configs.get("dataset", {})),
            "preprocessing_config": compute_fingerprint(configs.get("preprocessing", {})),
            "augmentation_config": compute_fingerprint(configs.get("augmentation", {}))
        },
        "seed": getattr(args, "seed", 42),
    }
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        metadata["environment"]["gpu"] = {
            "name": props.name,
            "memory_gb": props.total_memory / 1024**3,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version()
        }
        
    try:
        metadata["git"]["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        metadata["git"]["branch"] = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
    except Exception:
        metadata["git"]["error"] = "Could not retrieve git info"
        
    with open(reports_dir / "Experiment_Metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    LOGGER.info(f"Experiment metadata saved to {reports_dir / 'Experiment_Metadata.json'}")

# ==========================================================
# Environment Summary
# ==========================================================

def print_environment():

    print()

    print("=" * 80)
    print("HybridSegFormer-UMamba Research Framework")
    print("=" * 80)

    print(f"Python Version : {platform.python_version()}")

    print(f"PyTorch        : {torch.__version__}")

    print(f"CUDA Available : {torch.cuda.is_available()}")

    if torch.cuda.is_available():

        props = torch.cuda.get_device_properties(0)

        print(f"GPU            : {props.name}")

        print(
            f"GPU Memory     : "
            f"{props.total_memory / 1024**3:.2f} GB"
        )

        print(f"CUDA Version   : {torch.version.cuda}")

        print(
            f"cuDNN          : "
            f"{torch.backends.cudnn.version()}"
        )

        print(
            f"BF16 Support   : "
            f"{torch.cuda.is_bf16_supported()}"
        )

        print(
            f"TF32 Enabled   : "
            f"{torch.backends.cuda.matmul.allow_tf32}"
        )

    print("=" * 80)
    print()


# ==========================================================
# Parameter Counter
# ==========================================================

def print_model_summary(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    LOGGER.info(
        "Total Parameters     : %s",
        f"{total:,}",
    )

    LOGGER.info(
        "Trainable Parameters : %s",
        f"{trainable:,}",
    )


# ==========================================================
# Command Line
# ==========================================================

def build_parser():

    parser = argparse.ArgumentParser(
        description="HybridMedNeXt++ Training"
    )

    parser.add_argument(
        "--config_dir",
        default="configs",
        type=str,
    )

    parser.add_argument(
        "--resume",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu", "mps"],
    )

    parser.add_argument(
        "--experiment",
        default="Hemorrhage_SOTA",
        type=str,
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Research Debug Mode (intrusive hooks, gradient stats) for 1-epoch runs."
    )

    return parser

# ==========================================================
# Configuration Loader
# ==========================================================

def load_all_configs(config_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load every YAML configuration file required by the framework.
    """

    LOGGER.info("Loading configuration files...")

    configs = {
        "dataset": load_yaml(config_dir / "dataset.yaml"),

        "model": load_yaml(config_dir / "model.yaml"),

        "training": load_yaml(config_dir / "training.yaml"),

        "optimizer": load_yaml(config_dir / "optimizer.yaml"),

        "scheduler": load_yaml(config_dir / "scheduler.yaml"),

        "loss": load_yaml(config_dir / "loss.yaml"),

        "metrics": load_yaml(config_dir / "metrics.yaml"),

        "augmentation": load_yaml(config_dir / "augmentation.yaml"),

        "preprocessing": load_yaml(config_dir / "preprocessing.yaml"),

    }

    LOGGER.info("Configuration loading completed.")

    return configs


# ==========================================================
# Build Trainer Configuration
# ==========================================================

def build_trainer_config(cfg: Dict[str, Any]) -> TrainerConfig:

    return TrainerConfig(

        epochs=cfg.get("epochs", 250),

        batch_size=cfg.get("batch_size", 2),

        grad_accum_steps=cfg.get("grad_accum_steps", 1),

        max_grad_norm=cfg.get("max_grad_norm", 1.0),

        mixed_precision=cfg.get("mixed_precision", True),

        amp_dtype=cfg.get("amp_dtype", "bfloat16"),

        compile_model=cfg.get("compile_model", True),

        channels_last=cfg.get("channels_last", True),

        log_freq=cfg.get("log_freq", 10),

        save_dir=cfg.get("save_dir", "checkpoints"),

        patience=cfg.get("patience", 30),

        enable_profiler=cfg.get("enable_profiler", False),

        ema_enabled=cfg.get("ema_enabled", False),

        swa_enabled=cfg.get("swa_enabled", False),

        disable_dashboard=cfg.get("disable_dashboard", False),
        
        full_config=cfg.get("full_config", None),

    )


# ==========================================================
# Build Entire Framework
# ==========================================================

def build_framework(configs):

    LOGGER.info("=" * 80)
    LOGGER.info("Building Framework")
    LOGGER.info("=" * 80)

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    LOGGER.info("Building HybridSegFormer-UMamba")

    model = build_model(configs["model"])

    print_model_summary(model)

    # ------------------------------------------------------
    # Transform Factory
    # ------------------------------------------------------

    LOGGER.info("Building MONAI Transform Pipelines")

    transform_factory = TransformFactory(
        "configs/augmentation.yaml"
    )

    # ------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------

    LOGGER.info("Building DataModule")

    loader_cfg = DataLoaderConfig(

        batch_size=configs["training"].get(
            "batch_size",
            2,
        ),

        num_workers=configs["training"].get(
            "num_workers",
            min(os.cpu_count(), 8),
        ),

        pin_memory=True,

        persistent_workers=True,

        prefetch_factor=2,

        drop_last=True,

        seed=configs["training"].get(
            "seed",
            42,
        ),

        dataset_config=configs["dataset"],
        
        fold=configs["training"].get("fold", None),

    )

    data_module = BrainHemorrhageDataModule(

        config=loader_cfg,

        transform_factory=transform_factory,

    )

    train_loader = data_module.build_train_loader()

    val_loader = data_module.build_validation_loader()

    # ------------------------------------------------------
    # Loss
    # ------------------------------------------------------

    LOGGER.info("Building Loss Engine")

    loss_fn = LossFactory.build(
        configs["loss"]
    )

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    LOGGER.info("Building Metric Engine")

    metric_manager = ResearchMetricEngine()

    # ------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------

    LOGGER.info("Building Optimizer")

    optimizer = OptimizerFactory.build(

        model,

        configs["optimizer"],

    )

    # ------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------

    LOGGER.info("Building Scheduler")

    scheduler = SchedulerFactory.build(

        optimizer,

        configs["scheduler"],

    )

    # Pass full configs into trainer config
    full_cfg = {k: v for k, v in configs.items() if k != "training"}
    full_cfg["training"] = {k: v for k, v in configs["training"].items()}
    # Ensure augmentation config dictionary is present instead of just path
    full_cfg["augmentation"] = transform_factory.get_config_dict()
    configs["training"]["full_config"] = full_cfg
    
    trainer_cfg = build_trainer_config(
        configs["training"]
    )

    LOGGER.info("=" * 80)
    LOGGER.info("Framework Ready")
    LOGGER.info("=" * 80)

    return (

        model,

        train_loader,

        val_loader,

        optimizer,

        scheduler,

        loss_fn,

        metric_manager,

        trainer_cfg,

    )
# ==========================================================
# Main
# ==========================================================

def main():

    args = build_parser().parse_args()

    configure_cuda()

    seed_everything(args.seed)

    print_environment()

    config_dir = Path(args.config_dir)

    configs = load_all_configs(config_dir)

    if args.debug:
        if "health_monitor" not in configs["training"]:
            configs["training"]["health_monitor"] = {}
        configs["training"]["health_monitor"]["enabled"] = True
        configs["training"]["epochs"] = 1  # Debug mode is only for 1 epoch

    import datetime
    exp_id = f"EXP_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = Path("outputs") / exp_id
    
    for sub in ["checkpoints", "logs", "reports", "curves", "qualitative", "metrics", "failure_cases"]:
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)
        
    configs["training"]["save_dir"] = str(exp_dir / "checkpoints")
    configs["training"]["exp_dir"] = str(exp_dir)

    (
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        metric_manager,
        trainer_cfg,
    ) = build_framework(configs)

    save_experiment_metadata(configs, args, exp_dir)

    LOGGER.info("Creating Segmentation Trainer")

    trainer = SegmentationTrainer(

        model=model,

        optimizer=optimizer,

        loss_fn=loss_fn,

        train_loader=train_loader,

        val_loader=val_loader,

        config=trainer_cfg,

        metric_manager=metric_manager,

        scheduler=scheduler,

        device=args.device,

    )

    if args.resume is not None:

        LOGGER.info(
            "Resuming from checkpoint : %s",
            args.resume,
        )

        trainer.resume(
            Path(args.resume)
        )

    LOGGER.info("=" * 80)
    LOGGER.info("Starting Training")
    LOGGER.info("=" * 80)

    try:

        trainer.fit()

    except KeyboardInterrupt:

        LOGGER.warning(
            "Training interrupted by user."
        )

    except RuntimeError as e:

        LOGGER.exception(
            "RuntimeError occurred during training."
        )

        raise e

    except Exception as e:

        LOGGER.exception(
            "Unexpected exception during training."
        )

        raise e

    LOGGER.info("=" * 80)
    LOGGER.info("Training Finished Successfully")
    LOGGER.info("=" * 80)
    
    # Trigger Milestone D
    try:
        from evaluation.post_training_visualizer import trigger_visualization_pipeline
        try:
            trigger_visualization_pipeline(exp_dir, configs)
        except Exception as e:
            LOGGER.exception(f"Post-training visualization failed: {e}")
    except ImportError:
        LOGGER.warning("evaluation.post_training_visualizer not found. Skipping Milestone D.")

# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()