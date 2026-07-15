from __future__ import annotations
import logging
import platform
from pathlib import Path
from typing import Any, Dict
import torch
import psutil
from training.callbacks import TrainerCallback

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger("WandbLogger")

def is_main_process() -> bool:
    if not torch.distributed.is_initialized(): return True
    return torch.distributed.get_rank() == 0

class WandbLogger(TrainerCallback):
    def __init__(self, output_dir: Path, config: Dict[str, Any], global_config: Dict[str, Any] = None):
        self.output_dir = output_dir
        self.config = config
        self.global_config = global_config or {}
        self.enabled = config.get("enabled", False)
        if self.enabled and not WANDB_AVAILABLE:
            logger.warning("W&B enabled but wandb not installed.")
            self.enabled = False
        self._active = False

    def on_fit_begin(self, trainer) -> None:
        if not self.enabled or not is_main_process(): return
        try:
            import monai
            self.global_config["system"] = {
                "hostname": platform.node(),
                "pytorch_version": torch.__version__,
                "monai_version": monai.__version__,
                "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
            }
            self._run = wandb.init(
                project=self.config.get("project", "Hemorrhage_SOTA_V2"),
                entity=self.config.get("entity", None),
                name=self.config.get("run_name", None),
                tags=self.config.get("tags", []),
                config=self.global_config,
                resume=self.config.get("resume", "allow"),
                dir=str(self.output_dir),
                save_code=self.config.get("save_code", True)
            )
            self._active = True
        except Exception as e: logger.warning(f"W&B Init Failed: {e}")

    def on_epoch_end(self, trainer, epoch: int, metrics: dict) -> None:
        if not self._active or not is_main_process(): return
        try:
            log_dict = dict(metrics)
            log_dict["system/cpu_usage"] = psutil.cpu_percent()
            log_dict["system/ram_usage"] = psutil.virtual_memory().percent
            if torch.cuda.is_available():
                log_dict["system/gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
            wandb.log(log_dict, step=epoch)
        except Exception as e: pass

    def on_checkpoint_saved(self, trainer, ckpt_path: str, save_time: float, is_best: bool) -> None:
        if not self._active or not is_main_process() or not is_best or not self.config.get("log_model", True): return
        try:
            artifact = wandb.Artifact(name="model-best", type="model")
            artifact.add_file(ckpt_path)
            wandb.log_artifact(artifact)
        except Exception: pass

    def on_fit_end(self, trainer) -> None:
        if not self._active or not is_main_process(): return
        try:
            plots_dir = self.output_dir / "plots"
            if plots_dir.exists():
                for plot_file in plots_dir.glob("*.png"):
                    wandb.log({f"plots/{plot_file.stem}": wandb.Image(str(plot_file))})
            wandb.finish()
            self._active = False
        except Exception: pass
