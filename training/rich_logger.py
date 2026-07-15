"""
training/rich_logger.py
Decoupled UI, Logging, and Reporting components for HybridMedNeXt++
"""
from __future__ import annotations
import logging
import time
import os
import csv
import json
import traceback
import subprocess
from pathlib import Path

import torch

# Safe import of rich for graceful degradation
try:
    import rich
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import (
        Progress, TextColumn, BarColumn, TimeRemainingColumn
    )
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    Panel = None
    Table = None
    Progress = None
    TextColumn = None
    BarColumn = None
    TimeRemainingColumn = None
    box = None
    RichHandler = None

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from training.callbacks import TrainerCallback

console = Console() if RICH_AVAILABLE else None

def print_ui(renderable):
    if not RICH_AVAILABLE:
        if isinstance(renderable, str):
            logging.info(renderable.replace("[", "").replace("]", ""))
        return
    console.print(renderable)

def is_main_process():
    return int(os.environ.get("LOCAL_RANK", "0")) == 0

def setup_rich_logger(log_dir: str):
    if not is_main_process(): return
    os.makedirs(log_dir, exist_ok=True)
    log_file = Path(log_dir) / "training.log"
    
    file_formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    
    handlers = [file_handler]
    
    if RICH_AVAILABLE:
        rich_handler = RichHandler(rich_tracebacks=True, markup=True, show_path=False)
        handlers.insert(0, rich_handler)
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(file_formatter)
        handlers.insert(0, stream_handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s" if RICH_AVAILABLE else "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="[%X]" if RICH_AVAILABLE else "%H:%M:%S",
        handlers=handlers,
        force=True
    )


class MetadataLogger(TrainerCallback):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def on_fit_begin(self, trainer) -> None:
        if not is_main_process(): return
        
        try:
            from version import __version__
        except ImportError:
            __version__ = "1.0.0"

        try:
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
            git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
        except Exception:
            git_commit = "unknown"
            git_branch = "unknown"

        import platform
        import psutil
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode('utf-8')
        except Exception:
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"

        run_info = {
            "framework": "HybridMedNeXt++",
            "version": __version__,
            "experiment_name": getattr(trainer.config, "experiment_name", "experiment"),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hostname": platform.node(),
            "git_branch": git_branch,
            "git_commit": git_commit,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "gpu": gpu_name,
            "seed": getattr(trainer.config, "seed", 42)
        }

        with open(self.output_dir / "run_info.json", "w") as f:
            json.dump(run_info, f, indent=4)

        try:
            pip_freeze = subprocess.check_output(['pip', 'freeze'], stderr=subprocess.DEVNULL).decode('ascii')
        except Exception:
            pip_freeze = "Unavailable"

        env_txt = f"""HybridMedNeXt++ Environment Snapshot
======================================================
Framework Version : {__version__}
Hostname          : {platform.node()}
OS                : {platform.platform()}
CPU               : {platform.processor()} ({psutil.cpu_count(logical=True)} threads)
System RAM        : {psutil.virtual_memory().total / (1024**3):.2f} GB
GPU               : {gpu_name}

Python            : {platform.python_version()}
PyTorch           : {torch.__version__}
CUDA              : {run_info['cuda']}
Git Branch        : {git_branch}
Git Commit        : {git_commit}

--- PIP FREEZE ---
{pip_freeze}
"""
        with open(self.output_dir / "environment.txt", "w") as f:
            f.write(env_txt)

        # Model Summary
        try:
            total_params = sum(p.numel() for p in trainer.model.parameters())
            trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
            with open(self.output_dir / "model_summary.txt", "w") as f:
                f.write(f"Total Parameters: {total_params:,}\n")
                f.write(f"Trainable Parameters: {trainable_params:,}\n")
                f.write("\n" + str(trainer.model) + "\n")
        except Exception:
            pass


class CSVLogger(TrainerCallback):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        self.best_metrics = {}

    def on_epoch_end(self, trainer, epoch: int, metrics: dict) -> None:
        if not is_main_process(): return
        
        metrics_dict = {"epoch": epoch}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                metrics_dict[k] = v

        csv_path = self.logs_dir / "metrics.csv"
        json_path = self.logs_dir / "metrics.json"
        
        file_exists = csv_path.is_file()
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics_dict)
            
        data = []
        if json_path.is_file():
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
            except Exception:
                pass
        data.append(metrics_dict)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
            
        self.best_metrics = metrics_dict

    def on_checkpoint_saved(self, trainer, ckpt_path: str, save_time: float, is_best: bool) -> None:
        if not is_main_process() or not is_best: return
        
        best = {}
        requested = ["dice", "val_loss", "epoch"]
        for k, v in self.best_metrics.items():
            if any(req in k.lower() for req in requested) and isinstance(v, (int, float)):
                best[f"best_{k}"] = v
        best["checkpoint"] = ckpt_path
        
        with open(self.logs_dir / "best_metrics.json", "w") as f:
            json.dump(best, f, indent=4)


class RichLogger(TrainerCallback):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.progress = None
        self.train_task = None
        self.val_task = None
        self.best_dice = 0.0
        self.best_epoch = 1
        self.prev_val_metrics = {}

    def on_fit_begin(self, trainer) -> None:
        if not RICH_AVAILABLE or not is_main_process(): return
        
        total_params = sum(p.numel() for p in trainer.model.parameters())
        table = Table(title="STARTUP SUMMARY", box=box.DOUBLE_EDGE, title_style="bold cyan")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        mixed_prec = "Disabled"
        if getattr(trainer.config, "mixed_precision", False):
            mixed_prec = "BF16" if trainer.amp_dtype == torch.bfloat16 else "FP16"
            
        table.add_row("Framework", "HybridMedNeXt++")
        table.add_row("Model Parameters", f"{total_params:,}")
        table.add_row("Optimizer", trainer.optimizer.__class__.__name__)
        table.add_row("Loss", trainer.loss_fn.__class__.__name__)
        table.add_row("GPU", torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else "CPU")
        table.add_row("Mixed Precision", mixed_prec)
        
        print_ui(Panel(table, border_style="cyan"))

    def on_epoch_begin(self, trainer, epoch: int) -> None:
        if not RICH_AVAILABLE or not is_main_process(): return
        
        lr = trainer.optimizer.param_groups[0]['lr']
        patience_str = f"{trainer.epochs_without_improvement} / {trainer.config.patience}"
        
        header = f"""[bold cyan]================================================================================[/]
[bold white]Epoch {epoch} / {trainer.config.epochs}[/]
[bold cyan]================================================================================[/]

[white]Learning Rate       :[/] [blue]{lr:.2e}[/]
[white]Best Validation Dice:[/] [green]{self.best_dice:.4f}[/]
[white]Best Epoch          :[/] [green]{self.best_epoch}[/]
[white]Early Stopping      :[/] [yellow]{patience_str}[/]"""
        print_ui(header)
        
        if self.progress is not None and self.progress.live.is_started:
            self.progress.stop()
            
        self.progress = Progress(
            TextColumn("[bold cyan]Epoch {task.fields[epoch]}[/]"),
            BarColumn(complete_style="green", finished_style="green", bar_width=20),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("  [bold]Loss[/] {task.fields[loss]:.4f}"),
            TextColumn("  [bold]Dice[/] {task.fields[dice]:.4f}"),
            TextColumn("  [bold]ETA[/]"),
            TimeRemainingColumn(),
            console=console,
            transient=True
        )
        self.progress.start()
        self.train_task = self.progress.add_task(
            "Training ...", total=len(trainer.train_loader), epoch=epoch, loss=0.0, dice=0.0
        )

    def on_train_batch_end(self, trainer, batch_idx: int, loss: float) -> None:
        if not RICH_AVAILABLE or not is_main_process() or not self.progress: return
        
        tracker = trainer.live_tracker
        self.progress.update(self.train_task, advance=1, loss=loss, dice=tracker.train_dice)

    def on_validation_begin(self, trainer) -> None:
        if not RICH_AVAILABLE or not is_main_process() or not self.progress: return
        
        self.val_task = self.progress.add_task(
            "Validation ...", total=len(trainer.val_loader), epoch=trainer.current_epoch, loss=0.0, dice=0.0
        )

    def on_validation_batch_end(self, trainer, batch_idx: int) -> None:
        if not RICH_AVAILABLE or not is_main_process() or not self.progress or self.val_task is None: return
        self.progress.update(self.val_task, advance=1, loss=getattr(trainer, "current_val_loss", 0.0))

    def on_validation_end(self, trainer) -> None:
        if not RICH_AVAILABLE or not is_main_process() or not self.progress or self.val_task is None: return
        # Keep the progress bar active, it will be destroyed in on_epoch_end
        pass

    def on_epoch_end(self, trainer, epoch: int, metrics: dict) -> None:
        if not RICH_AVAILABLE or not is_main_process(): return
        
        self._cleanup_progress()
            
        val_dice = metrics.get("val_dice", metrics.get("dice", 0.0))
        val_loss = metrics.get("val_loss", 0.0)
        
        pred_mean = metrics.get("Pred Mean", 0.0)
        pred_std = metrics.get("Pred Std", 0.0)
        pred_fg = metrics.get("Pred FG %", 0.0)
        
        prev_dice = self.prev_val_metrics.get("val_dice", val_dice)
        prev_loss = self.prev_val_metrics.get("val_loss", val_loss)
        
        diff_dice = val_dice - prev_dice
        diff_loss = val_loss - prev_loss
        
        comp_table = Table(title="EPOCH COMPARISON", box=box.DOUBLE_EDGE, title_style="bold magenta")
        comp_table.add_column("Metric", style="cyan")
        comp_table.add_column("Previous", style="white")
        comp_table.add_column("Current", style="white")
        comp_table.add_column("Change", style="bold")
        
        dice_color = "green" if diff_dice > 0 else ("red" if diff_dice < 0 else "white")
        loss_color = "green" if diff_loss < 0 else ("red" if diff_loss > 0 else "white")
        
        comp_table.add_row("Validation Dice", f"{prev_dice:.4f}", f"{val_dice:.4f}", f"[{dice_color}]{diff_dice:+.4f}[/]")
        comp_table.add_row("Validation Loss", f"{prev_loss:.4f}", f"{val_loss:.4f}", f"[{loss_color}]{diff_loss:+.4f}[/]")
        
        # Add prediction diagnostics if available
        if "Pred Mean" in metrics:
            comp_table.add_row("Pred Prob Mean", "-", f"{pred_mean:.4f}", "-")
            comp_table.add_row("Pred Prob Std", "-", f"{pred_std:.4f}", "-")
            comp_table.add_row("Pred FG %", "-", f"{pred_fg:.2f}%", "-")
            
        if PYNVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_mem = mem_info.used / (1024**3)
                comp_table.add_row("GPU Mem Used", "-", f"{gpu_mem:.2f} GB", "-")
                comp_table.add_row("GPU Util", "-", f"{util.gpu}%", "-")
            except Exception:
                pass
            
        print_ui(Panel(comp_table, border_style="magenta"))
        
        self.prev_val_metrics = metrics

    def on_checkpoint_saved(self, trainer, ckpt_path: str, save_time: float, is_best: bool) -> None:
        if not RICH_AVAILABLE or not is_main_process() or not is_best: return
        
        self.best_dice = self.prev_val_metrics.get("val_dice", self.best_dice)
        self.best_epoch = trainer.current_epoch
        
        best_panel = Panel(
            f"[bold green]✓ Best Model Saved[/]\n\n"
            f"[white]Epoch      :[/] {self.best_epoch}\n"
            f"[white]Dice       :[/] {self.best_dice:.4f}\n"
            f"[white]Checkpoint :[/] {ckpt_path}\n",
            border_style="green"
        )
        print_ui(best_panel)

    def on_fit_end(self, trainer) -> None:
        self._cleanup_progress()

    def on_exception(self, trainer, exception: Exception) -> None:
        self._cleanup_progress()
        
    def _cleanup_progress(self) -> None:
        if self.progress is not None:
            if hasattr(self.progress, 'live') and self.progress.live is not None and self.progress.live.is_started:
                self.progress.stop()
            self.progress = None
            self.train_task = None
            self.val_task = None


class PlotGenerator(TrainerCallback):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def on_fit_end(self, trainer) -> None:
        if not MATPLOTLIB_AVAILABLE or not is_main_process(): return
        
        logs_dir = self.output_dir / "logs"
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        try:
            with open(logs_dir / "metrics.json", "r") as f:
                data = json.load(f)
            epochs = [d.get("epoch", i) for i, d in enumerate(data)]
            
            def plot_curve(key_train, key_val, title, ylabel, filename):
                plt.figure(figsize=(8, 5))
                if key_train and key_train in data[0]:
                    train_vals = [d.get(key_train, 0) for d in data]
                    plt.plot(epochs, train_vals, label="Train", color='blue')
                if key_val and key_val in data[0]:
                    val_vals = [d.get(key_val, 0) for d in data]
                    plt.plot(epochs, val_vals, label="Validation", color='orange')
                plt.title(title)
                plt.xlabel("Epoch")
                plt.ylabel(ylabel)
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.legend()
                plt.tight_layout()
                plt.savefig(plots_dir / filename)
                plt.close()

            plot_curve("train_loss", "val_loss", "Loss Curve", "Loss", "loss_curve.png")
            plot_curve("train_dice", "val_dice", "Dice Curve", "Dice", "dice_curve.png")
            plot_curve("Pred Mean", None, "Prediction Mean", "Probability", "pred_mean.png")
            plot_curve("Pred FG %", None, "Foreground Percentage", "%", "pred_fg.png")
            
        except Exception as e:
            logging.warning(f"Failed to generate plots: {e}")


class CrashHandler(TrainerCallback):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def on_exception(self, trainer, exception: Exception) -> None:
        if not is_main_process(): return
        logs_dir = self.output_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        crash_log = logs_dir / "crash.log"
        with open(crash_log, "w") as f:
            f.write("FATAL ERROR EXCEPTION DUMP\n")
            f.write("==========================\n")
            f.write(traceback.format_exc())
            f.write(f"\nException type: {type(exception)}\n")
            f.write(f"Message: {str(exception)}\n")
        logging.error(f"Crash details saved to {crash_log}")


class ReportGenerator(TrainerCallback):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def on_fit_end(self, trainer) -> None:
        if not is_main_process(): return
        
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        try:
            with open(self.output_dir / "logs" / "best_metrics.json", "r") as f:
                best_metrics = json.load(f)
        except Exception:
            best_metrics = {"Error": "best_metrics.json not found"}
            
        try:
            with open(self.output_dir / "run_info.json", "r") as f:
                run_info = json.load(f)
        except Exception:
            run_info = {"Error": "run_info.json not found"}
            
        report = f"""# HybridMedNeXt++ Experiment Report

## Overview
- **Experiment**: {run_info.get("experiment_name", "N/A")}
- **Framework Version**: {run_info.get("version", "N/A")}
- **Start Time**: {run_info.get("start_time", "N/A")}

## Hardware & Environment
- **GPU**: {run_info.get("gpu", "N/A")}
- **CUDA**: {run_info.get("cuda", "N/A")}
- **PyTorch**: {run_info.get("torch", "N/A")}
- **Git Commit**: {run_info.get("git_commit", "N/A")}

## Results
- **Best Validation Dice**: {best_metrics.get("best_val_dice", best_metrics.get("best_dice", "N/A"))}
- **Best Epoch**: {best_metrics.get("best_epoch", "N/A")}
- **Checkpoint Path**: `{best_metrics.get("checkpoint", "N/A")}`

## Training Curves
![Loss Curve](../plots/loss_curve.png)
![Dice Curve](../plots/dice_curve.png)
![Prediction Mean](../plots/pred_mean.png)

## Configuration Snapshot
```yaml
# Please refer to configs/ in the output directory
```
"""
        with open(reports_dir / "report.md", "w") as f:
            f.write(report)
