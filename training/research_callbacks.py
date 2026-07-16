import os
import csv
import json
import logging
import time
import math
from typing import Dict, Any, List
import torch
from pathlib import Path
from training.callbacks import TrainerCallback
from evaluation.metric_engine import ResearchMetricEngine
from evaluation.validator import ResearchValidator
from evaluation.prediction_analysis import PredictionAnalyzer
from training.health_monitor import HealthMonitor
from evaluation.visualize import Visualizer

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

logger = logging.getLogger(__name__)

class ResearchFrameworkCallback(TrainerCallback):
    """
    Integrates all research capabilities without polluting trainer.py
    (Phases 2-9)
    """
    def __init__(self, metric_engine: ResearchMetricEngine, health_monitor: HealthMonitor, config: Any):
        self.metric_engine = metric_engine
        self.health_monitor = health_monitor
        self.config = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history = []
        
        # Best model config
        self.best_criterion = getattr(config, "best_model_criterion", "val_dice")
        self.best_is_min = getattr(config, "best_model_criterion_min", False) # True for HD95, False for Dice
        self.best_value = float("inf") if self.best_is_min else -float("inf")
        self.best_epoch = -1

        # Dashboard state
        self.current_epoch = 1
        self.current_train_loss = 0.0
        self.current_lr = 0.0
        self.val_metrics = {}
        self.train_metrics = {}
        self.epoch_start_time = 0.0
        self.samples_per_sec = 0.0
        self.batch_count = 0
        self.last_batch_time = 0.0
        
        # Progress
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn()
        )
        self.task_id = None
        self.live = None

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=5)
        )
        layout["main"].split_row(
            Layout(name="hardware"),
            Layout(name="metrics"),
            Layout(name="stats")
        )
        
        # Header
        exp_name = getattr(self.config, "experiment_name", "Research Dashboard")
        if hasattr(self.config, 'get'):
            exp_name = self.config.get("experiment_name", exp_name)
        layout["header"].update(Panel(f"[bold cyan]{exp_name} | Epoch {self.current_epoch}/{self.config.epochs}[/]", style="bold blue"))
        
        # Hardware
        mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        hw_table = Table(show_header=False, expand=True, box=None)
        hw_table.add_row("GPU Memory", f"{mem:.1f} GB")
        hw_table.add_row("Samples/sec", f"{self.samples_per_sec:.2f}")
        hw_table.add_row("Learning Rate", f"{self.current_lr:.2e}")
        layout["hardware"].update(Panel(hw_table, title="[yellow]Hardware & Opt[/yellow]"))
        
        # Metrics
        metric_table = Table(expand=True, box=None)
        metric_table.add_column("Metric", style="cyan")
        metric_table.add_column("Train", style="green")
        metric_table.add_column("Validation", style="magenta")
        
        metric_table.add_row("Loss (Total)", f"{self.current_train_loss:.4f}", f"{self.val_metrics.get('val_loss_total', self.val_metrics.get('val_loss', 0.0)):.4f}")
        metric_table.add_row("  ↳ Dice (unweighted)", f"{self.train_metrics.get('train_loss_dice', 0.0):.4f}", f"{self.val_metrics.get('val_loss_dice', 0.0):.4f}")
        metric_table.add_row("  ↳ Focal (unweighted)", f"{self.train_metrics.get('train_loss_focal', 0.0):.4f}", f"{self.val_metrics.get('val_loss_focal', 0.0):.4f}")
        metric_table.add_row("  ↳ Boundary (unweighted)", f"{self.train_metrics.get('train_loss_boundary', 0.0):.4f}", f"{self.val_metrics.get('val_loss_boundary', 0.0):.4f}")
        
        metric_table.add_row("Weight: Dice", f"{self.train_metrics.get('train_loss_weight_dice', 0.0):.4f}", "-")
        metric_table.add_row("Weight: Focal", f"{self.train_metrics.get('train_loss_weight_focal', 0.0):.4f}", "-")
        metric_table.add_row("Weight: Boundary", f"{self.train_metrics.get('train_loss_weight_boundary', 0.0):.4f}", "-")
        
        metric_table.add_row("Dice", "-", f"{self.val_metrics.get('val_dice', 0.0):.4f}")
        metric_table.add_row("IoU", "-", f"{self.val_metrics.get('val_iou', 0.0):.4f}")
        metric_table.add_row("Precision", "-", f"{self.val_metrics.get('val_precision', 0.0):.4f}")
        metric_table.add_row("Recall", "-", f"{self.val_metrics.get('val_recall', 0.0):.4f}")
        metric_table.add_row("Sensitivity", "-", f"{self.val_metrics.get('val_sensitivity', 0.0):.4f}")
        metric_table.add_row("Specificity", "-", f"{self.val_metrics.get('val_specificity', 0.0):.4f}")
        metric_table.add_row("HD95", "-", f"{self.val_metrics.get('val_hd95', 0.0):.4f}")
        metric_table.add_row("Surface Dice", "-", f"{self.val_metrics.get('val_asd', 0.0):.4f}")
        
        layout["metrics"].update(Panel(metric_table, title="[magenta]Performance[/magenta]"))

        # Stats
        stats_table = Table(show_header=False, expand=True, box=None)
        epochs_since_best = max(0, self.current_epoch - self.best_epoch) if self.best_epoch > 0 else 0
        patience = getattr(self.config, "early_stopping_patience", 50)
        
        stats_table.add_row("Early Stop Counter", f"{epochs_since_best} / {patience}")
        stats_table.add_row("Current Best Epoch", str(self.best_epoch if self.best_epoch > 0 else "-"))
        
        if self.best_value != float("inf") and self.best_value != -float("inf"):
            curr_val = self.val_metrics.get(self.best_criterion, 0.0)
            imp = curr_val - self.best_value if not self.best_is_min else self.best_value - curr_val
            imp_str = f"[green]+{imp:.4f}[/]" if imp > 0 else f"[red]{imp:.4f}[/]"
            stats_table.add_row("Improv. Since Best", imp_str)
        else:
            stats_table.add_row("Improv. Since Best", "-")
            
        stats_table.add_row("Gradient Norm", f"{self.train_metrics.get('grad_norm', 0.0):.4f}")
        stats_table.add_row("Checkpoint Status", "Saved" if epochs_since_best == 0 else "Skipped")
            
        layout["stats"].update(Panel(stats_table, title="[blue]Experiment Stats[/blue]"))
        
        # Footer
        footer_table = Table(show_header=False, expand=True, box=None)
        best_str = f"Best {self.best_criterion}: {self.best_value:.4f} (Epoch {self.best_epoch})"
        footer_table.add_row(self.progress, best_str)
        layout["footer"].update(Panel(footer_table, title="[green]Progress & Best Model[/green]"))
        
        return layout

    def on_fit_begin(self, trainer) -> None:
        if getattr(self.config, "disable_dashboard", False):
            self.live = None
            logger.info("Dashboard disabled by configuration.")
        else:
            self.live = Live(self._build_layout(), refresh_per_second=4, screen=False)
            self.live.start()

    def on_fit_end(self, trainer) -> None:
        if getattr(self, "live", None) is not None:
            self.live.stop()
            
        # Phase 10: Automatic Post-Training Qualitative Visualization
        logger.info("Generating Final Post-Training Visualizations...")
        if hasattr(trainer, '_vis_batch') and trainer._vis_batch is not None:
            images, preds, masks = trainer._vis_batch
            
            if isinstance(preds, dict):
                pred_tensor = preds.get("full", list(preds.values())[-1])
            elif isinstance(preds, (list, tuple)):
                pred_tensor = preds[-1]
            else:
                pred_tensor = preds
                
            Visualizer.generate_qualitative_report(
                save_dir=self.save_dir,
                patient_id="val_sample_0",
                image=images,
                pred=pred_tensor,
                target=masks,
                metrics=self.val_metrics,
                epoch=self.current_epoch,
                dataset_name=getattr(self.config, "dataset_name", "CQ500")
            )
            logger.info(f"Post-Training Visualizations saved to {self.save_dir}/qualitative")

    def on_epoch_begin(self, trainer, epoch: int):
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        self.last_batch_time = time.time()
        self.batch_count = 0
        total_batches = len(trainer.train_loader)
        if self.task_id is None:
            self.task_id = self.progress.add_task("Training", total=total_batches)
        else:
            self.progress.reset(self.task_id, total=total_batches, description="Training")
        
        if self.live:
            self.live.update(self._build_layout())

    def on_train_batch_end(self, trainer, batch_idx: int, loss: float):
        self.current_train_loss = loss
        self.current_lr = trainer.optimizer.param_groups[0]['lr']
        self.batch_count += 1
        
        now = time.time()
        elapsed = now - self.last_batch_time
        # Batch size defaults to 1 if not explicitly found
        batch_size = 1
        if hasattr(trainer, 'train_loader') and hasattr(trainer.train_loader, 'batch_size') and trainer.train_loader.batch_size is not None:
            batch_size = trainer.train_loader.batch_size
            
        if elapsed > 0:
            self.samples_per_sec = batch_size / elapsed
        self.last_batch_time = now
        
        self.progress.advance(self.task_id, 1)
        
        if self.live:
            self.live.update(self._build_layout())

    def on_validation_begin(self, trainer):
        self.metric_engine.reset()
        self.progress.update(self.task_id, description="Validation...")

    def on_epoch_end(self, trainer, epoch: int, log_dict: Dict[str, Any]):
        # 1. Health Monitoring (Phase 7)
        try:
            health_stats = self.health_monitor.check_health()
            log_dict.update(health_stats)
        except Exception as e:
            logger.error(f"Health monitor check failed: {e}", exc_info=True)
            
        # 2. Prediction Analysis & Validation Metrics (Phase 5, 2)
        try:
            metrics = self.metric_engine.compute(mode="val")
            log_dict.update(metrics)
            self.val_metrics = metrics
        except Exception as e:
            logger.error(f"Metric engine validation compute failed: {e}", exc_info=True)
            
        try:
            train_metrics = self.metric_engine.compute(mode="train")
            if hasattr(self, "health_monitor") and self.health_monitor.enabled:
                # Merge health stats (which includes grad_norm) into train_metrics so dashboard can read them
                if 'health_stats' in locals():
                    train_metrics.update(health_stats)
            self.train_metrics = train_metrics
            log_dict.update(train_metrics)
        except Exception as e:
            logger.error(f"Metric engine train compute failed: {e}", exc_info=True)
            
        # 3. Best Model Selection (Phase 9)
        current_val = log_dict.get(self.best_criterion, 0.0)
        is_best = False
        if self.best_is_min:
            if current_val < self.best_value:
                self.best_value = current_val
                is_best = True
        else:
            if current_val > self.best_value:
                self.best_value = current_val
                is_best = True
                
        if is_best:
            self.best_epoch = epoch
            trainer.ckpt_manager.save("best_research_model.pt", trainer._create_state())
            
            # Milestone B: Log qualitative tracking info for best models
            exp_dir = Path(getattr(self.config, "exp_dir", "outputs"))
            best_log_path = exp_dir / "reports" / "Best_Model_Tracking.jsonl"
            best_log_path.parent.mkdir(parents=True, exist_ok=True)
            
            tracking_info = {
                "epoch": epoch,
                "best_criterion": self.best_criterion,
                "best_value": self.best_value,
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                "metrics": {k: v for k, v in self.val_metrics.items() if isinstance(v, (int, float)) and not math.isnan(v)}
            }
            try:
                with open(best_log_path, "a") as f:
                    f.write(json.dumps(tracking_info) + "\n")
            except Exception as e:
                logger.error(f"Failed to log best model tracking info: {e}")
            
        log_dict["best_epoch"] = self.best_epoch
        log_dict["best_criterion_val"] = self.best_value
        
        # 4. Statistical Reporting (Phase 8)
        self.history.append(log_dict)
        try:
            self._write_history(epoch, log_dict)
        except Exception as e:
            logger.error(f"Failed to write history: {e}", exc_info=True)
            
        if self.live:
            self.live.update(self._build_layout())
            
        # 5. Scientific Collapse Checks (Phase 4.2)
        fg_ratio = log_dict.get("val_pred_foreground_ratio", log_dict.get("val_fg_ratio", -1.0))
        val_dice = log_dict.get("val_dice", 1.0)
        grad_norm = log_dict.get("grad_norm", 1.0)
        
        collapse_reasons = []
        if fg_ratio != -1.0 and (fg_ratio < 0.00001 or fg_ratio > 0.9) and epoch > 5:
            collapse_reasons.append(f"Prediction foreground ratio ({fg_ratio:.4f}) collapsed or exploded")
        if grad_norm > 10000:
            collapse_reasons.append(f"Gradient explosion (norm: {grad_norm:.4f})")
        if grad_norm < 1e-8 and epoch > 1:
            collapse_reasons.append(f"Gradient vanishing (norm: {grad_norm:.4f})")
        if math.isnan(val_dice):
            collapse_reasons.append("Validation metric (Dice) is NaN")
            
        if collapse_reasons:
            self._generate_failure_report(epoch, collapse_reasons, log_dict)
            raise RuntimeError("Scientific Collapse Detected:\n" + "\n".join(collapse_reasons))
            
    def _generate_failure_report(self, epoch: int, reasons: List[str], log_dict: Dict[str, Any]):
        exp_dir = Path(getattr(self.config, "exp_dir", "outputs"))
        reports_dir = exp_dir / "failure_cases"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"Failure_Report_Epoch_{epoch}.md"
        with open(report_path, "w") as f:
            f.write(f"# Scientific Collapse Report - Epoch {epoch}\n\n")
            f.write("## Reasons for Collapse\n")
            for r in reasons:
                f.write(f"- {r}\n")
            f.write("\n## Metrics at Collapse\n")
            for k, v in log_dict.items():
                f.write(f"- **{k}**: {v}\n")
        logger.error(f"Scientific collapse. Report saved to {report_path}")

    def _write_history(self, epoch: int, log_dict: Dict[str, Any]):
        csv_path = self.save_dir / "epoch_metrics.csv"
        file_exists = csv_path.exists()
        
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_dict.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_dict)
            
        with open(self.save_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=4)
