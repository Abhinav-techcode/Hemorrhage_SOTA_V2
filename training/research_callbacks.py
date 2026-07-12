import os
import csv
import json
import logging
from typing import Dict, Any, List
import torch
from pathlib import Path
from training.callbacks import TrainerCallback
from evaluation.metric_engine import ResearchMetricEngine
from evaluation.validator import ResearchValidator
from evaluation.prediction_analysis import PredictionAnalyzer
from training.health_monitor import HealthMonitor
from evaluation.visualize import Visualizer

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

    def on_train_batch_end(self, trainer, batch_idx: int, loss: float):
        # We don't spam train batch logs, trainer handles that via ProgressBar callback
        pass

    def on_validation_begin(self, trainer):
        self.metric_engine.reset()

    def on_epoch_end(self, trainer, epoch: int, log_dict: Dict[str, Any]):
        # 1. Health Monitoring (Phase 7)
        health_stats = self.health_monitor.check_health()
        log_dict.update(health_stats)
        
        # 2. Prediction Analysis & Validation Metrics (Phase 5, 2)
        # Assuming metrics were computed inside trainer before this callback
        metrics = self.metric_engine.compute(mode="val")
        log_dict.update(metrics)
        
        # Train losses
        train_metrics = self.metric_engine.compute(mode="train")
        log_dict.update(train_metrics)
        
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
            
        log_dict["best_epoch"] = self.best_epoch
        log_dict["best_criterion_val"] = self.best_value
        
        # 4. Statistical Reporting (Phase 8)
        self.history.append(log_dict)
        self._write_history(epoch, log_dict)
        
        # 5. Epoch Summary (Phase 3)
        self._print_epoch_summary(epoch, log_dict)
        
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

    def _print_epoch_summary(self, epoch: int, log_dict: Dict[str, Any]):
        print(f"\n{'='*57}")
        print(f"Epoch {epoch} / {self.config.epochs}")
        print(f"{'='*57}")
        print("TRAIN")
        print(f"Loss             : {log_dict.get('train_loss', 0.0):.4f}")
        print(f"Learning Rate    : {log_dict.get('learning_rate', 0.0):.6f}")
        print(f"Epoch Time       : {log_dict.get('time_sec', 0.0):.2f}s")
        print(f"{'-'*57}")
        print("VALIDATION")
        print(f"Loss             : {log_dict.get('val_loss', 0.0):.4f}")
        print(f"Dice             : {log_dict.get('val_dice', 0.0):.4f}")
        print(f"IoU              : {log_dict.get('val_iou', 0.0):.4f}")
        print(f"Hausdorff95      : {log_dict.get('val_hd95', 0.0):.4f}")
        print(f"Foreground %     : {log_dict.get('val_foreground_percentage', 0.0):.2f}%")
        print(f"Empty Masks      : {log_dict.get('val_empty_masks', 0)}")
        print(f"Mean Confidence  : {log_dict.get('val_mean_confidence', 0.0):.4f}")
        print(f"\nBest {self.best_criterion}: {self.best_value:.4f} (Epoch {self.best_epoch})")
        print(f"{'='*57}\n")
