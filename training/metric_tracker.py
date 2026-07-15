"""
training/metric_tracker.py
Dedicated component for live metric tracking and timing.
"""
from __future__ import annotations
import time
import torch
import torch.distributed as dist
from typing import Dict, Any

class LiveMetricTracker:
    def __init__(self, update_interval: int = 10, task_type: str = "binary"):
        self.update_interval = update_interval
        self.task_type = task_type
        
        self.train_dice = 0.0
        self.train_loss = 0.0
        self.train_pred_mean = 0.0
        self.train_fg_ratio = 0.0
        
        self.val_dice = 0.0
        self.val_loss = 0.0
        
        # Timing statistics
        self.data_time = 0.0
        self.forward_time = 0.0
        self.backward_time = 0.0
        
        self._train_batches = 0
        self._val_batches = 0
        self._timing_steps = 0

    def start_epoch(self):
        """Called at the beginning of each epoch."""
        self.reset()

    def end_epoch(self):
        """Aggregates metrics across GPUs if DDP is enabled."""
        if dist.is_available() and dist.is_initialized():
            tensors = torch.tensor([
                self.train_dice, self.train_loss, self.train_pred_mean, self.train_fg_ratio,
                self.val_dice, self.val_loss,
                self.data_time, self.forward_time, self.backward_time
            ], dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
            
            dist.all_reduce(tensors, op=dist.ReduceOp.AVG)
            
            self.train_dice = tensors[0].item()
            self.train_loss = tensors[1].item()
            self.train_pred_mean = tensors[2].item()
            self.train_fg_ratio = tensors[3].item()
            self.val_dice = tensors[4].item()
            self.val_loss = tensors[5].item()
            self.data_time = tensors[6].item()
            self.forward_time = tensors[7].item()
            self.backward_time = tensors[8].item()
    def summary(self) -> dict:
        """Returns the scalar metrics."""
        return {
            "train_dice": self.train_dice,
            "train_loss": self.train_loss,
            "train_pred_mean": self.train_pred_mean,
            "train_fg_ratio": self.train_fg_ratio,
            "val_dice": self.val_dice,
            "val_loss": self.val_loss,
            "data_time": self.data_time,
            "forward_time": self.forward_time,
            "backward_time": self.backward_time
        }

    def _resolve_preds(self, outputs: Any) -> torch.Tensor:
        if isinstance(outputs, dict):
            if "full" in outputs:
                return outputs["full"].detach()
            return list(outputs.values())[-1].detach()
        if isinstance(outputs, (list, tuple)):
            return outputs[-1].detach()
        return outputs.detach()

    def update_train(self, outputs: Any, targets: torch.Tensor, loss: float, batch_idx: int):
        if batch_idx % self.update_interval != 0:
            return
            
        preds = self._resolve_preds(outputs)
        targets = targets.detach()
        dice, pred_mean, fg_ratio = self._compute(preds, targets, self.task_type)
        
        self._train_batches += 1
        self.train_dice += (dice - self.train_dice) / self._train_batches
        self.train_loss += (loss - self.train_loss) / self._train_batches
        self.train_pred_mean += (pred_mean - self.train_pred_mean) / self._train_batches
        self.train_fg_ratio += (fg_ratio - self.train_fg_ratio) / self._train_batches

    def update_validation(self, outputs: Any, targets: torch.Tensor, loss: float, batch_idx: int):
        preds = self._resolve_preds(outputs)
        targets = targets.detach()
        dice, _, _ = self._compute(preds, targets, self.task_type)
        
        self._val_batches += 1
        self.val_dice += (dice - self.val_dice) / self._val_batches
        self.val_loss += (loss - self.val_loss) / self._val_batches

    def record_timing(self, data_t: float, forward_t: float, backward_t: float):
        self._timing_steps += 1
        self.data_time += (data_t - self.data_time) / self._timing_steps
        self.forward_time += (forward_t - self.forward_time) / self._timing_steps
        self.backward_time += (backward_t - self.backward_time) / self._timing_steps

    def reset(self):
        self.train_dice = 0.0
        self.train_loss = 0.0
        self.train_pred_mean = 0.0
        self.train_fg_ratio = 0.0
        self.val_dice = 0.0
        self.val_loss = 0.0
        self.data_time = 0.0
        self.forward_time = 0.0
        self.backward_time = 0.0
        self._train_batches = 0
        self._val_batches = 0
        self._timing_steps = 0

    @staticmethod
    @torch.no_grad()
    def _compute(preds: torch.Tensor, target: torch.Tensor, task_type: str = "binary") -> tuple[float, float, float]:
        if task_type in ["binary", "multilabel"]:
            probs = torch.sigmoid(preds)
            binary_preds = (probs > 0.5).float()
        else:
            probs = torch.softmax(preds, dim=1)
            binary_preds = torch.argmax(probs, dim=1, keepdim=True) == target

        target = target.float()
        
        tp = (binary_preds * target).sum().item()
        fp = (binary_preds * (1 - target)).sum().item()
        fn = ((1 - binary_preds) * target).sum().item()
        
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        pred_mean = probs.mean().item()
        fg_ratio = binary_preds.mean().item()
        
        return dice, pred_mean, fg_ratio
