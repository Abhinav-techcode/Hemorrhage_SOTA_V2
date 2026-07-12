"""training/callbacks.py - Extensible modular hooks."""
from __future__ import annotations
import logging
import torch
from tqdm import tqdm

class TrainerCallback:
    def on_fit_begin(self, trainer) -> None: pass
    def on_fit_end(self, trainer) -> None: pass
    def on_epoch_begin(self, trainer, epoch: int) -> None: pass
    def on_epoch_end(self, trainer, epoch: int, metrics: dict) -> None: pass
    def on_train_batch_begin(self, trainer, batch_idx: int) -> None: pass
    def on_train_batch_end(self, trainer, batch_idx: int, loss: float) -> None: pass
    def on_validation_begin(self, trainer) -> None: pass
    def on_validation_end(self, trainer) -> None: pass

class ProgressBar(TrainerCallback):
    def on_epoch_begin(self, trainer, epoch: int):
        self.pbar = tqdm(trainer.train_loader, desc=f"Epoch {epoch}/{trainer.config.epochs}", leave=False)
        
    def on_train_batch_end(self, trainer, batch_idx: int, loss: float):
        self.pbar.update(1)
        lr = trainer.optimizer.param_groups[0]['lr']
        mem = torch.cuda.memory_allocated() / 1e9 if "cuda" in trainer.device else 0
        self.pbar.set_postfix({'Loss': f"{loss:.4f}", 'LR': f"{lr:.1e}", 'Mem': f"{mem:.1f}G"})
        
    def on_epoch_end(self, trainer, epoch: int, metrics: dict):
        self.pbar.close()

class EarlyStopping(TrainerCallback):
    def on_epoch_end(self, trainer, epoch: int, metrics: dict):
        if trainer.epochs_without_improvement >= trainer.config.patience:
            trainer.should_stop = True