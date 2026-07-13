"""
training/trainer.py
Production-Grade Orchestrator for 3D Brain Hemorrhage Segmentation.
Agnostic to model outputs (Tensor, List, Tuple, Dict).
"""
from __future__ import annotations

import gc
import logging
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.validator import ResearchValidator as Validator
from training.checkpoint_manager import CheckpointManager
from training.logger import ExperimentLogger
from training.callbacks import TrainerCallback, ProgressBar, EarlyStopping
from evaluation.metric_engine import ResearchMetricEngine
from training.health_monitor import HealthMonitor
from training.research_callbacks import ResearchFrameworkCallback
from evaluation.visualize import Visualizer

logger = logging.getLogger(__name__)

class SegmentationTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module, 
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any, 
        metric_manager: ResearchMetricEngine,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.config = config
        
        mem_format = torch.channels_last_3d if hasattr(torch, "channels_last_3d") and getattr(self.config, "channels_last", False) else torch.preserve_format
        model = model.to(device, memory_format=mem_format)

        if getattr(self.config, "compile_model", False) and hasattr(torch, "compile"):
            self.model = torch.compile(model)
        else:
            self.model = model

        self.optimizer = optimizer
        self.loss_fn = loss_fn 
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.metric_manager = metric_manager
        
        self.health_monitor = HealthMonitor(self.model)
        self.callbacks: list[TrainerCallback] = [
            EarlyStopping(),
            ResearchFrameworkCallback(self.metric_manager, self.health_monitor, self.config)
        ]
        
        save_dir = Path(config.save_dir)
        self.ckpt_manager = CheckpointManager(save_dir)
        self.exp_logger = ExperimentLogger(save_dir)
        self.exp_logger.save_metadata(asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else dict(self.config))

        self.amp_dtype = torch.bfloat16 if (getattr(config, "amp_dtype", "bfloat16") == "bfloat16" and torch.cuda.is_bf16_supported()) else torch.float16
        self.device_type = "cuda" if "cuda" in self.device else "cpu"
        self.scaler = torch.amp.GradScaler(self.device_type) if (getattr(config, "mixed_precision", False) and self.amp_dtype == torch.float16) else None
        
        if "cuda" in self.device:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.current_epoch = 1
        
        # Configurable best metric selection
        self.best_metric_name = getattr(self.config, "best_metric", "val_dice")
        self.best_metric_mode = getattr(self.config, "best_metric_mode", "max")
        
        self.best_metric_value = -float("inf") if self.best_metric_mode == "max" else float("inf")
        
        self.epochs_without_improvement = 0
        self.should_stop = False

    def _trigger(self, hook: str, *args, **kwargs):
        for cb in self.callbacks: getattr(cb, hook)(self, *args, **kwargs)

    def _train_step(self, batch: dict, batch_idx: int) -> float:
        mem_format = torch.channels_last_3d if hasattr(torch, "channels_last_3d") and getattr(self.config, "channels_last", False) else torch.preserve_format
        images = batch["image"].to(self.device, non_blocking=True, memory_format=mem_format)
        masks = batch["mask"].to(self.device, non_blocking=True)
        
        if self.current_epoch == 1 and batch_idx == 0:
            logger.info(f"Epoch 1, Batch 1 initialized. Input shape: {images.shape}, Mask shape: {masks.shape}")
        
        Validator.validate_input(images, masks)

        try:
            amp_ctx = torch.autocast(self.device_type, dtype=self.amp_dtype) if getattr(self.config, "mixed_precision", False) else nullcontext()
            with amp_ctx:
                outputs = self.model(images)
                Validator.validate_model_output(outputs, masks)
                loss_dict = self.loss_fn(outputs, masks)
                loss = loss_dict["total"] / self.config.grad_accum_steps

            Validator.validate_loss(loss)
            try:
                self.metric_manager.update_loss(loss_dict, mode="train")
            except Exception as e:
                logger.error(f"Research metric loss update failed: {e}", exc_info=True)

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.config.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            return loss.item() * self.config.grad_accum_steps

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM. Skipping batch & recovering.")
            torch.cuda.empty_cache()
            gc.collect()
            self.optimizer.zero_grad(set_to_none=True)
            return 0.0

    @torch.no_grad()
    def _val_step(self, batch: dict, batch_idx: int) -> float:
        images = batch["image"].to(self.device, non_blocking=True)
        masks = batch["mask"].to(self.device, non_blocking=True)
        Validator.validate_input(images, masks)
        
        amp_ctx = torch.autocast(self.device_type, dtype=self.amp_dtype) if getattr(self.config, "mixed_precision", False) else nullcontext()
        with amp_ctx:
            outputs = self.model(images)
            Validator.validate_model_output(outputs, masks)
            loss_dict = self.loss_fn(outputs, masks)
            Validator.validate_loss(loss_dict["total"])
            
        try:
            self.metric_manager.update_loss(loss_dict, mode="val")
            self.metric_manager.update(outputs, masks)
        except Exception as e:
            logger.error(f"Research metric validation update failed: {e}", exc_info=True)

        if batch_idx == 0:
            self._vis_batch = (
                images.detach().cpu(), 
                {k: v.detach().cpu() for k, v in outputs.items()} if isinstance(outputs, dict) else outputs.detach().cpu(),
                masks.detach().cpu()
            )

        return loss_dict["total"].item()

    def _create_state(self) -> dict:
        state = {
            "epoch": self.current_epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_metric_value": self.best_metric_value
        }
        if self.scheduler: state["scheduler"] = self.scheduler.state_dict()
        if self.scaler: state["scaler"] = self.scaler.state_dict()
        return state

    def resume(self, path: Path):
        ckpt = self.ckpt_manager.load(path, self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.current_epoch = ckpt["epoch"] + 1
        self.best_metric_value = ckpt.get("best_metric_value", -float("inf") if self.best_metric_mode == "max" else float("inf"))
        if self.scheduler and "scheduler" in ckpt: self.scheduler.load_state_dict(ckpt["scheduler"])
        if self.scaler and "scaler" in ckpt: self.scaler.load_state_dict(ckpt["scaler"])
        logger.info(f"Resumed training from epoch {self.current_epoch}.")

    def fit(self):
        logger.info(f"Starting Training on {self.device.upper()}")
        self._trigger("on_fit_begin")
        
        try:
            for epoch in range(self.current_epoch, self.config.epochs + 1):
                self.current_epoch = epoch
                self._trigger("on_epoch_begin", epoch)
                t0 = time.time()
                
                # TRAIN
                self.model.train()
                train_loss = 0.0
                for b_idx, batch in enumerate(self.train_loader):
                    self._trigger("on_train_batch_begin", b_idx)
                    loss = self._train_step(batch, b_idx)
                    train_loss += loss
                    self._trigger("on_train_batch_end", b_idx, loss)
                
                avg_train_loss = train_loss / max(len(self.train_loader), 1)

                # VALIDATE
                self.model.eval()
                val_loss = 0.0
                self.metric_manager.reset()
                self._trigger("on_validation_begin")
                for b_idx, batch in enumerate(self.val_loader):
                    val_loss += self._val_step(batch, b_idx)
                self._trigger("on_validation_end")
                
                avg_val_loss = val_loss / max(len(self.val_loader), 1)
                try:
                    metrics = self.metric_manager.compute()
                except Exception as e:
                    logger.error(f"Research metric computation failed: {e}", exc_info=True)
                    metrics = {}
                
                # METRICS & LOGGING
                epoch_time = time.time() - t0
                lr = self.optimizer.param_groups[0]['lr']
                log_dict = {"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "learning_rate": lr, "time_sec": epoch_time, **metrics}
                
                if epoch == 1: 
                    try:
                        self.exp_logger.init_csv(list(log_dict.keys()))
                    except Exception as e:
                        logger.error(f"CSV initialization failed: {e}")
                        
                try:
                    self.exp_logger.log_metrics(epoch, log_dict)
                except Exception as e:
                    logger.error(f"Metric logging failed: {e}", exc_info=True)
                    
                self._trigger("on_epoch_end", epoch, log_dict)
                
                if self.scheduler:
                    self.scheduler.step(avg_val_loss) if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else self.scheduler.step()

                # CHECKPOINTS & EARLY STOPPING
                current_metric = log_dict.get(self.best_metric_name, avg_val_loss)
                
                if self.best_metric_mode == "max":
                    is_best = current_metric > self.best_metric_value
                else:
                    is_best = current_metric < self.best_metric_value
                    
                if is_best:
                    self.best_metric_value = current_metric
                    self.epochs_without_improvement = 0
                    self.ckpt_manager.save("best_model.pt", self._create_state())
                else:
                    self.epochs_without_improvement += 1

                self.ckpt_manager.save("latest_checkpoint.pt", self._create_state())
                
                # VISUALIZATION (Epoch 1, Best, or Last)
                is_last = (epoch == self.config.epochs)
                if epoch == 1 or is_best or is_last:
                    if hasattr(self, '_vis_batch'):
                        v_imgs, v_outs, v_masks = self._vis_batch
                        try:
                            Visualizer.log_multi_plane(
                                self.exp_logger.writer, 
                                "Val/Predictions", 
                                v_imgs, v_outs, v_masks, 
                                self.current_epoch,
                                is_best=is_best
                            )
                        except Exception as e:
                            logger.warning(f"Visualization skipped: {e}")
                
                if self.should_stop:
                    logger.info("Early stopping triggered.")
                    break

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt! Saving emergency checkpoint.")
            self.ckpt_manager.save("emergency_checkpoint.pt", self._create_state())
        except Exception as e:
            logger.error(f"Fatal error: {traceback.format_exc()}")
            self.ckpt_manager.save("crash_checkpoint.pt", self._create_state())
            raise
        finally:
            self.exp_logger.close()
            self._trigger("on_fit_end")