"""
evaluation/test.py

Evaluation script for the test set of the Hemorrhage_SOTA_V2 dataset.
Generates an Overall Quantitative Report and a Per-Case Report (CSV).
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
from torch.utils.data import DataLoader

from datasets.dataset import BrainHemorrhageDataset
from models.mednext_block import MedNeXt
from evaluation.metric_engine import ResearchMetricEngine

logger = logging.getLogger(__name__)

def evaluate_test_set(config: Any, model: torch.nn.Module, test_loader: DataLoader, device: str = "cuda"):
    logger.info("Starting Final Evaluation on Test Set")
    
    metric_engine = ResearchMetricEngine(device=device)
    model.eval()
    
    per_case_metrics = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            case_ids = batch.get("case_id", ["unknown"] * images.shape[0])
            
            amp_ctx = torch.autocast(device, dtype=torch.bfloat16) if getattr(config, "mixed_precision", False) else torch.autocast(device, enabled=False)
            with amp_ctx:
                outputs = model(images)
                
            # Full validation pass inside metric engine
            metric_engine.update(outputs, masks, mode="val")
            
            # Per-case metrics for CSV
            y_pred = metric_engine._extract_highest_res(outputs).detach()
            probs = torch.sigmoid(y_pred)
            y_pred_bin = [metric_engine.post_pred(p) for p in y_pred]
            y_bin = [metric_engine.post_label(l) for l in masks]
            
            for b in range(images.shape[0]):
                mask_b = y_bin[b]
                pred_b = y_pred_bin[b]
                
                # Single sample metrics
                metric_engine.dice.reset()
                metric_engine.iou.reset()
                metric_engine.hd95.reset()
                
                metric_engine.dice([pred_b], [mask_b])
                metric_engine.iou([pred_b], [mask_b])
                
                dice_val = metric_engine._safe_agg(metric_engine.dice)
                iou_val = metric_engine._safe_agg(metric_engine.iou)
                
                try:
                    metric_engine.hd95([pred_b], [mask_b])
                    hd95_val = metric_engine._safe_agg(metric_engine.hd95)
                except Exception:
                    hd95_val = float('nan')
                    
                fg_voxels = mask_b.sum().item()
                pred_voxels = pred_b.sum().item()
                
                per_case_metrics.append({
                    "Case_ID": case_ids[b],
                    "Dice": dice_val,
                    "IoU": iou_val,
                    "HD95": hd95_val,
                    "Lesion_Volume": fg_voxels,
                    "Prediction_Volume": pred_voxels
                })
                
    # Restore full metric engine state compute
    final_metrics = metric_engine.compute(mode="val")
    
    # Save Overall Report
    save_dir = Path(getattr(config, "save_dir", "outputs/"))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    overall_path = save_dir / "test_overall_report.json"
    with open(overall_path, "w") as f:
        json.dump(final_metrics, f, indent=4)
        
    # Save Per-Case Report
    case_path = save_dir / "test_per_case_report.csv"
    if per_case_metrics:
        with open(case_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_case_metrics[0].keys())
            writer.writeheader()
            writer.writerows(per_case_metrics)
            
    logger.info(f"Evaluation complete. Reports saved to {save_dir}")
    return final_metrics
