"""training/logger.py"""
import csv
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class ExperimentLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=str(log_dir / "tb"))
        self.csv_path = log_dir / "metrics.csv"
        self._save_pip_freeze()

    def _save_pip_freeze(self):
        try:
            reqs = subprocess.check_output(["pip", "freeze"]).decode("utf-8")
            (self.log_dir / "requirements.txt").write_text(reqs)
        except Exception as e:
            logger.warning(f"Failed to dump pip freeze: {e}")

    def init_csv(self, fieldnames: list):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(fieldnames)

    def log_metrics(self, epoch: int, metrics: Dict[str, Any]):
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"Metrics/{k}", v, epoch)
                
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=list(metrics.keys()), extrasaction='ignore').writerow(metrics)
            
        # JSON Histories
        train_hist_path = self.log_dir / "training_history.json"
        val_hist_path = self.log_dir / "validation_history.json"
        
        train_metrics = {k: v for k, v in metrics.items() if "train" in k or k == "epoch"}
        val_metrics = {k: v for k, v in metrics.items() if "val" in k or k == "epoch"}
        
        def append_json(path, data):
            if path.exists():
                try:
                    with open(path, "r") as f:
                        history = json.load(f)
                except:
                    history = []
            else:
                history = []
            history.append(data)
            with open(path, "w") as f:
                json.dump(history, f, indent=4)
                
        append_json(train_hist_path, train_metrics)
        append_json(val_hist_path, val_metrics)
        
        # Markdown Summary
        md_path = self.log_dir / "epoch_summary.md"
        md_content = f"# Epoch {epoch} Summary\n\n"
        md_content += "## Training\n"
        for k, v in train_metrics.items():
            md_content += f"- **{k}**: {v}\n"
        md_content += "\n## Validation\n"
        for k, v in val_metrics.items():
            md_content += f"- **{k}**: {v}\n"
            
        with open(md_path, "w") as f:
            f.write(md_content)

    def save_metadata(self, meta: Dict[str, Any]):
        with open(self.log_dir / "experiment.json", "w") as f:
            json.dump(meta, f, indent=4)

    def close(self):
        self.writer.close()