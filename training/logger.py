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

    def save_metadata(self, meta: Dict[str, Any]):
        with open(self.log_dir / "experiment.json", "w") as f:
            json.dump(meta, f, indent=4)

    def close(self):
        self.writer.close()