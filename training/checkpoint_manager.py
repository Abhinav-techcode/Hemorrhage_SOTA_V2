"""training/checkpoint_manager.py"""
import random
import subprocess
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import monai

class CheckpointManager:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _get_env_metadata(self) -> Dict[str, str]:
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        except Exception:
            git_hash = "unknown"
        return {
            "git_commit": git_hash,
            "pytorch": torch.__version__,
            "monai": monai.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else "N/A"
        }

    def save(self, filename: str, state: Dict[str, Any]) -> None:
        state["rng"] = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(),
            "python": random.getstate()
        }
        state["metadata"] = self._get_env_metadata()
        torch.save(state, self.save_dir / filename)

    def load(self, filepath: Path, device: str) -> Dict[str, Any]:
        ckpt = torch.load(filepath, map_location=device)
        if "rng" in ckpt:
            torch.set_rng_state(ckpt["rng"]["torch"])
            if torch.cuda.is_available() and ckpt["rng"]["cuda"]:
                torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
            np.random.set_state(ckpt["rng"]["numpy"])
            random.setstate(ckpt["rng"]["python"])
        return ckpt