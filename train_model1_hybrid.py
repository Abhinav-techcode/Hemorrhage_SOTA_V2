#!/usr/bin/env python3
import sys

# Hardcode the exact arguments for Model 1 (Hybrid SegFormer)
sys.argv = [
    "training/train.py",
    "--config_dir", "configs"
]

print("==========================================================")
print("Starting MODEL 1: Hybrid SegFormer UMamba")
print("Using configs from: 'configs/'")
print("==========================================================")

from training.train import main

if __name__ == "__main__":
    main()
