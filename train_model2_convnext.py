#!/usr/bin/env python3
import sys

# Hardcode the exact arguments for Model 2 (ConvNeXtV2 + UMamba)
sys.argv = [
    "training/train.py",
    "--config_dir", "configs_umamba"
]

print("==========================================================")
print("Starting MODEL 2: ConvNeXtV2 UMamba")
print("Using configs from: 'configs_umamba/'")
print("==========================================================")

from training.train import main

if __name__ == "__main__":
    main()
