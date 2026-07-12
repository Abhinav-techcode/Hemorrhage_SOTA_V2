"""
training/optimizer.py

Research-grade Optimizer Factory.

Creates optimizers directly from YAML configuration.

Supported:

- Adam
- AdamW
- SGD
- RMSprop
- Adamax
- NAdam
- RAdam

"""

from __future__ import annotations

import logging
from typing import Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class OptimizerFactory:

    _OPTIMIZERS = {

        "adam": torch.optim.Adam,

        "adamw": torch.optim.AdamW,

        "sgd": torch.optim.SGD,

        "rmsprop": torch.optim.RMSprop,

        "adamax": torch.optim.Adamax,

        "nadam": torch.optim.NAdam,

        "radam": torch.optim.RAdam,

    }

    @classmethod
    def build(
        cls,
        model: nn.Module,
        config: Dict[str, Any],
    ) -> torch.optim.Optimizer:

        name = config["name"].lower()

        if name not in cls._OPTIMIZERS:

            raise ValueError(
                f"Unsupported optimizer : {config['name']}"
            )

        optimizer = cls._OPTIMIZERS[name](

            model.parameters(),

            **config.get("params", {})

        )

        logger.info(
            "Optimizer : %s",
            optimizer.__class__.__name__,
        )

        return optimizer