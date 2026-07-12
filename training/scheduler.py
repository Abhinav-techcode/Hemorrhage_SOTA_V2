"""
training/scheduler.py

Research-grade Learning Rate Scheduler Factory.

Supports:

- CosineAnnealingWarmRestarts
- CosineAnnealingLR
- ReduceLROnPlateau
- OneCycleLR
- PolynomialLR
- MultiStepLR
- StepLR
- ExponentialLR

Configuration is read directly from scheduler.yaml.
"""

from __future__ import annotations

import logging
from typing import Dict, Any

import torch

logger = logging.getLogger(__name__)


class SchedulerFactory:
    """
    Factory class for constructing learning-rate schedulers.
    """

    _SCHEDULERS = {
        "cosineannealingwarmrestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        "cosineannealinglr": torch.optim.lr_scheduler.CosineAnnealingLR,
        "reducelronplateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "onecyclelr": torch.optim.lr_scheduler.OneCycleLR,
        "polynomiallr": torch.optim.lr_scheduler.PolynomialLR,
        "multisteplr": torch.optim.lr_scheduler.MultiStepLR,
        "steplr": torch.optim.lr_scheduler.StepLR,
        "exponentiallr": torch.optim.lr_scheduler.ExponentialLR,
    }

    @classmethod
    def build(
        cls,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
    ):

        if config is None:
            logger.info("Scheduler disabled.")
            return None

        name = config.get("name", "").lower()

        if name not in cls._SCHEDULERS:
            raise ValueError(
                f"Unsupported scheduler: {config.get('name')}"
            )

        scheduler_cls = cls._SCHEDULERS[name]

        scheduler = scheduler_cls(
            optimizer,
            **config.get("params", {}),
        )

        logger.info(
            "Scheduler : %s",
            scheduler.__class__.__name__,
        )

        return scheduler


def scheduler_step(
    scheduler,
    validation_loss=None,
):
    """
    Unified scheduler stepping.

    ReduceLROnPlateau requires validation loss,
    while every other scheduler does not.
    """

    if scheduler is None:
        return

    if isinstance(
        scheduler,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    ):
        scheduler.step(validation_loss)
    else:
        scheduler.step()


def get_current_lr(
    optimizer: torch.optim.Optimizer,
) -> float:
    """
    Returns current learning rate.
    """

    return optimizer.param_groups[0]["lr"]


def print_lr(
    optimizer: torch.optim.Optimizer,
):

    lr = get_current_lr(optimizer)

    logger.info(
        "Current Learning Rate : %.8f",
        lr,
    )