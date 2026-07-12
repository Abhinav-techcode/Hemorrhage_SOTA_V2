"""
evaluation/metrics.py

Research-grade Metric Engine
Supports:
    - Tensor outputs
    - Deep supervision (list/tuple)
    - HybridMedNeXt++ dict outputs
    - YAML configurable metrics
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import torch

from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
    ConfusionMatrixMetric,
)

from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)

logger = logging.getLogger(__name__)


# ==========================================================
# Metric Registry
# ==========================================================

class MetricRegistry:

    _registry = {

        "dice": DiceMetric,

        "hd95": HausdorffDistanceMetric,

        "assd": SurfaceDistanceMetric,

        "confusion": ConfusionMatrixMetric,

    }

    @classmethod
    def get(cls, name: str):

        key = name.lower()

        if key not in cls._registry:
            raise KeyError(f"Unknown metric: {name}")

        return cls._registry[key]


# ==========================================================
# Metric Manager
# ==========================================================

class MetricManager:

    def __init__(
        self,
        config: List[Dict[str, Any]],
    ):

        self.metrics = {

            cfg["name"]:
            MetricRegistry.get(cfg["name"])(
                **cfg.get("params", {})
            )

            for cfg in config

        }

        self.post_pred = Compose([

            Activations(sigmoid=True),

            AsDiscrete(threshold=0.5),

        ])

        self.post_label = AsDiscrete(
            threshold=0.5,
        )
            # ==========================================================
    # Update Metrics
    # ==========================================================

    def update(
        self,
        y_pred: Any,
        y: torch.Tensor,
    ) -> None:
        """
        Update all configured metrics.

        Supported outputs:
            Tensor
            List/Tuple (deep supervision)
            Dict (HybridMedNeXt++)
        """

        # ------------------------------------------------------
        # Select highest-resolution prediction
        # ------------------------------------------------------

        if isinstance(y_pred, dict):

            if "full" in y_pred:

                y_pred = y_pred["full"]

            else:

                logger.warning(
                    "Model output does not contain 'full'. "
                    "Using last dictionary entry."
                )

                y_pred = list(y_pred.values())[-1]

        elif isinstance(y_pred, (list, tuple)):

            # Deep supervision:
            # last output = highest resolution
            y_pred = y_pred[-1]

        # ------------------------------------------------------
        # Post-processing
        # ------------------------------------------------------

        y_pred = y_pred.detach()

        y = y.detach()

        y_pred_post = [

            self.post_pred(pred)

            for pred in y_pred

        ]

        y_post = [

            self.post_label(label)

            for label in y

        ]

        # ------------------------------------------------------
        # Update all metrics
        # ------------------------------------------------------

        for name, metric in self.metrics.items():
            try:
                metric(
                    y_pred=y_pred_post,
                    y=y_post,
                )
            except Exception as e:
                # Log once per epoch per metric to avoid spam, or debug level
                logger.debug(f"Metric '{name}' update failed for batch: {e}")
                # ==========================================================
    # Compute Metrics
    # ==========================================================

    def compute(self) -> Dict[str, float]:
        """
        Aggregate all configured metrics and return a dictionary.
        """

        results: Dict[str, float] = {}

        for name, metric in self.metrics.items():

            try:

                value = metric.aggregate()

                # ------------------------------------------
                # Confusion Matrix Metrics
                # ------------------------------------------

                if isinstance(metric, ConfusionMatrixMetric):

                    if isinstance(value, (list, tuple)):

                        metric_names = [
                            "precision",
                            "recall",
                            "specificity",
                            "accuracy",
                            "f1_score",
                        ]

                        for idx, val in enumerate(value):

                            key = (
                                metric_names[idx]
                                if idx < len(metric_names)
                                else f"{name}_{idx}"
                            )

                            results[key] = (
                                val.item()
                                if isinstance(val, torch.Tensor)
                                else float(val)
                            )

                    else:

                        results[name] = (
                            value.item()
                            if isinstance(value, torch.Tensor)
                            else float(value)
                        )

                # ------------------------------------------
                # Standard Metrics
                # ------------------------------------------

                else:

                    results[name] = (
                        value.item()
                        if isinstance(value, torch.Tensor)
                        else float(value)
                    )

            except Exception as e:

                logger.warning(
                    f"Metric '{name}' failed: {e}"
                )

                results[name] = float("nan")

        return results

    # ==========================================================
    # Reset Metrics
    # ==========================================================

    def reset(self) -> None:
        """
        Reset all metric accumulators.
        """

        for metric in self.metrics.values():
            metric.reset()

    # ==========================================================
    # Summary
    # ==========================================================

    def summary(self) -> None:
        """
        Print all metrics to the logger.
        """

        results = self.compute()

        logger.info("=" * 60)
        logger.info("Validation Metrics")

        for key, value in results.items():
            logger.info(f"{key:<20}: {value:.6f}")

        logger.info("=" * 60)

    # ==========================================================
    # Export
    # ==========================================================

    def export_json(self, path: str) -> None:
        """
        Save metrics as JSON.
        """

        with open(path, "w") as f:
            json.dump(
                self.compute(),
                f,
                indent=4,
            )
        
    