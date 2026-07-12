"""
evaluation/visualize.py

Research-grade TensorBoard visualization for 3D Brain Hemorrhage Segmentation.
Displays middle Axial / Coronal / Sagittal slices of:
    CT Image | Prediction | Ground Truth
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter


class Visualizer:

    @staticmethod
    def log_multi_plane(
        writer: SummaryWriter,
        tag: str,
        image: torch.Tensor,
        pred: Any,
        target: torch.Tensor,
        step: int,
    ) -> None:
        """
        Log middle slices to TensorBoard.

        Supports:
            Tensor
            List
            Tuple
            Dict (HybridMedNeXt++)
        """

        # ---------------------------------------------------------
        # Select highest-resolution prediction
        # ---------------------------------------------------------
        if isinstance(pred, dict):
            pred = pred.get("full", list(pred.values())[-1])

        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]

        # ---------------------------------------------------------
        # First sample in batch
        # ---------------------------------------------------------
        img = image[0, 0].detach().cpu()

        pred = (
            pred[0, 0]
            .sigmoid()
            .ge(0.5)
            .float()
            .detach()
            .cpu()
        )

        gt = target[0, 0].float().detach().cpu()

        D, H, W = img.shape

        # ---------------------------------------------------------
        # Middle slices
        # ---------------------------------------------------------
        axial = torch.cat(
            [
                img[D // 2],
                pred[D // 2],
                gt[D // 2],
            ],
            dim=1,
        )

        coronal = torch.cat(
            [
                img[:, H // 2],
                pred[:, H // 2],
                gt[:, H // 2],
            ],
            dim=1,
        )

        sagittal = torch.cat(
            [
                img[:, :, W // 2],
                pred[:, :, W // 2],
                gt[:, :, W // 2],
            ],
            dim=1,
        )

        writer.add_image(
            f"{tag}/Axial",
            axial.unsqueeze(0),
            global_step=step,
        )

        writer.add_image(
            f"{tag}/Coronal",
            coronal.unsqueeze(0),
            global_step=step,
        )

        writer.add_image(
            f"{tag}/Sagittal",
            sagittal.unsqueeze(0),
            global_step=step,
        )