"""
evaluation/visualize.py

Research-grade TensorBoard visualization for 3D Brain Hemorrhage Segmentation.
Displays middle Axial / Coronal / Sagittal slices of:
    Input | Ground Truth | Prediction | Probability Map | FP | FN | Difference Map | Overlay
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
        is_best: bool = False
    ) -> None:
        """
        Log middle slices to TensorBoard.
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
        
        # Normalize image to [0, 1] for visualization overlay
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

        prob = pred[0, 0].sigmoid().detach().cpu()
        pred_bin = prob.ge(0.5).float()
        gt = target[0, 0].float().detach().cpu()

        fp = pred_bin * (1 - gt)
        fn = (1 - pred_bin) * gt
        diff = torch.abs(pred_bin - gt)
        overlay = torch.clamp(img_norm + pred_bin * 0.5, 0, 1.0)

        D, H, W = img.shape

        def build_row(*slices):
            return torch.cat(slices, dim=1)

        # ---------------------------------------------------------
        # Middle slices
        # ---------------------------------------------------------
        axial = build_row(
            img_norm[D // 2], gt[D // 2], pred_bin[D // 2], prob[D // 2], 
            fp[D // 2], fn[D // 2], diff[D // 2], overlay[D // 2]
        )

        coronal = build_row(
            img_norm[:, H // 2], gt[:, H // 2], pred_bin[:, H // 2], prob[:, H // 2], 
            fp[:, H // 2], fn[:, H // 2], diff[:, H // 2], overlay[:, H // 2]
        )

        sagittal = build_row(
            img_norm[:, :, W // 2], gt[:, :, W // 2], pred_bin[:, :, W // 2], prob[:, :, W // 2], 
            fp[:, :, W // 2], fn[:, :, W // 2], diff[:, :, W // 2], overlay[:, :, W // 2]
        )
        
        # Suffix tag if it's the best epoch
        full_tag = f"{tag}_Best" if is_best else tag

        writer.add_image(f"{full_tag}/Axial", axial.unsqueeze(0), global_step=step)
        writer.add_image(f"{full_tag}/Coronal", coronal.unsqueeze(0), global_step=step)
        writer.add_image(f"{full_tag}/Sagittal", sagittal.unsqueeze(0), global_step=step)