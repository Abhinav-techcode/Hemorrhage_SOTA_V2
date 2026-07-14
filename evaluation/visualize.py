"""
evaluation/visualize.py

Research-grade TensorBoard visualization for 3D Brain Hemorrhage Segmentation.
Outputs must be color-consistent: Green (TP), Blue (FP), Red (FN), Gray (TN), Yellow (GT Boundary), Cyan (Pred Boundary).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class Visualizer:
    
    @staticmethod
    def extract_boundaries(mask: torch.Tensor) -> torch.Tensor:
        """
        Extract boundaries from a binary mask (D, H, W).
        Uses a 3D MaxPool approach.
        """
        if mask.dim() == 3:
            mask = mask.unsqueeze(0).unsqueeze(0) # (1, 1, D, H, W)
        dilated = F.max_pool3d(mask, kernel_size=3, stride=1, padding=1)
        boundary = dilated - mask
        return boundary.squeeze()

    @staticmethod
    def create_color_overlay(img: torch.Tensor, gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """
        Creates an RGB overlay.
        img, gt, pred are (D, H, W) tensors.
        Returns (3, D, H, W) RGB tensor.
        Colors:
        - TP: Green (0, 1, 0)
        - FP: Blue (0, 0, 1)
        - FN: Red (1, 0, 0)
        - TN: Gray (image itself)
        - GT Boundary: Yellow (1, 1, 0)
        - Pred Boundary: Cyan (0, 1, 1)
        """
        # Normalize img to [0, 1]
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # Base RGB is grayscale image
        rgb = img_norm.unsqueeze(0).repeat(3, 1, 1, 1) # (3, D, H, W)
        
        tp = (pred * gt).bool()
        fp = (pred * (1 - gt)).bool()
        fn = ((1 - pred) * gt).bool()
        
        gt_boundary = Visualizer.extract_boundaries(gt).bool()
        pred_boundary = Visualizer.extract_boundaries(pred).bool()
        
        # Apply colors (R, G, B)
        # Red channel
        rgb[0, fn] = 1.0 # FN -> Red
        rgb[0, gt_boundary] = 1.0 # Yellow -> Red+Green
        
        # Green channel
        rgb[1, tp] = 1.0 # TP -> Green
        rgb[1, gt_boundary] = 1.0 # Yellow
        rgb[1, pred_boundary] = 1.0 # Cyan -> Green+Blue
        
        # Blue channel
        rgb[2, fp] = 1.0 # FP -> Blue
        rgb[2, pred_boundary] = 1.0 # Cyan
        
        return rgb

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
        if isinstance(pred, dict):
            pred = pred.get("full", list(pred.values())[-1])
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]

        # Extract first sample
        img = image[0, 0].detach().cpu()
        prob = pred[0, 0].sigmoid().detach().cpu()
        pred_bin = prob.ge(0.5).float()
        gt = target[0, 0].float().detach().cpu()

        # Create RGB volume
        rgb_vol = Visualizer.create_color_overlay(img, gt, pred_bin) # (3, D, H, W)
        
        D, H, W = img.shape
        
        # Extract slices (3, H, W) etc
        axial = rgb_vol[:, D // 2, :, :]
        coronal = rgb_vol[:, :, H // 2, :]
        sagittal = rgb_vol[:, :, :, W // 2]
        
        full_tag = f"{tag}_Best" if is_best else tag
        
        writer.add_image(f"{full_tag}/Axial_Overlay", axial, global_step=step)
        writer.add_image(f"{full_tag}/Coronal_Overlay", coronal, global_step=step)
        writer.add_image(f"{full_tag}/Sagittal_Overlay", sagittal, global_step=step)
        
        # Also log base metrics curves manually if needed, though they are usually handled by scaler add_scalar