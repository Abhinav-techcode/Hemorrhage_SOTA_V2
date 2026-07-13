# models/convnext_v2_encoder.py
"""3D ConvNeXt V2 Encoder for volumetric segmentation."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import List

from .utils import init_weights_kaiming


class LayerNorm(nn.Module):
    """Channel first layer norm for 3D tensors (B, C, D, H, W)."""
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
    
    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class GRN(nn.Module):
    """Global Response Normalization for 3D tensors."""
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        # L2 norm across spatial dimensions
        Gx = torch.norm(x, p=2, dim=(2, 3, 4), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class ConvNeXtV2Block(nn.Module):
    """3D ConvNeXt V2 Block."""
    def __init__(self, dim: int, drop_path: float = 0.):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        input_x = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = input_x + self.drop_path(x)
        return x


class DownsampleBlock(nn.Module):
    """2x2x2 Convolution for spatial downsampling."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = LayerNorm(in_channels, eps=1e-6)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.conv(x)
        return x


class ConvNeXtV2Encoder(nn.Module):
    """3D ConvNeXt V2 Encoder.
    
    Args:
        in_channels: Input channels (1 for CT).
        depths: Number of blocks per stage.
        dims: Channel dims per stage.
        drop_path_rate: Base drop path rate.
    """
    def __init__(self,
                 in_channels: int = 1,
                 depths: List[int] = [2, 2, 4, 2, 2],
                 dims: List[int] = [24, 48, 96, 192, 384],
                 drop_path_rate: float = 0.1):
        super().__init__()
        self.dims = dims
        # Stem: 3x3x3 stride 1 (as in MedNeXt)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, dims[0], kernel_size=3, padding=1),
            LayerNorm(dims[0], eps=1e-6)
        )
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(len(depths)):
            blocks = nn.ModuleList()
            for j in range(depths[i]):
                blocks.append(ConvNeXtV2Block(dims[i], drop_path=dp_rates[cur + j]))
            cur += depths[i]
            self.stages.append(blocks)
            if i < len(depths) - 1:
                self.stages.append(DownsampleBlock(dims[i], dims[i + 1]))
        self.apply(init_weights_kaiming)

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.stem(x)
        features = []
        for module in self.stages:
            if isinstance(module, nn.ModuleList):
                for block in module:
                    x = block(x)
                features.append(x)
            else:  # DownsampleBlock
                x = module(x)
        return features
