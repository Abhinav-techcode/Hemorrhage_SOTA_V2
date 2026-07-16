# models/segformer_baseline.py
"""
SegFormer 3D Baseline Architecture.
This model removes the U-Mamba bottleneck and acts as a pure SegFormer 3D
with an attention-based decoder. Used as a baseline for ablation studies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segformer_3d import SegFormer3DEncoder
from .attention_skip_fusion import AttentionSkipFusion3D

class BoundaryRefinementHead(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        # Lightweight boundary recovery via edge highlighting
        self.conv1 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm3d(in_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels // 2, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        return self.conv2(x)

class SegFormer3DBaseline(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 num_classes=1, 
                 embed_dims=[32, 64, 160, 256],
                 fusion_dim=256,
                 decoder_dims=[128, 64, 32]):
        super().__init__()
        
        # 1. 3D SegFormer Encoder
        self.encoder = SegFormer3DEncoder(in_chans=in_channels, embed_dims=embed_dims)
        
        # We replace UMamba with a standard Conv3d Bottleneck to process the deepest features (E4)
        self.bottleneck = nn.Sequential(
            nn.Conv3d(embed_dims[3], fusion_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(fusion_dim, fusion_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(fusion_dim),
            nn.ReLU(inplace=True)
        )
        
        # Decoder path
        self.up3 = nn.ConvTranspose3d(fusion_dim, decoder_dims[0], kernel_size=2, stride=2)
        self.att3 = AttentionSkipFusion3D(embed_dims[2], decoder_dims[0]) # E3 skip
        
        self.up2 = nn.ConvTranspose3d(decoder_dims[0], decoder_dims[1], kernel_size=2, stride=2)
        self.att2 = AttentionSkipFusion3D(embed_dims[1], decoder_dims[1]) # E2 skip
        
        self.up1 = nn.ConvTranspose3d(decoder_dims[1], decoder_dims[2], kernel_size=2, stride=2)
        self.att1 = AttentionSkipFusion3D(embed_dims[0], decoder_dims[2]) # E1 skip
        
        self.up0 = nn.ConvTranspose3d(decoder_dims[2], decoder_dims[2], kernel_size=2, stride=2)
        
        # 4. Final Output and Boundary Refinement Head
        self.boundary_head = BoundaryRefinementHead(in_channels=decoder_dims[2], out_channels=num_classes)
        
        # Deep Supervision Heads
        self.ds_quarter = nn.Conv3d(decoder_dims[0], num_classes, kernel_size=1)
        self.ds_half = nn.Conv3d(decoder_dims[1], num_classes, kernel_size=1)

    def forward(self, x):
        # Original size
        target_size = x.shape[2:]
        
        # Stage 1: Encoder
        # E1 (1/4), E2 (1/8), E3 (1/16), E4 (1/32)
        enc_feats = self.encoder(x)
        E1, E2, E3, E4 = enc_feats
        
        # Stage 3: Baseline Bottleneck (No Mamba)
        # Directly process the deepest feature map
        b_out = self.bottleneck(E4)
        
        # Decoder
        # D3 (1/16)
        d3 = self.up3(b_out)
        d3 = self.att3(E3, d3)
        
        # Quarter output
        out_quarter = self.ds_quarter(d3)
        
        # D2 (1/8)
        d2 = self.up2(d3)
        d2 = self.att2(E2, d2)
        
        # Half output
        out_half = self.ds_half(d2)
        
        # D1 (1/4)
        d1 = self.up1(d2)
        d1 = self.att1(E1, d1)
        
        # Up to original (1/1 or 1/2 depending on E1 stride. E1 is 1/4 because of stride 4 patch embed)
        # E1 is 1/4. up0 brings it to 1/2.
        d0 = self.up0(d1)
        
        # Then interpolate to full size
        d0 = F.interpolate(d0, size=target_size, mode='trilinear', align_corners=False)
        
        # Stage 5: Boundary Refinement Head (Final output)
        out_full = self.boundary_head(d0)
        
        # Ensure deep supervision outputs are upsampled to target size for loss computation
        out_quarter = F.interpolate(out_quarter, size=target_size, mode='trilinear', align_corners=False)
        out_half = F.interpolate(out_half, size=target_size, mode='trilinear', align_corners=False)
        
        if self.training:
            return out_full, out_half, out_quarter
        return out_full
