# models/attention_skip_fusion.py
"""
Attention Skip Fusion block for HybridSegFormer.
Uses an attention gate to combine encoder skip features with upsampled decoder features,
highlighting hemorrhage regions and preserving fine boundaries.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionSkipFusion3D(nn.Module):
    def __init__(self, encoder_channels: int, decoder_channels: int, inter_channels: int = None):
        super().__init__()
        self.inter_channels = inter_channels or (encoder_channels // 2)
        
        self.W_g = nn.Sequential(
            nn.Conv3d(decoder_channels, self.inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(self.inter_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(encoder_channels, self.inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(self.inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(self.inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
        # Final mix
        self.out_conv = nn.Sequential(
            nn.Conv3d(encoder_channels + decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(decoder_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_skip: torch.Tensor, g_dec: torch.Tensor) -> torch.Tensor:
        """
        x_skip: Features from the encoder.
        g_dec: Upsampled features from the decoder.
        """
        # Align spatial dimensions if necessary
        if g_dec.shape[2:] != x_skip.shape[2:]:
            g_dec = F.interpolate(g_dec, size=x_skip.shape[2:], mode='trilinear', align_corners=False)
            
        g1 = self.W_g(g_dec)
        x1 = self.W_x(x_skip)
        
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        
        # Apply attention to skip connection
        x_att = x_skip * alpha
        
        # Concatenate attended skip with decoder features
        out = torch.cat([x_att, g_dec], dim=1)
        out = self.out_conv(out)
        
        return out
