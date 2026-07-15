# models/umamba.py
"""
U-Mamba (Bot variant) Architecture.
Faithful implementation of the U-Mamba network from Ma et al. 
Uses 3D ResNet-style Convolutional blocks for the Encoder and Decoder, 
and the UMambaBridge (Mamba blocks) for the bottleneck to capture 
long-range dependencies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .umamba_bridge import UMambaBridge

class ResBlock3D(nn.Module):
    """Standard 3D Residual Convolutional Block used in U-Mamba."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.lrelu = nn.LeakyReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.lrelu(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.lrelu(out)
        return out

class UMambaBot(nn.Module):
    """
    U-Mamba_Bot Architecture.
    Encoder: 3D ResBlocks
    Bottleneck: Mamba (UMambaBridge)
    Decoder: 3D ResBlocks + Transpose Convolutions
    """
    def __init__(self, 
                 in_channels=1, 
                 num_classes=1, 
                 features=[32, 64, 128, 256, 320], 
                 d_state=16,
                 deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # Encoder
        self.encoder1 = ResBlock3D(in_channels, features[0], stride=1)
        self.encoder2 = ResBlock3D(features[0], features[1], stride=2)
        self.encoder3 = ResBlock3D(features[1], features[2], stride=2)
        self.encoder4 = ResBlock3D(features[2], features[3], stride=2)
        self.encoder5 = ResBlock3D(features[3], features[4], stride=2)
        
        # Bottleneck (Mamba)
        self.bottleneck = UMambaBridge(dim=features[4], d_state=d_state, num_blocks=2)
        
        # Decoder
        self.up4 = nn.ConvTranspose3d(features[4], features[3], kernel_size=2, stride=2)
        self.decoder4 = ResBlock3D(features[3] * 2, features[3], stride=1)
        
        self.up3 = nn.ConvTranspose3d(features[3], features[2], kernel_size=2, stride=2)
        self.decoder3 = ResBlock3D(features[2] * 2, features[2], stride=1)
        
        self.up2 = nn.ConvTranspose3d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = ResBlock3D(features[1] * 2, features[1], stride=1)
        
        self.up1 = nn.ConvTranspose3d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = ResBlock3D(features[0] * 2, features[0], stride=1)
        
        # Final Output Heads
        self.out_head = nn.Conv3d(features[0], num_classes, kernel_size=1)
        
        if self.deep_supervision:
            self.ds_head1 = nn.Conv3d(features[1], num_classes, kernel_size=1)
            self.ds_head2 = nn.Conv3d(features[2], num_classes, kernel_size=1)
            self.ds_head3 = nn.Conv3d(features[3], num_classes, kernel_size=1)

    def forward(self, x):
        target_size = x.shape[2:]
        
        # Encoder Pathway
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        
        # Bottleneck Pathway (Mamba)
        b = self.bottleneck(e5)
        
        # Decoder Pathway
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)
        
        # Main output
        out = self.out_head(d1)
        
        if self.training and self.deep_supervision:
            out_ds1 = self.ds_head1(d2)
            out_ds2 = self.ds_head2(d3)
            out_ds3 = self.ds_head3(d4)
            
            # Interpolate to match target size for deep supervision
            out_ds1 = F.interpolate(out_ds1, size=target_size, mode='trilinear', align_corners=False)
            out_ds2 = F.interpolate(out_ds2, size=target_size, mode='trilinear', align_corners=False)
            out_ds3 = F.interpolate(out_ds3, size=target_size, mode='trilinear', align_corners=False)
            
            return out, out_ds1, out_ds2
            
        return out
