# models/segformer_3d.py
"""
3D SegFormer Encoder.
Implements hierarchical feature extraction with Overlapping Patch Merging,
3D Efficient Self-Attention (sequence reduction), and Mix-FFN (3D convolutions).
"""
import torch
import torch.nn as nn
from typing import Tuple, List

class MixFFN3D(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        # 3x3x3 depthwise conv for spatial mixing
        self.dwconv = nn.Conv3d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x, shape):
        # x: (B, L, C), shape: (D, H, W)
        B, L, C = x.shape
        D, H, W = shape
        x = x.transpose(1, 2).view(B, C, D, H, W)
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.flatten(2).transpose(1, 2) # (B, L, C)
        return x

class EfficientSelfAttention3D(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
            
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, shape):
        # x: (B, L, C), shape: (D, H, W)
        B, L, C = x.shape
        D, H, W = shape
        
        q = self.q(x).reshape(B, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, D, H, W)
            x_ = self.sr(x_)
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class Block3D(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention3D(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixFFN3D(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, shape):
        x = x + self.attn(self.norm1(x), shape)
        x = x + self.mlp(self.norm2(x), shape)
        return x

class OverlapPatchEmbed3D(nn.Module):
    def __init__(self, patch_size=7, stride=4, padding=3, in_chans=3, embed_dim=768):
        super().__init__()
        # patch_size can be tuple or int
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (D, H, W)

class SegFormer3DEncoder(nn.Module):
    def __init__(self, 
                 in_chans=1, 
                 embed_dims=[32, 64, 160, 256],
                 num_heads=[1, 2, 5, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=True,
                 depths=[2, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 drop_rate=0.,
                 attn_drop_rate=0.):
        super().__init__()
        
        self.depths = depths
        
        # Patch Embeddings
        self.patch_embed1 = OverlapPatchEmbed3D(patch_size=7, stride=4, padding=3, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed3D(patch_size=3, stride=2, padding=1, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed3D(patch_size=3, stride=2, padding=1, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed3D(patch_size=3, stride=2, padding=1, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        
        # Transformer Blocks
        self.block1 = nn.ModuleList([Block3D(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, 
            drop=drop_rate, attn_drop=attn_drop_rate, sr_ratio=sr_ratios[0]) for _ in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])
        
        self.block2 = nn.ModuleList([Block3D(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, 
            drop=drop_rate, attn_drop=attn_drop_rate, sr_ratio=sr_ratios[1]) for _ in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        
        self.block3 = nn.ModuleList([Block3D(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, 
            drop=drop_rate, attn_drop=attn_drop_rate, sr_ratio=sr_ratios[2]) for _ in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        
        self.block4 = nn.ModuleList([Block3D(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, 
            drop=drop_rate, attn_drop=attn_drop_rate, sr_ratio=sr_ratios[3]) for _ in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward_features(self, x):
        outs = []
        
        # Stage 1
        x, shape = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, shape)
        x = self.norm1(x)
        x = x.reshape(x.shape[0], shape[0], shape[1], shape[2], -1).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x)
        
        # Stage 2
        x, shape = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, shape)
        x = self.norm2(x)
        x = x.reshape(x.shape[0], shape[0], shape[1], shape[2], -1).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x)
        
        # Stage 3
        x, shape = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, shape)
        x = self.norm3(x)
        x = x.reshape(x.shape[0], shape[0], shape[1], shape[2], -1).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x)
        
        # Stage 4
        x, shape = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, shape)
        x = self.norm4(x)
        x = x.reshape(x.shape[0], shape[0], shape[1], shape[2], -1).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x)
        
        return outs

    def forward(self, x):
        return self.forward_features(x)
