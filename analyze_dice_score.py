import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
from typing import Dict, Any
from monai.inferers import SlidingWindowInferer

# Ensure we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.hybrid_convnext_umamba import HybridConvNeXtV2_UMamba
from models.hybrid_segformer_umamba import HybridSegFormerUMamba

def analyze_predictions(model_name: str, checkpoint_path: str, image_path: str, mask_path: str):
    print("="*60)
    print(f"DIAGNOSTIC ANALYSIS: {model_name}")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1] Using Device: {device}")
    
    # 1. Initialize Model
    if model_name.lower() == "convnext":
        model = HybridConvNeXtV2_UMamba(in_channels=3, num_classes=1)
    elif model_name.lower() == "segformer":
        model = HybridSegFormerUMamba(in_channels=3, num_classes=1)
    else:
        raise ValueError("Unknown model name")
        
    model.to(device)
    model.eval()
    
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            out = self.model(x)
            if isinstance(out, dict):
                return out.get("full", list(out.values())[-1])
            if isinstance(out, (list, tuple)):
                return out[0]
            return out
            
    wrapped_model = ModelWrapper(model)
    inferer = SlidingWindowInferer(roi_size=(64, 256, 256), sw_batch_size=4, overlap=0.25)
    
    # 2. Load Checkpoint
    if os.path.exists(checkpoint_path):
        print(f"[2] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        # Handle DDP if needed
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print("    -> Checkpoint loaded successfully.")
    else:
        print(f"[2] WARNING: Checkpoint not found at {checkpoint_path}. Using random weights!")
        
    # 3. Load Data
    print(f"[3] Loading Image: {image_path}")
    img_sitk = sitk.ReadImage(image_path)
    # (Z, Y, X, C) or (Z, Y, X)
    img_arr = sitk.GetArrayFromImage(img_sitk)
    if img_arr.ndim == 4:
        img_arr = np.transpose(img_arr, (3, 0, 1, 2))  # (C, Z, Y, X)
    else:
        img_arr = img_arr[np.newaxis, ...]  # (1, Z, Y, X)
        
    print(f"[3] Loading Mask: {mask_path}")
    mask_sitk = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask_sitk)[np.newaxis, ...] # (1, Z, Y, X)
    
    # Add batch dimension
    img_tensor = torch.from_numpy(img_arr).unsqueeze(0).float().to(device)
    mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).long().to(device)
    
    print(f"    -> Image shape: {img_tensor.shape}, dtype: {img_tensor.dtype}, range: [{img_tensor.min():.2f}, {img_tensor.max():.2f}]")
    print(f"    -> Mask shape: {mask_tensor.shape}, dtype: {mask_tensor.dtype}, positive voxels: {mask_tensor.sum().item()}")
    
    if mask_tensor.sum().item() == 0:
        print("    -> WARNING: Mask is empty! Dice will be 0/1 depending on definition.")
        
    # 4. Inference
    print("\n[4] Running Inference...")
    with torch.no_grad(), torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=torch.bfloat16):
        logits = inferer(img_tensor, wrapped_model)
            
    # Ensure float32 for metric calculation
    logits = logits.float()
    probs = torch.sigmoid(logits)
    
    print(f"    -> Logits shape: {logits.shape}")
    print(f"    -> Logits range: [min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}]")
    print(f"    -> Probs range:  [min={probs.min().item():.4f}, max={probs.max().item():.4f}, mean={probs.mean().item():.4f}]")
    
    # Calculate positive voxel counts at various thresholds
    print("\n[5] Threshold Analysis:")
    for thresh in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]:
        pred_binary = (probs > thresh).long()
        pred_pos_voxels = pred_binary.sum().item()
        
        # Dice calculation
        intersection = (pred_binary * mask_tensor).sum().item()
        union = pred_binary.sum().item() + mask_tensor.sum().item()
        dice = (2.0 * intersection) / (union + 1e-6)
        
        print(f"    -> Threshold {thresh:.2f}: Positive Voxels = {pred_pos_voxels:<8} | Intersection = {intersection:<8} | Dice Score = {dice:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["convnext", "segformer"], help="Model architecture")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best checkpoint")
    parser.add_argument("--image", type=str, default="processed/images/BHSD_0045.nii.gz", help="Path to test image")
    parser.add_argument("--mask", type=str, default="processed/masks/BHSD_0045.nii.gz", help="Path to test mask")
    args = parser.parse_args()
    
    analyze_predictions(args.model, args.ckpt, args.image, args.mask)
