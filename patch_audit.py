import os
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
from monai.transforms import (
    Compose, EnsureTyped, CropForegroundd, SpatialPadd, CenterSpatialCropd
)

np.random.seed(42)
df = pd.read_csv('processed/reports/Processed_Metadata.csv')
non_empty = df[df['Positive_Voxels'] > 0]
samples = non_empty.sample(n=min(100, len(non_empty)), random_state=42)

p_val = Compose([
    EnsureTyped(keys=["image", "mask"], dtype=torch.float32),
    CropForegroundd(keys=["image", "mask"], source_key="image"),
    SpatialPadd(keys=["image", "mask"], spatial_size=(64, 256, 256)),
    CenterSpatialCropd(keys=["image", "mask"], roi_size=(64, 256, 256))
])

zeros_val = 0

for idx, row in samples.iterrows():
    img_path = f"processed/images/{row['Filename']}"
    mask_path = f"processed/masks/{row['Filename']}"
    
    mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))[np.newaxis, ...]
    
    img_arr = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
    if img_arr.ndim == 4:
        img_arr = np.transpose(img_arr, (3, 0, 1, 2))
    else:
        img_arr = img_arr[np.newaxis, ...]
        
    data = {"image": img_arr.copy(), "mask": mask_arr.copy()}
    out = p_val(data)
    if out["mask"].sum().item() == 0:
        zeros_val += 1

print(f"Lesions completely erased in Val (Center Crop 256x256): {zeros_val} / {len(samples)}")
