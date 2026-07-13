import numpy as np
import pandas as pd
import SimpleITK as sitk

df = pd.read_csv('processed/reports/Processed_Metadata.csv')
samples = df.sample(n=20, random_state=42)

sizes = []
for idx, row in samples.iterrows():
    img = sitk.GetArrayFromImage(sitk.ReadImage(f"processed/images/{row['Filename']}"))
    # img is (Z, Y, X) or (Z, Y, X, C)
    if img.ndim == 4:
        img = np.transpose(img, (3, 0, 1, 2))
    else:
        img = img[np.newaxis, ...]
    
    # Calculate foreground bounding box based on intensity > a threshold, or just non-zero
    # Since background is typically 0
    mask = img[0] > 0
    coords = np.argwhere(mask)
    if coords.size > 0:
        zmin, ymin, xmin = coords.min(axis=0)
        zmax, ymax, xmax = coords.max(axis=0)
        sizes.append((zmax - zmin, ymax - ymin, xmax - xmin))

sizes = np.array(sizes)
print("Average Foreground Size (Z, Y, X):", sizes.mean(axis=0))
print("Max Foreground Size (Z, Y, X):", sizes.max(axis=0))
