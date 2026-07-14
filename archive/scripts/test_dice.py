import torch
import warnings
from monai.losses import DiceLoss

print(f"DiceLoss include_background default: {DiceLoss().include_background}")

preds = torch.randn(1, 1, 64, 256, 256)
target = torch.ones(1, 1, 64, 256, 256)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    loss = DiceLoss(sigmoid=True)(preds, target)
    print(f"Warnings caught: {len(w)}")
    for warn in w:
        print(f"Warning: {warn.message}")
