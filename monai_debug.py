import torch
import warnings
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

print(f"DiceLoss include_background default: {DiceLoss().include_background}")
print(f"DiceMetric include_background default: {DiceMetric().include_background}")

preds = torch.randn(1, 1, 64, 256, 256)
target = torch.randint(0, 2, (1, 1, 64, 256, 256))

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    # Loss
    loss = DiceLoss(sigmoid=True)(preds, target)
    print(f"Loss Warnings caught: {len(w)}")
    for warn in w:
        print(f"Loss Warning: {warn.message}")
        
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    # Metric
    preds_bin = (torch.sigmoid(preds) > 0.5).float()
    metric = DiceMetric(reduction="mean")(preds_bin, target)
    print(f"Metric Warnings caught: {len(w)}")
    for warn in w:
        print(f"Metric Warning: {warn.message}")
