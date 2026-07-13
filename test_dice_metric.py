import torch
import warnings
from monai.metrics import DiceMetric

print(f"DiceMetric include_background default: {DiceMetric().include_background}")

preds = torch.randint(0, 2, (1, 1, 64, 256, 256))
target = torch.randint(0, 2, (1, 1, 64, 256, 256))

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    metric = DiceMetric()(preds, target)
    print(f"Warnings caught: {len(w)}")
    for warn in w:
        print(f"Warning: {warn.message}")
