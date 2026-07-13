import torch
from monai.transforms import RandCropByPosNegLabeld, Compose
data = {"image": torch.ones(1, 10, 10, 10), "mask": torch.ones(1, 10, 10, 10)}
t = Compose([
    RandCropByPosNegLabeld(keys=["image", "mask"], label_key="mask", spatial_size=(5, 5, 5), pos=1, neg=1, num_samples=1)
])
out = t(data)
print(type(out))
if isinstance(out, list):
    print(len(out))
    print(type(out[0]))
