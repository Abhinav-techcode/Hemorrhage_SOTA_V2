from datasets.transforms import TransformFactory
import logging
logging.basicConfig(level=logging.INFO)
factory = TransformFactory()
t = factory.build_train_pipeline()
import torch
dummy_data = {
    "image": torch.rand(3, 64, 256, 256, dtype=torch.float32),
    "mask": torch.randint(0, 2, (1, 64, 256, 256), dtype=torch.long),
    "metadata": {"case": "test"}
}
out = t(dummy_data)
print(type(out))
if isinstance(out, list):
    print("List length:", len(out))
    print(type(out[0]))
