# Training Failure Report

## Error

```
Expected a cuda device, but got: cpu
```

## Traceback

```
Traceback (most recent call last):
  File "/Users/abhinavgupta/Desktop/Hemorrhage_SOTA_V2_Full_Backup/training/trainer.py", line 301, in fit
    self._pre_flight_check()
  File "/Users/abhinavgupta/Desktop/Hemorrhage_SOTA_V2_Full_Backup/training/trainer.py", line 235, in _pre_flight_check
    torch.cuda.reset_peak_memory_stats(self.device)
  File "/Users/abhinavgupta/Desktop/Hemorrhage_SOTA_V2_Full_Backup/.venv/lib/python3.11/site-packages/torch/cuda/memory.py", line 395, in reset_peak_memory_stats
    device = _get_device_index(device, optional=True)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/abhinavgupta/Desktop/Hemorrhage_SOTA_V2_Full_Backup/.venv/lib/python3.11/site-packages/torch/cuda/_utils.py", line 582, in _get_device_index
    raise ValueError(f"Expected a cuda device, but got: {device}")
ValueError: Expected a cuda device, but got: cpu

```
