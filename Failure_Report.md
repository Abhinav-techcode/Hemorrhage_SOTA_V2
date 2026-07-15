# Training Failure Report

## Error

```
Deep supervision must output 3 tensors.
```

## Traceback

```
Traceback (most recent call last):
  File "/workspace/Hemorrhage_SOTA_V2/training/trainer.py", line 302, in fit
    self._pre_flight_check()
  File "/workspace/Hemorrhage_SOTA_V2/training/trainer.py", line 243, in _pre_flight_check
    assert len(outputs) == 3, "Deep supervision must output 3 tensors."
AssertionError: Deep supervision must output 3 tensors.

```
