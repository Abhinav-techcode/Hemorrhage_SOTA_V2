# Binary Segmentation Pipeline Audit

**Date:** July 2026
**Framework:** Hemorrhage_SOTA_V2
**Objective:** Document the correct parameterization of MONAI metrics and losses for single-channel (binary) segmentation tasks to ensure mathematical correctness and silence warning spam (`single channel prediction, include_background=False ignored`).

## 1. Output Tensor Mathematics

The `HybridMedNeXt++` architecture has a final output tensor of shape:
`[Batch, 1, Depth, Height, Width]`

The target (ground truth mask) has a matching shape:
`[Batch, 1, Depth, Height, Width]` containing discrete values `{0, 1}`.

Because the output is a *single channel*, that channel explicitly represents the logits for the foreground class (Hemorrhage). There is no dedicated channel dimension representing the background.

## 2. MONAI Metric Configuration

MONAI's multi-class metrics usually assume outputs of shape `[B, C, D, H, W]`. When `C > 1`, setting `include_background=False` tells MONAI to skip channel `c=0` (usually assumed to be the background).

However, when `C = 1`, passing `include_background=False` is semantically invalid because skipping `c=0` means skipping the *only* channel being predicted. MONAI catches this edge case, ignores the flag, evaluates the foreground channel, and prints a warning.

**Resolution:**
We have officially removed `include_background=False` from all metric instantiations in the framework:
- `DiceLoss`
- `DiceMetric`
- `MeanIoU`
- `HausdorffDistanceMetric`
- `SurfaceDistanceMetric`
- `ConfusionMatrixMetric`

## 3. Activation & Thresholding

- **Training (Losses):** 
  We use `sigmoid=True` in our loss configuration. `DiceLoss` natively applies the sigmoid activation to the single channel logits before computing the overlap.
- **Validation (Metrics):** 
  The `ResearchMetricEngine` correctly applies a sigmoid and a `0.5` binary threshold to discrete the predictions before passing them to the MONAI metric implementations:
  ```python
  y_probs = torch.sigmoid(y_logits)
  y_preds_bin = (y_probs >= 0.5).float()
  ```

## 4. Verification

The pipeline has been verified for consistency. Training, Validation, Metrics, Visualization, and Reports all now use the identical single-channel sigmoid-thresholding pipeline without arbitrary background channel assumptions.
