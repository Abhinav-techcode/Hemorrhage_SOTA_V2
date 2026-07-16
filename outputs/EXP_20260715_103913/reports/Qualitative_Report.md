# Qualitative & Quantitative Research Report

## Evaluation Protocol
Segmentation quality was assessed on 318 held-out cases. Predictions were thresholded at p>0.5 for the primary metrics; a sensitivity sweep across thresholds (0.3, 0.4, 0.5, 0.6, 0.7) was additionally run per case to report the Dice-optimal operating point. Overlap was scored with Dice and IoU; boundary agreement with the 95th-percentile Hausdorff distance (HD95) and average symmetric surface distance (ASSD), both computed in physical mm using per-case voxel spacing recovered from the original NIfTI headers. Volumes are reported in mL. Lesion-wise detection used 10% voxel-overlap as the match criterion between 26-connected components in the prediction and ground truth. Calibration was assessed via Brier score and Expected Calibration Error (ECE, 10 bins). 95% confidence intervals below are bootstrap estimates (n=2000 resamples).

## Table 1: Summary Statistics

| Metric | Mean ± Std | Median [IQR] | 95% CI |
|---|---|---|---|
| Dice | 0.1265 ± 0.2136 | 0.0000 [0.0000, 0.1727] | [0.1012, 0.1531] |
| IoU | 0.0850 ± 0.1552 | 0.0000 [0.0000, 0.0945] | [0.0666, 0.1043] |
| Precision | 0.1316 ± 0.2400 | 0.0000 [0.0000, 0.1497] | [0.1052, 0.1580] |
| Recall | 0.1383 ± 0.2417 | 0.0000 [0.0000, 0.1788] | [0.1096, 0.1691] |
| HD95 (mm) | 101.8119 ± 42.3240 | 97.8998 [69.6536, 130.1375] | [96.8649, 107.0425] |
| ASSD (mm) | 62.9632 ± 41.7544 | 61.6180 [26.5945, 93.6003] | [57.9215, 67.9990] |
| GT Volume (mL) | 19.8416 ± 40.1995 | 6.0972 [0.6829, 23.0467] | [15.7774, 24.6431] |
| Pred Volume (mL) | 15.7644 ± 17.4148 | 9.7917 [3.8819, 21.2197] | [13.9131, 17.6780] |
| RVD (%) | 304.0376 ± 1086.4089 | 14.8866 [-56.7764, 145.8460] | [185.7553, 439.7685] |
| Lesion Sensitivity | 0.1622 ± 0.3076 | 0.0000 [0.0000, 0.2000] | [0.1264, 0.2016] |
| Lesion Precision | 0.0848 ± 0.1719 | 0.0000 [0.0000, 0.1250] | [0.0653, 0.1049] |
| ECE | 0.0017 ± 0.0030 | 0.0011 [0.0007, 0.0018] | [0.0014, 0.0021] |
| Brier | 0.0015 ± 0.0031 | 0.0009 [0.0004, 0.0015] | [0.0013, 0.0019] |

## Aggregate Findings
Across 318 cases, mean Dice was 0.1265. Errors are relatively balanced between over- and under-segmentation. At the lesion level, 1023 of 1125 (90.9%) distinct hemorrhage foci across the dataset were missed entirely — this is the metric most relevant to missed-diagnosis risk, since voxel Dice can look acceptable while a whole secondary bleed is undetected.

Mean calibration error (ECE) was 0.0017; probabilities are reasonably well-calibrated.

312/318 cases had a Dice-optimal threshold different from the default 0.5, suggesting the fixed operating point may be leaving Dice on the table for a meaningful fraction of cases.

## Detailed Case Analysis

### Case: PHYSIO_0011
- **Overlap**: Dice 0.7842 | IoU 0.6450 | Precision 0.6952 | Recall 0.8994
- **Boundary**: HD95 30.75mm | ASSD 7.00mm
- **Volume**: GT 14.68mL | Pred 18.99mL | RVD +29.4%
- **Lesions**: GT 1 | Pred 4 | Sensitivity 1.00 | Precision 0.25
- **Optimal threshold**: 0.70 (Dice 0.8056 vs 0.7842 @0.5)
- **Diagnosis**: Tendency toward over-segmentation and false-positive boundaries. Boundary distance is large (HD95=30.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0023
- **Overlap**: Dice 0.7696 | IoU 0.6255 | Precision 0.7920 | Recall 0.7484
- **Boundary**: HD95 61.45mm | ASSD 14.00mm
- **Volume**: GT 53.40mL | Pred 50.46mL | RVD -5.5%
- **Lesions**: GT 2 | Pred 5 | Sensitivity 0.50 | Precision 0.20
- **Optimal threshold**: 0.70 (Dice 0.7758 vs 0.7696 @0.5)
- **Diagnosis**: Moderate agreement with mixed boundary discrepancies. Multi-focal case: missed 1 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=61.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0167
- **Overlap**: Dice 0.7535 | IoU 0.6045 | Precision 0.7467 | Recall 0.7605
- **Boundary**: HD95 60.02mm | ASSD 8.58mm
- **Volume**: GT 56.72mL | Pred 57.77mL | RVD +1.8%
- **Lesions**: GT 2 | Pred 9 | Sensitivity 0.50 | Precision 0.11
- **Optimal threshold**: 0.70 (Dice 0.7623 vs 0.7535 @0.5)
- **Diagnosis**: Moderate agreement with mixed boundary discrepancies. Multi-focal case: missed 1 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=60.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0042
- **Overlap**: Dice 0.7522 | IoU 0.6028 | Precision 0.6585 | Recall 0.8771
- **Boundary**: HD95 30.02mm | ASSD 5.30mm
- **Volume**: GT 32.24mL | Pred 42.94mL | RVD +33.2%
- **Lesions**: GT 1 | Pred 3 | Sensitivity 1.00 | Precision 0.33
- **Optimal threshold**: 0.70 (Dice 0.7678 vs 0.7522 @0.5)
- **Diagnosis**: Tendency toward over-segmentation and false-positive boundaries. Boundary distance is large (HD95=30.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0062
- **Overlap**: Dice 0.7472 | IoU 0.5964 | Precision 0.6595 | Recall 0.8618
- **Boundary**: HD95 57.46mm | ASSD 9.63mm
- **Volume**: GT 31.65mL | Pred 41.36mL | RVD +30.7%
- **Lesions**: GT 1 | Pred 3 | Sensitivity 1.00 | Precision 0.33
- **Optimal threshold**: 0.70 (Dice 0.7627 vs 0.7472 @0.5)
- **Diagnosis**: Tendency toward over-segmentation and false-positive boundaries. Boundary distance is large (HD95=57.5mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0097
- **Overlap**: Dice 0.7322 | IoU 0.5775 | Precision 0.8612 | Recall 0.6368
- **Boundary**: HD95 31.92mm | ASSD 4.91mm
- **Volume**: GT 96.93mL | Pred 71.68mL | RVD -26.0%
- **Lesions**: GT 2 | Pred 4 | Sensitivity 0.50 | Precision 0.25
- **Optimal threshold**: 0.30 (Dice 0.7457 vs 0.7322 @0.5)
- **Diagnosis**: Tendency toward under-segmentation; missing lesion boundaries. Multi-focal case: missed 1 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=31.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0064
- **Overlap**: Dice 0.7285 | IoU 0.5729 | Precision 0.7131 | Recall 0.7445
- **Boundary**: HD95 20.82mm | ASSD 4.45mm
- **Volume**: GT 39.27mL | Pred 41.00mL | RVD +4.4%
- **Lesions**: GT 1 | Pred 5 | Sensitivity 1.00 | Precision 0.20
- **Optimal threshold**: 0.70 (Dice 0.7334 vs 0.7285 @0.5)
- **Diagnosis**: Moderate agreement with mixed boundary discrepancies. Boundary distance is large (HD95=20.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0045
- **Overlap**: Dice 0.7084 | IoU 0.5485 | Precision 0.5920 | Recall 0.8817
- **Boundary**: HD95 19.24mm | ASSD 3.19mm
- **Volume**: GT 44.22mL | Pred 65.86mL | RVD +48.9%
- **Lesions**: GT 4 | Pred 1 | Sensitivity 0.25 | Precision 1.00
- **Optimal threshold**: 0.70 (Dice 0.7183 vs 0.7084 @0.5)
- **Diagnosis**: Tendency toward over-segmentation and false-positive boundaries. Multi-focal case: missed 3 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=19.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0044
- **Overlap**: Dice 0.6838 | IoU 0.5195 | Precision 0.6145 | Recall 0.7707
- **Boundary**: HD95 47.16mm | ASSD 10.93mm
- **Volume**: GT 10.94mL | Pred 13.72mL | RVD +25.4%
- **Lesions**: GT 2 | Pred 4 | Sensitivity 0.50 | Precision 0.25
- **Optimal threshold**: 0.70 (Dice 0.6981 vs 0.6838 @0.5)
- **Diagnosis**: Tendency toward over-segmentation and false-positive boundaries. Multi-focal case: missed 1 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=47.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0025
- **Overlap**: Dice 0.6637 | IoU 0.4966 | Precision 0.5984 | Recall 0.7450
- **Boundary**: HD95 91.31mm | ASSD 33.39mm
- **Volume**: GT 41.55mL | Pred 51.73mL | RVD +24.5%
- **Lesions**: GT 3 | Pred 7 | Sensitivity 0.33 | Precision 0.14
- **Optimal threshold**: 0.70 (Dice 0.6973 vs 0.6637 @0.5)
- **Diagnosis**: Moderate agreement with mixed boundary discrepancies. Multi-focal case: missed 2 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=91.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0103
- **Overlap**: Dice 0.6593 | IoU 0.4917 | Precision 0.5433 | Recall 0.8382
- **Boundary**: HD95 10.11mm | ASSD 4.02mm
- **Volume**: GT 25.35mL | Pred 39.10mL | RVD +54.3%
- **Lesions**: GT 2 | Pred 1 | Sensitivity 0.50 | Precision 1.00
- **Optimal threshold**: 0.70 (Dice 0.6814 vs 0.6593 @0.5)
- **Diagnosis**: Tendency toward over-segmentation and false-positive boundaries. Multi-focal case: missed 1 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=10.1mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0149
- **Overlap**: Dice 0.6500 | IoU 0.4815 | Precision 0.7974 | Recall 0.5486
- **Boundary**: HD95 97.50mm | ASSD 9.55mm
- **Volume**: GT 104.40mL | Pred 71.83mL | RVD -31.2%
- **Lesions**: GT 3 | Pred 3 | Sensitivity 0.67 | Precision 0.67
- **Optimal threshold**: 0.30 (Dice 0.6510 vs 0.6500 @0.5)
- **Diagnosis**: Tendency toward under-segmentation; missing lesion boundaries. Multi-focal case: missed 1 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=97.5mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0032
- **Overlap**: Dice 0.6443 | IoU 0.4752 | Precision 0.5122 | Recall 0.8680
- **Boundary**: HD95 79.12mm | ASSD 18.37mm
- **Volume**: GT 14.57mL | Pred 24.68mL | RVD +69.5%
- **Lesions**: GT 1 | Pred 5 | Sensitivity 1.00 | Precision 0.20
- **Optimal threshold**: 0.70 (Dice 0.6719 vs 0.6443 @0.5)
- **Diagnosis**: Tendency toward over-segmentation and false-positive boundaries. Boundary distance is large (HD95=79.1mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0018
- **Overlap**: Dice 0.6358 | IoU 0.4661 | Precision 0.6334 | Recall 0.6382
- **Boundary**: HD95 91.00mm | ASSD 18.14mm
- **Volume**: GT 43.54mL | Pred 43.86mL | RVD +0.8%
- **Lesions**: GT 2 | Pred 7 | Sensitivity 0.50 | Precision 0.14
- **Optimal threshold**: 0.70 (Dice 0.6466 vs 0.6358 @0.5)
- **Diagnosis**: Moderate agreement with mixed boundary discrepancies. Multi-focal case: missed 1 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=91.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0052
- **Overlap**: Dice 0.6343 | IoU 0.4644 | Precision 0.7224 | Recall 0.5653
- **Boundary**: HD95 26.91mm | ASSD 5.70mm
- **Volume**: GT 76.16mL | Pred 59.60mL | RVD -21.8%
- **Lesions**: GT 2 | Pred 4 | Sensitivity 0.50 | Precision 0.25
- **Optimal threshold**: 0.30 (Dice 0.6364 vs 0.6343 @0.5)
- **Diagnosis**: Tendency toward under-segmentation; missing lesion boundaries. Multi-focal case: missed 1 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=26.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0179
- **Overlap**: Dice 0.6240 | IoU 0.4534 | Precision 0.6044 | Recall 0.6448
- **Boundary**: HD95 14.98mm | ASSD 4.60mm
- **Volume**: GT 13.63mL | Pred 14.54mL | RVD +6.7%
- **Lesions**: GT 1 | Pred 4 | Sensitivity 1.00 | Precision 0.25
- **Optimal threshold**: 0.70 (Dice 0.6271 vs 0.6240 @0.5)
- **Diagnosis**: Moderate agreement with mixed boundary discrepancies. Boundary distance is large (HD95=15.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0022
- **Overlap**: Dice 0.6154 | IoU 0.4444 | Precision 0.5395 | Recall 0.7161
- **Boundary**: HD95 50.45mm | ASSD 14.11mm
- **Volume**: GT 6.29mL | Pred 8.35mL | RVD +32.7%
- **Lesions**: GT 1 | Pred 4 | Sensitivity 1.00 | Precision 0.25
- **Optimal threshold**: 0.70 (Dice 0.6381 vs 0.6154 @0.5)
- **Diagnosis**: Tendency toward over-segmentation and false-positive boundaries. Boundary distance is large (HD95=50.5mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0054
- **Overlap**: Dice 0.6137 | IoU 0.4427 | Precision 0.7973 | Recall 0.4988
- **Boundary**: HD95 90.55mm | ASSD 12.74mm
- **Volume**: GT 33.05mL | Pred 20.68mL | RVD -37.4%
- **Lesions**: GT 4 | Pred 4 | Sensitivity 0.25 | Precision 0.25
- **Optimal threshold**: 0.30 (Dice 0.6158 vs 0.6137 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 3 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=90.6mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0030
- **Overlap**: Dice 0.6026 | IoU 0.4312 | Precision 0.9001 | Recall 0.4529
- **Boundary**: HD95 91.03mm | ASSD 10.57mm
- **Volume**: GT 15.03mL | Pred 7.56mL | RVD -49.7%
- **Lesions**: GT 1 | Pred 2 | Sensitivity 1.00 | Precision 0.50
- **Optimal threshold**: 0.30 (Dice 0.6400 vs 0.6026 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=91.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0004
- **Overlap**: Dice 0.5988 | IoU 0.4273 | Precision 0.7314 | Recall 0.5069
- **Boundary**: HD95 47.65mm | ASSD 3.49mm
- **Volume**: GT 41.78mL | Pred 28.95mL | RVD -30.7%
- **Lesions**: GT 2 | Pred 3 | Sensitivity 0.50 | Precision 0.33
- **Optimal threshold**: 0.30 (Dice 0.6040 vs 0.5988 @0.5)
- **Diagnosis**: Tendency toward under-segmentation; missing lesion boundaries. Multi-focal case: missed 1 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=47.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0003
- **Overlap**: Dice 0.5846 | IoU 0.4130 | Precision 0.4719 | Recall 0.7680
- **Boundary**: HD95 32.36mm | ASSD 6.59mm
- **Volume**: GT 58.33mL | Pred 94.93mL | RVD +62.7%
- **Lesions**: GT 7 | Pred 3 | Sensitivity 0.14 | Precision 0.33
- **Optimal threshold**: 0.70 (Dice 0.5968 vs 0.5846 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 6 of 7 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=32.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0008
- **Overlap**: Dice 0.5626 | IoU 0.3914 | Precision 0.5671 | Recall 0.5581
- **Boundary**: HD95 124.93mm | ASSD 52.41mm
- **Volume**: GT 11.12mL | Pred 10.94mL | RVD -1.6%
- **Lesions**: GT 1 | Pred 2 | Sensitivity 1.00 | Precision 0.50
- **Optimal threshold**: 0.70 (Dice 0.6073 vs 0.5626 @0.5)
- **Diagnosis**: Moderate agreement with mixed boundary discrepancies. Boundary distance is large (HD95=124.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0176
- **Overlap**: Dice 0.5545 | IoU 0.3836 | Precision 0.5887 | Recall 0.5241
- **Boundary**: HD95 106.79mm | ASSD 11.55mm
- **Volume**: GT 41.73mL | Pred 37.16mL | RVD -11.0%
- **Lesions**: GT 4 | Pred 5 | Sensitivity 0.25 | Precision 0.20
- **Optimal threshold**: 0.70 (Dice 0.5605 vs 0.5545 @0.5)
- **Diagnosis**: Moderate agreement with mixed boundary discrepancies. Multi-focal case: missed 3 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=106.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0025
- **Overlap**: Dice 0.5484 | IoU 0.3778 | Precision 0.4740 | Recall 0.6505
- **Boundary**: HD95 120.09mm | ASSD 44.20mm
- **Volume**: GT 25.78mL | Pred 35.38mL | RVD +37.2%
- **Lesions**: GT 1 | Pred 7 | Sensitivity 1.00 | Precision 0.14
- **Optimal threshold**: 0.70 (Dice 0.5743 vs 0.5484 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=120.1mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0032
- **Overlap**: Dice 0.5264 | IoU 0.3572 | Precision 0.5292 | Recall 0.5236
- **Boundary**: HD95 37.17mm | ASSD 11.72mm
- **Volume**: GT 30.29mL | Pred 29.97mL | RVD -1.1%
- **Lesions**: GT 1 | Pred 4 | Sensitivity 1.00 | Precision 0.25
- **Optimal threshold**: 0.70 (Dice 0.5342 vs 0.5264 @0.5)
- **Diagnosis**: Moderate agreement with mixed boundary discrepancies. Boundary distance is large (HD95=37.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0001
- **Overlap**: Dice 0.5160 | IoU 0.3478 | Precision 0.5332 | Recall 0.5000
- **Boundary**: HD95 64.18mm | ASSD 29.69mm
- **Volume**: GT 3.07mL | Pred 2.88mL | RVD -6.2%
- **Lesions**: GT 1 | Pred 6 | Sensitivity 1.00 | Precision 0.17
- **Optimal threshold**: 0.50 (Dice 0.5160 vs 0.5160 @0.5)
- **Diagnosis**: Moderate agreement with mixed boundary discrepancies. Boundary distance is large (HD95=64.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0049
- **Overlap**: Dice 0.5119 | IoU 0.3440 | Precision 0.5614 | Recall 0.4705
- **Boundary**: HD95 55.56mm | ASSD 7.36mm
- **Volume**: GT 5.89mL | Pred 4.94mL | RVD -16.2%
- **Lesions**: GT 4 | Pred 2 | Sensitivity 0.25 | Precision 0.50
- **Optimal threshold**: 0.70 (Dice 0.5125 vs 0.5119 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 3 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=55.6mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0024
- **Overlap**: Dice 0.5099 | IoU 0.3422 | Precision 0.8940 | Recall 0.3567
- **Boundary**: HD95 91.34mm | ASSD 13.53mm
- **Volume**: GT 21.23mL | Pred 8.47mL | RVD -60.1%
- **Lesions**: GT 1 | Pred 4 | Sensitivity 1.00 | Precision 0.25
- **Optimal threshold**: 0.30 (Dice 0.5337 vs 0.5099 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=91.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0050
- **Overlap**: Dice 0.5075 | IoU 0.3401 | Precision 0.3740 | Recall 0.7893
- **Boundary**: HD95 55.08mm | ASSD 15.15mm
- **Volume**: GT 1.69mL | Pred 3.58mL | RVD +111.0%
- **Lesions**: GT 1 | Pred 2 | Sensitivity 1.00 | Precision 0.50
- **Optimal threshold**: 0.70 (Dice 0.5392 vs 0.5075 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=55.1mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0123
- **Overlap**: Dice 0.4849 | IoU 0.3200 | Precision 0.5469 | Recall 0.4355
- **Boundary**: HD95 40.18mm | ASSD 3.84mm
- **Volume**: GT 96.48mL | Pred 76.82mL | RVD -20.4%
- **Lesions**: GT 13 | Pred 3 | Sensitivity 0.15 | Precision 1.00
- **Optimal threshold**: 0.30 (Dice 0.4906 vs 0.4849 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 11 of 13 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=40.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0170
- **Overlap**: Dice 0.4563 | IoU 0.2956 | Precision 0.5345 | Recall 0.3981
- **Boundary**: HD95 81.03mm | ASSD 36.82mm
- **Volume**: GT 37.44mL | Pred 27.89mL | RVD -25.5%
- **Lesions**: GT 15 | Pred 10 | Sensitivity 0.07 | Precision 0.12
- **Optimal threshold**: 0.70 (Dice 0.4708 vs 0.4563 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 14 of 15 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=81.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0137
- **Overlap**: Dice 0.4547 | IoU 0.2943 | Precision 0.3110 | Recall 0.8458
- **Boundary**: HD95 114.95mm | ASSD 58.57mm
- **Volume**: GT 10.50mL | Pred 28.56mL | RVD +172.0%
- **Lesions**: GT 1 | Pred 3 | Sensitivity 1.00 | Precision 0.33
- **Optimal threshold**: 0.70 (Dice 0.5165 vs 0.4547 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=114.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0152
- **Overlap**: Dice 0.4443 | IoU 0.2856 | Precision 0.3688 | Recall 0.5585
- **Boundary**: HD95 95.77mm | ASSD 39.14mm
- **Volume**: GT 30.55mL | Pred 46.26mL | RVD +51.4%
- **Lesions**: GT 3 | Pred 12 | Sensitivity 0.33 | Precision 0.08
- **Optimal threshold**: 0.70 (Dice 0.4817 vs 0.4443 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 2 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=95.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0098
- **Overlap**: Dice 0.4267 | IoU 0.2712 | Precision 0.5270 | Recall 0.3584
- **Boundary**: HD95 48.23mm | ASSD 22.95mm
- **Volume**: GT 18.48mL | Pred 12.57mL | RVD -32.0%
- **Lesions**: GT 1 | Pred 4 | Sensitivity 1.00 | Precision 0.25
- **Optimal threshold**: 0.40 (Dice 0.4301 vs 0.4267 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=48.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0129
- **Overlap**: Dice 0.4209 | IoU 0.2665 | Precision 0.2961 | Recall 0.7273
- **Boundary**: HD95 19.08mm | ASSD 7.37mm
- **Volume**: GT 13.11mL | Pred 32.21mL | RVD +145.6%
- **Lesions**: GT 3 | Pred 2 | Sensitivity 0.33 | Precision 0.50
- **Optimal threshold**: 0.70 (Dice 0.4439 vs 0.4209 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 2 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=19.1mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0086
- **Overlap**: Dice 0.4175 | IoU 0.2638 | Precision 0.8912 | Recall 0.2726
- **Boundary**: HD95 33.62mm | ASSD 2.63mm
- **Volume**: GT 136.87mL | Pred 41.87mL | RVD -69.4%
- **Lesions**: GT 1 | Pred 1 | Sensitivity 1.00 | Precision 1.00
- **Optimal threshold**: 0.30 (Dice 0.4298 vs 0.4175 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=33.6mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0181
- **Overlap**: Dice 0.4148 | IoU 0.2617 | Precision 0.4659 | Recall 0.3738
- **Boundary**: HD95 109.59mm | ASSD 18.78mm
- **Volume**: GT 33.13mL | Pred 26.58mL | RVD -19.8%
- **Lesions**: GT 9 | Pred 3 | Sensitivity 0.22 | Precision 0.50
- **Optimal threshold**: 0.50 (Dice 0.4148 vs 0.4148 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 7 of 9 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=109.6mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0001
- **Overlap**: Dice 0.4097 | IoU 0.2576 | Precision 0.2850 | Recall 0.7282
- **Boundary**: HD95 112.09mm | ASSD 33.79mm
- **Volume**: GT 10.28mL | Pred 26.27mL | RVD +155.5%
- **Lesions**: GT 1 | Pred 16 | Sensitivity 1.00 | Precision 0.07
- **Optimal threshold**: 0.70 (Dice 0.4609 vs 0.4097 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=112.1mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0118
- **Overlap**: Dice 0.4023 | IoU 0.2518 | Precision 0.4291 | Recall 0.3787
- **Boundary**: HD95 113.79mm | ASSD 58.92mm
- **Volume**: GT 14.17mL | Pred 12.50mL | RVD -11.7%
- **Lesions**: GT 1 | Pred 7 | Sensitivity 1.00 | Precision 0.14
- **Optimal threshold**: 0.70 (Dice 0.4272 vs 0.4023 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=113.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0158
- **Overlap**: Dice 0.3989 | IoU 0.2491 | Precision 0.3362 | Recall 0.4904
- **Boundary**: HD95 91.02mm | ASSD 43.71mm
- **Volume**: GT 13.46mL | Pred 19.63mL | RVD +45.9%
- **Lesions**: GT 2 | Pred 3 | Sensitivity 0.50 | Precision 0.33
- **Optimal threshold**: 0.70 (Dice 0.4257 vs 0.3989 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 1 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=91.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0046
- **Overlap**: Dice 0.3839 | IoU 0.2376 | Precision 0.7837 | Recall 0.2542
- **Boundary**: HD95 61.93mm | ASSD 18.76mm
- **Volume**: GT 123.49mL | Pred 40.06mL | RVD -67.6%
- **Lesions**: GT 1 | Pred 6 | Sensitivity 1.00 | Precision 0.17
- **Optimal threshold**: 0.30 (Dice 0.3947 vs 0.3839 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=61.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0143
- **Overlap**: Dice 0.3729 | IoU 0.2292 | Precision 0.2529 | Recall 0.7092
- **Boundary**: HD95 49.91mm | ASSD 7.20mm
- **Volume**: GT 24.31mL | Pred 68.15mL | RVD +180.4%
- **Lesions**: GT 7 | Pred 3 | Sensitivity 0.29 | Precision 0.50
- **Optimal threshold**: 0.70 (Dice 0.3772 vs 0.3729 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 5 of 7 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=49.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0055
- **Overlap**: Dice 0.3691 | IoU 0.2263 | Precision 0.7857 | Recall 0.2412
- **Boundary**: HD95 38.40mm | ASSD 4.06mm
- **Volume**: GT 55.36mL | Pred 17.00mL | RVD -69.3%
- **Lesions**: GT 3 | Pred 4 | Sensitivity 0.67 | Precision 0.67
- **Optimal threshold**: 0.30 (Dice 0.4048 vs 0.3691 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 1 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=38.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0082
- **Overlap**: Dice 0.3486 | IoU 0.2111 | Precision 0.2565 | Recall 0.5440
- **Boundary**: HD95 108.03mm | ASSD 65.40mm
- **Volume**: GT 13.88mL | Pred 29.43mL | RVD +112.1%
- **Lesions**: GT 4 | Pred 4 | Sensitivity 0.25 | Precision 0.25
- **Optimal threshold**: 0.70 (Dice 0.3715 vs 0.3486 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 3 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=108.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0122
- **Overlap**: Dice 0.3370 | IoU 0.2027 | Precision 0.2699 | Recall 0.4485
- **Boundary**: HD95 146.48mm | ASSD 93.98mm
- **Volume**: GT 9.17mL | Pred 15.24mL | RVD +66.2%
- **Lesions**: GT 1 | Pred 6 | Sensitivity 1.00 | Precision 0.17
- **Optimal threshold**: 0.70 (Dice 0.3613 vs 0.3370 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=146.5mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0006
- **Overlap**: Dice 0.3337 | IoU 0.2003 | Precision 0.2278 | Recall 0.6233
- **Boundary**: HD95 69.82mm | ASSD 38.18mm
- **Volume**: GT 12.13mL | Pred 33.17mL | RVD +173.6%
- **Lesions**: GT 5 | Pred 4 | Sensitivity 0.20 | Precision 0.25
- **Optimal threshold**: 0.70 (Dice 0.3627 vs 0.3337 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 4 of 5 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=69.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0009
- **Overlap**: Dice 0.3179 | IoU 0.1890 | Precision 0.4387 | Recall 0.2492
- **Boundary**: HD95 139.72mm | ASSD 84.41mm
- **Volume**: GT 19.78mL | Pred 11.24mL | RVD -43.2%
- **Lesions**: GT 1 | Pred 12 | Sensitivity 1.00 | Precision 0.08
- **Optimal threshold**: 0.70 (Dice 0.3242 vs 0.3179 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=139.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0020
- **Overlap**: Dice 0.3080 | IoU 0.1820 | Precision 0.5527 | Recall 0.2135
- **Boundary**: HD95 89.75mm | ASSD 43.30mm
- **Volume**: GT 44.27mL | Pred 17.10mL | RVD -61.4%
- **Lesions**: GT 4 | Pred 13 | Sensitivity 0.25 | Precision 0.08
- **Optimal threshold**: 0.30 (Dice 0.3100 vs 0.3080 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 3 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=89.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0010
- **Overlap**: Dice 0.3030 | IoU 0.1786 | Precision 0.4991 | Recall 0.2176
- **Boundary**: HD95 43.18mm | ASSD 2.45mm
- **Volume**: GT 29.89mL | Pred 13.03mL | RVD -56.4%
- **Lesions**: GT 15 | Pred 6 | Sensitivity 0.07 | Precision 0.20
- **Optimal threshold**: 0.30 (Dice 0.3047 vs 0.3030 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 14 of 15 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=43.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0016
- **Overlap**: Dice 0.2992 | IoU 0.1759 | Precision 0.8370 | Recall 0.1822
- **Boundary**: HD95 61.47mm | ASSD 7.51mm
- **Volume**: GT 114.87mL | Pred 25.00mL | RVD -78.2%
- **Lesions**: GT 1 | Pred 8 | Sensitivity 1.00 | Precision 0.12
- **Optimal threshold**: 0.30 (Dice 0.3079 vs 0.2992 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=61.5mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0018
- **Overlap**: Dice 0.2912 | IoU 0.1704 | Precision 0.2816 | Recall 0.3014
- **Boundary**: HD95 67.99mm | ASSD 8.37mm
- **Volume**: GT 6.02mL | Pred 6.44mL | RVD +7.0%
- **Lesions**: GT 6 | Pred 4 | Sensitivity 0.17 | Precision 0.25
- **Optimal threshold**: 0.70 (Dice 0.3038 vs 0.2912 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 5 of 6 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=68.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0085
- **Overlap**: Dice 0.2775 | IoU 0.1611 | Precision 0.4465 | Recall 0.2013
- **Boundary**: HD95 104.57mm | ASSD 43.26mm
- **Volume**: GT 59.50mL | Pred 26.83mL | RVD -54.9%
- **Lesions**: GT 4 | Pred 2 | Sensitivity 0.25 | Precision 0.50
- **Optimal threshold**: 0.70 (Dice 0.2785 vs 0.2775 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 3 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=104.6mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0177
- **Overlap**: Dice 0.2737 | IoU 0.1585 | Precision 0.2111 | Recall 0.3888
- **Boundary**: HD95 55.69mm | ASSD 6.34mm
- **Volume**: GT 15.73mL | Pred 28.97mL | RVD +84.1%
- **Lesions**: GT 2 | Pred 2 | Sensitivity 0.50 | Precision 0.50
- **Optimal threshold**: 0.70 (Dice 0.2823 vs 0.2737 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 1 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=55.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0132
- **Overlap**: Dice 0.2714 | IoU 0.1570 | Precision 0.3515 | Recall 0.2210
- **Boundary**: HD95 102.15mm | ASSD 52.43mm
- **Volume**: GT 26.02mL | Pred 16.36mL | RVD -37.1%
- **Lesions**: GT 3 | Pred 4 | Sensitivity 0.33 | Precision 0.25
- **Optimal threshold**: 0.70 (Dice 0.2862 vs 0.2714 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 2 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=102.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0088
- **Overlap**: Dice 0.2655 | IoU 0.1531 | Precision 0.2002 | Recall 0.3938
- **Boundary**: HD95 92.81mm | ASSD 64.27mm
- **Volume**: GT 12.59mL | Pred 24.77mL | RVD +96.7%
- **Lesions**: GT 5 | Pred 5 | Sensitivity 0.20 | Precision 0.20
- **Optimal threshold**: 0.70 (Dice 0.3044 vs 0.2655 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 4 of 5 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=92.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0183
- **Overlap**: Dice 0.2650 | IoU 0.1527 | Precision 0.3166 | Recall 0.2278
- **Boundary**: HD95 33.04mm | ASSD 6.28mm
- **Volume**: GT 54.15mL | Pred 38.96mL | RVD -28.1%
- **Lesions**: GT 4 | Pred 16 | Sensitivity 0.25 | Precision 0.14
- **Optimal threshold**: 0.30 (Dice 0.2683 vs 0.2650 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 3 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=33.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0027
- **Overlap**: Dice 0.2649 | IoU 0.1527 | Precision 0.3115 | Recall 0.2304
- **Boundary**: HD95 93.87mm | ASSD 24.77mm
- **Volume**: GT 26.66mL | Pred 19.72mL | RVD -26.0%
- **Lesions**: GT 8 | Pred 6 | Sensitivity 0.25 | Precision 0.33
- **Optimal threshold**: 0.30 (Dice 0.2693 vs 0.2649 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 6 of 8 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=93.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0151
- **Overlap**: Dice 0.2593 | IoU 0.1490 | Precision 0.2021 | Recall 0.3619
- **Boundary**: HD95 97.90mm | ASSD 57.13mm
- **Volume**: GT 11.69mL | Pred 20.93mL | RVD +79.1%
- **Lesions**: GT 1 | Pred 4 | Sensitivity 1.00 | Precision 0.25
- **Optimal threshold**: 0.70 (Dice 0.3006 vs 0.2593 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=97.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0065
- **Overlap**: Dice 0.2431 | IoU 0.1384 | Precision 0.1976 | Recall 0.3158
- **Boundary**: HD95 98.21mm | ASSD 38.22mm
- **Volume**: GT 23.27mL | Pred 37.18mL | RVD +59.8%
- **Lesions**: GT 3 | Pred 7 | Sensitivity 0.33 | Precision 0.14
- **Optimal threshold**: 0.70 (Dice 0.2586 vs 0.2431 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 2 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=98.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0182
- **Overlap**: Dice 0.2398 | IoU 0.1363 | Precision 0.2153 | Recall 0.2706
- **Boundary**: HD95 24.39mm | ASSD 10.22mm
- **Volume**: GT 58.60mL | Pred 73.66mL | RVD +25.7%
- **Lesions**: GT 4 | Pred 7 | Sensitivity 0.25 | Precision 0.33
- **Optimal threshold**: 0.30 (Dice 0.2506 vs 0.2398 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 3 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=24.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0107
- **Overlap**: Dice 0.2374 | IoU 0.1347 | Precision 0.2075 | Recall 0.2774
- **Boundary**: HD95 68.57mm | ASSD 30.16mm
- **Volume**: GT 35.16mL | Pred 47.00mL | RVD +33.7%
- **Lesions**: GT 6 | Pred 9 | Sensitivity 0.17 | Precision 0.14
- **Optimal threshold**: 0.70 (Dice 0.2498 vs 0.2374 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 5 of 6 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=68.6mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0036
- **Overlap**: Dice 0.2321 | IoU 0.1313 | Precision 0.4912 | Recall 0.1519
- **Boundary**: HD95 75.62mm | ASSD 7.21mm
- **Volume**: GT 180.56mL | Pred 55.85mL | RVD -69.1%
- **Lesions**: GT 14 | Pred 4 | Sensitivity 0.07 | Precision 0.50
- **Optimal threshold**: 0.30 (Dice 0.2375 vs 0.2321 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 13 of 14 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=75.6mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0116
- **Overlap**: Dice 0.2319 | IoU 0.1312 | Precision 0.3510 | Recall 0.1732
- **Boundary**: HD95 129.19mm | ASSD 72.44mm
- **Volume**: GT 24.09mL | Pred 11.89mL | RVD -50.7%
- **Lesions**: GT 3 | Pred 12 | Sensitivity 0.33 | Precision 0.08
- **Optimal threshold**: 0.70 (Dice 0.2399 vs 0.2319 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 2 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=129.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0112
- **Overlap**: Dice 0.2297 | IoU 0.1298 | Precision 0.1444 | Recall 0.5620
- **Boundary**: HD95 39.20mm | ASSD 16.57mm
- **Volume**: GT 17.56mL | Pred 68.36mL | RVD +289.3%
- **Lesions**: GT 8 | Pred 5 | Sensitivity 0.12 | Precision 0.20
- **Optimal threshold**: 0.70 (Dice 0.2349 vs 0.2297 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 7 of 8 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=39.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0035
- **Overlap**: Dice 0.2146 | IoU 0.1202 | Precision 0.3030 | Recall 0.1661
- **Boundary**: HD95 42.11mm | ASSD 19.98mm
- **Volume**: GT 18.53mL | Pred 10.16mL | RVD -45.2%
- **Lesions**: GT 3 | Pred 5 | Sensitivity 0.33 | Precision 0.25
- **Optimal threshold**: 0.30 (Dice 0.2182 vs 0.2146 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 2 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=42.1mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0144
- **Overlap**: Dice 0.1786 | IoU 0.0981 | Precision 0.1029 | Recall 0.6776
- **Boundary**: HD95 136.77mm | ASSD 65.24mm
- **Volume**: GT 2.32mL | Pred 15.30mL | RVD +558.7%
- **Lesions**: GT 1 | Pred 10 | Sensitivity 1.00 | Precision 0.10
- **Optimal threshold**: 0.70 (Dice 0.2152 vs 0.1786 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=136.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0148
- **Overlap**: Dice 0.1780 | IoU 0.0977 | Precision 0.2412 | Recall 0.1410
- **Boundary**: HD95 71.90mm | ASSD 35.51mm
- **Volume**: GT 19.41mL | Pred 11.35mL | RVD -41.5%
- **Lesions**: GT 13 | Pred 10 | Sensitivity 0.08 | Precision 0.12
- **Optimal threshold**: 0.50 (Dice 0.1780 vs 0.1780 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 12 of 13 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=71.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0019
- **Overlap**: Dice 0.1710 | IoU 0.0935 | Precision 0.1250 | Recall 0.2705
- **Boundary**: HD95 76.76mm | ASSD 21.24mm
- **Volume**: GT 11.28mL | Pred 24.42mL | RVD +116.4%
- **Lesions**: GT 13 | Pred 7 | Sensitivity 0.08 | Precision 0.14
- **Optimal threshold**: 0.40 (Dice 0.1743 vs 0.1710 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 12 of 13 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=76.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0161
- **Overlap**: Dice 0.1577 | IoU 0.0856 | Precision 0.8432 | Recall 0.0870
- **Boundary**: HD95 59.82mm | ASSD 3.61mm
- **Volume**: GT 106.34mL | Pred 10.97mL | RVD -89.7%
- **Lesions**: GT 7 | Pred 5 | Sensitivity 0.14 | Precision 0.33
- **Optimal threshold**: 0.30 (Dice 0.1707 vs 0.1577 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 6 of 7 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=59.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0026
- **Overlap**: Dice 0.1541 | IoU 0.0835 | Precision 0.8559 | Recall 0.0847
- **Boundary**: HD95 63.72mm | ASSD 10.31mm
- **Volume**: GT 9.42mL | Pred 0.93mL | RVD -90.1%
- **Lesions**: GT 1 | Pred 7 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.1921 vs 0.1541 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=63.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0021
- **Overlap**: Dice 0.1509 | IoU 0.0816 | Precision 0.5491 | Recall 0.0875
- **Boundary**: HD95 77.89mm | ASSD 9.01mm
- **Volume**: GT 83.77mL | Pred 13.35mL | RVD -84.1%
- **Lesions**: GT 1 | Pred 2 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.1574 vs 0.1509 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=77.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0043
- **Overlap**: Dice 0.1380 | IoU 0.0741 | Precision 0.0945 | Recall 0.2563
- **Boundary**: HD95 76.67mm | ASSD 51.66mm
- **Volume**: GT 2.66mL | Pred 7.22mL | RVD +171.3%
- **Lesions**: GT 4 | Pred 3 | Sensitivity 0.25 | Precision 0.33
- **Optimal threshold**: 0.70 (Dice 0.1443 vs 0.1380 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 3 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=76.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0004
- **Overlap**: Dice 0.1348 | IoU 0.0723 | Precision 0.1515 | Recall 0.1214
- **Boundary**: HD95 35.08mm | ASSD 24.05mm
- **Volume**: GT 0.55mL | Pred 0.44mL | RVD -19.9%
- **Lesions**: GT 1 | Pred 3 | Sensitivity 1.00 | Precision 0.33
- **Optimal threshold**: 0.40 (Dice 0.1422 vs 0.1348 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.55 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=35.1mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0019
- **Overlap**: Dice 0.1312 | IoU 0.0702 | Precision 0.2803 | Recall 0.0856
- **Boundary**: HD95 93.89mm | ASSD 61.62mm
- **Volume**: GT 19.37mL | Pred 5.92mL | RVD -69.4%
- **Lesions**: GT 3 | Pred 4 | Sensitivity 0.33 | Precision 0.25
- **Optimal threshold**: 0.30 (Dice 0.1466 vs 0.1312 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 2 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=93.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0124
- **Overlap**: Dice 0.1281 | IoU 0.0685 | Precision 0.4005 | Recall 0.0763
- **Boundary**: HD95 68.70mm | ASSD 8.97mm
- **Volume**: GT 29.57mL | Pred 5.63mL | RVD -81.0%
- **Lesions**: GT 21 | Pred 2 | Sensitivity 0.05 | Precision 0.50
- **Optimal threshold**: 0.30 (Dice 0.1302 vs 0.1281 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 20 of 21 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=68.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0034
- **Overlap**: Dice 0.1276 | IoU 0.0681 | Precision 0.2732 | Recall 0.0832
- **Boundary**: HD95 130.15mm | ASSD 74.51mm
- **Volume**: GT 63.40mL | Pred 19.31mL | RVD -69.5%
- **Lesions**: GT 1 | Pred 6 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.1674 vs 0.1276 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=130.1mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0007
- **Overlap**: Dice 0.1223 | IoU 0.0651 | Precision 0.3682 | Recall 0.0733
- **Boundary**: HD95 81.03mm | ASSD 35.28mm
- **Volume**: GT 151.08mL | Pred 30.08mL | RVD -80.1%
- **Lesions**: GT 5 | Pred 8 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.1289 vs 0.1223 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 5 of 5 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=81.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0153
- **Overlap**: Dice 0.1210 | IoU 0.0644 | Precision 0.8340 | Recall 0.0652
- **Boundary**: HD95 59.28mm | ASSD 7.31mm
- **Volume**: GT 88.36mL | Pred 6.91mL | RVD -92.2%
- **Lesions**: GT 4 | Pred 6 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.1353 vs 0.1210 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 4 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=59.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0084
- **Overlap**: Dice 0.1122 | IoU 0.0595 | Precision 0.1841 | Recall 0.0807
- **Boundary**: HD95 86.89mm | ASSD 38.23mm
- **Volume**: GT 53.75mL | Pred 23.57mL | RVD -56.2%
- **Lesions**: GT 28 | Pred 10 | Sensitivity 0.07 | Precision 0.22
- **Optimal threshold**: 0.40 (Dice 0.1141 vs 0.1122 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 26 of 28 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=86.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0025
- **Overlap**: Dice 0.1105 | IoU 0.0585 | Precision 0.1442 | Recall 0.0896
- **Boundary**: HD95 40.40mm | ASSD 13.20mm
- **Volume**: GT 14.28mL | Pred 8.87mL | RVD -37.9%
- **Lesions**: GT 5 | Pred 16 | Sensitivity 0.20 | Precision 0.11
- **Optimal threshold**: 0.30 (Dice 0.1353 vs 0.1105 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 4 of 5 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=40.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0067
- **Overlap**: Dice 0.1069 | IoU 0.0565 | Precision 0.1412 | Recall 0.0861
- **Boundary**: HD95 47.23mm | ASSD 9.42mm
- **Volume**: GT 9.00mL | Pred 5.49mL | RVD -39.1%
- **Lesions**: GT 5 | Pred 4 | Sensitivity 0.20 | Precision 0.33
- **Optimal threshold**: 0.30 (Dice 0.1148 vs 0.1069 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 4 of 5 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=47.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0010
- **Overlap**: Dice 0.1058 | IoU 0.0558 | Precision 0.1130 | Recall 0.0994
- **Boundary**: HD95 93.68mm | ASSD 70.46mm
- **Volume**: GT 6.75mL | Pred 5.93mL | RVD -12.1%
- **Lesions**: GT 1 | Pred 3 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.1189 vs 0.1058 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=93.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0066
- **Overlap**: Dice 0.0997 | IoU 0.0525 | Precision 0.1717 | Recall 0.0703
- **Boundary**: HD95 154.92mm | ASSD 109.20mm
- **Volume**: GT 15.47mL | Pred 6.33mL | RVD -59.1%
- **Lesions**: GT 1 | Pred 4 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.1178 vs 0.0997 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=154.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0026
- **Overlap**: Dice 0.0895 | IoU 0.0468 | Precision 0.3300 | Recall 0.0517
- **Boundary**: HD95 58.40mm | ASSD 18.98mm
- **Volume**: GT 56.38mL | Pred 8.84mL | RVD -84.3%
- **Lesions**: GT 6 | Pred 12 | Sensitivity 0.17 | Precision 0.14
- **Optimal threshold**: 0.30 (Dice 0.1155 vs 0.0895 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 5 of 6 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=58.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0100
- **Overlap**: Dice 0.0829 | IoU 0.0433 | Precision 0.1331 | Recall 0.0602
- **Boundary**: HD95 117.82mm | ASSD 70.92mm
- **Volume**: GT 9.72mL | Pred 4.40mL | RVD -54.7%
- **Lesions**: GT 2 | Pred 5 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.1009 vs 0.0829 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=117.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0020
- **Overlap**: Dice 0.0782 | IoU 0.0407 | Precision 0.8423 | Recall 0.0410
- **Boundary**: HD95 43.72mm | ASSD 5.35mm
- **Volume**: GT 74.31mL | Pred 3.62mL | RVD -95.1%
- **Lesions**: GT 1 | Pred 4 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0887 vs 0.0782 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=43.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0072
- **Overlap**: Dice 0.0684 | IoU 0.0354 | Precision 0.0396 | Recall 0.2528
- **Boundary**: HD95 38.03mm | ASSD 18.68mm
- **Volume**: GT 5.13mL | Pred 32.77mL | RVD +539.0%
- **Lesions**: GT 2 | Pred 3 | Sensitivity 0.50 | Precision 0.33
- **Optimal threshold**: 0.30 (Dice 0.0748 vs 0.0684 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 1 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=38.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0172
- **Overlap**: Dice 0.0608 | IoU 0.0314 | Precision 0.0382 | Recall 0.1501
- **Boundary**: HD95 122.04mm | ASSD 73.47mm
- **Volume**: GT 14.51mL | Pred 57.12mL | RVD +293.5%
- **Lesions**: GT 3 | Pred 8 | Sensitivity 0.33 | Precision 0.11
- **Optimal threshold**: 0.70 (Dice 0.0680 vs 0.0608 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 2 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=122.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0048
- **Overlap**: Dice 0.0564 | IoU 0.0290 | Precision 0.0417 | Recall 0.0869
- **Boundary**: HD95 71.45mm | ASSD 22.87mm
- **Volume**: GT 19.54mL | Pred 40.71mL | RVD +108.4%
- **Lesions**: GT 10 | Pred 7 | Sensitivity 0.30 | Precision 0.33
- **Optimal threshold**: 0.50 (Dice 0.0564 vs 0.0564 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 7 of 10 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=71.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0111
- **Overlap**: Dice 0.0543 | IoU 0.0279 | Precision 0.0966 | Recall 0.0377
- **Boundary**: HD95 60.15mm | ASSD 15.18mm
- **Volume**: GT 39.89mL | Pred 15.58mL | RVD -61.0%
- **Lesions**: GT 20 | Pred 8 | Sensitivity 0.05 | Precision 0.33
- **Optimal threshold**: 0.30 (Dice 0.0657 vs 0.0543 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 19 of 20 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=60.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0003
- **Overlap**: Dice 0.0539 | IoU 0.0277 | Precision 0.0909 | Recall 0.0383
- **Boundary**: HD95 150.48mm | ASSD 119.69mm
- **Volume**: GT 4.45mL | Pred 1.87mL | RVD -57.9%
- **Lesions**: GT 1 | Pred 5 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0662 vs 0.0539 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=150.5mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0174
- **Overlap**: Dice 0.0538 | IoU 0.0276 | Precision 0.0494 | Recall 0.0591
- **Boundary**: HD95 79.71mm | ASSD 29.73mm
- **Volume**: GT 39.06mL | Pred 46.73mL | RVD +19.6%
- **Lesions**: GT 13 | Pred 11 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0555 vs 0.0538 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 13 of 13 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=79.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0119
- **Overlap**: Dice 0.0513 | IoU 0.0263 | Precision 0.0806 | Recall 0.0376
- **Boundary**: HD95 101.92mm | ASSD 50.69mm
- **Volume**: GT 52.49mL | Pred 24.52mL | RVD -53.3%
- **Lesions**: GT 11 | Pred 5 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0545 vs 0.0513 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 11 of 11 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=101.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0002
- **Overlap**: Dice 0.0499 | IoU 0.0256 | Precision 0.0885 | Recall 0.0347
- **Boundary**: HD95 109.02mm | ASSD 66.15mm
- **Volume**: GT 34.20mL | Pred 13.41mL | RVD -60.8%
- **Lesions**: GT 1 | Pred 7 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0601 vs 0.0499 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=109.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0039
- **Overlap**: Dice 0.0448 | IoU 0.0229 | Precision 0.0375 | Recall 0.0555
- **Boundary**: HD95 118.11mm | ASSD 98.68mm
- **Volume**: GT 23.19mL | Pred 34.30mL | RVD +47.9%
- **Lesions**: GT 4 | Pred 3 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0544 vs 0.0448 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 4 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=118.1mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0093
- **Overlap**: Dice 0.0446 | IoU 0.0228 | Precision 0.0255 | Recall 0.1777
- **Boundary**: HD95 89.21mm | ASSD 25.42mm
- **Volume**: GT 8.16mL | Pred 56.80mL | RVD +596.1%
- **Lesions**: GT 6 | Pred 3 | Sensitivity 0.17 | Precision 0.25
- **Optimal threshold**: 0.50 (Dice 0.0446 vs 0.0446 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 5 of 6 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=89.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0060
- **Overlap**: Dice 0.0427 | IoU 0.0218 | Precision 0.0783 | Recall 0.0293
- **Boundary**: HD95 97.14mm | ASSD 64.88mm
- **Volume**: GT 25.05mL | Pred 9.38mL | RVD -62.5%
- **Lesions**: GT 10 | Pred 16 | Sensitivity 0.10 | Precision 0.07
- **Optimal threshold**: 0.30 (Dice 0.0428 vs 0.0427 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 9 of 10 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=97.1mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0142
- **Overlap**: Dice 0.0385 | IoU 0.0196 | Precision 0.0333 | Recall 0.0456
- **Boundary**: HD95 85.42mm | ASSD 57.87mm
- **Volume**: GT 6.18mL | Pred 8.45mL | RVD +36.9%
- **Lesions**: GT 9 | Pred 7 | Sensitivity 0.11 | Precision 0.14
- **Optimal threshold**: 0.30 (Dice 0.0426 vs 0.0385 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 8 of 9 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=85.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0164
- **Overlap**: Dice 0.0359 | IoU 0.0183 | Precision 0.0276 | Recall 0.0516
- **Boundary**: HD95 67.16mm | ASSD 17.70mm
- **Volume**: GT 11.97mL | Pred 22.41mL | RVD +87.2%
- **Lesions**: GT 6 | Pred 4 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0369 vs 0.0359 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 6 of 6 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=67.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0057
- **Overlap**: Dice 0.0270 | IoU 0.0137 | Precision 0.0144 | Recall 0.2200
- **Boundary**: HD95 63.37mm | ASSD 28.89mm
- **Volume**: GT 2.57mL | Pred 39.46mL | RVD +1432.4%
- **Lesions**: GT 3 | Pred 3 | Sensitivity 0.33 | Precision 0.25
- **Optimal threshold**: 0.70 (Dice 0.0285 vs 0.0270 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 2 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=63.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0121
- **Overlap**: Dice 0.0262 | IoU 0.0133 | Precision 0.0165 | Recall 0.0649
- **Boundary**: HD95 91.10mm | ASSD 30.05mm
- **Volume**: GT 19.11mL | Pred 75.35mL | RVD +294.3%
- **Lesions**: GT 6 | Pred 9 | Sensitivity 0.33 | Precision 0.18
- **Optimal threshold**: 0.30 (Dice 0.0272 vs 0.0262 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 4 of 6 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=91.1mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0154
- **Overlap**: Dice 0.0240 | IoU 0.0122 | Precision 0.0184 | Recall 0.0345
- **Boundary**: HD95 51.97mm | ASSD 28.32mm
- **Volume**: GT 33.07mL | Pred 61.78mL | RVD +86.8%
- **Lesions**: GT 6 | Pred 8 | Sensitivity 0.33 | Precision 0.29
- **Optimal threshold**: 0.30 (Dice 0.0332 vs 0.0240 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 4 of 6 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=52.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0150
- **Overlap**: Dice 0.0224 | IoU 0.0113 | Precision 0.0389 | Recall 0.0157
- **Boundary**: HD95 85.41mm | ASSD 29.21mm
- **Volume**: GT 6.00mL | Pred 2.43mL | RVD -59.6%
- **Lesions**: GT 16 | Pred 5 | Sensitivity 0.06 | Precision 0.20
- **Optimal threshold**: 0.30 (Dice 0.0239 vs 0.0224 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 15 of 16 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=85.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0180
- **Overlap**: Dice 0.0204 | IoU 0.0103 | Precision 0.0535 | Recall 0.0126
- **Boundary**: HD95 95.64mm | ASSD 56.78mm
- **Volume**: GT 39.73mL | Pred 9.34mL | RVD -76.5%
- **Lesions**: GT 9 | Pred 10 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0273 vs 0.0204 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 9 of 9 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=95.6mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0083
- **Overlap**: Dice 0.0186 | IoU 0.0094 | Precision 0.0721 | Recall 0.0107
- **Boundary**: HD95 114.27mm | ASSD 45.57mm
- **Volume**: GT 23.98mL | Pred 3.55mL | RVD -85.2%
- **Lesions**: GT 5 | Pred 7 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0204 vs 0.0186 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 5 of 5 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=114.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0070
- **Overlap**: Dice 0.0156 | IoU 0.0078 | Precision 0.0469 | Recall 0.0093
- **Boundary**: HD95 87.11mm | ASSD 38.58mm
- **Volume**: GT 71.94mL | Pred 14.33mL | RVD -80.1%
- **Lesions**: GT 22 | Pred 13 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0222 vs 0.0156 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 22 of 22 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=87.1mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0073
- **Overlap**: Dice 0.0131 | IoU 0.0066 | Precision 0.2304 | Recall 0.0068
- **Boundary**: HD95 101.17mm | ASSD 50.43mm
- **Volume**: GT 219.27mL | Pred 6.44mL | RVD -97.1%
- **Lesions**: GT 9 | Pred 4 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0180 vs 0.0131 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 9 of 9 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=101.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0115
- **Overlap**: Dice 0.0127 | IoU 0.0064 | Precision 0.0074 | Recall 0.0470
- **Boundary**: HD95 95.38mm | ASSD 67.33mm
- **Volume**: GT 4.77mL | Pred 30.39mL | RVD +536.7%
- **Lesions**: GT 7 | Pred 9 | Sensitivity 0.14 | Precision 0.11
- **Optimal threshold**: 0.70 (Dice 0.0138 vs 0.0127 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 6 of 7 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=95.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0021
- **Overlap**: Dice 0.0113 | IoU 0.0057 | Precision 0.0084 | Recall 0.0175
- **Boundary**: HD95 68.98mm | ASSD 15.32mm
- **Volume**: GT 9.24mL | Pred 19.29mL | RVD +108.8%
- **Lesions**: GT 15 | Pred 3 | Sensitivity 0.07 | Precision 0.25
- **Optimal threshold**: 0.30 (Dice 0.0145 vs 0.0113 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 14 of 15 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=69.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0110
- **Overlap**: Dice 0.0108 | IoU 0.0054 | Precision 0.0441 | Recall 0.0062
- **Boundary**: HD95 75.77mm | ASSD 16.77mm
- **Volume**: GT 153.33mL | Pred 21.42mL | RVD -86.0%
- **Lesions**: GT 20 | Pred 9 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0156 vs 0.0108 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 20 of 20 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=75.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0043
- **Overlap**: Dice 0.0105 | IoU 0.0053 | Precision 0.0123 | Recall 0.0092
- **Boundary**: HD95 117.77mm | ASSD 106.92mm
- **Volume**: GT 3.87mL | Pred 2.92mL | RVD -24.6%
- **Lesions**: GT 1 | Pred 4 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0212 vs 0.0105 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=117.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0095
- **Overlap**: Dice 0.0075 | IoU 0.0038 | Precision 0.0055 | Recall 0.0115
- **Boundary**: HD95 79.44mm | ASSD 43.86mm
- **Volume**: GT 26.62mL | Pred 55.05mL | RVD +106.8%
- **Lesions**: GT 27 | Pred 4 | Sensitivity 0.04 | Precision 0.20
- **Optimal threshold**: 0.30 (Dice 0.0089 vs 0.0075 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 26 of 27 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=79.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0106
- **Overlap**: Dice 0.0053 | IoU 0.0026 | Precision 0.0046 | Recall 0.0062
- **Boundary**: HD95 41.32mm | ASSD 27.92mm
- **Volume**: GT 6.18mL | Pred 8.31mL | RVD +34.5%
- **Lesions**: GT 4 | Pred 5 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.50 (Dice 0.0053 vs 0.0053 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 4 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=41.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0013
- **Overlap**: Dice 0.0023 | IoU 0.0012 | Precision 0.0026 | Recall 0.0021
- **Boundary**: HD95 84.51mm | ASSD 52.77mm
- **Volume**: GT 22.62mL | Pred 18.17mL | RVD -19.7%
- **Lesions**: GT 11 | Pred 12 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0039 vs 0.0023 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 11 of 11 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=84.5mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0061
- **Overlap**: Dice 0.0012 | IoU 0.0006 | Precision 0.0018 | Recall 0.0009
- **Boundary**: HD95 81.57mm | ASSD 32.31mm
- **Volume**: GT 5.43mL | Pred 2.60mL | RVD -52.1%
- **Lesions**: GT 13 | Pred 12 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0029 vs 0.0012 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 13 of 13 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=81.6mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0173
- **Overlap**: Dice 0.0009 | IoU 0.0004 | Precision 0.0033 | Recall 0.0005
- **Boundary**: HD95 95.42mm | ASSD 75.86mm
- **Volume**: GT 37.56mL | Pred 5.84mL | RVD -84.4%
- **Lesions**: GT 5 | Pred 6 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0017 vs 0.0009 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 5 of 5 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=95.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0068
- **Overlap**: Dice 0.0003 | IoU 0.0001 | Precision 0.0003 | Recall 0.0003
- **Boundary**: HD95 149.80mm | ASSD 132.04mm
- **Volume**: GT 17.71mL | Pred 14.97mL | RVD -15.5%
- **Lesions**: GT 20 | Pred 3 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0005 vs 0.0003 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 20 of 20 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=149.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0087
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 68.29mm | ASSD 48.17mm
- **Volume**: GT 15.47mL | Pred 23.25mL | RVD +50.2%
- **Lesions**: GT 10 | Pred 8 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 10 of 10 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=68.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0187
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 205.91mm | ASSD 88.43mm
- **Volume**: GT 0.12mL | Pred 2.00mL | RVD +1542.9%
- **Lesions**: GT 1 | Pred 6 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.12 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=205.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0188
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 108.42mm | ASSD 59.01mm
- **Volume**: GT 50.82mL | Pred 9.99mL | RVD -80.3%
- **Lesions**: GT 1 | Pred 6 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=108.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0189
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 91.37mm | ASSD 35.23mm
- **Volume**: GT 58.37mL | Pred 41.26mL | RVD -29.3%
- **Lesions**: GT 11 | Pred 2 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 11 of 11 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=91.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0190
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 35.86mm | ASSD 26.14mm
- **Volume**: GT 3.11mL | Pred 36.03mL | RVD +1057.6%
- **Lesions**: GT 3 | Pred 2 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 3 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=35.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0186
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 127.00mm | ASSD 93.48mm
- **Volume**: GT 1.94mL | Pred 22.75mL | RVD +1072.5%
- **Lesions**: GT 4 | Pred 10 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0006 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 4 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=127.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0192
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 69.65mm | ASSD 63.56mm
- **Volume**: GT 1.38mL | Pred 2.19mL | RVD +58.6%
- **Lesions**: GT 2 | Pred 4 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=69.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0081
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 57.97mm | ASSD 35.51mm
- **Volume**: GT 1.53mL | Pred 21.18mL | RVD +1281.7%
- **Lesions**: GT 5 | Pred 5 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 5 of 5 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=58.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0002
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 153.75mm | ASSD 81.40mm
- **Volume**: GT 493.72mL | Pred 0.50mL | RVD -99.9%
- **Lesions**: GT 1 | Pred 1 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=153.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0080
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 99.40mm | ASSD 71.11mm
- **Volume**: GT 13.91mL | Pred 6.58mL | RVD -52.7%
- **Lesions**: GT 3 | Pred 7 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 3 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=99.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0006
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 130.81mm | ASSD 62.27mm
- **Volume**: GT 26.25mL | Pred 23.68mL | RVD -9.8%
- **Lesions**: GT 4 | Pred 9 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 4 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=130.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0007
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 163.88mm | ASSD 126.65mm
- **Volume**: GT 2.65mL | Pred 10.57mL | RVD +298.8%
- **Lesions**: GT 1 | Pred 7 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=163.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0078
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 132.48mm | ASSD 104.84mm
- **Volume**: GT 1.20mL | Pred 7.72mL | RVD +542.9%
- **Lesions**: GT 4 | Pred 7 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 4 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=132.5mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0077
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 83.62mm | ASSD 59.77mm
- **Volume**: GT 2.01mL | Pred 13.53mL | RVD +572.5%
- **Lesions**: GT 1 | Pred 7 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=83.6mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0191
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 114.04mm | ASSD 86.12mm
- **Volume**: GT 11.54mL | Pred 15.26mL | RVD +32.3%
- **Lesions**: GT 9 | Pred 7 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 9 of 9 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=114.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0089
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 124.37mm | ASSD 78.25mm
- **Volume**: GT 4.46mL | Pred 10.51mL | RVD +135.8%
- **Lesions**: GT 3 | Pred 5 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 3 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=124.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0184
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 87.97mm | ASSD 42.21mm
- **Volume**: GT 4.49mL | Pred 10.71mL | RVD +138.8%
- **Lesions**: GT 5 | Pred 7 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 5 of 5 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=88.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0090
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 119.92mm | ASSD 105.81mm
- **Volume**: GT 4.08mL | Pred 1.24mL | RVD -69.7%
- **Lesions**: GT 1 | Pred 4 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=119.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0091
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 84.99mm | ASSD 73.30mm
- **Volume**: GT 13.74mL | Pred 3.96mL | RVD -71.2%
- **Lesions**: GT 8 | Pred 7 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 8 of 8 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=85.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0178
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 61.03mm | ASSD 32.17mm
- **Volume**: GT 2.18mL | Pred 3.86mL | RVD +76.9%
- **Lesions**: GT 2 | Pred 2 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=61.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0175
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 163.09mm | ASSD 105.75mm
- **Volume**: GT 4.56mL | Pred 10.11mL | RVD +121.5%
- **Lesions**: GT 5 | Pred 1 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 5 of 5 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=163.1mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0008
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 114.03mm | ASSD 81.63mm
- **Volume**: GT 22.26mL | Pred 1.55mL | RVD -93.1%
- **Lesions**: GT 1 | Pred 8 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=114.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0171
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 145.75mm | ASSD 59.24mm
- **Volume**: GT 3.37mL | Pred 1.78mL | RVD -47.2%
- **Lesions**: GT 2 | Pred 16 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=145.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0102
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 140.50mm | ASSD 81.81mm
- **Volume**: GT 6.99mL | Pred 4.18mL | RVD -40.2%
- **Lesions**: GT 4 | Pred 13 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 4 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=140.5mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0101
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 151.68mm | ASSD 144.87mm
- **Volume**: GT 0.19mL | Pred 0.35mL | RVD +85.0%
- **Lesions**: GT 1 | Pred 6 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.19 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=151.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0092
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 105.32mm | ASSD 82.22mm
- **Volume**: GT 1.24mL | Pred 1.13mL | RVD -8.7%
- **Lesions**: GT 3 | Pred 3 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 3 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=105.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0094
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 111.34mm | ASSD 87.04mm
- **Volume**: GT 4.64mL | Pred 5.40mL | RVD +16.2%
- **Lesions**: GT 2 | Pred 7 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=111.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0099
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 144.26mm | ASSD 85.25mm
- **Volume**: GT 14.83mL | Pred 20.69mL | RVD +39.5%
- **Lesions**: GT 15 | Pred 4 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 15 of 15 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=144.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0096
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 90.19mm | ASSD 68.34mm
- **Volume**: GT 0.90mL | Pred 7.54mL | RVD +740.4%
- **Lesions**: GT 1 | Pred 5 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.90 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=90.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0185
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 150.01mm | ASSD 70.69mm
- **Volume**: GT 30.52mL | Pred 0.17mL | RVD -99.4%
- **Lesions**: GT 11 | Pred 2 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 11 of 11 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=150.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0027
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 111.31mm | ASSD 87.24mm
- **Volume**: GT 6.60mL | Pred 17.21mL | RVD +160.6%
- **Lesions**: GT 4 | Pred 5 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 4 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=111.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0028
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 155.76mm | ASSD 53.34mm
- **Volume**: GT 69.78mL | Pred 0.65mL | RVD -99.1%
- **Lesions**: GT 2 | Pred 1 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=155.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0029
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 151.85mm | ASSD 119.14mm
- **Volume**: GT 7.34mL | Pred 6.19mL | RVD -15.7%
- **Lesions**: GT 1 | Pred 6 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=151.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0031
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 124.49mm | ASSD 116.91mm
- **Volume**: GT 0.68mL | Pred 0.01mL | RVD -98.7%
- **Lesions**: GT 2 | Pred 1 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.68 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=124.5mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0033
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 163.72mm | ASSD 132.63mm
- **Volume**: GT 0.26mL | Pred 10.12mL | RVD +3727.3%
- **Lesions**: GT 1 | Pred 5 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.26 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=163.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0036
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 114.66mm | ASSD 78.20mm
- **Volume**: GT 39.83mL | Pred 8.59mL | RVD -78.4%
- **Lesions**: GT 10 | Pred 11 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 10 of 10 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=114.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0037
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 144.10mm | ASSD 119.05mm
- **Volume**: GT 115.47mL | Pred 9.33mL | RVD -91.9%
- **Lesions**: GT 1 | Pred 16 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=144.1mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0038
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 119.20mm | ASSD 103.55mm
- **Volume**: GT 5.97mL | Pred 7.19mL | RVD +20.4%
- **Lesions**: GT 1 | Pred 3 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=119.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0039
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 158.56mm | ASSD 116.38mm
- **Volume**: GT 10.26mL | Pred 28.72mL | RVD +179.8%
- **Lesions**: GT 1 | Pred 5 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=158.6mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0040
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 121.04mm | ASSD 90.27mm
- **Volume**: GT 2.40mL | Pred 5.55mL | RVD +131.4%
- **Lesions**: GT 6 | Pred 6 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 6 of 6 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=121.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0041
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 143.26mm | ASSD 124.07mm
- **Volume**: GT 1.44mL | Pred 8.98mL | RVD +523.2%
- **Lesions**: GT 1 | Pred 6 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=143.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0045
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 144.87mm | ASSD 112.19mm
- **Volume**: GT 11.00mL | Pred 37.33mL | RVD +239.5%
- **Lesions**: GT 1 | Pred 5 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=144.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0033
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 68.53mm | ASSD 48.75mm
- **Volume**: GT 7.30mL | Pred 0.55mL | RVD -92.4%
- **Lesions**: GT 1 | Pred 1 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=68.5mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0034
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 173.26mm | ASSD 153.70mm
- **Volume**: GT 7.82mL | Pred 9.24mL | RVD +18.2%
- **Lesions**: GT 1 | Pred 6 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=173.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0035
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 129.69mm | ASSD 88.74mm
- **Volume**: GT 2.42mL | Pred 7.43mL | RVD +206.9%
- **Lesions**: GT 2 | Pred 2 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=129.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0036
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 141.29mm | ASSD 108.91mm
- **Volume**: GT 6.48mL | Pred 28.76mL | RVD +343.7%
- **Lesions**: GT 2 | Pred 3 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=141.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0037
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 179.17mm | ASSD 110.09mm
- **Volume**: GT 9.00mL | Pred 4.61mL | RVD -48.7%
- **Lesions**: GT 9 | Pred 10 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 9 of 9 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=179.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0038
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 181.65mm | ASSD 114.28mm
- **Volume**: GT 1.13mL | Pred 15.88mL | RVD +1308.7%
- **Lesions**: GT 2 | Pred 11 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=181.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0039
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 138.42mm | ASSD 82.15mm
- **Volume**: GT 60.63mL | Pred 15.32mL | RVD -74.7%
- **Lesions**: GT 4 | Pred 1 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 4 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=138.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0040
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 195.56mm | ASSD 176.80mm
- **Volume**: GT 0.18mL | Pred 6.70mL | RVD +3690.9%
- **Lesions**: GT 1 | Pred 4 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.18 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=195.6mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0012
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 165.90mm | ASSD 137.04mm
- **Volume**: GT 37.74mL | Pred 0.66mL | RVD -98.3%
- **Lesions**: GT 4 | Pred 5 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 4 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=165.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0038
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 154.13mm | ASSD 101.84mm
- **Volume**: GT 11.22mL | Pred 11.13mL | RVD -0.7%
- **Lesions**: GT 10 | Pred 6 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 10 of 10 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=154.1mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0076
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 138.38mm | ASSD 91.47mm
- **Volume**: GT 5.30mL | Pred 14.23mL | RVD +168.7%
- **Lesions**: GT 4 | Pred 9 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 4 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=138.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0041
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 83.17mm | ASSD 67.45mm
- **Volume**: GT 0.79mL | Pred 1.95mL | RVD +146.6%
- **Lesions**: GT 2 | Pred 4 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.79 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=83.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0042
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 150.76mm | ASSD 111.13mm
- **Volume**: GT 1.96mL | Pred 0.18mL | RVD -90.7%
- **Lesions**: GT 3 | Pred 3 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 3 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=150.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0044
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 97.01mm | ASSD 80.75mm
- **Volume**: GT 10.47mL | Pred 15.66mL | RVD +49.6%
- **Lesions**: GT 13 | Pred 11 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 13 of 13 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=97.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0046
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 87.61mm | ASSD 45.54mm
- **Volume**: GT 3.37mL | Pred 14.17mL | RVD +321.0%
- **Lesions**: GT 1 | Pred 4 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=87.6mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0047
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 108.42mm | ASSD 39.37mm
- **Volume**: GT 2.01mL | Pred 2.96mL | RVD +47.4%
- **Lesions**: GT 4 | Pred 6 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 4 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=108.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0049
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 100.40mm | ASSD 52.28mm
- **Volume**: GT 3.37mL | Pred 4.62mL | RVD +37.1%
- **Lesions**: GT 2 | Pred 4 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=100.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0020
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 106.63mm | ASSD 69.21mm
- **Volume**: GT 0.61mL | Pred 0.85mL | RVD +39.0%
- **Lesions**: GT 1 | Pred 3 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.61 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=106.6mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0053
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 63.29mm | ASSD 20.46mm
- **Volume**: GT 1.74mL | Pred 0.07mL | RVD -96.2%
- **Lesions**: GT 3 | Pred 2 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 3 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=63.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0056
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 70.51mm | ASSD 17.61mm
- **Volume**: GT 5.39mL | Pred 0.63mL | RVD -88.4%
- **Lesions**: GT 5 | Pred 1 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 5 of 5 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=70.5mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0058
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 63.16mm | ASSD 51.68mm
- **Volume**: GT 0.68mL | Pred 16.42mL | RVD +2303.8%
- **Lesions**: GT 1 | Pred 4 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.68 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=63.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0059
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 143.24mm | ASSD 127.34mm
- **Volume**: GT 5.96mL | Pred 5.36mL | RVD -10.0%
- **Lesions**: GT 4 | Pred 7 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 4 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=143.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0063
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 180.03mm | ASSD 77.96mm
- **Volume**: GT 6.31mL | Pred 0.07mL | RVD -98.8%
- **Lesions**: GT 8 | Pred 1 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 8 of 8 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=180.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0069
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 124.50mm | ASSD 94.87mm
- **Volume**: GT 5.46mL | Pred 22.13mL | RVD +305.6%
- **Lesions**: GT 4 | Pred 3 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 4 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=124.5mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0013
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 116.62mm | ASSD 97.79mm
- **Volume**: GT 4.00mL | Pred 7.70mL | RVD +92.7%
- **Lesions**: GT 1 | Pred 10 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=116.6mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0014
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 117.82mm | ASSD 101.44mm
- **Volume**: GT 6.90mL | Pred 17.07mL | RVD +147.2%
- **Lesions**: GT 1 | Pred 8 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=117.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0015
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 174.07mm | ASSD 96.17mm
- **Volume**: GT 2.21mL | Pred 2.39mL | RVD +7.7%
- **Lesions**: GT 2 | Pred 7 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=174.1mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0017
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision nan | Recall 0.0000
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.48mL | Pred 0.00mL | RVD -100.0%
- **Lesions**: GT 1 | Pred 0 | Sensitivity 0.00 | Precision nan
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.48 mL) — low Dice here is partly a size-effect artifact, not purely a model failure.

### Case: BHSD_0051
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 74.80mm | ASSD 40.73mm
- **Volume**: GT 13.30mL | Pred 2.98mL | RVD -77.6%
- **Lesions**: GT 5 | Pred 2 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 5 of 5 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=74.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0001
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 85.92mm | ASSD 61.56mm
- **Volume**: GT 5.19mL | Pred 5.39mL | RVD +3.8%
- **Lesions**: GT 3 | Pred 6 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 3 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=85.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0002
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 83.22mm | ASSD 57.94mm
- **Volume**: GT 4.27mL | Pred 0.31mL | RVD -92.7%
- **Lesions**: GT 2 | Pred 2 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=83.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0003
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 122.31mm | ASSD 81.19mm
- **Volume**: GT 32.45mL | Pred 19.94mL | RVD -38.5%
- **Lesions**: GT 3 | Pred 8 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 3 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=122.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0004
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 125.01mm | ASSD 101.94mm
- **Volume**: GT 5.16mL | Pred 6.75mL | RVD +30.9%
- **Lesions**: GT 1 | Pred 3 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=125.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0005
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 147.79mm | ASSD 105.97mm
- **Volume**: GT 7.85mL | Pred 5.65mL | RVD -28.1%
- **Lesions**: GT 4 | Pred 7 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 4 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=147.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0010
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 134.01mm | ASSD 85.19mm
- **Volume**: GT 0.25mL | Pred 4.08mL | RVD +1505.7%
- **Lesions**: GT 1 | Pred 6 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.25 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=134.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0012
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 111.06mm | ASSD 86.04mm
- **Volume**: GT 0.64mL | Pred 7.22mL | RVD +1020.1%
- **Lesions**: GT 1 | Pred 4 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.64 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=111.1mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0013
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 195.32mm | ASSD 180.91mm
- **Volume**: GT 10.45mL | Pred 2.37mL | RVD -77.3%
- **Lesions**: GT 1 | Pred 3 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=195.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0014
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 92.98mm | ASSD 79.01mm
- **Volume**: GT 0.37mL | Pred 0.80mL | RVD +114.8%
- **Lesions**: GT 1 | Pred 10 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.37 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=93.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0169
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 70.17mm | ASSD 46.65mm
- **Volume**: GT 1.15mL | Pred 0.65mL | RVD -43.8%
- **Lesions**: GT 1 | Pred 4 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=70.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0017
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision nan | Recall 0.0000
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.50mL | Pred 0.00mL | RVD -100.0%
- **Lesions**: GT 1 | Pred 0 | Sensitivity 0.00 | Precision nan
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.50 mL) — low Dice here is partly a size-effect artifact, not purely a model failure.

### Case: PHYSIO_0018
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 209.33mm | ASSD 177.93mm
- **Volume**: GT 4.77mL | Pred 8.49mL | RVD +78.0%
- **Lesions**: GT 1 | Pred 5 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=209.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0019
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 190.30mm | ASSD 145.41mm
- **Volume**: GT 7.58mL | Pred 18.20mL | RVD +140.3%
- **Lesions**: GT 2 | Pred 7 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=190.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0021
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 109.31mm | ASSD 79.01mm
- **Volume**: GT 14.46mL | Pred 20.91mL | RVD +44.7%
- **Lesions**: GT 12 | Pred 7 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 12 of 12 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=109.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0022
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 86.13mm | ASSD 59.63mm
- **Volume**: GT 24.62mL | Pred 9.86mL | RVD -59.9%
- **Lesions**: GT 1 | Pred 8 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=86.1mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0023
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 103.10mm | ASSD 86.35mm
- **Volume**: GT 5.73mL | Pred 3.24mL | RVD -43.5%
- **Lesions**: GT 2 | Pred 9 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=103.1mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0024
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 143.51mm | ASSD 105.27mm
- **Volume**: GT 0.13mL | Pred 8.94mL | RVD +6876.2%
- **Lesions**: GT 1 | Pred 4 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.13 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=143.5mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0026
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 130.00mm | ASSD 75.36mm
- **Volume**: GT 40.65mL | Pred 26.08mL | RVD -35.8%
- **Lesions**: GT 1 | Pred 5 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=130.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0027
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 67.36mm | ASSD 47.91mm
- **Volume**: GT 1.69mL | Pred 1.73mL | RVD +2.6%
- **Lesions**: GT 1 | Pred 2 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=67.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0015
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 144.84mm | ASSD 95.82mm
- **Volume**: GT 30.48mL | Pred 1.90mL | RVD -93.8%
- **Lesions**: GT 1 | Pred 4 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=144.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0005
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 59.17mm | ASSD 29.08mm
- **Volume**: GT 5.88mL | Pred 19.87mL | RVD +237.7%
- **Lesions**: GT 3 | Pred 3 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 3 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=59.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0009
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision nan | Recall 0.0000
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.21mL | Pred 0.00mL | RVD -100.0%
- **Lesions**: GT 1 | Pred 0 | Sensitivity 0.00 | Precision nan
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.21 mL) — low Dice here is partly a size-effect artifact, not purely a model failure.

### Case: BHSD_0011
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 129.53mm | ASSD 111.56mm
- **Volume**: GT 2.66mL | Pred 6.38mL | RVD +140.0%
- **Lesions**: GT 1 | Pred 4 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=129.5mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0012
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 94.74mm | ASSD 64.20mm
- **Volume**: GT 2.47mL | Pred 2.92mL | RVD +18.0%
- **Lesions**: GT 2 | Pred 11 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=94.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0014
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 142.50mm | ASSD 117.27mm
- **Volume**: GT 0.90mL | Pred 10.18mL | RVD +1035.6%
- **Lesions**: GT 1 | Pred 6 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.90 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=142.5mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0015
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 43.54mm | ASSD 25.21mm
- **Volume**: GT 11.91mL | Pred 99.06mL | RVD +731.7%
- **Lesions**: GT 8 | Pred 1 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 8 of 8 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=43.5mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0016
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 85.63mm | ASSD 48.99mm
- **Volume**: GT 3.06mL | Pred 14.76mL | RVD +382.1%
- **Lesions**: GT 2 | Pred 8 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=85.6mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0017
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 157.90mm | ASSD 74.56mm
- **Volume**: GT 5.46mL | Pred 31.65mL | RVD +480.2%
- **Lesions**: GT 13 | Pred 11 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 13 of 13 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=157.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0023
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 159.86mm | ASSD 120.78mm
- **Volume**: GT 0.48mL | Pred 3.81mL | RVD +690.1%
- **Lesions**: GT 1 | Pred 5 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.48 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=159.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: CQ500_0051
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 111.40mm | ASSD 94.75mm
- **Volume**: GT 8.87mL | Pred 6.63mL | RVD -25.2%
- **Lesions**: GT 1 | Pred 7 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=111.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0028
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 59.51mm | ASSD 34.99mm
- **Volume**: GT 16.35mL | Pred 3.22mL | RVD -80.3%
- **Lesions**: GT 21 | Pred 6 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 21 of 21 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=59.5mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0029
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 179.38mm | ASSD 128.09mm
- **Volume**: GT 10.94mL | Pred 12.42mL | RVD +13.6%
- **Lesions**: GT 1 | Pred 2 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=179.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0030
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 55.45mm | ASSD 47.40mm
- **Volume**: GT 2.82mL | Pred 1.13mL | RVD -60.1%
- **Lesions**: GT 2 | Pred 3 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=55.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0031
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 172.30mm | ASSD 122.04mm
- **Volume**: GT 0.51mL | Pred 6.97mL | RVD +1265.4%
- **Lesions**: GT 2 | Pred 9 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.51 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=172.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0032
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 180.99mm | ASSD 146.98mm
- **Volume**: GT 84.82mL | Pred 6.49mL | RVD -92.3%
- **Lesions**: GT 1 | Pred 4 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=181.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0033
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 87.21mm | ASSD 71.14mm
- **Volume**: GT 0.67mL | Pred 18.64mL | RVD +2663.0%
- **Lesions**: GT 1 | Pred 8 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.67 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=87.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0034
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 121.72mm | ASSD 95.47mm
- **Volume**: GT 0.58mL | Pred 3.56mL | RVD +511.8%
- **Lesions**: GT 1 | Pred 4 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.58 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=121.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0035
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 104.19mm | ASSD 62.25mm
- **Volume**: GT 21.14mL | Pred 4.98mL | RVD -76.5%
- **Lesions**: GT 1 | Pred 3 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=104.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0037
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 137.44mm | ASSD 98.60mm
- **Volume**: GT 4.36mL | Pred 5.54mL | RVD +27.0%
- **Lesions**: GT 2 | Pred 1 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=137.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0024
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 106.92mm | ASSD 93.76mm
- **Volume**: GT 1.23mL | Pred 7.16mL | RVD +484.4%
- **Lesions**: GT 1 | Pred 8 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=106.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0140
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 95.77mm | ASSD 74.60mm
- **Volume**: GT 3.10mL | Pred 7.04mL | RVD +126.7%
- **Lesions**: GT 3 | Pred 3 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 3 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=95.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0141
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 152.76mm | ASSD 120.93mm
- **Volume**: GT 2.07mL | Pred 18.83mL | RVD +809.7%
- **Lesions**: GT 1 | Pred 3 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=152.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0145
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 71.84mm | ASSD 35.04mm
- **Volume**: GT 6.54mL | Pred 3.76mL | RVD -42.5%
- **Lesions**: GT 15 | Pred 12 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 15 of 15 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=71.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0146
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 65.68mm | ASSD 24.68mm
- **Volume**: GT 4.87mL | Pred 21.36mL | RVD +338.8%
- **Lesions**: GT 3 | Pred 5 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 3 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=65.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0147
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 113.79mm | ASSD 85.53mm
- **Volume**: GT 5.64mL | Pred 1.03mL | RVD -81.8%
- **Lesions**: GT 10 | Pred 3 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 10 of 10 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=113.8mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0155
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 86.17mm | ASSD 60.15mm
- **Volume**: GT 11.92mL | Pred 4.39mL | RVD -63.2%
- **Lesions**: GT 5 | Pred 6 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 5 of 5 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=86.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0156
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 167.25mm | ASSD 102.62mm
- **Volume**: GT 66.98mL | Pred 9.72mL | RVD -85.5%
- **Lesions**: GT 1 | Pred 5 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=167.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0157
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 189.19mm | ASSD 150.33mm
- **Volume**: GT 0.48mL | Pred 9.13mL | RVD +1796.0%
- **Lesions**: GT 1 | Pred 6 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.48 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=189.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0159
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 140.01mm | ASSD 126.51mm
- **Volume**: GT 2.29mL | Pred 2.96mL | RVD +28.9%
- **Lesions**: GT 1 | Pred 4 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=140.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0028
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 111.05mm | ASSD 81.94mm
- **Volume**: GT 3.76mL | Pred 2.88mL | RVD -23.3%
- **Lesions**: GT 1 | Pred 5 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=111.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0162
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 87.65mm | ASSD 69.17mm
- **Volume**: GT 1.03mL | Pred 0.25mL | RVD -75.9%
- **Lesions**: GT 2 | Pred 5 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=87.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0163
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 111.61mm | ASSD 96.62mm
- **Volume**: GT 3.54mL | Pred 18.39mL | RVD +419.8%
- **Lesions**: GT 2 | Pred 2 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=111.6mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0165
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 118.89mm | ASSD 91.89mm
- **Volume**: GT 0.13mL | Pred 15.07mL | RVD +11607.4%
- **Lesions**: GT 1 | Pred 6 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.13 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=118.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0166
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 160.39mm | ASSD 111.21mm
- **Volume**: GT 7.56mL | Pred 6.82mL | RVD -9.7%
- **Lesions**: GT 1 | Pred 10 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=160.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0168
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 173.11mm | ASSD 143.54mm
- **Volume**: GT 0.31mL | Pred 6.40mL | RVD +1998.4%
- **Lesions**: GT 1 | Pred 7 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.31 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=173.1mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0042
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 81.71mm | ASSD 63.37mm
- **Volume**: GT 6.44mL | Pred 6.22mL | RVD -3.4%
- **Lesions**: GT 1 | Pred 3 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Boundary distance is large (HD95=81.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0071
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 91.28mm | ASSD 67.35mm
- **Volume**: GT 76.87mL | Pred 7.12mL | RVD -90.7%
- **Lesions**: GT 3 | Pred 11 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0002 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 3 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=91.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0074
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 213.46mm | ASSD 167.60mm
- **Volume**: GT 4.79mL | Pred 7.80mL | RVD +62.8%
- **Lesions**: GT 1 | Pred 2 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=213.5mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0075
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 121.04mm | ASSD 87.56mm
- **Volume**: GT 3.30mL | Pred 19.99mL | RVD +505.9%
- **Lesions**: GT 3 | Pred 9 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 3 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=121.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0160
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 179.03mm | ASSD 163.58mm
- **Volume**: GT 0.54mL | Pred 4.32mL | RVD +702.7%
- **Lesions**: GT 1 | Pred 5 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.54 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=179.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0029
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 96.03mm | ASSD 77.38mm
- **Volume**: GT 4.02mL | Pred 11.62mL | RVD +189.2%
- **Lesions**: GT 1 | Pred 5 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=96.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0030
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 115.32mm | ASSD 93.60mm
- **Volume**: GT 0.74mL | Pred 1.68mL | RVD +126.1%
- **Lesions**: GT 1 | Pred 3 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.74 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=115.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: PHYSIO_0031
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 109.05mm | ASSD 86.83mm
- **Volume**: GT 3.11mL | Pred 14.81mL | RVD +376.2%
- **Lesions**: GT 1 | Pred 11 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Boundary distance is large (HD95=109.1mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0104
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 102.59mm | ASSD 68.98mm
- **Volume**: GT 0.54mL | Pred 3.02mL | RVD +462.2%
- **Lesions**: GT 1 | Pred 10 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.54 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=102.6mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0105
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 128.02mm | ASSD 101.14mm
- **Volume**: GT 13.21mL | Pred 3.27mL | RVD -75.2%
- **Lesions**: GT 5 | Pred 4 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 5 of 5 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=128.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0108
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 141.98mm | ASSD 98.25mm
- **Volume**: GT 0.40mL | Pred 13.81mL | RVD +3389.2%
- **Lesions**: GT 1 | Pred 6 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.40 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=142.0mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0113
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 138.16mm | ASSD 106.47mm
- **Volume**: GT 0.64mL | Pred 9.16mL | RVD +1323.7%
- **Lesions**: GT 4 | Pred 9 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.64 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Multi-focal case: missed 4 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=138.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0114
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 68.25mm | ASSD 41.84mm
- **Volume**: GT 0.78mL | Pred 0.45mL | RVD -42.8%
- **Lesions**: GT 2 | Pred 4 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.78 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=68.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0120
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 75.38mm | ASSD 44.20mm
- **Volume**: GT 0.33mL | Pred 23.86mL | RVD +7047.1%
- **Lesions**: GT 1 | Pred 1 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.33 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=75.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0139
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 108.25mm | ASSD 59.24mm
- **Volume**: GT 17.94mL | Pred 1.98mL | RVD -89.0%
- **Lesions**: GT 10 | Pred 7 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 10 of 10 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=108.3mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0127
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 98.11mm | ASSD 86.68mm
- **Volume**: GT 0.52mL | Pred 1.19mL | RVD +128.4%
- **Lesions**: GT 1 | Pred 5 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.52 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=98.1mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0128
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 169.71mm | ASSD 107.16mm
- **Volume**: GT 10.80mL | Pred 24.63mL | RVD +128.0%
- **Lesions**: GT 2 | Pred 7 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 2 of 2 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=169.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0130
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 66.65mm | ASSD 26.59mm
- **Volume**: GT 17.30mL | Pred 25.67mL | RVD +48.4%
- **Lesions**: GT 13 | Pred 6 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 13 of 13 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=66.6mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0131
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 142.67mm | ASSD 131.82mm
- **Volume**: GT 0.33mL | Pred 7.14mL | RVD +2071.0%
- **Lesions**: GT 1 | Pred 3 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.33 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=142.7mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0133
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 186.44mm | ASSD 142.30mm
- **Volume**: GT 41.05mL | Pred 7.47mL | RVD -81.8%
- **Lesions**: GT 3 | Pred 1 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 3 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=186.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0134
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 130.14mm | ASSD 100.49mm
- **Volume**: GT 0.40mL | Pred 6.74mL | RVD +1602.4%
- **Lesions**: GT 1 | Pred 4 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.40 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=130.1mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0135
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 132.38mm | ASSD 111.35mm
- **Volume**: GT 0.13mL | Pred 5.21mL | RVD +3800.0%
- **Lesions**: GT 1 | Pred 1 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Model struggled on a small lesion (GT volume 0.13 mL) — low Dice here is partly a size-effect artifact, not purely a model failure. Boundary distance is large (HD95=132.4mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0136
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 119.19mm | ASSD 74.19mm
- **Volume**: GT 53.34mL | Pred 13.24mL | RVD -75.2%
- **Lesions**: GT 3 | Pred 13 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe under-segmentation; missed significant hemorrhage volume. Multi-focal case: missed 3 of 3 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=119.2mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0138
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 87.91mm | ASSD 64.04mm
- **Volume**: GT 4.29mL | Pred 62.33mL | RVD +1353.0%
- **Lesions**: GT 4 | Pred 3 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 4 of 4 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=87.9mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0126
- **Overlap**: Dice 0.0000 | IoU 0.0000 | Precision 0.0000 | Recall 0.0000
- **Boundary**: HD95 98.64mm | ASSD 65.19mm
- **Volume**: GT 5.79mL | Pred 10.47mL | RVD +80.7%
- **Lesions**: GT 5 | Pred 5 | Sensitivity 0.00 | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs 0.0000 @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present. Multi-focal case: missed 5 of 5 distinct hemorrhage foci entirely, despite decent voxel-level overlap on the detected ones. Boundary distance is large (HD95=98.6mm), indicating localized regions of substantial disagreement even where bulk overlap looks fine.

### Case: BHSD_0022
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 0.99mL | RVD +nan%
- **Lesions**: GT 0 | Pred 5 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: BHSD_0050
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 14.52mL | RVD +nan%
- **Lesions**: GT 0 | Pred 3 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: BHSD_0079
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 11.52mL | RVD +nan%
- **Lesions**: GT 0 | Pred 6 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: BHSD_0109
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 0.81mL | RVD +nan%
- **Lesions**: GT 0 | Pred 1 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: BHSD_0117
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 5.55mL | RVD +nan%
- **Lesions**: GT 0 | Pred 5 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: BHSD_0125
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 17.61mL | RVD +nan%
- **Lesions**: GT 0 | Pred 5 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: CQ500_0005
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 0.13mL | RVD +nan%
- **Lesions**: GT 0 | Pred 2 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: CQ500_0011
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 2.96mL | RVD +nan%
- **Lesions**: GT 0 | Pred 7 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: CQ500_0016
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 11.93mL | RVD +nan%
- **Lesions**: GT 0 | Pred 16 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: CQ500_0047
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 0.86mL | RVD +nan%
- **Lesions**: GT 0 | Pred 4 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: CQ500_0048
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 12.50mL | RVD +nan%
- **Lesions**: GT 0 | Pred 2 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0006
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 11.65mL | RVD +nan%
- **Lesions**: GT 0 | Pred 12 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0007
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 15.26mL | RVD +nan%
- **Lesions**: GT 0 | Pred 14 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0008
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 0.08mL | RVD +nan%
- **Lesions**: GT 0 | Pred 1 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0009
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 13.30mL | RVD +nan%
- **Lesions**: GT 0 | Pred 9 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0040
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 3.95mL | RVD +nan%
- **Lesions**: GT 0 | Pred 5 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0041
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 0.31mL | RVD +nan%
- **Lesions**: GT 0 | Pred 1 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0043
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 12.86mL | RVD +nan%
- **Lesions**: GT 0 | Pred 8 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0044
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 21.23mL | RVD +nan%
- **Lesions**: GT 0 | Pred 12 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0045
- **Overlap**: Dice nan | IoU nan | Precision nan | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 0.00mL | RVD +nan%
- **Lesions**: GT 0 | Pred 0 | Sensitivity nan | Precision nan
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Moderate agreement with mixed boundary discrepancies.

### Case: PHYSIO_0046
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 4.28mL | RVD +nan%
- **Lesions**: GT 0 | Pred 4 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0047
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 1.91mL | RVD +nan%
- **Lesions**: GT 0 | Pred 6 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0048
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 10.63mL | RVD +nan%
- **Lesions**: GT 0 | Pred 1 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0049
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 2.37mL | RVD +nan%
- **Lesions**: GT 0 | Pred 6 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0050
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 21.68mL | RVD +nan%
- **Lesions**: GT 0 | Pred 7 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0051
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 6.07mL | RVD +nan%
- **Lesions**: GT 0 | Pred 4 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0052
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 2.74mL | RVD +nan%
- **Lesions**: GT 0 | Pred 3 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0053
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 4.31mL | RVD +nan%
- **Lesions**: GT 0 | Pred 8 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0054
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 0.02mL | RVD +nan%
- **Lesions**: GT 0 | Pred 2 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0055
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 25.12mL | RVD +nan%
- **Lesions**: GT 0 | Pred 3 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0056
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 4.38mL | RVD +nan%
- **Lesions**: GT 0 | Pred 10 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0057
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 2.51mL | RVD +nan%
- **Lesions**: GT 0 | Pred 1 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0058
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 3.14mL | RVD +nan%
- **Lesions**: GT 0 | Pred 3 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0059
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 4.70mL | RVD +nan%
- **Lesions**: GT 0 | Pred 6 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0060
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 2.78mL | RVD +nan%
- **Lesions**: GT 0 | Pred 5 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0061
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 23.93mL | RVD +nan%
- **Lesions**: GT 0 | Pred 2 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0062
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 10.66mL | RVD +nan%
- **Lesions**: GT 0 | Pred 1 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0063
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 1.55mL | RVD +nan%
- **Lesions**: GT 0 | Pred 4 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0064
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 25.02mL | RVD +nan%
- **Lesions**: GT 0 | Pred 3 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0065
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 1.94mL | RVD +nan%
- **Lesions**: GT 0 | Pred 2 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0066
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 0.56mL | RVD +nan%
- **Lesions**: GT 0 | Pred 1 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0067
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 6.49mL | RVD +nan%
- **Lesions**: GT 0 | Pred 5 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0068
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 20.27mL | RVD +nan%
- **Lesions**: GT 0 | Pred 8 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0069
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 2.43mL | RVD +nan%
- **Lesions**: GT 0 | Pred 5 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0070
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 25.48mL | RVD +nan%
- **Lesions**: GT 0 | Pred 7 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0071
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 0.94mL | RVD +nan%
- **Lesions**: GT 0 | Pred 4 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0072
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 7.19mL | RVD +nan%
- **Lesions**: GT 0 | Pred 4 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0073
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 6.97mL | RVD +nan%
- **Lesions**: GT 0 | Pred 3 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0074
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 1.22mL | RVD +nan%
- **Lesions**: GT 0 | Pred 3 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

### Case: PHYSIO_0075
- **Overlap**: Dice nan | IoU nan | Precision 0.0000 | Recall nan
- **Boundary**: HD95 nanmm | ASSD nanmm
- **Volume**: GT 0.00mL | Pred 10.24mL | RVD +nan%
- **Lesions**: GT 0 | Pred 3 | Sensitivity nan | Precision 0.00
- **Optimal threshold**: 0.30 (Dice 0.0000 vs nan @0.5)
- **Diagnosis**: Severe over-segmentation; spurious false-positive clusters present.

