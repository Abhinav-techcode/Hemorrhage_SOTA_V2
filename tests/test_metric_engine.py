import unittest
import torch
import numpy as np

try:
    from evaluation.metric_engine import ResearchMetricEngine
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False

class TestResearchMetricEngine(unittest.TestCase):
    
    @unittest.skipIf(not HAS_MONAI, "MONAI is required for Metric Engine")
    def test_canonical_prediction_pipeline(self):
        engine = ResearchMetricEngine(device="cpu")
        
        # Batch size 2, Channels 1, D 4, H 4, W 4
        logits = torch.zeros((2, 1, 4, 4, 4))
        masks = torch.zeros((2, 1, 4, 4, 4))
        
        # --- Batch 0: Perfect prediction ---
        # Logits > 0 means prob > 0.5 -> prediction 1
        logits[0, 0, 0:2, 0:2, 0:2] = 10.0
        masks[0, 0, 0:2, 0:2, 0:2] = 1.0
        
        # --- Batch 1: Complete Mismatch ---
        # Predicts foreground in bottom corner, ground truth is top corner
        logits[1, 0, 2:4, 2:4, 2:4] = 10.0
        masks[1, 0, 0:2, 0:2, 0:2] = 1.0
        
        # Execute canonical update
        engine.update(logits, masks, mode="val")
        
        # Compute metrics
        metrics = engine.compute(mode="val")
        
        # Validate that no exceptions occur and basic keys exist
        self.assertIn("val_dice", metrics)
        self.assertIn("val_iou", metrics)
        self.assertIn("val_accuracy", metrics)
        self.assertIn("val_precision", metrics)
        self.assertIn("val_recall", metrics)
        self.assertIn("val_specificity", metrics)
        self.assertIn("val_f1_score", metrics)
        self.assertIn("val_sensitivity", metrics)
        self.assertIn("val_hd95", metrics)
        self.assertIn("val_asd", metrics)
        
        # Values should be valid floats, not NaN
        for k, v in metrics.items():
            self.assertFalse(np.isnan(v), f"Metric {k} is NaN!")
            self.assertIsInstance(v, float, f"Metric {k} is not a float!")
            
        # Basic sanity checks based on batch setup
        # One perfect batch, one terrible batch. Accuracy should be ~0.875
        self.assertGreater(metrics["val_accuracy"], 0.0)
        self.assertGreater(metrics["val_dice"], 0.0)

if __name__ == '__main__':
    unittest.main()
