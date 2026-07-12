import unittest
import torch
import numpy as np

from evaluation.prediction_analysis import PredictionAnalyzer

class TestPredictionAnalyzer(unittest.TestCase):
    
    def test_prediction_analyzer(self):
        # Create a mock batch: B=2, D=4, H=4, W=4
        probs = torch.zeros((2, 1, 4, 4, 4))
        preds_bin = torch.zeros((2, 1, 4, 4, 4))
        masks = torch.zeros((2, 1, 4, 4, 4))
        
        # Batch 0: Perfect prediction
        probs[0, 0, 0:2, 0:2, 0:2] = 0.9
        preds_bin[0, 0, 0:2, 0:2, 0:2] = 1.0
        masks[0, 0, 0:2, 0:2, 0:2] = 1.0
        
        # Batch 1: Terrible prediction
        probs[1, 0, 2:4, 2:4, 2:4] = 0.9
        preds_bin[1, 0, 2:4, 2:4, 2:4] = 1.0
        masks[1, 0, 0:2, 0:2, 0:2] = 1.0
        
        stats = PredictionAnalyzer.analyze(probs, preds_bin, masks)
        
        self.assertIn("tp", stats)
        self.assertIn("fp", stats)
        self.assertIn("fn", stats)
        self.assertIn("tn", stats)
        
        # Batch 0 has 8 TP, Batch 1 has 8 FP, 8 FN, and some TN
        self.assertEqual(stats["tp"], 8)
        self.assertEqual(stats["fp"], 8)
        self.assertEqual(stats["fn"], 8)
        
        # Lesion count tracking
        self.assertIn("pred_lesion_count", stats)
        self.assertIn("pred_largest_lesion", stats)
        self.assertIn("pred_smallest_lesion", stats)

if __name__ == '__main__':
    unittest.main()
