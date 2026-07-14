import unittest
from unittest.mock import MagicMock, call
import torch
import numpy as np

from evaluation.visualize import Visualizer
from training.logger import ExperimentLogger

class TestVisualizationPipeline(unittest.TestCase):

    def setUp(self):
        self.D, self.H, self.W = 16, 32, 32
        
        # Create a mock grayscale image
        self.img = torch.rand(self.D, self.H, self.W)
        
        # Create a mock ground truth
        self.gt = torch.zeros(self.D, self.H, self.W)
        self.gt[4:12, 8:24, 8:24] = 1.0
        
        # Create a mock prediction (slightly offset)
        self.pred = torch.zeros(self.D, self.H, self.W)
        self.pred[6:14, 10:26, 10:26] = 1.0

    def test_2d_slice_color_overlay(self):
        """Test the overlay rules: TP Green, FP Blue, FN Red, TN Gray, GT Bound Yellow, Pred Bound Cyan."""
        rgb = Visualizer.create_color_overlay(self.img, self.gt, self.pred)
        
        # Check shape (3, D, H, W)
        self.assertEqual(rgb.shape, (3, self.D, self.H, self.W))
        
        # Calculate expected TP, FP, FN
        tp = (self.pred * self.gt).bool()
        fp = (self.pred * (1 - self.gt)).bool()
        fn = ((1 - self.pred) * self.gt).bool()
        
        gt_bound = Visualizer.extract_boundaries(self.gt).bool()
        pred_bound = Visualizer.extract_boundaries(self.pred).bool()
        
        # Ensure mutually exclusive masks for strict color check where boundaries don't overlap
        pure_tp = tp & ~gt_bound & ~pred_bound
        pure_fp = fp & ~gt_bound & ~pred_bound
        pure_fn = fn & ~gt_bound & ~pred_bound
        
        # Check Green (TP)
        # Green channel should be 1.0, Red/Blue should be 0 or derived from img
        if pure_tp.any():
            self.assertTrue(torch.all(rgb[1, pure_tp] == 1.0))
            
        # Check Blue (FP)
        if pure_fp.any():
            self.assertTrue(torch.all(rgb[2, pure_fp] == 1.0))
            
        # Check Red (FN)
        if pure_fn.any():
            self.assertTrue(torch.all(rgb[0, pure_fn] == 1.0))
            
        # Check Yellow (GT Boundary: R=1, G=1)
        pure_gt_bound = gt_bound & ~pred_bound
        if pure_gt_bound.any():
            self.assertTrue(torch.all(rgb[0, pure_gt_bound] == 1.0))
            self.assertTrue(torch.all(rgb[1, pure_gt_bound] == 1.0))
            
        # Check Cyan (Pred Boundary: G=1, B=1)
        pure_pred_bound = pred_bound & ~gt_bound
        if pure_pred_bound.any():
            self.assertTrue(torch.all(rgb[1, pure_pred_bound] == 1.0))
            self.assertTrue(torch.all(rgb[2, pure_pred_bound] == 1.0))

    def test_3d_volume_output(self):
        """Test multi-plane logging without crashing."""
        mock_writer = MagicMock()
        
        image_batch = self.img.unsqueeze(0).unsqueeze(0) # (1, 1, D, H, W)
        target_batch = self.gt.unsqueeze(0).unsqueeze(0)
        # Mock prediction as logits (so sigmoid > 0.5 where it's 1.0)
        pred_logits = (self.pred * 10 - 5).unsqueeze(0).unsqueeze(0)
        
        Visualizer.log_multi_plane(
            writer=mock_writer,
            tag="TestVol",
            image=image_batch,
            pred=pred_logits,
            target=target_batch,
            step=1
        )
        
        # Verify 3 planes were logged
        self.assertEqual(mock_writer.add_image.call_count, 3)
        
        calls = mock_writer.add_image.call_args_list
        tags = [c[0][0] for c in calls]
        self.assertIn("TestVol/Axial_Overlay", tags)
        self.assertIn("TestVol/Coronal_Overlay", tags)
        self.assertIn("TestVol/Sagittal_Overlay", tags)

    def test_log_curves_metrics(self):
        """Test ExperimentLogger logs metrics correctly."""
        # Using MagicMock for SummaryWriter inside Logger
        logger = ExperimentLogger(MagicMock())
        logger.writer = MagicMock()
        logger.csv_path = MagicMock()
        
        metrics = {
            "val_dice": 0.85,
            "val_loss_total": 0.2,
            "train_loss_total": 0.3
        }
        
        # We need to mock open since logger tries to write to file
        with unittest.mock.patch('builtins.open', unittest.mock.mock_open()):
            logger.log_metrics(10, metrics)
        
        # Check that add_scalar was called 3 times
        self.assertEqual(logger.writer.add_scalar.call_count, 3)
        
        calls = logger.writer.add_scalar.call_args_list
        tags = [c[0][0] for c in calls]
        
        self.assertIn("Metrics/val_dice", tags)
        self.assertIn("Metrics/val_loss_total", tags)
        self.assertIn("Metrics/train_loss_total", tags)

if __name__ == "__main__":
    unittest.main()
