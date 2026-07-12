import unittest
import torch
import torch.nn as nn

from training.health_monitor import HealthMonitor

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(4 * 16 * 16 * 16, 1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)

class TestHealthMonitor(unittest.TestCase):
    
    def test_health_monitor_hooks_and_stats(self):
        model = DummyModel()
        monitor = HealthMonitor(model)
        
        # Simulate forward pass
        x = torch.randn(2, 1, 16, 16, 16)
        out = model(x)
        
        # Simulate backward pass (to populate gradients)
        loss = out.sum()
        loss.backward()
        
        stats = monitor.check_health()
        
        # Validation
        self.assertIn("grad_max", stats)
        self.assertIn("grad_min", stats)
        self.assertIn("grad_norm", stats)
        
        # Dead gradients
        self.assertIn("dead_gradients_ratio", stats)
        self.assertGreaterEqual(stats["dead_gradients_ratio"], 0.0)
        
        # Weights
        self.assertIn("weight_drift", stats)
        self.assertEqual(stats["weight_drift"], 0.0) # Should be 0 since no optimizer step happened
        
        # Activations
        self.assertIn("activation_mean", stats)
        self.assertIn("dead_activations_ratio", stats)

if __name__ == '__main__':
    unittest.main()
