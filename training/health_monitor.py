import torch
import torch.nn as nn
from typing import Dict, Any
import numpy as np

class HealthMonitor:
    """
    Model Health Monitoring Module (Phase 7)
    Tracks gradient, weight, and activation statistics once per epoch.
    """
    def __init__(self, model: nn.Module, enabled: bool = False):
        self.model = model
        self.enabled = enabled
        self.initial_weights = {}
        
        if self.enabled:
            self._capture_initial_weights()
        
        self.activation_means = []
        self.activation_stds = []
        self.dead_activations_count = 0
        self.total_activations = 0
        self.hooks = []
        
        # Phase 4.3: Intrusive forward hooks only enabled in Debug mode
        if self.enabled:
            self._register_hooks()
            
    def _register_hooks(self):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor) and output.requires_grad:
                out = output.detach()
                self.activation_means.append(out.mean().item())
                self.activation_stds.append(out.std().item())
                dead = (out == 0).sum().item()
                self.dead_activations_count += dead
                self.total_activations += out.numel()

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv3d, nn.Linear, nn.ReLU, nn.GELU)):
                self.hooks.append(module.register_forward_hook(hook_fn))
        
    def _capture_initial_weights(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.initial_weights[name] = param.detach().clone().cpu()
                
    def check_health(self) -> Dict[str, float]:
        if not self.enabled:
            return {}
            
        stats = {}
        
        # 1. Gradient Statistics
        grad_max, grad_min, grad_mean = -float('inf'), float('inf'), []
        total_norm = 0.0
        dead_gradients = 0
        total_params = 0
        
        # 2. Weight Statistics
        weight_means, weight_stds = [], []
        weight_drift = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Gradients
                if param.grad is not None:
                    g = param.grad.detach()
                    grad_max = max(grad_max, g.max().item())
                    grad_min = min(grad_min, g.min().item())
                    grad_mean.append(g.mean().item())
                    total_norm += g.norm(2).item() ** 2
                    if g.abs().max().item() < 1e-8:
                        dead_gradients += 1
                    total_params += 1
                    
                # Weights
                w = param.detach().cpu()
                weight_means.append(w.mean().item())
                weight_stds.append(w.std().item())
                
                # Drift
                if name in self.initial_weights:
                    drift = torch.norm(w - self.initial_weights[name]).item()
                    weight_drift.append(drift)
                    
        stats["grad_max"] = float(grad_max) if grad_max != -float('inf') else 0.0
        stats["grad_min"] = float(grad_min) if grad_min != float('inf') else 0.0
        stats["grad_mean"] = float(np.mean(grad_mean)) if grad_mean else 0.0
        stats["grad_norm"] = float(total_norm ** 0.5)
        
        stats["weight_mean"] = float(np.mean(weight_means)) if weight_means else 0.0
        stats["weight_std"] = float(np.mean(weight_stds)) if weight_stds else 0.0
        stats["weight_drift"] = float(np.mean(weight_drift)) if weight_drift else 0.0
        
        stats["dead_gradients_ratio"] = float(dead_gradients / total_params) if total_params > 0 else 0.0
        
        # Activations
        stats["activation_mean"] = float(np.mean(self.activation_means)) if self.activation_means else 0.0
        stats["activation_std"] = float(np.mean(self.activation_stds)) if self.activation_stds else 0.0
        stats["dead_activations_ratio"] = float(self.dead_activations_count / self.total_activations) if self.total_activations > 0 else 0.0
        
        # Reset activation trackers for the next epoch to avoid accumulating across the entire training run
        self.activation_means = []
        self.activation_stds = []
        self.dead_activations_count = 0
        self.total_activations = 0
        
        return stats
