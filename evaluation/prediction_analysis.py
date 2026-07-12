import torch
import numpy as np
from typing import Dict, Any

class PredictionAnalyzer:
    """
    Prediction Analysis Module (Phase 5)
    Computes deep statistics on raw predictions and binary segmentations.
    """
    @staticmethod
    def analyze(probs: torch.Tensor, preds_bin: torch.Tensor, masks: torch.Tensor) -> Dict[str, Any]:
        stats = {}
        
        # 1. Prediction Statistics (on probabilities)
        p_np = probs.detach().cpu().numpy()
        stats["prob_mean"] = float(np.mean(p_np))
        stats["prob_std"] = float(np.std(p_np))
        stats["prob_min"] = float(np.min(p_np))
        stats["prob_max"] = float(np.max(p_np))
        stats["prob_median"] = float(np.median(p_np))
        stats["prob_95th"] = float(np.percentile(p_np, 95))
        
        # 2. Binary Statistics
        p_b = preds_bin.detach().bool()
        m_b = masks.detach().bool()
        
        tp = (p_b & m_b).sum().item()
        fp = (p_b & ~m_b).sum().item()
        fn = (~p_b & m_b).sum().item()
        tn = (~p_b & ~m_b).sum().item()
        
        stats["tp"] = tp
        stats["fp"] = fp
        stats["fn"] = fn
        stats["tn"] = tn
        
        # 3. Segmentation Statistics (Batch level averages)
        B = preds_bin.shape[0]
        fg_percentages = []
        lesion_counts = []
        avg_lesion_sizes = []
        
        for b in range(B):
            pred_vol = p_b[b].squeeze() # [D, H, W]
            
            fg_voxels = pred_vol.sum().item()
            total_voxels = pred_vol.numel()
            fg_percentages.append(fg_voxels / total_voxels if total_voxels > 0 else 0)
            
            # Simple connected components using PyTorch (if available) or NumPy/SciPy
            # Since MONAI / PyTorch doesn't have a built-in 3D CC without scipy, 
            # we approximate lesion count by non-zero elements or just use SciPy if we want exact counts.
            # For performance and to keep it PyTorch native, we'll skip rigorous 3D CC here
            # and just report the total foreground as 1 'lesion' if not using SciPy.
            
            try:
                from scipy.ndimage import label
                labeled_array, num_features = label(pred_vol.cpu().numpy())
                lesion_counts.append(num_features)
                if num_features > 0:
                    avg_lesion_sizes.append(fg_voxels / num_features)
                else:
                    avg_lesion_sizes.append(0)
            except ImportError:
                # Fallback
                lesion_counts.append(1 if fg_voxels > 0 else 0)
                avg_lesion_sizes.append(fg_voxels)
                
        stats["pred_foreground_percentage"] = float(np.mean(fg_percentages) * 100)
        stats["pred_lesion_count"] = float(np.mean(lesion_counts))
        stats["pred_avg_lesion_size"] = float(np.mean(avg_lesion_sizes))
        
        return stats
