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
        largest_lesions = []
        smallest_lesions = []
        
        for b in range(B):
            pred_vol = p_b[b].squeeze() # [D, H, W]
            
            fg_voxels = pred_vol.sum().item()
            total_voxels = pred_vol.numel()
            fg_percentages.append(fg_voxels / total_voxels if total_voxels > 0 else 0)
            
            try:
                from scipy.ndimage import label
                labeled_array, num_features = label(pred_vol.cpu().numpy())
                lesion_counts.append(num_features)
                if num_features > 0:
                    component_sizes = [np.sum(labeled_array == i) for i in range(1, num_features + 1)]
                    avg_lesion_sizes.append(np.mean(component_sizes))
                    largest_lesions.append(np.max(component_sizes))
                    smallest_lesions.append(np.min(component_sizes))
                else:
                    avg_lesion_sizes.append(0)
                    largest_lesions.append(0)
                    smallest_lesions.append(0)
            except ImportError:
                # Fallback
                lesion_counts.append(1 if fg_voxels > 0 else 0)
                avg_lesion_sizes.append(fg_voxels)
                largest_lesions.append(fg_voxels)
                smallest_lesions.append(fg_voxels)
                
        stats["pred_foreground_percentage"] = float(np.mean(fg_percentages) * 100)
        stats["pred_lesion_count"] = float(np.mean(lesion_counts))
        stats["pred_avg_lesion_size"] = float(np.mean(avg_lesion_sizes))
        stats["pred_largest_lesion"] = float(np.mean(largest_lesions))
        stats["pred_smallest_lesion"] = float(np.mean(smallest_lesions))
        
        return stats
