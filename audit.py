import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
import json
import traceback

# Setup env
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# MIG Workarounds
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"

from training.train import load_all_configs, build_framework, configure_cuda, seed_everything
from evaluation.losses import LossFactory

def write_md(filename, content):
    with open(filename, "w") as f:
        f.write(content)

def hook_fn(module, input, output, name, registry):
    if isinstance(output, tuple):
        registry.append(f"**{name}**\n- Input shape: {[i.shape for i in input if isinstance(i, torch.Tensor)]}\n- Output shape: {output[0].shape}, Metadata: {output[1:]}\n")
    else:
        registry.append(f"**{name}**\n- Input shape: {[i.shape for i in input if isinstance(i, torch.Tensor)]}\n- Output shape: {output.shape}\n")

def run_audit():
    try:
        configure_cuda()
        seed_everything(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Loading configs...", flush=True)
        config_dir = PROJECT_ROOT / "configs"
        configs = load_all_configs(config_dir)
        
        configs["training"]["batch_size"] = 2
        configs["training"]["compile_model"] = False
        
        print("Building framework...", flush=True)
        model, train_loader, val_loader, optimizer, scheduler, loss_fn, metric_manager, trainer_cfg = build_framework(configs)
        model = model.to(device)
        model.train()
        
        print("Attaching hooks...", flush=True)
        arch_registry = []
        
        encoder = model.encoder
        for i in range(1, 5):
            getattr(encoder, f"patch_embed{i}").register_forward_hook(lambda m, i, o, name=f"Stage {i} Patch Embed": hook_fn(m, i, o, name, arch_registry))
            blocks = getattr(encoder, f"block{i}")
            for j, blk in enumerate(blocks):
                blk.attn.register_forward_hook(lambda m, i, o, name=f"Stage {i} Block {j} Attention": hook_fn(m, i, o, name, arch_registry))
                blk.mlp.register_forward_hook(lambda m, i, o, name=f"Stage {i} Block {j} MixFFN": hook_fn(m, i, o, name, arch_registry))

        model.up3.register_forward_hook(lambda m, i, o, name="Decoder Up3": hook_fn(m, i, o, name, arch_registry))
        model.boundary_head.register_forward_hook(lambda m, i, o, name="Boundary Refinement Head": hook_fn(m, i, o, name, arch_registry))

        print("Fetching batch...", flush=True)
        batch = next(iter(train_loader))
        images = batch["image"][0:1, :, 16:48, 64:192, 64:192].to(device)
        masks = batch["mask"][0:1, :, 16:48, 64:192, 64:192].to(device)
        
        print(f"Running forward pass on device {device}...", flush=True)
        outputs = model(images)
        
        if isinstance(outputs, tuple):
            out_full, out_half, out_quarter = outputs
        else:
            out_full = outputs
            
        print("Computing loss...", flush=True)
        loss_dict = loss_fn(outputs, masks)
        loss = loss_dict["total"]
        
        print("Running backward pass...", flush=True)
        loss.backward()
        
        arch_md = "# Architecture Verification Audit\n\n## 3D SegFormer Encoder Stages\n\n"
        arch_md += "\n".join(arch_registry)
        arch_md += "\n## Output Verification\n"
        arch_md += f"- Final Output Shape: {out_full.shape}\n"
        arch_md += f"- Expected Shape: {masks.shape}\n"
        write_md("/Users/abhinavgupta/.gemini/antigravity-ide/brain/73af64ef-7fc0-4911-944a-6d2366f5ddc5/Architecture_Audit.md", arch_md)
        
        loss_md = "# Loss Verification Audit\n\n## Loss Components\n"
        loss_md += f"- Total Loss: {loss.item():.4f}\n"
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                loss_md += f"- {k}: {v.item():.4f}\n"
            else:
                loss_md += f"- {k}: {v:.4f}\n"
        write_md("/Users/abhinavgupta/.gemini/antigravity-ide/brain/73af64ef-7fc0-4911-944a-6d2366f5ddc5/Loss_Audit.md", loss_md)
        
        grad_md = "# Gradient Flow Audit\n\n"
        zero_grad_layers = []
        
        orig_weights = {name: p.clone().detach() for name, p in model.named_parameters()}
        
        for name, p in model.named_parameters():
            if p.grad is not None:
                mean_grad = p.grad.abs().mean().item()
                max_grad = p.grad.abs().max().item()
                norm_grad = p.grad.norm().item()
                if mean_grad == 0:
                    zero_grad_layers.append(name)
                grad_md += f"### {name}\n- Mean: {mean_grad:.6e}\n- Max: {max_grad:.6e}\n- Norm: {norm_grad:.6e}\n\n"
            else:
                zero_grad_layers.append(f"{name} (None)")
                grad_md += f"### {name}\n- GRADIENT IS NONE\n\n"
                
        grad_md += "## Zero Gradient Layers\n"
        for l in zero_grad_layers:
            grad_md += f"- {l}\n"
        write_md("/Users/abhinavgupta/.gemini/antigravity-ide/brain/73af64ef-7fc0-4911-944a-6d2366f5ddc5/Gradient_Audit.md", grad_md)
        
        optimizer.step()
        
        pred_md = "# Prediction Audit\n\n"
        probs = torch.sigmoid(out_full)
        preds = (probs > 0.5).float()
        
        pred_md += f"- Min Prob: {probs.min().item():.4f}\n"
        pred_md += f"- Max Prob: {probs.max().item():.4f}\n"
        pred_md += f"- Mean Prob: {probs.mean().item():.4f}\n"
        pred_md += f"- Median Prob: {probs.median().item():.4f}\n"
        pred_md += f"- Foreground Pred %: {(preds.sum() / preds.numel()).item() * 100:.4f}%\n"
        pred_md += f"- Ground Truth %: {(masks.sum() / masks.numel()).item() * 100:.4f}%\n"
        write_md("/Users/abhinavgupta/.gemini/antigravity-ide/brain/73af64ef-7fc0-4911-944a-6d2366f5ddc5/Prediction_Audit.md", pred_md)
        
        metric_md = "# Metric Verification Audit\n\n"
        metrics = metric_manager(preds, masks)
        for k, v in metrics.items():
            metric_md += f"- {k}: {v:.4f}\n"
        write_md("/Users/abhinavgupta/.gemini/antigravity-ide/brain/73af64ef-7fc0-4911-944a-6d2366f5ddc5/Metric_Audit.md", metric_md)
        
        pipe_md = "# Training Pipeline Audit\n\n"
        pipe_md += f"1. Input Shape: {images.shape}\n"
        pipe_md += f"2. Mask Shape: {masks.shape}\n"
        pipe_md += f"3. Forward Output: {out_full.shape}\n"
        pipe_md += f"4. Loss Computed: {loss.item():.4f}\n"
        pipe_md += f"5. Zero Grad Layers: {len(zero_grad_layers)}\n"
        write_md("/Users/abhinavgupta/.gemini/antigravity-ide/brain/73af64ef-7fc0-4911-944a-6d2366f5ddc5/Training_Pipeline_Audit.md", pipe_md)
        
        print("Audit Complete. Markdown reports generated.", flush=True)

    except Exception as e:
        print("EXCEPTION CAUGHT IN AUDIT.PY:", flush=True)
        traceback.print_exc()

if __name__ == "__main__":
    run_audit()
