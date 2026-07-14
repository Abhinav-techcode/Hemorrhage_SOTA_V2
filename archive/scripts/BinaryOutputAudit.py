import torch
import logging
import sys
from rich.console import Console
from rich.table import Table
from evaluation.losses import LossFactory
from evaluation.metric_engine import ResearchMetricEngine

def run_audit():
    console = Console()
    console.print("[bold cyan]Binary Output & Metric Flow Audit: Phase 4.2[/]")
    
    B, C, D, H, W = 2, 1, 64, 128, 128
    console.print(f"Generating dummy batch of size [B={B}, C={C}, D={D}, H={H}, W={W}]")
    
    y_logits = torch.randn(B, C, D, H, W)
    y_logits_ds = [torch.randn(B, C, D//2, H//2, W//2), y_logits]
    y_true = torch.zeros(B, C, D, H, W)
    y_true[0, 0, 10:20, 10:20, 10:20] = 1.0
    
    # Simulate a bug where mask has invalid values
    # y_true[0, 0, 0, 0, 0] = 0.5 
    
    unique_vals = torch.unique(y_true)
    if not torch.all((unique_vals == 0) | (unique_vals == 1)):
        console.print("[bold red]FATAL ERROR: Ground truth mask contains values other than 0 and 1![/]")
        sys.exit(1)
        
    if torch.isnan(y_logits).any():
        console.print("[bold red]FATAL ERROR: Predictions contain NaN![/]")
        sys.exit(1)
        
    loss_cfg = {
        "weighting_strategy": "static",
        "losses": [
            {"name": "dice_focal", "params": {"sigmoid": True}, "weight": 1.0}
        ]
    }
    loss_engine = LossFactory.build(loss_cfg)
    
    table = Table(title="Loss Forward Pass")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")
    
    try:
        loss_dict = loss_engine(y_logits_ds, y_true)
        table.add_row("Output Type", str(type(loss_dict)))
        table.add_row("Total Loss", f"{loss_dict['total'].item():.4f}")
        for k, v in loss_dict.items():
            if k != "total":
                table.add_row(f"Component: {k}", f"{v.item():.4f}")
    except Exception as e:
        table.add_row("Error", str(e))
    console.print(table)
    
    metric_engine = ResearchMetricEngine()
    metric_table = Table(title="Metric Forward Pass")
    metric_table.add_column("Property", style="cyan")
    metric_table.add_column("Value", style="magenta")
    
    try:
        metric_engine.update(y_logits_ds, y_true, mode="val")
        metrics = metric_engine.compute(mode="val")
        metric_table.add_row("Update Success", "True")
        for k, v in metrics.items():
            if k in ["val_dice", "val_hd95", "val_case_tp"]:
                metric_table.add_row(k, str(v))
    except Exception as e:
        metric_table.add_row("Error", str(e))
        
    console.print(metric_table)

if __name__ == "__main__":
    run_audit()
