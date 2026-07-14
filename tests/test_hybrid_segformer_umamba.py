import os
import tempfile
import torch
from pathlib import Path
from models.hybrid_segformer_umamba import HybridSegFormerUMamba
from evaluation.visualize import Visualizer

def test_comprehensive_smoke():
    print("Initializing HybridSegFormerUMamba Comprehensive Smoke Test...")
    model = HybridSegFormerUMamba(
        in_channels=1, 
        num_classes=1, 
        embed_dims=[32, 64, 160, 256],
        fusion_dim=256,
        decoder_dims=[128, 64, 32],
        d_state=16
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    x = torch.randn(1, 1, 32, 64, 64, device=device)
    target = torch.randint(0, 2, (1, 1, 32, 64, 64), device=device).float()
    
    print(f"1. Tensor Shapes Verified: Input {x.shape}")
    
    model.train()
    
    # 2. AMP & Forward Pass
    print("2. Testing Forward Pass & AMP...")
    amp_ctx = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else torch.autocast("cpu", dtype=torch.bfloat16)
    with amp_ctx:
        out_full, out_half, out_quarter = model(x)
        loss = out_full.sum() + out_half.sum() + out_quarter.sum()
        
    assert out_full.shape == (1, 1, 32, 64, 64)
    assert out_half.shape == (1, 1, 32, 64, 64)
    assert out_quarter.shape == (1, 1, 32, 64, 64)
    print("-> Forward pass and Deep Supervision shapes passed.")
    
    # 3. Backward Pass & Gradient Flow
    print("3. Testing Backward Pass & Gradient Flow...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "Gradient flow failed!"
    optimizer.step()
    print("-> Backward pass passed.")
    
    # 4. Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"4. Total Parameters: {total_params / 1e6:.2f} M")
    
    # 5. Checkpoint Save/Load
    print("5. Testing Checkpointing...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_ckpt.pth"
        torch.save(model.state_dict(), ckpt_path)
        assert ckpt_path.exists()
        
        new_model = HybridSegFormerUMamba(
            in_channels=1, num_classes=1, 
            embed_dims=[32, 64, 160, 256], fusion_dim=256, decoder_dims=[128, 64, 32], d_state=16
        )
        new_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        print("-> Checkpointing passed.")
        
        # 6. Visualization Generation
        print("6. Testing Visualization Pipeline...")
        Visualizer.generate_qualitative_report(
            save_dir=Path(tmp_dir),
            patient_id="smoke_test",
            image=x,
            pred=out_full.sigmoid(),
            target=target,
            metrics={"val_dice": 0.95},
            epoch=1
        )
        assert (Path(tmp_dir) / "qualitative" / "smoke_test_epoch_1.png").exists()
        print("-> Visualization passed.")
        
    print("========================================")
    print("ALL SMOKE TESTS PASSED!")
    print("========================================")

if __name__ == "__main__":
    test_comprehensive_smoke()
