import torch
import sys
import gc

from models.hybrid_mednext import HybridMedNeXt_PlusPlus

def test_forward():
    print("==================================================")
    print("Phase 4: Standalone Forward/Backward Verification")
    print("==================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Mem Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
    try:
        print("\n1. Building Model...")
        model = HybridMedNeXt_PlusPlus(
            in_channels=3,
            num_classes=1,
            spatial_dims=3
        ).to(device)
        model.train()
        print("Model built and moved to device successfully.")
        
        print("\n2. Creating Synthetic Tensor...")
        # 1 Batch, 3 Channels, 64 Depth, 256 Height, 256 Width
        x = torch.randn(1, 3, 64, 256, 256, device=device)
        print(f"Synthetic tensor created: {x.shape}, dtype: {x.dtype}")
        
        print("\n3. Running Forward Pass...")
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(x)
        print(f"Forward pass successful. Output shape: {outputs.shape}")
        
        print("\n4. Computing Mock Loss...")
        mock_target = torch.ones_like(outputs)
        loss = torch.nn.functional.mse_loss(outputs, mock_target)
        print(f"Loss computed: {loss.item():.4f}")
        
        print("\n5. Running Backward Pass...")
        loss.backward()
        print("Backward pass successful.")
        
        print("\n==================================================")
        print("Verification PASSED: Framework independent run successful.")
        print("==================================================")
        
    except Exception as e:
        print("\n==================================================")
        print("Verification FAILED: Model/Runtime issue detected.")
        print("==================================================")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_forward()
