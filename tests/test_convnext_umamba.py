# tests/test_convnext_umamba.py
import torch
import torch.nn as nn
from models.hybrid_convnext_umamba import HybridConvNeXtV2_UMamba

def main():
    print("=" * 80)
    print("ConvNeXt V2 + UMamba Architecture Verification")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Instantiate the Model
    model = HybridConvNeXtV2_UMamba(
        in_channels=1,
        num_classes=1,
        encoder_depths=[2, 2, 4, 2, 2],
        encoder_dims=[24, 48, 96, 192, 384],
        drop_path_rate=0.1,
        fusion_dim=384,
        fusion_strategy="weighted",
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_blocks=2,
        decoder_dims=[192, 96, 48, 24]
    ).to(device)

    # 2. Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters: {total_params:,} (Trainable: {trainable_params:,})")

    # 3. Create dummy data
    batch_size = 1
    # Small patch size for fast CPU testing
    dummy_input = torch.randn(batch_size, 1, 16, 16, 16).to(device)
    dummy_target = torch.randint(0, 2, (batch_size, 1, 16, 16, 16), dtype=torch.float32).to(device)
    
    print(f"\nInput shape: {dummy_input.shape}")

    # 4. Forward Pass Verification
    model.train()
    outputs = model(dummy_input)
    
    print("\nForward Pass Successful!")
    print("Verifying Deep Supervision Outputs:")
    
    assert isinstance(outputs, dict), "Output must be a dictionary"
    assert "quarter" in outputs, "Missing 'quarter' output"
    assert "half" in outputs, "Missing 'half' output"
    assert "full" in outputs, "Missing 'full' output"
    
    print(f" - Quarter Output Shape: {outputs['quarter'].shape}")
    print(f" - Half Output Shape:    {outputs['half'].shape}")
    print(f" - Full Output Shape:    {outputs['full'].shape}")
    
    # Verify spatial dimensions match (1/4, 1/2, 1/1)
    D, H, W = dummy_input.shape[2:]
    assert outputs['full'].shape[2:] == (D, H, W)
    assert outputs['half'].shape[2:] == (D // 2, H // 2, W // 2)
    assert outputs['quarter'].shape[2:] == (D // 4, H // 4, W // 4)

    # 5. Backward Pass and Optimizer Verification
    print("\nVerifying Backward Pass and Optimizer Step...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    
    # We simulate a simple loss by interpolating predictions to target size
    from torch.nn.functional import interpolate
    quarter_up = interpolate(outputs['quarter'], size=(D, H, W), mode='trilinear')
    half_up = interpolate(outputs['half'], size=(D, H, W), mode='trilinear')
    
    criterion = nn.BCEWithLogitsLoss()
    loss = (
        1.0 * criterion(outputs['full'], dummy_target) +
        0.5 * criterion(half_up, dummy_target) +
        0.25 * criterion(quarter_up, dummy_target)
    )
    
    print(f"Computed simulated loss: {loss.item():.4f}")
    loss.backward()
    
    # Check gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            break
            
    assert has_grad, "No gradients were computed!"
    print("Backward Pass Successful! Gradients computed properly.")
    
    optimizer.step()
    print("Optimizer Step Successful!")

    # 6. Validation Step
    print("\nVerifying Validation Step (Inference Mode)...")
    model.eval()
    with torch.no_grad():
        val_outputs = model(dummy_input)
        
    print(f"Validation Output Shape (Full): {val_outputs['full'].shape}")
    print("\nAll architecture verifications passed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
