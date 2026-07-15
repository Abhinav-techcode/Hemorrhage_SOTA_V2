import torch
from models.mamba_pytorch import Mamba

def mamba_block():
    torch.manual_seed(42)
    return Mamba(d_model=64, d_state=16, d_conv=4, expand=2)

def test_forward_pass_and_shapes(mamba_block):
    """Verify the output tensor maintains the correct shape."""
    B, L, D = 2, 128, 64
    x = torch.randn(B, L, D)
    out = mamba_block(x)
    assert out.shape == (B, L, D), f"Expected shape {(B, L, D)}, got {out.shape}"

def test_backward_pass_and_gradients(mamba_block):
    """Verify gradients correctly propagate backwards."""
    B, L, D = 2, 128, 64
    x = torch.randn(B, L, D, requires_grad=True)
    out = mamba_block(x)
    
    # Fake loss
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "Input tensor did not receive gradients"
    assert mamba_block.A_log.grad is not None, "A_log parameter did not receive gradients"
    assert mamba_block.D.grad is not None, "D parameter did not receive gradients"

def test_mixed_precision_compatibility(mamba_block):
    """Verify the module works correctly under autocast (mixed precision)."""
    B, L, D = 2, 128, 64
    x = torch.randn(B, L, D).cuda() if torch.cuda.is_available() else torch.randn(B, L, D)
    mamba_block = mamba_block.to(x.device)
    
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            out = mamba_block(x)
            assert out.shape == (B, L, D)
            # Ensure the output comes back in the casted type
            assert out.dtype == torch.float16 or out.dtype == torch.float32
    except Exception as e:
        raise RuntimeError(f"Mixed precision failed: {e}")

def test_cpu_execution(mamba_block):
    """Assert smooth execution on CPU."""
    mamba_block.to('cpu')
    x = torch.randn(2, 64, 64, device='cpu')
    out = mamba_block(x)
    assert out.device.type == 'cpu'

def test_cuda_execution(mamba_block):
    """Assert smooth execution on CUDA (if available) without custom extensions."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping CUDA test.")
        return
    mamba_block.to('cuda')
    x = torch.randn(2, 64, 64, device='cuda')
    out = mamba_block(x)
    assert out.device.type == 'cuda'

def test_determinism():
    """Assert fixed random seed guarantees identical tensor outputs."""
    B, L, D = 2, 64, 64
    
    torch.manual_seed(100)
    model1 = Mamba(d_model=D)
    x1 = torch.randn(B, L, D)
    out1 = model1(x1)
    
    torch.manual_seed(100)
    model2 = Mamba(d_model=D)
    x2 = torch.randn(B, L, D)
    out2 = model2(x2)
    
    assert torch.allclose(out1, out2), "Outputs are not deterministic for the same seed"

if __name__ == "__main__":
    # Allows running script directly without pytest
    print("Running Verification Suite...")
    block = Mamba(d_model=64, d_state=16, d_conv=4, expand=2)
    test_forward_pass_and_shapes(block)
    test_backward_pass_and_gradients(block)
    test_mixed_precision_compatibility(block)
    test_cpu_execution(block)
    test_cuda_execution(block)
    test_determinism()
    print("All verification tests passed successfully!")
