# Architecture

## Original U-Mamba
The architecture in this repository is the **ORIGINAL U-Mamba architecture** exactly as described in the official paper. 
The tensor flow, encoder-decoder design, skip connections, residual structure, and integration of the Mamba blocks all follow the original paper precisely.

## Pure PyTorch Mamba Backend
To ensure cross-platform compatibility, portability, and ease of installation, the backend implementation of the Mamba Selective State Space Model block has been replaced with a **pure PyTorch implementation**.

- No architectural changes have been introduced.
- No `mamba-ssm` or `causal-conv1d` dependencies are required.
- No custom CUDA kernels or Triton kernels are used.
- The mathematical behavior of the original Mamba block is reproduced perfectly in PyTorch using associative scanning and numerically stable parameter discretizations.

This guarantees full reproducibility across CPU and GPU environments without relying on fragile build processes for custom C++/CUDA extensions.
