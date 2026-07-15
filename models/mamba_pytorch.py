import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Mamba(nn.Module):
    """
    Pure PyTorch implementation of the Mamba bottleneck.
    Mathematically equivalent to mamba_ssm but lacks custom CUDA kernels.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        Returns: (B, L, D)
        """
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        x_proj = x_proj.transpose(1, 2)
        x_proj = self.conv1d(x_proj)[:, :, :L]
        x_proj = x_proj.transpose(1, 2)
        x_proj = F.silu(x_proj)

        x_dbl = self.x_proj(x_proj)
        delta, B_t, C_t = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)

        delta = F.softplus(self.dt_proj(delta))
        
        # Ensure everything is in float32 for numerical stability in the sequential scan
        delta = delta.float()
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        
        deltaA = torch.exp(torch.einsum('b l d, d n -> b l d n', delta, A))
        deltaB_u = torch.einsum('b l d, b l n, b l d -> b l d n', delta, B_t.float(), x_proj.float())
        
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=torch.float32)
        y = []
        for i in range(L):
            h = deltaA[:, i] * h + deltaB_u[:, i]
            y.append(torch.einsum('b d n, b n -> b d', h, C_t[:, i].float()))
            
        y = torch.stack(y, dim=1).to(dtype=x.dtype)
        y = y + x_proj * D
        y = y * F.silu(z)
        out = self.out_proj(y)
        return out
