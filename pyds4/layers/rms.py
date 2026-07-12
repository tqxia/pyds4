"""RMSNorm — one learned gain vector, scalar epsilon.

`x_i / sqrt(mean(x^2) + eps) * weight_i`. The `weight` tensor is the only
parameter; `eps` is a config constant. ds4 stores all norm gains as F32 in
the GGUF (see `tensor_expect_layout(..., DS4_TENSOR_F32, 1, dim, 0, 0)` in
ds4.c around line 2398), so we default to fp32.
"""

from __future__ import annotations

import torch
from torch import nn


def rms_norm_weight(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Weighted RMSNorm. x and weight must be same-length 1-D tensors or last-dim align.

    x: (*, dim), weight: (dim,)
    Returns: (*, dim)
    """
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normed = x.float() * torch.rsqrt(variance + eps)
    return (normed * weight).to(x.dtype)


def rms_norm_no_weight(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Weight-less RMSNorm. Used for HC flat normalization and per-head Q norm.

    x: (*, any)
    Returns: (*, same shape) — normalized in float then cast back to input dtype.
    """
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps)).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float,
        *,
        device: torch.device | str | None = "meta",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(
            torch.empty(dim, device=device, dtype=dtype),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (*, dim), weight: (dim,). Returns (*, dim)."""
        return rms_norm_weight(x, self.weight, self.eps)

    @staticmethod
    def norm(x: torch.Tensor, eps: float) -> torch.Tensor:
        """Convenience alias for the weight-less version."""
        return rms_norm_no_weight(x, eps)
