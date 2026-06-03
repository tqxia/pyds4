"""RMSNorm — one learned gain vector, scalar epsilon.

`x_i / sqrt(mean(x^2) + eps) * weight_i`. The `weight` tensor is the only
parameter; `eps` is a config constant. ds4 stores all norm gains as F32 in
the GGUF (see `tensor_expect_layout(..., DS4_TENSOR_F32, 1, dim, 0, 0)` in
ds4.c around line 2398), so we default to fp32.
"""

from __future__ import annotations

import torch
from torch import nn


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
