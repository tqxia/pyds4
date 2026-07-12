"""Hyper-connection (mHC) routing parameters.

DeepSeek V4 Flash splits the residual stream into `n_hc=4` parallel streams.
Each layer mixes them with a Sinkhorn-normalized mixing matrix computed
from the input. The mixing factor isn't a free parameter — it's the output
of a small projection (`hc_*_fn`) plus a bias (`hc_*_base`) plus three
scalar gates (`hc_*_scale`).

From ds4.c (lines 2379-2429):

    hc_dim     = n_embd * n_hc                            = 16384
    hc_mix_dim = 2*n_hc + n_hc*n_hc                       = 24
    hc_*_fn    shape: (hc_dim,     hc_mix_dim)            (16384, 24)  F16
    hc_*_base  shape: (hc_mix_dim,)                       (24,)        F32
    hc_*_scale shape: (3,)                                (3,)         F32

Forward semantics match ds4.c's `hc_pre_from_state_one_scratch` (line 4377),
`hc_split_sinkhorn_one` (line 4279), `hc_weighted_sum_one` (line 4425),
`hc_post_one` (line 4459), and `output_hc_head_one` (line 8029).

All HC control computation runs in fp32 to match ds4.c's precision.
Activations flow at the caller's dtype (bf16).
"""

from __future__ import annotations

import torch
from torch import nn

from pyds4.layers.rms import rms_norm_no_weight


def _hc_dim(n_embd: int, n_hc: int) -> int:
    return n_embd * n_hc


def _hc_mix_dim(n_hc: int) -> int:
    return 2 * n_hc + n_hc * n_hc


class HyperConnections(nn.Module):
    """Per-layer HC bundle (attention or FFN). 3 parameters + forward_pre/post."""

    def __init__(
        self,
        n_embd: int,
        n_hc: int,
        *,
        device: torch.device | str | None = "meta",
    ) -> None:
        super().__init__()
        hc_dim = _hc_dim(n_embd, n_hc)
        mix_dim = _hc_mix_dim(n_hc)
        self.n_hc = n_hc
        self.fn = nn.Parameter(
            torch.empty(hc_dim, mix_dim, device=device, dtype=torch.float16),
            requires_grad=False,
        )
        self.base = nn.Parameter(
            torch.empty(mix_dim, device=device, dtype=torch.float32),
            requires_grad=False,
        )
        self.scale = nn.Parameter(
            torch.empty(3, device=device, dtype=torch.float32),
            requires_grad=False,
        )

    def forward_pre(
        self,
        hc_residual: torch.Tensor,
        rms_eps: float,
        hc_eps: float,
        sinkhorn_iters: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split HC streams into a single sub-layer input via Sinkhorn mixing.

        hc_residual: (seq, n_hc, n_embd)
        Returns:
            x:    (seq, n_embd)        — weighted-sum input to sub-layer
            post: (seq, n_hc)          — post-gate values for forward_post
            comb: (seq, n_hc, n_hc)    — Sinkhorn mixing matrix (fp32)
        """
        seq, n_hc, n_embd = hc_residual.shape
        eps = hc_eps
        dtype = hc_residual.dtype

        # 1. RMSNorm without weight — ds4.c line 4394
        flat = hc_residual.reshape(seq, n_hc * n_embd).float()
        flat = rms_norm_no_weight(flat, rms_eps)

        # 2. matvec fn — ds4.c line 4395 (all fp32)
        mix = torch.matmul(flat, self.fn.float())   # (seq, mix_dim)

        # 3. Pre-gates — ds4.c lines 4291-4294
        s = self.scale.float()
        b = self.base.float()
        pre = torch.sigmoid(mix[:, :n_hc] * s[0] + b[:n_hc]) + eps

        # 4. Post-gates — ds4.c lines 4296-4300
        post = 2.0 * torch.sigmoid(mix[:, n_hc:2*n_hc] * s[1] + b[n_hc:2*n_hc])

        # 5. Sinkhorn on combine matrix — ds4.c lines 4302-4355
        comb_raw = mix[:, 2*n_hc:].reshape(seq, n_hc, n_hc)
        comb_z = comb_raw * s[2] + b[2*n_hc:].reshape(1, n_hc, n_hc)

        # Row-softmax init: exp(x - row_max) / row_sum + eps
        comb_max = comb_z.amax(dim=-1, keepdim=True)
        comb = torch.exp(comb_z - comb_max)
        comb = comb / comb.sum(dim=-1, keepdim=True) + eps

        # Column-normalize — ds4 lines 4329-4335
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

        # Sinkhorn iterate: col-norm is last operation → cols = 1.0, rows ≈ 1.0
        # ds4 lines 4337-4353
        for _ in range(1, sinkhorn_iters):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)   # row
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)   # column

        # 6. Weighted sum — ds4.c line 4404
        x = (pre.unsqueeze(-1) * hc_residual.float()).sum(dim=1)

        return x.to(dtype), post.to(dtype), comb  # comb stays fp32

    def forward_post(
        self,
        block_out: torch.Tensor,
        hc_residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Recombine sub-layer output into HC residual. ds4.c line 4459-4479.

        block_out:   (seq, n_embd)
        hc_residual: (seq, n_hc, n_embd)
        post:        (seq, n_hc)
        comb:        (seq, n_hc, n_hc) — fp32 from forward_pre
        Returns:     (seq, n_hc, n_embd)
        """
        # ds4 addresses the flattened matrix as comb[dst + src * n_hc],
        # the transpose of the row-major Sinkhorn view above.
        mixed = torch.bmm(comb.float().transpose(-2, -1), hc_residual.float())
        contrib = block_out.float().unsqueeze(1) * post.float().unsqueeze(-1)
        return (contrib + mixed).to(block_out.dtype)


class OutputHC(nn.Module):
    """Output-head HC: collapse `n_hc` streams to 1. 3 parameters + forward."""

    def __init__(
        self,
        n_embd: int,
        n_hc: int,
        *,
        device: torch.device | str | None = "meta",
    ) -> None:
        super().__init__()
        hc_dim = _hc_dim(n_embd, n_hc)
        self.n_hc = n_hc
        self.fn = nn.Parameter(
            torch.empty(hc_dim, n_hc, device=device, dtype=torch.float16),
            requires_grad=False,
        )
        self.base = nn.Parameter(
            torch.empty(n_hc, device=device, dtype=torch.float32),
            requires_grad=False,
        )
        self.scale = nn.Parameter(
            torch.empty(1, device=device, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(
        self,
        hc_state: torch.Tensor,
        rms_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        """Collapse n_hc streams to a single vector. ds4.c lines 8029-8054.

        hc_state: (seq, n_hc, n_embd)
        Returns:  (seq, n_embd)
        """
        seq, n_hc, n_embd = hc_state.shape
        dtype = hc_state.dtype

        # RMSNorm (no weight) + matvec fn in fp32 — ds4 lines 8040-8041
        flat = hc_state.reshape(seq, n_hc * n_embd).float()
        flat = rms_norm_no_weight(flat, rms_eps)
        pre = torch.matmul(flat, self.fn.float())  # (seq, n_hc)

        # Sigmoid gate — ds4 lines 8045-8047
        gate = torch.sigmoid(pre * self.scale.float() + self.base.float()) + hc_eps

        # Weighted sum — ds4 line 8049
        x = (gate.unsqueeze(-1) * hc_state.float()).sum(dim=1)

        return x.to(dtype)
