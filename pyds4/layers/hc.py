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

Each transformer block has TWO HC bundles — one for the attention input
(`hc_attn_*`) and one for the FFN input (`hc_ffn_*`). The final output head
has ONE simpler bundle that collapses `n_hc` streams down to a single
vector for the LM head:

    output_hc_fn    shape: (hc_dim, n_hc)                  (16384, 4)   F16
    output_hc_base  shape: (n_hc,)                         (4,)         F32
    output_hc_scale shape: (1,)                            (1,)         F32
"""

from __future__ import annotations

import torch
from torch import nn


def _hc_dim(n_embd: int, n_hc: int) -> int:
    return n_embd * n_hc


def _hc_mix_dim(n_hc: int) -> int:
    # ds4.c: 2*n_hc + n_hc*n_hc — the offsets+matrix for the Sinkhorn mixer.
    return 2 * n_hc + n_hc * n_hc


class HyperConnections(nn.Module):
    """Per-layer HC bundle (attention or FFN). 3 parameters."""

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


class OutputHC(nn.Module):
    """Output-head HC: collapse `n_hc` streams to 1. 3 parameters."""

    def __init__(
        self,
        n_embd: int,
        n_hc: int,
        *,
        device: torch.device | str | None = "meta",
    ) -> None:
        super().__init__()
        hc_dim = _hc_dim(n_embd, n_hc)
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
