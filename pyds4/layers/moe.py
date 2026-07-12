"""MoE FFN parameters: routed experts + shared expert + router.

Each transformer block has a Mixture-of-Experts feed-forward:

  Routed (n_expert = 256, top-K = 6 used per token):
    ffn_gate_exps: (n_embd, ff_exp, n_expert)        IQ2_XXS
    ffn_up_exps:   (n_embd, ff_exp, n_expert)        IQ2_XXS
    ffn_down_exps: (ff_exp, n_embd, n_expert)        Q2_K

  Shared (always-on):
    ffn_gate_shexp: (n_embd, ff_exp)                 Q8_0
    ffn_up_shexp:   (n_embd, ff_exp)                 Q8_0
    ffn_down_shexp: (ff_exp, n_embd)                 Q8_0

  Router:
    ffn_norm:       (n_embd,)                        F32
    ffn_gate_inp:   (n_embd, n_expert)               F16  -- always present
    exp_probs_b.bias: (n_expert,)                    F32  -- only learned layers
    ffn_gate_tid2eid: (n_expert_used, vocab_size)    I32  -- only hash layers

Router formula (ds4.c line 5273):
  probs[i] = sqrt(softplus(logits[i]))
  logits = ffn_gate_inp.T @ x

SwiGLU (ds4.c line 5115):
  silu(gate) * up  with gate, up clamped to [-clamp, +clamp]

M8c: full routed MoE + shared expert forward.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from pyds4.config import DS4Config


class MoEFFN(nn.Module):
    def __init__(
        self,
        cfg: DS4Config,
        *,
        is_hash_routed: bool,
        has_exp_bias: bool,
        device: torch.device | str | None = "meta",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.is_hash_routed = is_hash_routed
        self.has_exp_bias = has_exp_bias

        # --- norm + router input -------------------------------------------
        self.norm = nn.Parameter(
            torch.empty(cfg.n_embd, device=device, dtype=torch.float32),
            requires_grad=False,
        )
        self.gate_inp = nn.Parameter(
            torch.empty(cfg.n_embd, cfg.n_expert, device=device, dtype=torch.float16),
            requires_grad=False,
        )

        # --- routed experts (3D: stacked across expert axis) ---------------
        self.gate_exps = nn.Parameter(
            torch.empty(cfg.n_embd, cfg.ff_exp, cfg.n_expert, device=device, dtype=torch.float32),
            requires_grad=False,
        )
        self.up_exps = nn.Parameter(
            torch.empty(cfg.n_embd, cfg.ff_exp, cfg.n_expert, device=device, dtype=torch.float32),
            requires_grad=False,
        )
        self.down_exps = nn.Parameter(
            torch.empty(cfg.ff_exp, cfg.n_embd, cfg.n_expert, device=device, dtype=torch.float32),
            requires_grad=False,
        )

        # --- shared experts ------------------------------------------------
        self.gate_shexp = nn.Parameter(
            torch.empty(cfg.n_embd, cfg.ff_exp, device=device, dtype=torch.float32),
            requires_grad=False,
        )
        self.up_shexp = nn.Parameter(
            torch.empty(cfg.n_embd, cfg.ff_exp, device=device, dtype=torch.float32),
            requires_grad=False,
        )
        self.down_shexp = nn.Parameter(
            torch.empty(cfg.ff_exp, cfg.n_embd, device=device, dtype=torch.float32),
            requires_grad=False,
        )

        # --- optional routing aids ----------------------------------------
        if has_exp_bias:
            self.exp_probs_b = nn.Parameter(
                torch.empty(cfg.n_expert, device=device, dtype=torch.float32),
                requires_grad=False,
            )

        if is_hash_routed:
            self.register_buffer(
                "tid2eid",
                torch.empty(
                    cfg.n_expert_used, cfg.vocab_size, device=device, dtype=torch.int32
                ),
                persistent=True,
            )

    # ------------------------------------------------------------------
    # Forward — MoE FFN (M8c)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        expert_weights_scale: float,
        swiglu_clamp_exp: float,
        token_ids: Optional[torch.Tensor] = None,
        expert_load_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """MoE FFN: router + top-K routed experts + shared expert + SwiGLU.

        x:                    (seq, n_embd) — pre-RMS-normed input
        expert_weights_scale: from cfg (1.5)
        swiglu_clamp_exp:     from cfg.swiglu_clamp_exp[il] (10.0)
        expert_load_fn:       callable(expert_id) -> (gate_w, up_w, down_w)
                              for lazy expert dequant. If None, experts must
                              be pre-loaded as nn.Parameters.
        Returns: (seq, n_embd)
        """
        seq = x.shape[0]
        device = x.device
        dtype = x.dtype
        n_expert = self.cfg.n_expert
        n_used = self.cfg.n_expert_used

        # ---- Router ----
        # logits = x @ gate_inp: (seq, n_embd) @ (n_embd, n_expert) → (seq, n_expert)
        logits = torch.matmul(x, self.gate_inp.to(dtype))
        probs = torch.sqrt(F.softplus(logits))  # (seq, n_expert)

        # Early layers select from the fixed token-id hash table. Learned
        # router probabilities still provide their mixture weights.
        if self.is_hash_routed and token_ids is not None:
            topk_ids = self.tid2eid[:, token_ids].T.long()
        elif self.has_exp_bias:
            selection = probs + self.exp_probs_b.to(dtype).unsqueeze(0)
            _, topk_ids = torch.topk(selection, k=n_used, dim=-1)
        else:
            selection = probs
            _, topk_ids = torch.topk(selection, k=n_used, dim=-1)

        # Expert weights from UNBIASED probs (ds4.c line 5375)
        weights = torch.gather(probs, -1, topk_ids)  # (seq, 6)
        denom = weights.sum(dim=-1, keepdim=True).clamp(min=1.0 / 16384)
        weights = weights / denom * expert_weights_scale  # (seq, 6)

        # ---- Routed expert compute ----
        out = torch.zeros(seq, self.cfg.n_embd, device=device, dtype=dtype)

        route_tokens, route_slots = torch.nonzero(
            torch.ones_like(topk_ids, dtype=torch.bool), as_tuple=True
        )
        route_experts = topk_ids[route_tokens, route_slots]

        if expert_load_fn is not None:
            # Group all routes for an expert so it is dequantized only once.
            for e_id_t in torch.unique(route_experts):
                e_id = int(e_id_t.item())
                mask = route_experts == e_id
                token_idx = route_tokens[mask]
                slot_idx = route_slots[mask]
                x_e = x[token_idx]
                w_e = weights[token_idx, slot_idx]
                gate_w, up_w, down_w = expert_load_fn(e_id)
                gate = torch.matmul(x_e, gate_w)
                up = torch.matmul(x_e, up_w)
                mid = _swiglu(gate, up, swiglu_clamp_exp) * w_e.unsqueeze(-1)
                out.index_add_(0, token_idx, torch.matmul(mid, down_w))
        else:
            # Pre-loaded mode: experts are nn.Parameters
            gate_exps = self.gate_exps.to(dtype)
            up_exps = self.up_exps.to(dtype)
            down_exps = self.down_exps.to(dtype)
            for e_id_t in torch.unique(route_experts):
                e_id = int(e_id_t.item())
                mask = route_experts == e_id
                token_idx = route_tokens[mask]
                slot_idx = route_slots[mask]
                x_e = x[token_idx]
                w_e = weights[token_idx, slot_idx]
                gate = torch.matmul(x_e, gate_exps[:, :, e_id])
                up = torch.matmul(x_e, up_exps[:, :, e_id])
                mid = _swiglu(gate, up, swiglu_clamp_exp) * w_e.unsqueeze(-1)
                out.index_add_(0, token_idx, torch.matmul(mid, down_exps[:, :, e_id]))

        # ---- Shared expert (always-on) ----
        gate_s = torch.matmul(x, self.gate_shexp.to(dtype))  # (seq, ff_exp)
        up_s = torch.matmul(x, self.up_shexp.to(dtype))      # (seq, ff_exp)
        mid_s = _swiglu(gate_s, up_s, swiglu_clamp_exp)
        out_s = torch.matmul(mid_s, self.down_shexp.to(dtype))  # (seq, n_embd)

        return out + out_s


def _swiglu(gate: torch.Tensor, up: torch.Tensor, clamp_val: float) -> torch.Tensor:
    """DS4 SwiGLU: upper-clamp gate, two-sided-clamp up, then SiLU(gate)*up."""
    if clamp_val > 1e-6:
        gate = gate.clamp(max=clamp_val)
        up = up.clamp(-clamp_val, clamp_val)
    return F.silu(gate) * up
