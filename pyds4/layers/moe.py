"""MoE FFN parameters: routed experts + shared expert + router (with optional hash table or bias).

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
    exp_probs_b.bias: (n_expert,)                    F32  -- only when layer
                                                            uses learned routing
    ffn_gate_tid2eid: (n_expert_used, vocab_size)    I32  -- only layers
                                                            il < n_hash_layer

Hash routing is used for the first `n_hash_layer = 3` layers (0, 1, 2): instead
of running the soft router, the layer looks up token IDs in a precomputed
expert assignment table. Layers 3.. use the learned router + `exp_probs_b` bias.
Layer 2 carries BOTH (the hash table and the gate_inp matrix), per the GGUF.
"""

from __future__ import annotations

import torch
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
        # `exp_probs_b` is a per-expert additive bias used in the DeepSeek-V3
        # top-K router; carried for layers 3..n_layer-1.
        if has_exp_bias:
            self.exp_probs_b = nn.Parameter(
                torch.empty(cfg.n_expert, device=device, dtype=torch.float32),
                requires_grad=False,
            )

        # `ffn_gate_tid2eid` is the precomputed hash-routing table for the
        # first `n_hash_layer` layers. It's an int32 lookup, not a learnable
        # weight — but ds4's model_summary counts every tensor's element
        # count toward "logical parameters", so we register it the same way
        # for parity. Buffer (not Parameter) since dtype is integer.
        if is_hash_routed:
            self.register_buffer(
                "tid2eid",
                torch.empty(
                    cfg.n_expert_used, cfg.vocab_size, device=device, dtype=torch.int32
                ),
                persistent=True,
            )
