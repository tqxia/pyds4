"""Attention path parameters: main MLA-style + optional HCA compressor + optional CSA indexer.

DeepSeek V4 Flash has three attention paths that read the same KV cache:

  1. **Main** (always present) — MLA-style low-rank Q + low-rank KV:
       attn_q_a:       (n_embd, lora_q)              Q8_0
       attn_q_a_norm:  (lora_q,)                     F32
       attn_q_b:       (lora_q, n_head*head_dim)     Q8_0
       attn_kv:        (n_embd, head_dim)            Q8_0  -- ratio-1 KV (MLA-flavor)
       attn_kv_a_norm: (head_dim,)                   F32
       attn_sinks:     (n_head,)                     F32   -- attention sink bias
       attn_output_a:  (head_dim*n_head/out_group, lora_o)  Q8_0
       attn_output_b:  (lora_o, n_embd)              Q8_0
       attn_norm:      (n_embd,)                     F32

  2. **Compressor** (HCA, ratio != 0) — pooled view of the long past. ds4.c:
       coff = ratio == 4 ? 2 : 1
       comp_width = coff * head_dim
       attn_compressor_norm:  (head_dim,)                F32
       attn_compressor_kv:    (n_embd, comp_width)       F16
       attn_compressor_gate:  (n_embd, comp_width)       F16
       attn_compressor_ape:   (comp_width, ratio)        F16

  3. **Indexer** (CSA, only ratio==4) — scores past compressor rows and
     selects top-K. Has its own (smaller) compressor:
       indexer.attn_q_b:      (lora_q, ix_head*ix_hdim)  F16
       indexer.proj:          (n_embd, ix_head)          F16
       indexer_compressor_norm: (ix_hdim,)               F32
       indexer_compressor_kv:   (n_embd, 2*ix_hdim)      F16
       indexer_compressor_gate: (n_embd, 2*ix_hdim)      F16
       indexer_compressor_ape:  (2*ix_hdim, 4)           F16

`ratio` is the per-layer compress ratio from cfg.compress_ratios.

Shape conventions match the GGUF tensor descs verbatim so the
GGUF→model weight loader is an identity on shape.
"""

from __future__ import annotations

import torch
from torch import nn

from pyds4.config import DS4Config


def _compressor_width(cfg: DS4Config, ratio: int) -> int:
    """Per ds4.c: comp_width = (ratio == 4 ? 2 : 1) * head_dim."""
    coff = 2 if ratio == 4 else 1
    return coff * cfg.head_dim


class Compressor(nn.Module):
    """HCA compressor — 4 parameters (norm, kv, gate, ape)."""

    def __init__(
        self,
        *,
        n_embd: int,
        head_dim: int,
        comp_width: int,
        ape_ratio: int,
        device: torch.device | str | None = "meta",
    ) -> None:
        super().__init__()
        self.norm = nn.Parameter(
            torch.empty(head_dim, device=device, dtype=torch.float32),
            requires_grad=False,
        )
        self.kv = nn.Parameter(
            torch.empty(n_embd, comp_width, device=device, dtype=torch.float16),
            requires_grad=False,
        )
        self.gate = nn.Parameter(
            torch.empty(n_embd, comp_width, device=device, dtype=torch.float16),
            requires_grad=False,
        )
        self.ape = nn.Parameter(
            torch.empty(comp_width, ape_ratio, device=device, dtype=torch.float16),
            requires_grad=False,
        )


class Indexer(nn.Module):
    """CSA indexer — Q LoRA tail, embedding projection, and its own ratio-4 compressor."""

    def __init__(
        self,
        cfg: DS4Config,
        *,
        device: torch.device | str | None = "meta",
    ) -> None:
        super().__init__()
        ix_q_dim = cfg.indexer_n_head * cfg.indexer_head_dim  # 64 * 128 = 8192
        ix_width = 2 * cfg.indexer_head_dim                   # 256
        self.attn_q_b = nn.Parameter(
            torch.empty(cfg.lora_q, ix_q_dim, device=device, dtype=torch.float16),
            requires_grad=False,
        )
        self.proj = nn.Parameter(
            torch.empty(cfg.n_embd, cfg.indexer_n_head, device=device, dtype=torch.float16),
            requires_grad=False,
        )
        # Indexer compressor is always ratio-4 (matches the CSA path).
        self.compressor = Compressor(
            n_embd=cfg.n_embd,
            head_dim=cfg.indexer_head_dim,
            comp_width=ix_width,
            ape_ratio=4,
            device=device,
        )


class Attention(nn.Module):
    """Per-layer attention parameters: main MLA + optional compressor + optional indexer."""

    def __init__(
        self,
        cfg: DS4Config,
        ratio: int,
        *,
        device: torch.device | str | None = "meta",
    ) -> None:
        super().__init__()
        self.ratio = ratio

        # ds4.c: attn_output_a shape = (head_dim * (n_head / out_group), lora_o).
        # 512 * (64/8) = 512 * 8 = 4096... wait that's not 4096; it's 64/8=8 so
        # 512*8 = 4096. GGUF says (4096, 8192). The 8192 is lora_o? No, lora_o=1024.
        # Actually the GGUF shape (4096, 8192) tells us:
        #   in_dim=4096 = head_dim*n_head/out_group  ✓
        #   out_dim=8192 = lora_o*n_hc? Or just an output low rank dim.
        # ds4.c calls this `out_low_dim`; in the model that's 8*lora_o = 8192.
        # We bake the shape in from the GGUF, no need to re-derive symbolically.
        attn_out_a_in = cfg.head_dim * (cfg.n_head // cfg.out_group)  # 4096
        attn_out_a_out = 8 * cfg.lora_o                               # 8192  (matches GGUF)
        attn_out_b_out = cfg.n_embd                                   # 4096

        self.norm = nn.Parameter(
            torch.empty(cfg.n_embd, device=device, dtype=torch.float32),
            requires_grad=False,
        )
        self.q_a = nn.Parameter(
            torch.empty(cfg.n_embd, cfg.lora_q, device=device, dtype=torch.float32),
            requires_grad=False,
        )
        self.q_a_norm = nn.Parameter(
            torch.empty(cfg.lora_q, device=device, dtype=torch.float32),
            requires_grad=False,
        )
        self.q_b = nn.Parameter(
            torch.empty(cfg.lora_q, cfg.n_head * cfg.head_dim, device=device, dtype=torch.float32),
            requires_grad=False,
        )
        self.kv = nn.Parameter(
            torch.empty(cfg.n_embd, cfg.head_dim, device=device, dtype=torch.float32),
            requires_grad=False,
        )
        self.kv_a_norm = nn.Parameter(
            torch.empty(cfg.head_dim, device=device, dtype=torch.float32),
            requires_grad=False,
        )
        self.sinks = nn.Parameter(
            torch.empty(cfg.n_head, device=device, dtype=torch.float32),
            requires_grad=False,
        )
        self.output_a = nn.Parameter(
            torch.empty(attn_out_a_in, attn_out_a_out, device=device, dtype=torch.float32),
            requires_grad=False,
        )
        self.output_b = nn.Parameter(
            torch.empty(attn_out_a_out, attn_out_b_out, device=device, dtype=torch.float32),
            requires_grad=False,
        )

        if ratio != 0:
            self.compressor = Compressor(
                n_embd=cfg.n_embd,
                head_dim=cfg.head_dim,
                comp_width=_compressor_width(cfg, ratio),
                ape_ratio=ratio,
                device=device,
            )
        if ratio == 4:
            self.indexer = Indexer(cfg, device=device)
