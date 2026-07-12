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

M9 implements the raw MLA path. M10 adds ratio-4/128 HCA compressor prefill
and exact decode-frontier construction. The CSA indexer lands in M11; combining
raw and compressed rows in the attention softmax remains M12.

Shape conventions match the GGUF tensor descs verbatim so the
GGUF→model weight loader is an identity on shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from pyds4.config import DS4Config
from pyds4.layers.rms import rms_norm_no_weight, rms_norm_weight
from pyds4.layers.rope import rope_forward, rope_inverse


def _compressor_width(cfg: DS4Config, ratio: int) -> int:
    """Per ds4.c: comp_width = (ratio == 4 ? 2 : 1) * head_dim."""
    coff = 2 if ratio == 4 else 1
    return coff * cfg.head_dim


_E4M3FN_LEVELS = (
    0.0, 0.001953125, 0.00390625, 0.005859375, 0.0078125, 0.009765625,
    0.01171875, 0.013671875,
) + tuple(
    (1.0 + mant * 0.125) * (2.0 ** (exp - 7))
    for exp in range(1, 16)
    for mant in range(8)
)[:-1]  # ds4 accepts finite E4M3 codes 0..126 (maximum 448).


def quantize_fp8_kv(kv: torch.Tensor, n_rot: int) -> torch.Tensor:
    """Simulate ds4's block-scaled E4M3 round trip for raw KV rows.

    The non-RoPE prefix is quantized in independent 64-value blocks. The
    rotated tail is preserved. The returned cache tensor is float32, matching
    ds4's dequantized activation cache rather than a packed byte format.
    """
    head_dim = kv.shape[-1]
    n_nope = head_dim - n_rot
    if n_nope < 0 or n_nope % 64 != 0:
        raise ValueError("non-RoPE KV width must be non-negative and 64-aligned")

    out = kv.float().clone()
    if n_nope == 0:
        return out

    prefix = out[..., :n_nope]
    blocks = prefix.reshape(*prefix.shape[:-1], n_nope // 64, 64)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1.0e-4)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    normalized = (blocks / scale).clamp(-448.0, 448.0)

    levels = torch.tensor(_E4M3FN_LEVELS, device=kv.device, dtype=torch.float32)
    absolute = normalized.abs().contiguous()
    lo = torch.searchsorted(levels, absolute, right=True) - 1
    lo = lo.clamp(0, len(_E4M3FN_LEVELS) - 1)
    hi = (lo + 1).clamp(max=len(_E4M3FN_LEVELS) - 1)
    lo_value = levels[lo]
    hi_value = levels[hi]
    lo_diff = (absolute - lo_value).abs()
    hi_diff = (hi_value - absolute).abs()
    ties_to_even = (hi_diff == lo_diff) & ((hi & 1) == 0) & ((lo & 1) != 0)
    use_hi = (hi_diff < lo_diff) | ties_to_even
    magnitude = torch.where(use_hi, hi_value, lo_value)
    quantized = torch.where(normalized < 0.0, -magnitude, magnitude) * scale
    out[..., :n_nope] = quantized.reshape_as(prefix)
    return out


def raw_sliding_window_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    sinks: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """DS4 raw prefill attention over the causal trailing window.

    ``q`` is ``(seq, n_head, head_dim)`` and ``kv`` is ``(seq, head_dim)``.
    The learned sink is an extra zero-valued softmax row.
    """
    seq, n_head, head_dim = q.shape
    if kv.shape != (seq, head_dim):
        raise ValueError(f"expected kv shape {(seq, head_dim)}, got {tuple(kv.shape)}")
    if window < 0:
        raise ValueError("window must be non-negative")

    scores = torch.matmul(
        q.float().transpose(0, 1),
        kv.float().T.unsqueeze(0),
    ) * (head_dim ** -0.5)

    query = torch.arange(seq, device=q.device).unsqueeze(1)
    key = torch.arange(seq, device=q.device).unsqueeze(0)
    distance = query - key
    valid = distance >= 0
    if window:
        valid &= distance < window
    scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))

    sink_scores = sinks.float().reshape(n_head, 1, 1).expand(-1, seq, -1)
    probabilities = torch.softmax(torch.cat([scores, sink_scores], dim=-1), dim=-1)
    real_probabilities = probabilities[..., :seq]
    return torch.matmul(real_probabilities, kv.float().unsqueeze(0)).transpose(0, 1)


@dataclass(frozen=True)
class CompressorPrefillOutput:
    """Compressed rows plus the decode frontier left after prompt prefill."""

    rows: torch.Tensor
    state_kv: torch.Tensor
    state_score: torch.Tensor
    counts: torch.Tensor


def compressor_prefill_from_projected(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    norm: torch.Tensor,
    *,
    ratio: int,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    n_rot: int,
    rms_eps: float,
) -> CompressorPrefillOutput:
    """Pool projected HCA rows and construct ds4's post-prefill frontier."""
    if ratio not in (4, 128):
        raise ValueError("HCA compressor ratio must be 4 or 128")
    if kv.shape != score.shape or kv.ndim != 2:
        raise ValueError("compressor kv and score must be equal 2-D tensors")

    seq, width = kv.shape
    coff = 2 if ratio == 4 else 1
    if width % coff != 0:
        raise ValueError("compressor width is incompatible with its ratio")
    head_dim = width // coff
    if ape.shape != (width, ratio):
        raise ValueError(f"expected APE shape {(width, ratio)}, got {tuple(ape.shape)}")
    if norm.shape != (head_dim,):
        raise ValueError(f"expected norm shape {(head_dim,)}, got {tuple(norm.shape)}")
    if positions.shape != (seq,):
        raise ValueError(f"expected positions shape {(seq,)}, got {tuple(positions.shape)}")
    if seq > 1 and not torch.all(positions[1:] == positions[:-1] + 1):
        raise ValueError("compressor prefill positions must be contiguous")

    kv_f32 = kv.float()
    phase = positions.remainder(ratio).long()
    score_f32 = score.float() + ape.float()[:, phase].T
    n_comp = seq // ratio
    cutoff = n_comp * ratio

    if n_comp == 0:
        pooled = kv_f32.new_empty((0, head_dim))
    elif ratio == 4:
        blocks_kv = kv_f32[:cutoff].reshape(n_comp, ratio, width)
        blocks_score = score_f32[:cutoff].reshape(n_comp, ratio, width)
        values = kv_f32.new_zeros((n_comp, 2 * ratio, head_dim))
        scores = kv_f32.new_full((n_comp, 2 * ratio, head_dim), float("-inf"))
        values[:, ratio:] = blocks_kv[:, :, head_dim:]
        scores[:, ratio:] = blocks_score[:, :, head_dim:]
        if n_comp > 1:
            values[1:, :ratio] = blocks_kv[:-1, :, :head_dim]
            scores[1:, :ratio] = blocks_score[:-1, :, :head_dim]
        weights = torch.softmax(scores, dim=1)
        pooled = (weights * values).sum(dim=1)
    else:
        values = kv_f32[:cutoff, :head_dim].reshape(n_comp, ratio, head_dim)
        scores = score_f32[:cutoff, :head_dim].reshape(n_comp, ratio, head_dim)
        pooled = (torch.softmax(scores, dim=1) * values).sum(dim=1)

    rows = rms_norm_weight(pooled, norm.float(), rms_eps)
    if n_comp:
        comp_positions = positions[:cutoff:ratio]
        nope = head_dim - n_rot
        rotated = rope_forward(rows[:, nope:], comp_positions, inv_freq)
        rows = torch.cat([rows[:, :nope], rotated], dim=-1)
        rows = quantize_fp8_kv(rows, n_rot)

    state_rows = coff * ratio
    state_kv = kv_f32.new_zeros((state_rows, width))
    state_score = kv_f32.new_full((state_rows, width), float("-inf"))
    remainder = seq - cutoff
    if ratio == 4:
        if seq >= ratio:
            # The CUDA release path replays the last four projections with the
            # small-batch kernel, placing them in the primary frontier lane.
            state_kv[:ratio] = kv_f32[-ratio:]
            state_score[:ratio] = score_f32[-ratio:]
        elif seq:
            frontier_rows = ratio + phase
            state_kv[frontier_rows] = kv_f32
            state_score[frontier_rows] = score_f32
    elif remainder:
        state_kv[:remainder] = kv_f32[cutoff:]
        state_score[:remainder] = score_f32[cutoff:]

    counts = torch.arange(1, seq + 1, device=kv.device).div(
        ratio, rounding_mode="floor"
    )
    return CompressorPrefillOutput(rows, state_kv, state_score, counts)


class Compressor(nn.Module):
    """HCA compressor parameters plus M10 prompt-prefill implementation."""

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
        self.head_dim = head_dim
        self.comp_width = comp_width
        self.ratio = ape_ratio
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

    def prefill(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        inv_freq: torch.Tensor,
        *,
        n_rot: int,
        rms_eps: float,
    ) -> CompressorPrefillOutput:
        """Project a prompt and build HCA rows plus the decode frontier."""
        if self.head_dim != 512:
            raise ValueError("M10 HCA prefill expects the 512-wide attention compressor")
        dtype = x.dtype
        kv = torch.matmul(x, self.kv.to(dtype))
        score = torch.matmul(x, self.gate.to(dtype))
        return compressor_prefill_from_projected(
            kv,
            score,
            self.ape,
            self.norm,
            ratio=self.ratio,
            positions=positions,
            inv_freq=inv_freq,
            n_rot=n_rot,
            rms_eps=rms_eps,
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
    """Per-layer raw attention plus optional HCA/CSA components.

    M10 compressor prefill is available through ``self.compressor.prefill``.
    The indexer and mixed-attention path land in M11 and M12.
    """

    def __init__(
        self,
        cfg: DS4Config,
        ratio: int,
        *,
        device: torch.device | str | None = "meta",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.ratio = ratio

        # ds4.c: attn_output_a shape = (head_dim * (n_head / out_group), lora_o).
        # 512 * (64/8) = 512 * 8 = 4096. The output is 8 * lora_o = 8192.
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

    # ------------------------------------------------------------------
    # Forward — raw sliding-window MLA attention (M9)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        inv_freq: torch.Tensor,
        n_head: int,
        head_dim: int,
        n_rot: int,
        out_group: int,
        lora_o: int,
    ) -> torch.Tensor:
        """Raw causal sliding-window attention (no compressor/indexer yet).

        x:         (seq, n_embd) — pre-RMS-normed input
        positions: (seq,)        — position IDs
        inv_freq:  (n_rot//2,)   — precomputed RoPE frequencies
        Returns:   (seq, n_embd) — attention output (pre-HC-post)
        """
        seq = x.shape[0]
        dtype = x.dtype
        nope = head_dim - n_rot  # 448

        # ---- Q projection (MLA low-rank) ----
        # x: (seq, n_embd) @ q_a: (n_embd, lora_q) → qr: (seq, lora_q)
        qr = torch.matmul(x, self.q_a.to(dtype))
        qr = rms_norm_weight(qr, self.q_a_norm.to(dtype), self.cfg.rms_eps)
        # qr: (seq, lora_q) @ q_b: (lora_q, n_head*head_dim) → q: (seq, n_head*head_dim)
        q = torch.matmul(qr, self.q_b.to(dtype))
        q = q.reshape(seq, n_head, head_dim)
        # Per-head RMSNorm (no learned weight) — ds4.c line 7977
        q = rms_norm_no_weight(q, self.cfg.rms_eps)

        # ---- KV projection (MLA shared, 1 KV head) ----
        # x: (seq, n_embd) @ kv: (n_embd, head_dim) → kv: (seq, head_dim)
        kv = torch.matmul(x, self.kv.to(dtype))
        kv = rms_norm_weight(kv, self.kv_a_norm.to(dtype), self.cfg.rms_eps)
        kv = kv.unsqueeze(1)  # (seq, 1, head_dim) — 1 KV head

        # ---- Split no-pe / rope ----
        q_nope = q[..., :nope]   # (seq, n_head, nope)
        q_rope = q[..., nope:]   # (seq, n_head, n_rot)
        kv_nope = kv[..., :nope]  # (seq, 1, nope)
        kv_rope = kv[..., nope:]  # (seq, 1, n_rot)

        # ---- RoPE (forward on Q and K) ----
        q_rope_flat = q_rope.reshape(-1, n_rot)     # (seq*n_head, n_rot)
        q_pos = positions.repeat_interleave(n_head)  # (seq*n_head,)
        q_rope_rot = rope_forward(q_rope_flat, q_pos, inv_freq).reshape(seq, n_head, n_rot)

        kv_rope_rot = rope_forward(
            kv_rope.reshape(-1, n_rot), positions, inv_freq
        ).reshape(seq, 1, n_rot)

        # Recombine and simulate the raw-cache E4M3 round trip.
        q_full = torch.cat([q_nope, q_rope_rot], dim=-1)       # (seq, n_head, head_dim)
        kv_full = torch.cat([kv_nope, kv_rope_rot], dim=-1).squeeze(1)  # (seq, head_dim)
        kv_full = quantize_fp8_kv(kv_full, n_rot)

        # ---- Raw causal sliding-window attention (M9) ----
        attn_out = raw_sliding_window_attention(
            q_full, kv_full, self.sinks, self.cfg.n_swa
        ).to(dtype)

        # ---- Inverse RoPE on output ----
        attn_nope = attn_out[..., :nope]             # (seq, n_head, nope)
        attn_rope = attn_out[..., nope:]             # (seq, n_head, n_rot)
        attn_rope_flat = attn_rope.reshape(-1, n_rot)
        attn_rope_inv = rope_inverse(attn_rope_flat, q_pos, inv_freq).reshape(seq, n_head, n_rot)
        attn_full = torch.cat([attn_nope, attn_rope_inv], dim=-1)  # (seq, n_head, head_dim)

        # ---- Grouped output projection ----
        attn_flat = attn_full.reshape(seq, n_head * head_dim)  # (seq, 32768)
        chunk_size = head_dim * n_head // out_group            # 4096
        low_parts = []
        out_a = self.output_a.to(dtype)
        for g in range(out_group):
            group = attn_flat[:, g * chunk_size : (g + 1) * chunk_size]  # (seq, 4096)
            slice_w = out_a[:, g * lora_o : (g + 1) * lora_o]            # (4096, 1024)
            low_g = torch.matmul(group, slice_w)  # (seq, 1024)
            low_parts.append(low_g)
        low = torch.cat(low_parts, dim=-1)  # (seq, 8192)
        out = torch.matmul(low, self.output_b.to(dtype))  # (seq, 8192) @ (8192, 4096) = (seq, 4096)
        return out
