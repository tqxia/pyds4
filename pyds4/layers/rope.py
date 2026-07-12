"""RoPE (Rotary Position Embedding) utilities.

DeepSeek V4 Flash applies RoPE only to the **last** n_rot=64 dimensions of
each attention head. The first nope=448 dims are passed through unchanged.

ds4.c reference: `rope_tail_ext_inplace` (line 4787).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pyds4.config import DS4Config


def precompute_rope_freqs(
    n_rot: int,
    freq_base: float,
    device: torch.device | str,
    *,
    freq_scale: float = 1.0,
    yarn_orig_ctx: int | None = None,
    yarn_beta_fast: float = 32.0,
    yarn_beta_slow: float = 1.0,
) -> torch.Tensor:
    """Return the effective frequency of each adjacent RoPE pair.

    With ``yarn_orig_ctx`` unset this is ordinary RoPE, optionally interpolated
    by ``freq_scale``. With it set, reproduce ds4's YaRN interpolation ramp.
    DS4 cancels YaRN's magnitude multiplier, so only the angle changes.
    """
    i = torch.arange(0, n_rot, 2, device=device, dtype=torch.float32)
    inv_freq = freq_base ** (-i / n_rot)
    if yarn_orig_ctx is None:
        return inv_freq * freq_scale

    denom = 2.0 * math.log(freq_base)
    low = math.floor(
        n_rot * math.log(yarn_orig_ctx / (yarn_beta_fast * 2.0 * math.pi)) / denom
    )
    high = math.ceil(
        n_rot * math.log(yarn_orig_ctx / (yarn_beta_slow * 2.0 * math.pi)) / denom
    )
    low = max(0.0, float(low))
    high = min(float(n_rot - 1), float(high))
    ramp = 1.0 - ((i / 2.0 - low) / max(0.001, high - low)).clamp(0.0, 1.0)
    return inv_freq * (freq_scale * (1.0 - ramp) + ramp)


def precompute_layer_rope_freqs(
    cfg: "DS4Config",
    ratio: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Return the exact RoPE frequencies selected by ds4 for one layer."""
    if ratio == 0:
        return precompute_rope_freqs(cfg.n_rot, cfg.rope_freq_base, device)

    factor = cfg.yarn_factor
    freq_scale = 1.0 / factor if factor > 0.0 else 1.0
    return precompute_rope_freqs(
        cfg.n_rot,
        cfg.compress_rope_freq_base,
        device,
        freq_scale=freq_scale,
        yarn_orig_ctx=cfg.yarn_orig_ctx if factor > 1.0 else None,
        yarn_beta_fast=cfg.yarn_beta_fast,
        yarn_beta_slow=cfg.yarn_beta_slow,
    )


def rope_forward(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
) -> torch.Tensor:
    """Apply forward RoPE rotation to the last dimension.

    x: (..., n_rot) — the rotated tail
    position_ids: (...,) — broadcast-compatible position indices
    inv_freq: (n_rot//2,)
    Returns: (..., n_rot)
    """
    return _apply_rope(x, position_ids, inv_freq, inverse=False)


def rope_inverse(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
) -> torch.Tensor:
    """Apply inverse RoPE rotation (sin sign negated).

    x: (..., n_rot)
    position_ids: (...,)
    inv_freq: (n_rot//2,)
    Returns: (..., n_rot)
    """
    return _apply_rope(x, position_ids, inv_freq, inverse=True)


def _apply_rope(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    inverse: bool,
) -> torch.Tensor:
    """Core RoPE: (x0, x1) → (x0*cos - x1*sin, x0*sin + x1*cos).

    position_ids shape: (batch,) or scalar. We unsqueeze to align with
    x's leading dims: (..., n_rot).
    """
    # position_ids: (batch,) → (batch, 1) for broadcasting
    pos = position_ids.to(dtype=inv_freq.dtype)
    # Compute (batch, n_rot//2) frequencies
    while pos.dim() < x.dim():
        pos = pos.unsqueeze(-1)
    freqs = torch.outer(pos.reshape(-1), inv_freq).reshape(
        *pos.shape[:-1], -1
    )  # (..., n_rot//2)
    freqs = freqs.to(x.dtype)

    cos = freqs.cos()
    sin = -(freqs.sin()) if inverse else freqs.sin()

    # Reshape x into pairs: (..., n_rot) → (..., n_rot//2, 2)
    orig_shape = x.shape
    x_pairs = x.reshape(*orig_shape[:-1], -1, 2)
    x0, x1 = x_pairs.unbind(-1)  # each (..., n_rot//2)

    r0 = x0 * cos - x1 * sin
    r1 = x0 * sin + x1 * cos

    return torch.stack([r0, r1], dim=-1).reshape(orig_shape)
