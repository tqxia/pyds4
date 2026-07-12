"""M9 raw sliding-window attention tests."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pyds4 import gguf
from pyds4.config import DS4Config
from pyds4.layers.attention import (
    compressor_prefill_from_projected,
    quantize_fp8_kv,
    raw_sliding_window_attention,
)
from pyds4.layers.rope import (
    precompute_layer_rope_freqs,
    precompute_rope_freqs,
)
from pyds4.model import DS4Model


def _e4m3_levels() -> list[float]:
    levels = [i * 0.001953125 for i in range(8)]
    levels.extend(
        (1.0 + mant * 0.125) * (2.0 ** (exp - 7))
        for exp in range(1, 16)
        for mant in range(8)
    )
    return levels[:127]


def _quantize_fp8_scalar(kv: np.ndarray, n_rot: int) -> np.ndarray:
    """Literal translation of ds4.c::dsv4_fp8_kv_quantize_row_inplace_cpu."""
    out = kv.astype(np.float32, copy=True)
    n_nope = out.shape[-1] - n_rot
    levels = _e4m3_levels()
    for row in out.reshape(-1, out.shape[-1]):
        for off in range(0, n_nope, 64):
            amax = max(float(np.abs(row[off : off + 64]).max()), 1.0e-4)
            scale = np.float32(2.0 ** math.ceil(math.log2(amax / 448.0)))
            for i in range(off, off + 64):
                value = np.float32(np.clip(row[i] / scale, -448.0, 448.0))
                absolute = abs(float(value))
                # C chooses the even code on an exact midpoint.
                best = min(
                    range(127),
                    key=lambda code: (abs(absolute - levels[code]), code & 1),
                )
                sign = -1.0 if value < 0.0 else 1.0
                row[i] = np.float32(sign * levels[best] * scale)
    return out


def _raw_attention_scalar(
    q: np.ndarray,
    kv: np.ndarray,
    sinks: np.ndarray,
    window: int,
) -> np.ndarray:
    """Literal ds4 raw-row attention, including the zero-valued sink."""
    seq, n_head, head_dim = q.shape
    out = np.zeros_like(q, dtype=np.float32)
    scale = 1.0 / math.sqrt(head_dim)
    for token in range(seq):
        start = max(0, token + 1 - window) if window else 0
        for head in range(n_head):
            scores = [
                float(np.dot(q[token, head], kv[key])) * scale
                for key in range(start, token + 1)
            ]
            maximum = max(float(sinks[head]), *scores)
            weights = [math.exp(score - maximum) for score in scores]
            denominator = math.exp(float(sinks[head]) - maximum) + sum(weights)
            for weight, key in zip(weights, range(start, token + 1)):
                out[token, head] += kv[key] * np.float32(weight / denominator)
    return out


def test_fp8_kv_quantization_matches_ds4_scalar_oracle() -> None:
    values = torch.linspace(-700.0, 700.0, 2 * 136, dtype=torch.float32).reshape(2, 136)
    # Include exact midpoint cases to exercise code-index tie-to-even.
    values[0, :8] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.75])

    actual = quantize_fp8_kv(values, n_rot=8)
    expected = torch.from_numpy(_quantize_fp8_scalar(values.numpy(), n_rot=8))

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[:, -8:], values[:, -8:], rtol=0.0, atol=0.0)


def test_raw_window_attention_matches_ds4_scalar_oracle() -> None:
    generator = torch.Generator().manual_seed(9)
    q = torch.randn(7, 3, 16, generator=generator)
    kv = torch.randn(7, 16, generator=generator)
    sinks = torch.tensor([-0.75, 0.0, 1.25])

    actual = raw_sliding_window_attention(q, kv, sinks, window=3)
    expected = torch.from_numpy(
        _raw_attention_scalar(q.numpy(), kv.numpy(), sinks.numpy(), window=3)
    )

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def test_raw_window_excludes_expired_tokens() -> None:
    generator = torch.Generator().manual_seed(10)
    window = 4
    q = torch.randn(9, 2, 8, generator=generator)
    kv = torch.randn(9, 8, generator=generator)
    changed = kv.clone()
    changed[0] += 1000.0
    sinks = torch.tensor([0.25, -0.5])

    baseline = raw_sliding_window_attention(q, kv, sinks, window)
    perturbed = raw_sliding_window_attention(q, changed, sinks, window)

    assert not torch.allclose(baseline[:window], perturbed[:window])
    torch.testing.assert_close(
        baseline[window:], perturbed[window:], rtol=0.0, atol=0.0
    )


def test_compressed_layer_rope_frequencies_match_ds4_yarn_formula() -> None:
    cfg = SimpleNamespace(
        n_rot=64,
        rope_freq_base=10_000.0,
        compress_rope_freq_base=160_000.0,
        yarn_factor=16.0,
        yarn_orig_ctx=65_536,
        yarn_beta_fast=32.0,
        yarn_beta_slow=1.0,
    )
    raw = precompute_layer_rope_freqs(cfg, ratio=0, device="cpu")
    compressed = precompute_layer_rope_freqs(cfg, ratio=4, device="cpu")

    expected_raw = torch.tensor(
        [cfg.rope_freq_base ** (-i / cfg.n_rot) for i in range(0, cfg.n_rot, 2)]
    )
    denominator = 2.0 * math.log(cfg.compress_rope_freq_base)
    low = max(
        0.0,
        math.floor(
            cfg.n_rot
            * math.log(cfg.yarn_orig_ctx / (cfg.yarn_beta_fast * 2.0 * math.pi))
            / denominator
        ),
    )
    high = min(
        cfg.n_rot - 1,
        math.ceil(
            cfg.n_rot
            * math.log(cfg.yarn_orig_ctx / (cfg.yarn_beta_slow * 2.0 * math.pi))
            / denominator
        ),
    )
    expected_compressed = []
    for i in range(0, cfg.n_rot, 2):
        extrapolated = cfg.compress_rope_freq_base ** (-i / cfg.n_rot)
        ramp = 1.0 - min(1.0, max(0.0, (i / 2.0 - low) / max(0.001, high - low)))
        expected_compressed.append(
            extrapolated * ((1.0 / cfg.yarn_factor) * (1.0 - ramp) + ramp)
        )

    torch.testing.assert_close(raw, expected_raw, rtol=1e-6, atol=0.0)
    torch.testing.assert_close(
        compressed, torch.tensor(expected_compressed), rtol=1e-6, atol=0.0
    )
    assert not torch.equal(raw, compressed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="M9 tap replay needs CUDA")
def test_layer0_attention_matches_ds4_activation_tap() -> None:
    """Replay ds4's layer-0 normalized input through pyds4 attention."""
    root = Path(__file__).parent
    gguf_path = Path("/home/tqxia/workspace/ds4/ds4flash.gguf")
    if not gguf_path.exists():
        pytest.skip(f"ds4 GGUF not available at {gguf_path}")

    x_np = np.fromfile(root / "data/m9_layer0_attn_norm.bin", dtype=np.float32)
    expected_np = np.fromfile(root / "data/m9_layer0_attn_out.bin", dtype=np.float32)
    x = torch.from_numpy(x_np.reshape(-1, 4096)).cuda()
    expected = expected_np.reshape(-1, 4096)

    with gguf.parse(gguf_path) as g:
        cfg = DS4Config.from_gguf(g)
        model = DS4Model(cfg, device="meta")
        names = [
            "blk.0.attn_q_a.weight",
            "blk.0.attn_q_a_norm.weight",
            "blk.0.attn_q_b.weight",
            "blk.0.attn_kv.weight",
            "blk.0.attn_kv_a_norm.weight",
            "blk.0.attn_sinks.weight",
            "blk.0.attn_output_a.weight",
            "blk.0.attn_output_b.weight",
        ]
        model.load_weights(g, device="cuda", dtype=torch.float32, names=names)

    positions = torch.arange(x.shape[0], device="cuda")
    inv_freq = precompute_layer_rope_freqs(cfg, ratio=0, device="cuda")
    with torch.no_grad():
        actual = model.blocks[0].attn(
            x,
            positions,
            inv_freq,
            cfg.n_head,
            cfg.head_dim,
            cfg.n_rot,
            cfg.out_group,
            cfg.lora_o,
        ).float().cpu().numpy()

    difference = np.abs(actual - expected)
    assert difference.max() < 0.05
    assert difference.mean() < 0.005


def test_ratio128_compressor_prefill_matches_scalar_pool_and_frontier() -> None:
    generator = torch.Generator().manual_seed(128)
    ratio = 128
    seq = 2 * ratio + 2
    head_dim = 72
    n_rot = 8
    kv = torch.randn(seq, head_dim, generator=generator)
    score = torch.randn(seq, head_dim, generator=generator)
    ape = torch.randn(head_dim, ratio, generator=generator)
    norm = torch.randn(head_dim, generator=generator)
    positions = torch.arange(seq)
    inv_freq = precompute_rope_freqs(n_rot, 10_000.0, "cpu")

    actual = compressor_prefill_from_projected(
        kv,
        score,
        ape,
        norm,
        ratio=ratio,
        positions=positions,
        inv_freq=inv_freq,
        n_rot=n_rot,
        rms_eps=1e-6,
    )

    adjusted = score.numpy() + ape.numpy()[:, positions.remainder(ratio)].T
    expected_rows = []
    for block in range(seq // ratio):
        start = block * ratio
        values = kv.numpy()[start : start + ratio]
        scores = adjusted[start : start + ratio]
        maximum = scores.max(axis=0, keepdims=True)
        weights = np.exp(scores - maximum)
        pooled = (weights * values).sum(axis=0) / weights.sum(axis=0)
        rms = 1.0 / math.sqrt(float(np.mean(pooled.astype(np.float64) ** 2)) + 1e-6)
        row = (pooled * rms * norm.numpy()).astype(np.float32)
        position = start
        for pair, frequency in enumerate(inv_freq.numpy()):
            i = head_dim - n_rot + 2 * pair
            angle = position * float(frequency)
            x0, x1 = float(row[i]), float(row[i + 1])
            row[i] = x0 * math.cos(angle) - x1 * math.sin(angle)
            row[i + 1] = x0 * math.sin(angle) + x1 * math.cos(angle)
        expected_rows.append(row)
    expected_rows_np = _quantize_fp8_scalar(np.stack(expected_rows), n_rot)

    torch.testing.assert_close(
        actual.rows, torch.from_numpy(expected_rows_np), rtol=2e-5, atol=2e-5
    )
    torch.testing.assert_close(actual.state_kv[:2], kv[-2:].float())
    torch.testing.assert_close(
        actual.state_score[:2], torch.from_numpy(adjusted[-2:]).float()
    )
    assert torch.isneginf(actual.state_score[2:]).all()
    assert actual.counts[-1].item() == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="M10 tap replay needs CUDA")
@pytest.mark.parametrize(
    ("layer", "ratio", "rows_max", "rows_mean"),
    [(2, 4, 0.07, 1e-4), (3, 128, 2e-4, 2e-6)],
)
def test_compressor_matches_ds4_activation_tap(
    layer: int, ratio: int, rows_max: float, rows_mean: float
) -> None:
    """Replay ds4 HCA inputs and compare compressed rows plus frontier."""
    root = Path(__file__).parent
    gguf_path = Path("/home/tqxia/workspace/ds4/ds4flash.gguf")
    if not gguf_path.exists():
        pytest.skip(f"ds4 GGUF not available at {gguf_path}")

    stem = f"data/m10_layer{layer}"
    x_np = np.fromfile(root / f"{stem}_attn_norm.bin", dtype=np.float32)
    rows_expected = np.fromfile(
        root / f"{stem}_compressed.bin", dtype=np.float32
    ).reshape(-1, 512)
    coff = 2 if ratio == 4 else 1
    state_shape = (coff * ratio, coff * 512)
    state_kv_expected = np.fromfile(
        root / f"{stem}_state_kv.bin", dtype=np.float32
    ).reshape(state_shape)
    state_score_expected = np.fromfile(
        root / f"{stem}_state_score.bin", dtype=np.float32
    ).reshape(state_shape)
    x = torch.from_numpy(x_np.reshape(-1, 4096)).cuda()

    with gguf.parse(gguf_path) as g:
        cfg = DS4Config.from_gguf(g)
        model = DS4Model(cfg, device="meta")
        names = [
            f"blk.{layer}.attn_compressor_norm.weight",
            f"blk.{layer}.attn_compressor_kv.weight",
            f"blk.{layer}.attn_compressor_gate.weight",
            f"blk.{layer}.attn_compressor_ape.weight",
        ]
        model.load_weights(g, device="cuda", dtype=torch.float32, names=names)

    positions = torch.arange(x.shape[0], device="cuda")
    inv_freq = precompute_layer_rope_freqs(cfg, ratio=ratio, device="cuda")
    with torch.no_grad():
        actual = model.blocks[layer].attn.compressor.prefill(
            x, positions, inv_freq, n_rot=cfg.n_rot, rms_eps=cfg.rms_eps
        )

    rows_difference = np.abs(actual.rows.cpu().numpy() - rows_expected)
    state_kv_difference = np.abs(actual.state_kv.cpu().numpy() - state_kv_expected)
    state_score_np = actual.state_score.cpu().numpy()
    finite = np.isfinite(state_score_expected)
    state_score_difference = np.abs(
        state_score_np[finite] - state_score_expected[finite]
    )
    assert rows_difference.max() < rows_max
    assert rows_difference.mean() < rows_mean
    assert state_kv_difference.max() < 1e-4
    assert state_score_difference.max() < 5e-4
    assert np.array_equal(
        np.isneginf(state_score_np), np.isneginf(state_score_expected)
    )
    assert actual.counts.tolist() == [
        (token + 1) // ratio for token in range(x.shape[0])
    ]
