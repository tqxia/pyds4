"""M4 parity tests for Q8_0 dequant.

Oracle: `scripts/capture_q8_0_oracle.py` slurps the first 256 blocks
(8192 elements) of `blk.0.attn_kv.weight` from ds4flash.gguf, pipes them
through `scripts/q8_0_oracle.c` (a verbatim re-implementation of ds4.c's
f16_to_f32 + the Q8_0 dequant formula), and saves the fp32 output to
`tests/data/q8_0_expected.bin`.

Bit-exact equality is the goal: NumPy's f16→f32 is IEEE 754, and so is the
oracle's hand-rolled version. Any disagreement points at a layout or
endianness bug in our reader, not a numeric drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pyds4 import gguf
from pyds4.quant import (
    IQ2_XXS_BLOCK_BYTES,
    IQ2_XXS_BLOCK_ELEMS,
    Q2_K_BLOCK_BYTES,
    Q2_K_BLOCK_ELEMS,
    Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_ELEMS,
    dequant_iq2_xxs,
    dequant_q2_k,
    dequant_q8_0,
    quantize_q8_0,
)


DATA_DIR = Path(__file__).parent / "data"
INPUT_BIN = DATA_DIR / "q8_0_input.bin"
EXPECTED_BIN = DATA_DIR / "q8_0_expected.bin"
META_JSON = DATA_DIR / "q8_0_meta.json"
IQ2_INPUT_BIN = DATA_DIR / "iq2_xxs_input.bin"
IQ2_EXPECTED_BIN = DATA_DIR / "iq2_xxs_expected.bin"
IQ2_META_JSON = DATA_DIR / "iq2_xxs_meta.json"
Q2K_INPUT_BIN = DATA_DIR / "q2_k_input.bin"
Q2K_EXPECTED_BIN = DATA_DIR / "q2_k_expected.bin"
Q2K_META_JSON = DATA_DIR / "q2_k_meta.json"
GGUF_PATH = Path("/home/tqxia/workspace/ds4/ds4flash.gguf")


oracle_available = pytest.mark.skipif(
    not (INPUT_BIN.exists() and EXPECTED_BIN.exists() and META_JSON.exists()),
    reason="Q8_0 oracle not captured (run scripts/capture_q8_0_oracle.py)",
)

iq2_oracle_available = pytest.mark.skipif(
    not (IQ2_INPUT_BIN.exists() and IQ2_EXPECTED_BIN.exists() and IQ2_META_JSON.exists()),
    reason="IQ2_XXS oracle not captured (run scripts/capture_iq2_xxs_oracle.py)",
)

q2k_oracle_available = pytest.mark.skipif(
    not (Q2K_INPUT_BIN.exists() and Q2K_EXPECTED_BIN.exists() and Q2K_META_JSON.exists()),
    reason="Q2_K oracle not captured (run scripts/capture_q2_k_oracle.py)",
)


@oracle_available
def test_dequant_q8_0_bit_exact_vs_c_oracle() -> None:
    """Our fp32 output must match the C oracle byte-for-byte."""
    meta = json.loads(META_JSON.read_text())
    n_elements = meta["n_elements"]
    raw = INPUT_BIN.read_bytes()
    expected = np.frombuffer(EXPECTED_BIN.read_bytes(), dtype=np.float32)

    got = dequant_q8_0(raw, n_elements)

    assert got.shape == (n_elements,)
    assert got.dtype == np.float32
    # Byte-exact: IEEE 754 fp16→fp32 + fp32 multiply has zero implementation
    # freedom.
    assert got.tobytes() == expected.tobytes(), (
        "Q8_0 dequant disagrees with C oracle "
        f"(max abs diff: {np.max(np.abs(got - expected))})"
    )


def test_dequant_q8_0_round_trip_random() -> None:
    """quantize → dequant must keep error bounded by amax/127 per block.

    Q8_0 is an affine 8-bit quantizer with no offset: the max representable
    magnitude per block is `127 * d`, and `d = amax / 127`, so the worst-case
    reconstruction error of any sample is `amax / 254` (the quantizer rounds
    to nearest of 256 levels spanning ±amax). We allow `amax / 127` as a
    loose bound to absorb f16 scale rounding.
    """
    rng = np.random.default_rng(0xD54)
    nb = 64
    x = rng.standard_normal(nb * Q8_0_BLOCK_ELEMS).astype(np.float32) * 0.7

    buf = quantize_q8_0(x)
    assert len(buf) == nb * Q8_0_BLOCK_BYTES
    y = dequant_q8_0(buf, x.size)

    per_block_amax = np.abs(x.reshape(nb, Q8_0_BLOCK_ELEMS)).max(axis=1)
    err = np.abs(x - y).reshape(nb, Q8_0_BLOCK_ELEMS).max(axis=1)
    # Per-block bound: ~amax/127 plus a tiny slack for fp16 scale rounding.
    bound = per_block_amax / 127.0 + 1e-4
    assert np.all(err <= bound), (
        f"block error exceeds bound: max ratio = {(err / bound).max()}"
    )


def test_dequant_q8_0_zero_block() -> None:
    """A pure-zero input must round-trip exactly (no division-by-zero)."""
    x = np.zeros(Q8_0_BLOCK_ELEMS, dtype=np.float32)
    buf = quantize_q8_0(x)
    y = dequant_q8_0(buf, x.size)
    assert np.all(y == 0)


def test_dequant_q8_0_empty() -> None:
    """Empty input → empty output (no buffer required)."""
    y = dequant_q8_0(b"", 0)
    assert y.shape == (0,)
    assert y.dtype == np.float32


def test_dequant_q8_0_buf_too_small() -> None:
    """Caller error: too few bytes for the requested element count."""
    with pytest.raises(ValueError, match="too small"):
        dequant_q8_0(b"\x00" * 33, Q8_0_BLOCK_ELEMS)


@pytest.mark.skipif(
    not GGUF_PATH.exists(),
    reason=f"ds4 GGUF not available at {GGUF_PATH}",
)
def test_dequant_q8_0_real_tensor_sanity() -> None:
    """End-to-end: dequant a real Q8_0 tensor row, sanity-check the result.

    `blk.0.attn_kv.weight` is a Q8_0 weight matrix (4096, 512). We dequant
    the first row (512 elements = 16 blocks) and assert basic properties.
    """
    with gguf.parse(GGUF_PATH) as g:
        t = g.tensors["blk.0.attn_kv.weight"]
        assert t.dtype == 8  # Q8_0
        n_per_row = t.shape[1]
        nblocks = n_per_row // Q8_0_BLOCK_ELEMS
        raw = bytes(g.tensor_bytes(t.name)[: nblocks * Q8_0_BLOCK_BYTES])

    y = dequant_q8_0(raw, n_per_row)
    assert y.shape == (n_per_row,)
    assert np.isfinite(y).all()
    # Trained weights are tiny but non-zero on average. A loose bound rules
    # out gross misreads (e.g. accidentally reading the scale as int8).
    assert 0.0 < np.abs(y).mean() < 1.0


# ---------------------------------------------------------------------------
# IQ2_XXS
# ---------------------------------------------------------------------------


@iq2_oracle_available
def test_dequant_iq2_xxs_bit_exact_vs_c_oracle() -> None:
    """Our fp32 output must match the C oracle byte-for-byte.

    The IQ2_XXS dequant chain — f16→f32, four uint8 grid lookups, four 7-bit
    sign lookups, and a single sub-block scalar multiply — is all IEEE 754
    fp32 once `d` is upcast. Same operation order in our NumPy and the C
    oracle ⇒ same bits.
    """
    meta = json.loads(IQ2_META_JSON.read_text())
    n_elements = meta["n_elements"]
    raw = IQ2_INPUT_BIN.read_bytes()
    expected = np.frombuffer(IQ2_EXPECTED_BIN.read_bytes(), dtype=np.float32)

    got = dequant_iq2_xxs(raw, n_elements)

    assert got.shape == (n_elements,)
    assert got.dtype == np.float32
    assert got.tobytes() == expected.tobytes(), (
        "IQ2_XXS dequant disagrees with C oracle "
        f"(max abs diff: {np.max(np.abs(got - expected))})"
    )


def test_dequant_iq2_xxs_empty() -> None:
    """Empty input → empty output (no buffer required)."""
    y = dequant_iq2_xxs(b"", 0)
    assert y.shape == (0,)
    assert y.dtype == np.float32


def test_dequant_iq2_xxs_buf_too_small() -> None:
    """Caller error: too few bytes for the requested element count."""
    with pytest.raises(ValueError, match="too small"):
        dequant_iq2_xxs(b"\x00" * 65, IQ2_XXS_BLOCK_ELEMS)


def test_dequant_iq2_xxs_zero_block() -> None:
    """A pure-zero block: scale = 0 ⇒ every output is exactly 0.

    With `d_bits = 0` the fp16 super-scale is +0.0, so every product is +0.0
    regardless of grid / sign / ls_raw. Cheapest possible smoke test that the
    block-size arithmetic is right.
    """
    raw = b"\x00" * IQ2_XXS_BLOCK_BYTES
    y = dequant_iq2_xxs(raw, IQ2_XXS_BLOCK_ELEMS)
    assert y.shape == (IQ2_XXS_BLOCK_ELEMS,)
    assert np.all(y == 0.0)


@pytest.mark.skipif(
    not GGUF_PATH.exists(),
    reason=f"ds4 GGUF not available at {GGUF_PATH}",
)
def test_dequant_iq2_xxs_real_tensor_sanity() -> None:
    """End-to-end: dequant a real IQ2_XXS slice, sanity-check the result.

    `blk.0.ffn_gate_exps.weight` is IQ2_XXS. We dequant 4 blocks (1024 elems)
    and assert basic properties of MoE expert weights: finite, mostly small,
    not all zero.
    """
    nblocks = 4
    with gguf.parse(GGUF_PATH) as g:
        t = g.tensors["blk.0.ffn_gate_exps.weight"]
        assert t.dtype == 16  # IQ2_XXS
        raw = bytes(g.tensor_bytes(t.name)[: nblocks * IQ2_XXS_BLOCK_BYTES])

    y = dequant_iq2_xxs(raw, nblocks * IQ2_XXS_BLOCK_ELEMS)
    assert y.shape == (nblocks * IQ2_XXS_BLOCK_ELEMS,)
    assert np.isfinite(y).all()
    assert np.abs(y).max() < 1.0
    assert np.abs(y).mean() > 0.0


# ---------------------------------------------------------------------------
# Q2_K
# ---------------------------------------------------------------------------


@q2k_oracle_available
def test_dequant_q2_k_bit_exact_vs_c_oracle() -> None:
    """Our fp32 output must match the C oracle byte-for-byte.

    The Q2_K dequant chain — two fp16→fp32 casts, two fp32 multiplies into
    `d_sc` and `dmin_mn`, one fp32 multiply by `q`, one fp32 subtract —
    has zero implementation freedom under IEEE 754 once the operation order
    is fixed. Our NumPy uses the same hoisting as the C oracle, so the
    output must be byte-equal.
    """
    meta = json.loads(Q2K_META_JSON.read_text())
    n_elements = meta["n_elements"]
    raw = Q2K_INPUT_BIN.read_bytes()
    expected = np.frombuffer(Q2K_EXPECTED_BIN.read_bytes(), dtype=np.float32)

    got = dequant_q2_k(raw, n_elements)

    assert got.shape == (n_elements,)
    assert got.dtype == np.float32
    assert got.tobytes() == expected.tobytes(), (
        "Q2_K dequant disagrees with C oracle "
        f"(max abs diff: {np.max(np.abs(got - expected))})"
    )


def test_dequant_q2_k_empty() -> None:
    """Empty input → empty output (no buffer required)."""
    y = dequant_q2_k(b"", 0)
    assert y.shape == (0,)
    assert y.dtype == np.float32


def test_dequant_q2_k_buf_too_small() -> None:
    """Caller error: too few bytes for the requested element count."""
    with pytest.raises(ValueError, match="too small"):
        dequant_q2_k(b"\x00" * 83, Q2_K_BLOCK_ELEMS)


def test_dequant_q2_k_zero_block() -> None:
    """A pure-zero block: d = 0, dmin = 0 ⇒ every output is exactly 0.

    With all bytes zero, scales/qs are zero (so `sc = mn = q = 0`) and both
    fp16 super-values are +0.0. Each per-element product is +0.0 - +0.0 =
    +0.0. Cheapest possible smoke test that the block-size arithmetic and
    the field offsets are right.
    """
    raw = b"\x00" * Q2_K_BLOCK_BYTES
    y = dequant_q2_k(raw, Q2_K_BLOCK_ELEMS)
    assert y.shape == (Q2_K_BLOCK_ELEMS,)
    assert np.all(y == 0.0)


@pytest.mark.skipif(
    not GGUF_PATH.exists(),
    reason=f"ds4 GGUF not available at {GGUF_PATH}",
)
def test_dequant_q2_k_real_tensor_sanity() -> None:
    """End-to-end: dequant a real Q2_K slice, sanity-check the result.

    `blk.0.ffn_down_exps.weight` is Q2_K. We dequant 4 blocks (1024 elems)
    and assert basic properties of MoE expert weights: finite, mostly
    small, not all zero.
    """
    nblocks = 4
    with gguf.parse(GGUF_PATH) as g:
        t = g.tensors["blk.0.ffn_down_exps.weight"]
        assert t.dtype == 10  # Q2_K
        raw = bytes(g.tensor_bytes(t.name)[: nblocks * Q2_K_BLOCK_BYTES])

    y = dequant_q2_k(raw, nblocks * Q2_K_BLOCK_ELEMS)
    assert y.shape == (nblocks * Q2_K_BLOCK_ELEMS,)
    assert np.isfinite(y).all()
    assert np.abs(y).max() < 1.0
    assert np.abs(y).mean() > 0.0
