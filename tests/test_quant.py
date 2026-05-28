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
    Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_ELEMS,
    dequant_q8_0,
    quantize_q8_0,
)


DATA_DIR = Path(__file__).parent / "data"
INPUT_BIN = DATA_DIR / "q8_0_input.bin"
EXPECTED_BIN = DATA_DIR / "q8_0_expected.bin"
META_JSON = DATA_DIR / "q8_0_meta.json"
GGUF_PATH = Path("/home/tqxia/workspace/ds4/ds4flash.gguf")


oracle_available = pytest.mark.skipif(
    not (INPUT_BIN.exists() and EXPECTED_BIN.exists() and META_JSON.exists()),
    reason="Q8_0 oracle not captured (run scripts/capture_q8_0_oracle.py)",
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
