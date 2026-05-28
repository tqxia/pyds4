"""Dequantization kernels — CPU/NumPy implementations.

Phase B (M4–M6) covers Q8_0, IQ2_XXS, and Q2_K. These are the only three
quant types DeepSeek V4 Flash actually uses for routed weight data; the rest
of the GGUF is F16, F32, BF16, or I32 and is read raw via `tensor_bytes`.

Each dequant function takes a buffer of packed quant bytes plus the logical
element count and returns a 1-D `np.ndarray[np.float32]`. The caller is
expected to reshape to whatever 2-D layout the tensor descriptor declares.

We stay in NumPy for the CPU reference; PyTorch tensors get built on top in
the model-loading layer (M7). Triton GPU kernels arrive in Phase E and must
keep parity with these functions.

References:
- Format definitions: ds4.c::dtype_table at line 884 (block sizes).
- Q8_0 inner loop: ds4.c::dot_q8_0_row_2 at line 3042.
- f16→f32 spec: ds4.c::f16_to_f32 at line 1535 (plain IEEE 754 binary16).
"""

from __future__ import annotations

import numpy as np


# Q8_0: 32 elements per block. Layout per block (34 bytes):
#   [0..2)   fp16 scale `d`     (little-endian)
#   [2..34)  32 int8 quantized values
# Dequant: x[i] = d * q[i], with d cast f16 → f32 via IEEE 754.
Q8_0_BLOCK_ELEMS = 32
Q8_0_BLOCK_BYTES = 34


def dequant_q8_0(buf: bytes | memoryview, n_elements: int) -> np.ndarray:
    """Dequantize a Q8_0 byte buffer to fp32.

    `buf` must be at least `ceil(n_elements / 32) * 34` bytes long. We read
    only the first `n_blocks` blocks; trailing bytes are ignored. The
    returned array is exactly `n_elements` long (the last block is truncated
    if `n_elements` is not a multiple of 32 — which happens when a row's
    column count isn't 32-aligned; ds4 itself only uses 32-aligned shapes,
    but we don't rely on that).

    Numerically bit-exact against ds4: the only float operations are an
    IEEE-754 fp16→fp32 cast (NumPy's native conversion) and an fp32 multiply.
    """
    if n_elements < 0:
        raise ValueError(f"n_elements must be >= 0, got {n_elements}")
    if n_elements == 0:
        return np.empty(0, dtype=np.float32)

    n_blocks = (n_elements + Q8_0_BLOCK_ELEMS - 1) // Q8_0_BLOCK_ELEMS
    need = n_blocks * Q8_0_BLOCK_BYTES
    if len(buf) < need:
        raise ValueError(
            f"buf too small: have {len(buf)} bytes, need {need} for "
            f"{n_blocks} blocks ({n_elements} elements)"
        )

    # View the bytes as a (n_blocks, 34) uint8 matrix. From there we slice
    # out the 2-byte fp16 scales and the 32-byte int8 payloads in one shot
    # each, no Python loop over blocks.
    raw = np.frombuffer(buf, dtype=np.uint8, count=need).reshape(n_blocks, Q8_0_BLOCK_BYTES)

    # Scales: bytes [0:2] of each block, reinterpreted as little-endian fp16,
    # then promoted to fp32. `view` keeps it zero-copy; `astype` does the
    # IEEE-754 conversion (bit-exact with ds4's f16_to_f32).
    scales = raw[:, :2].copy().view(np.float16).astype(np.float32).reshape(n_blocks)

    # Quants: bytes [2:34] of each block as signed int8.
    quants = raw[:, 2:].view(np.int8).astype(np.float32)  # (n_blocks, 32)

    out = (scales[:, None] * quants).reshape(-1)
    return out[:n_elements]


def quantize_q8_0(x: np.ndarray) -> bytes:
    """Quantize fp32 → Q8_0. Used by the round-trip test, not the inference path.

    Mirrors ds4's quantizer (`ds4q_quantize_q8_0` in gguf-tools/quants.c at
    line 341). Per block: scale d = amax/127, quants = round(x / d).
    """
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    n = x.size
    if n == 0:
        return b""
    if n % Q8_0_BLOCK_ELEMS != 0:
        raise ValueError(
            f"quantize_q8_0: element count {n} is not a multiple of "
            f"{Q8_0_BLOCK_ELEMS}"
        )
    nb = n // Q8_0_BLOCK_ELEMS
    x = x.reshape(nb, Q8_0_BLOCK_ELEMS)

    amax = np.abs(x).max(axis=1)                         # (nb,)
    d = amax / 127.0                                     # (nb,) fp32
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
    q = np.round(x * inv[:, None]).astype(np.int8)       # (nb, 32)
    d_fp16 = d.astype(np.float16)                        # IEEE 754 round-to-nearest

    out = np.empty((nb, Q8_0_BLOCK_BYTES), dtype=np.uint8)
    out[:, :2] = d_fp16.view(np.uint8).reshape(nb, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()
