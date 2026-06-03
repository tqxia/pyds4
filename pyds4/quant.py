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

from pyds4.quant_tables import IQ2XXS_GRID_BYTES, KSIGNS_IQ2XS_MASK


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


# IQ2_XXS: 256 elements per block. Layout per block (66 bytes):
#   [0..2)   fp16 super-scale `d`
#   [2..66)  qs[32] uint16, equivalently 16 uint32, equivalently
#            8 sub-blocks of 8 bytes each (one sub-block per 32 elements).
#
# Per sub-block (8 bytes = aux32[0], aux32[1] little-endian):
#   aux32[0]: 4 grid indices, one per byte → 4 × 8 = 32 magnitudes.
#   aux32[1]: 4 sign indices (7 bits each, bits 0..6, 7..13, 14..20, 21..27)
#             + a 4-bit local-scale `ls_raw` (bits 28..31).
#
# The local scale lives at half-integer steps:
#   ls = 2 * ls_raw + 1            (an odd integer in {1, 3, 5, ..., 31})
#   per-element factor = ls / 8    (so range {1/8, 3/8, ..., 31/8})
#
# Dequant per element:
#   d * (2 * ls_raw + 1) * 0.125 * sign * grid_byte
# where `sign ∈ {-1, +1}` and `grid_byte` is a small positive integer.
# Cross-referenced with ds4.c::ds4_vec_dot_iq2_xxs_q8_K (line 1910): the
# scalar path computes `0.125 * d * grid_byte * sign * q8_byte * ls`, so the
# factor we apply during dequant matches what ds4 applies during the dot.
IQ2_XXS_BLOCK_ELEMS = 256
IQ2_XXS_BLOCK_BYTES = 66


def dequant_iq2_xxs(buf: bytes | memoryview, n_elements: int) -> np.ndarray:
    """Dequantize an IQ2_XXS byte buffer to fp32.

    `buf` must be at least `ceil(n_elements / 256) * 66` bytes long. Output is
    exactly `n_elements` long; the tail of the last block is dropped if
    `n_elements` is not a multiple of 256 (ds4's tensors are always 256-
    aligned along the contiguous axis, so this only matters for tests).
    """
    if n_elements < 0:
        raise ValueError(f"n_elements must be >= 0, got {n_elements}")
    if n_elements == 0:
        return np.empty(0, dtype=np.float32)

    n_blocks = (n_elements + IQ2_XXS_BLOCK_ELEMS - 1) // IQ2_XXS_BLOCK_ELEMS
    need = n_blocks * IQ2_XXS_BLOCK_BYTES
    if len(buf) < need:
        raise ValueError(
            f"buf too small: have {len(buf)} bytes, need {need} for "
            f"{n_blocks} blocks ({n_elements} elements)"
        )

    raw = np.frombuffer(buf, dtype=np.uint8, count=need).reshape(
        n_blocks, IQ2_XXS_BLOCK_BYTES
    )

    # Super-scale d (fp16 → fp32, IEEE 754, bit-exact with ds4's f16_to_f32).
    d = raw[:, :2].copy().view(np.float16).astype(np.float32).reshape(n_blocks)

    # qs as (n_blocks, 8 sub-blocks, 8 bytes each). Sub-block = 32 elements.
    qs = raw[:, 2:].reshape(n_blocks, 8, 8)

    # Bytes 0..3 of each sub-block: four grid indices (0..255).
    grid_idx = qs[:, :, 0:4]                                # (nb, 8, 4) uint8

    # Bytes 4..7 of each sub-block: aux32[1] little-endian. Hold as uint32 so
    # the shifts that pull out four 7-bit sign indices and the 4-bit local
    # scale don't overflow.
    aux_hi = (qs[:, :, 4].astype(np.uint32)
              | (qs[:, :, 5].astype(np.uint32) << 8)
              | (qs[:, :, 6].astype(np.uint32) << 16)
              | (qs[:, :, 7].astype(np.uint32) << 24))      # (nb, 8) uint32

    ls_raw = (aux_hi >> 28).astype(np.float32)              # (nb, 8) ∈ 0..15
    # Pre-multiply d and the sub-block scale (2*ls_raw + 1)/8 into a single
    # factor — one fp32 broadcast per sub-block instead of two.
    sub_scale = d[:, None] * (2.0 * ls_raw + 1.0) * 0.125   # (nb, 8)

    # Four 7-bit sign indices stacked along a new axis: bits 0..6, 7..13, 14..20, 21..27.
    sign_idx = np.stack(
        [(aux_hi >> (7 * g)) & np.uint32(0x7F) for g in range(4)],
        axis=-1,
    ).astype(np.uint8)                                      # (nb, 8, 4) uint8

    # Vectorized lookups. The grid lookup yields the 8 magnitude bytes for
    # each of the 4 groups in each sub-block; the sign lookup yields the
    # matching 8 sign values (+1 / -1).
    grid_vals = IQ2XXS_GRID_BYTES[grid_idx]                 # (nb, 8, 4, 8) int8
    sign_vals = KSIGNS_IQ2XS_MASK[sign_idx]                 # (nb, 8, 4, 8) int8

    # Cast to fp32 once. The product never exceeds ~|grid|*1 = ~43, well
    # within int8, but doing the multiply in fp32 keeps the chain monotonic
    # with the C oracle (scale * (float)signed_byte).
    signed = (grid_vals * sign_vals).astype(np.float32)     # (nb, 8, 4, 8)

    out = signed * sub_scale[:, :, None, None]              # (nb, 8, 4, 8)
    return out.reshape(-1)[:n_elements]


# Q2_K: 256 elements per block. Layout per block (84 bytes):
#   [0..16)   scales[16]  uint8  — each byte holds (sub-min : 4) << 4 | (sub-scale : 4)
#   [16..80)  qs[64]      uint8  — 2-bit quants, four elements packed per byte
#   [80..82)  d           fp16   — super-scale
#   [82..84)  dmin        fp16   — super-min
#
# The 256 elements split into 16 sub-blocks of 16 elements each. The packing in
# qs is interleaved so that one byte contributes one element to each of four
# sub-blocks (at bit-shifts 0, 2, 4, 6). The dot-product loop in
# ds4.c::ds4_vec_dot_q2_K_q8_K (line 1888 onward, non-NEON path) makes the
# mapping explicit. Reading element s (sub-block 0..15) at position i (0..15):
#
#   k                = s // 8         # which 32-byte half of qs
#   second_in_pair   = s & 1          # which 16-byte slice of that half
#   pair_idx         = (s >> 1) & 3   # which 2-bit slot inside each byte
#   shift            = 2 * pair_idx
#   byte             = qs[32*k + 16*second_in_pair + i]
#   q                = (byte >> shift) & 0x3
#
# Per-element dequant: x = d * sc * q - dmin * mn, where
#   sc = scales[s] & 0x0f, mn = scales[s] >> 4.
Q2_K_BLOCK_ELEMS = 256
Q2_K_BLOCK_BYTES = 84


def dequant_q2_k(buf: bytes | memoryview, n_elements: int) -> np.ndarray:
    """Dequantize a Q2_K byte buffer to fp32.

    `buf` must be at least `ceil(n_elements / 256) * 84` bytes long. Output is
    exactly `n_elements` long; the tail of the last block is dropped if
    `n_elements` is not a multiple of 256 (ds4's tensors are always 256-
    aligned along the contiguous axis, so this only matters for tests).

    Numerically bit-exact against ds4: two IEEE-754 fp16→fp32 casts, two
    fp32 multiplies into `d_sc` / `dmin_mn`, one multiply by `q`, and one
    subtract. Hoisting `d_sc` and `dmin_mn` once per sub-block matches the
    C oracle and pins the operation order — no associativity wiggle room.
    """
    if n_elements < 0:
        raise ValueError(f"n_elements must be >= 0, got {n_elements}")
    if n_elements == 0:
        return np.empty(0, dtype=np.float32)

    n_blocks = (n_elements + Q2_K_BLOCK_ELEMS - 1) // Q2_K_BLOCK_ELEMS
    need = n_blocks * Q2_K_BLOCK_BYTES
    if len(buf) < need:
        raise ValueError(
            f"buf too small: have {len(buf)} bytes, need {need} for "
            f"{n_blocks} blocks ({n_elements} elements)"
        )

    raw = np.frombuffer(buf, dtype=np.uint8, count=need).reshape(
        n_blocks, Q2_K_BLOCK_BYTES
    )

    # Per-sub-block 4-bit (scale, min) packed in scales[16]: low nibble = sub
    # scale, high nibble = sub min.
    scales = raw[:, :16]                                    # (nb, 16) uint8
    sub_scale = (scales & 0x0F).astype(np.float32)          # (nb, 16)
    sub_min = (scales >> 4).astype(np.float32)              # (nb, 16)

    # qs as (nb, 2 halves, 2 slices, 16 bytes). The four 2-bit slots inside
    # each byte (shifts 0, 2, 4, 6) belong to sub-blocks at the same
    # (half, slice) but at four different `pair_idx` values.
    qs = raw[:, 16:80].reshape(n_blocks, 2, 2, 16)          # (nb, k, sip, byte)
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    # Extract all four shifts at once → axis order (nb, k, sip, byte, pair).
    vals = (qs[..., None] >> shifts) & np.uint8(0x3)        # (nb, 2, 2, 16, 4)
    # Reorder to (nb, k, pair, sip, byte) so that flattening the middle three
    # axes yields sub-block index 8*k + 2*pair + sip.
    q_values = vals.transpose(0, 1, 4, 2, 3).reshape(
        n_blocks, 16, 16
    ).astype(np.float32)                                    # (nb, 16, 16)

    # Super-scale d and super-min dmin: trailing 4 bytes of the block as
    # two little-endian fp16 values.
    d = raw[:, 80:82].copy().view(np.float16).astype(np.float32).reshape(n_blocks)
    dmin = raw[:, 82:84].copy().view(np.float16).astype(np.float32).reshape(n_blocks)

    # Hoist the two multiplies that don't depend on `q` — the C oracle does
    # the same, so the operation order is identical and the output is
    # byte-equal.
    d_sc = d[:, None] * sub_scale                           # (nb, 16)
    dmin_mn = dmin[:, None] * sub_min                       # (nb, 16)

    out = d_sc[:, :, None] * q_values - dmin_mn[:, :, None] # (nb, 16, 16)
    return out.reshape(-1)[:n_elements]
