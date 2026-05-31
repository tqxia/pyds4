# Quantization in ds4 / pyds4

Notes on what quantization is, the three flavors ds4 uses (Q8_0, IQ2_XXS,
Q2_K), and how our NumPy dequant kernels in `pyds4/quant.py` connect to them.
This is the conceptual companion to the `M4`/`M5`/`M6` checklist entries in
[roadmap.md](roadmap.md).

---

## 1. Why quantize at all

DeepSeek V4 Flash is a **284-billion-parameter** MoE model. At fp32 that's
1.14 TB; at bf16, 568 GB. The ds4flash.gguf we work from is **86.7 GB**.
Two problems get easier as we shrink:

- **Capacity.** Whether the weights fit in VRAM / unified memory at all. The
  GB10 box has 121 GB unified RAM — fp16 (568 GB) is hopeless; 86.7 GB fits.
- **Bandwidth.** Matrix multiplication on a modern GPU is bandwidth-bound,
  not compute-bound: a B200 can do ~3.4 TB/s of HBM, but ~5 PFLOPS of fp16
  matmul. Halving the byte count of `W` roughly doubles the practical
  throughput of `y = W x`.

The cost is **representational precision**. We pick numeric formats that lose
as little as possible per *bit budgeted*.

A note on MoE param counts: only ~38 B of those 284 B are *active per token*
(6 of 256 routed experts + 1 shared expert + all the dense attention /
embeddings). But the storage bill is for the full 284 B — every expert has
to be resident even when cold. That's the headline reason MoE inference is
storage-hungry while still being compute-cheap.

## 2. The shape of any quantization scheme

Two things define a quantization format:

1. **What integer type stores each element** — typically `int4`, `int2`, or a
   pointer into a small codebook.
2. **How the integer is turned back into a real number** — almost always
   `x ≈ scale * q + zero_point`, with **scales shared across a block** so the
   amortized per-element cost approaches the integer's bit width.

The trade-off lives entirely in the block:

```
        small block  →  more scales/block-overhead  →  better fit, worse compression
        large block  →  fewer scales                →  better compression, worse fit
```

Common block sizes are 32 (Q8_0, Q4_0, Q5_0) and 256 (the "K-quants" and
"I-quants" — Q2_K, Q3_K, IQ2_*, IQ3_*).

### 2a. Symmetric vs asymmetric

- **Symmetric** (zero point = 0). Block stores only one scalar — a scale.
  Used when the data is roughly zero-mean — most neural-net weights are. The
  reconstruction is `x = d * q`.
- **Asymmetric** (zero point ≠ 0). Block stores a scale + a minimum (or zero
  offset). `x = d * q + m`. Used when the data is one-sided (activation
  outputs of ReLU, attention scores after softmax) or, in K-quants, when
  using *small block* affine in a second tier.

ds4's Q8_0 is symmetric. Q2_K is asymmetric. IQ2_XXS is a different beast
entirely (see §5).

### 2b. The "K-quant" two-tier trick

Llama.cpp's K-quants pack 256 elements into a "super-block" with **two
levels** of scaling:

```
super-block (256 elements)
├── super-scale d        (fp16, one per 256 elements)
├── super-min   m        (fp16, one per 256 elements, asymmetric variants)
├── 16 sub-blocks of 16 elements each
│   └── per sub-block: a tiny per-sub-block scale `sc` (4-6 bits)
│       and per sub-block: a tiny per-sub-block min `mn` (4-6 bits)
└── 2-, 3-, 4-, 5-, or 6-bit integer quants
```

So per element: `x = (d * sc) * q + (m * mn)`. The fp16 super-scale carries
*range*; the 4-bit sub-block scale carries *local detail*. You get more
fidelity than a flat 2-bit scheme without paying for one fp16 per 16
elements.

### 2c. The "I-quant" codebook trick

I-quants don't store integers that get scaled — they store **pointers into a
fixed lookup table** of small integer vectors. Two elements aren't free to
take arbitrary 2-bit values; together they're constrained to lie on the
codebook lattice. This gives much better accuracy at low bit widths because
the lattice can be optimized offline to match the actual distribution of
weights (it's IQ2's whole reason for existing — see §5).

## 3. The three formats in ds4

| Format    | Block size | Bytes/block | Bits/elem | Where it's used                          | Style       |
| --------- | ---------: | ----------: | --------: | ---------------------------------------- | ----------- |
| `Q8_0`    |         32 |          34 |      8.5  | attention K/V projections, embeddings, norms | symmetric  |
| `Q2_K`    |        256 |          84 |      2.625| routed FFN **down** experts              | K-quant     |
| `IQ2_XXS` |        256 |          66 |      2.0625 | routed FFN **gate/up** experts         | I-quant     |

The headline number "DeepSeek V4 Flash quantized" almost entirely refers to
those routed-expert matrices — they're 95%+ of the model's parameter count.
The rest stays at higher precision because (a) it's small, and (b) attention
and embeddings are more sensitive to noise than dense MLP weights.

### 3a. Where the 86.7 GB actually goes

Reading `ds4flash.gguf` directly (sum of bytes per dtype):

| dtype     | params      | bits/elem (effective) | bytes        | % file |
| --------- | ----------: | --------------------: | -----------: | -----: |
| F32       |        461K |                  32   |       1.8 MB |  0.0 % |
| F16       |       1.10B |                  16   |     2.19 GB  |  2.5 % |
| Q8_0      |       6.21B |                   8.5 |     6.60 GB  |  7.6 % |
| Q2_K      |      92.34B |                   2.625|    30.31 GB  | 34.9 % |
| IQ2_XXS   |     184.68B |                   2.0625|   47.65 GB  | 54.9 % |
| **total** |   **284.33B**| —                    |  **86.7 GB** |        |

If everything were at a flat 2 bits/elem the file would be ~71 GB. The
remaining ~16 GB pays for two things:

- **+13 GB**: the Q2_K and IQ2_XXS "2-bit" formats actually average
  2.625 and 2.0625 bits/elem once their fp16 super-scales and per-sub-block
  scales are amortized in.
- **+9 GB**: the parity-sensitive tensors (attention K/V, embeddings, norms,
  router weights) are kept at Q8_0 or F16 rather than dropped to 2-bit.

It is genuinely cheap to be more careful in the places where it matters.

## 4. Q8_0 — symmetric per-block 8-bit (the easy one)

```c
// ds4.c — not literally this struct, but logically equivalent
struct block_q8_0 {
    uint16_t d;        // fp16 scale (per-block amax / 127)
    int8_t   qs[32];   // 32 quantized samples in [-128, 127]
};                     // 34 bytes total
```

**Quantize:**

```
d  = max(|x|) / 127
qs = round(x / d)        // int8, clipped to [-128, 127]
```

**Dequant** (what `pyds4/quant.py::dequant_q8_0` does):

```
x ≈ d * qs
```

That's it. No zero point, no sub-blocks, no codebook. One IEEE 754 fp16→fp32
cast and one fp32 multiply per element — that's the entire numeric chain,
which is why our `test_dequant_q8_0_bit_exact_vs_c_oracle` is byte-equal to
the C oracle: there's no implementation freedom.

Used where accuracy matters most:
- Token & output embeddings
- `Wq`/`Wk`/`Wv` attention projections (Q8_0 packs `Wk` and `Wv` together
  into `blk.N.attn_kv.weight`)
- Norm weights & router weights

8 bits per element is wildly over-budget by 2025 standards, but on these
tensors the absolute byte cost is tiny so there's no reason to push further.

## 5. IQ2_XXS — codebook + 7-bit signs (2 bits per element)

Here's where it gets interesting. "Pure" 2 bits per element would give you
only 4 distinct values per weight — far too coarse for transformer weights.
IQ2_XXS sidesteps this by storing **groups of 8 elements together** as a
pointer into a 256-entry codebook of vectors. The codebook lives in
`ds4_iq2_tables_cuda.inc` and was optimized offline against the empirical
weight distribution.

**Block layout** (256 elements, 66 bytes):

```
struct block_iq2_xxs {
    uint16_t d;          // fp16 super-scale
    uint16_t qs[32];     // 8 sub-blocks of 8 bytes each
};
```

Each 32-element **sub-block** packs into 8 bytes (= two uint32):

```
aux32[0]: |  g0  |  g1  |  g2  |  g3  |    ← 4 grid indices (8 bits each)
aux32[1]: | s0      | s1      | s2      | s3      | ls_raw |
          bits 0-6   7-13     14-20    21-27     28-31
                  ↑ 4 sign indices (7 bits each)   ↑ local scale (4 bits)
```

Each grid index `gi` points into `iq2xxs_grid[256]`, a table of 8 small
positive integers (packed as one `uint64`). Each sign index `si` points into
`ksigns_iq2xs[128]`, an 8-bit mask that tells us which of the 8 grid
positions should be negated. So a sub-block produces:

```
for j in 0..3:
    grid    = iq2xxs_grid[gj]          # 8 magnitudes ∈ {8, 25, 43, ...}
    signs   = ksigns_iq2xs[sj]         # 8 ±1 from bit mask
    sub_scale = d * (2*ls_raw + 1) / 8 # per-sub-block scale, half-integer in {1/8, 3/8, ...}
    out[j*8 + k] = sub_scale * signs[k] * grid[k]
```

That's literally what `pyds4/quant.py::dequant_iq2_xxs` does, vectorized over
all blocks at once.

**Why this is more accurate than a naive 2-bit linear scheme**

In a flat 2-bit scheme, every weight is one of 4 values; consecutive weights
are independent. In IQ2_XXS, every group of 8 consecutive weights is
constrained to be one of `256 grids × 128 signs = 32768` rigid patterns. The
8-element lattice was tuned offline so most actually-occurring weight
patterns land near a lattice point. You get 2.0625 bits/elem (8 bits grid +
7 bits sign per 8 elements + 4 bits local scale per 32 elements + fp16
super-scale per 256 elements) and accuracy comparable to a hypothetical
3-bit linear scheme.

This is also why the dequant is table-driven, not arithmetic — there's no
closed-form `q → x` because `q` is a pointer, not a number. Triton kernels
get to enjoy this in M14.

## 6. Q2_K — asymmetric K-quant (2.625 bits per element)

```c
struct block_q2_K {
    uint8_t  scales[16];    // 16 sub-block (scale, min) pairs, 4 bits each
    uint8_t  qs[64];        // 256 elements × 2 bits, packed
    uint16_t d;             // fp16 super-scale
    uint16_t dmin;          // fp16 super-min
};                          // 84 bytes total
```

The super-block holds 256 elements split into 16 sub-blocks of 16 each. For
each sub-block we store a 4-bit `sc` (sub-scale) and a 4-bit `mn` (sub-min),
both packed into one byte of `scales[]`. Dequant per element:

```
x = (d * sc) * q  -  (dmin * mn)
```

where `q ∈ [0, 3]`. The asymmetry (`-dmin * mn`) buys back enough range to
make 2-bit affine quantization work on signed weights despite the unsigned
2-bit `q`. It's what an IQ2_*-free world would have to live with; ds4 uses
Q2_K specifically for the routed-expert **down** matrices, presumably
because their distribution maps better to affine than to a codebook.

M6 is implementing exactly this — same shape as M4/M5: a NumPy vectorized
dequant + a C oracle for bit-exact parity.

## 7. So how do the formats fit together in ds4?

Routed MoE FFN, per expert per layer:

```
hidden = SwiGLU(  W_gate · x  ,  W_up · x  )    ← IQ2_XXS gate, IQ2_XXS up
out    = W_down · hidden                         ← Q2_K down
```

Two 2-bit codebook matrices feed into a 2-bit affine matrix. With 256 experts
per layer, this is where ~95% of the parameters live, and the 2-bit budget is
what makes DeepSeek V4 Flash fit on a single GB10.

Everywhere else — attention, embeddings, norms — stays at Q8_0 (or fp16 for
RoPE tables and the like) because those tensors are small and parity-
sensitive.

## 8. How our dequant code maps to all this

`pyds4/quant.py`:

| Function           | Format    | Block size | Lines (approx) |
| ------------------ | --------- | ---------: | -------------- |
| `dequant_q8_0`     | Q8_0      |         32 | one fp16 + one int8→fp32 broadcast |
| `dequant_iq2_xxs`  | IQ2_XXS   |        256 | two `(nb, 8, 4, 8)` table lookups + scale |
| `dequant_q2_k`     | Q2_K      |        256 | (M6, pending)  |

`pyds4/quant_tables.py` holds the IQ2_XXS lookup tables (`IQ2XXS_GRID`,
`KSIGNS_IQ2XS`, plus precomputed `IQ2XXS_GRID_BYTES` and
`KSIGNS_IQ2XS_MASK`). These tables are constants — they ship with the format
itself, not with any particular model.

All three functions take a buffer of packed bytes plus `n_elements` and
return `np.ndarray[float32]`. The single-tensor API is uniform — the M7
loader in `pyds4/model.py` will dispatch by `gguf_tensor.dtype` and call the
right one.

## 9. Phase-B's dual life: production vs oracle

In Phase C/D (M7–M12) the NumPy dequant is **production**: it produces the
fp32 arrays that get cast to bf16 and fed to PyTorch as `nn.Parameter`s. The
model forward pass uses them like any other tensor — `F.linear(x, W)` etc.

In Phase E (M13–M15) we replace the PyTorch matmul with a Triton **fused
dequant + matmul** kernel that keeps weights packed (the 2-bit bytes never
leave the GGUF mapping) and dequants inside registers, just-in-time. At that
point the NumPy dequant **demotes from production to test oracle** — the
Triton kernel is verified against "NumPy dequant + PyTorch matmul" on a
fixed input. This is the same role pattern as M4/M5 today: the C oracle is
the source of truth for the NumPy dequant; the NumPy dequant will be the
source of truth for the Triton dequant.

So the bit-exact byte-equal tests we built in M4/M5 aren't just a quality
gate for Phase B — they're the bottom of a parity chain that runs all the
way up through Phase E.

## 10. Aside: how this compares to NVFP4

NVFP4 is NVIDIA's microscaling 4-bit float introduced with Blackwell. Worth
understanding because GB10 has Blackwell Tensor Cores — NVFP4 is a *native*
format on our hardware, the GGUF quants are not.

### Format

```
each element:    E2M1     (1 sign + 2 exponent + 1 mantissa = 4 bits)
                          encodes { 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6 }
per 16 elements: FP8 E4M3 block scale
per tensor:      FP32     overall scale

x ≈ s_tensor · s_block · element_e2m1
```

Two-level scaling — structurally the same idea as Q2_K's `(d * sc) * q -
(dmin * mn)`, just with different bit allocations and a float (not int)
element. Storage works out to **4.5 bits/elem** (4 element bits + 8 scale
bits / 16 elements), plus a vanishing per-tensor fp32.

### Comparison table

| Format        | Element                | Block | Block scale     | Bits/elem | Hardware                          |
| ------------- | ---------------------- | ----: | --------------- | --------: | --------------------------------- |
| FP16 / BF16   | 16-bit float           | —     | —               |    16     | Native everywhere                 |
| FP8 (E4M3)    | 8-bit float            | —     | per-tensor fp32 |     8     | Hopper, Blackwell native          |
| **NVFP4**     | E2M1                   |    16 | FP8 E4M3        |     4.5   | **Blackwell Tensor Cores native** |
| MXFP4 (OCP)   | E2M1                   |    32 | E8M0 (2ⁿ only)  |     4.25  | Blackwell native, partial elsewhere |
| Q8_0          | int8 (sym)             |    32 | fp16            |     8.5   | Software dequant only             |
| Q4_K          | int4 (affine, 2-tier)  |    32 | fp16 + 6-bit    |    ~4.5   | Software dequant only             |
| Q2_K          | int2 (affine, 2-tier)  |    16 | 4-bit + fp16    |     2.625 | Software dequant only             |
| IQ2_XXS       | 8-entry codebook       |    32 | 4-bit + fp16    |     2.0625| Software dequant only             |

### Why ds4 doesn't use NVFP4

NVFP4 is *better than 2-bit GGUF on accuracy* (NVIDIA reports ~1 % accuracy
loss vs FP8 on DeepSeek-R1 benchmarks; IQ2_XXS gives up a few percent more)
— but it can't compress below ~4 bits. For our 284 B-param model:

```
277 B routed-expert params × 4.5 bits / 8  ≈  156 GB    NVFP4
                                                +
                                              ~9 GB     attention/embeddings at Q8_0/F16
                                              ─────
                                              ~165 GB
```

vs. **86.7 GB** today. GB10 has 121 GB unified memory. The NVFP4 version
**literally would not fit**. The whole reason DeepSeek V4 Flash exists at
all is the 2-bit quant budget; NVFP4 sits at a different point on the
capacity/accuracy curve, not a strictly better one.

### The deeper observation: hardware vs software

Blackwell Tensor Cores natively consume NVFP4 operands — the hardware
dequants on the fly inside the MMA pipeline. No separate dequant step. This
is *structurally exactly* what M13–M15 in `roadmap.md` try to do **in
software**: Triton kernels that keep IQ2_XXS / Q2_K bytes packed and
dequant inside registers as part of the matmul. Blackwell does it in
silicon for NVFP4; we do it in Triton for the 2-bit GGUF formats Blackwell
doesn't support.

So the right benchmark for our Phase-E kernels isn't "how fast vs the
PyTorch baseline" (the bar will be embarrassingly easy) — it's "how close
to native NVFP4 throughput, given we're storing half the bytes". That ratio
is the honest measure of whether our software fused-dequant pays its way.

### If we ever shipped a Blackwell-optimized successor

Plausible hybrid: keep routed experts at 2-bit GGUF (capacity-limited,
where every bit counts), upgrade everything else from Q8_0 to NVFP4
(latency-limited, where hardware acceleration matters). That's a sensible
production target — not for this project, but worth noting because it
shows the two format families are complementary rather than competing.

## 11. Further reading

- The original GGUF format spec (llama.cpp repo, `ggml-quants.h` and
  `ggml-common.h`) defines all the block structures. ds4 inlines the subset
  it actually parses (`ds4.c` lines 130–170 for structs, 228–308 for IQ2
  tables).
- The K-quant design write-up: ikawrakow's PR series on llama.cpp that
  introduced Q2_K..Q6_K. The "two-tier scaling" motivation is explained
  there.
- The I-quant design write-up: ikawrakow's IQ2_XXS/IQ2_XS/IQ3_XXS series.
  Codebook construction uses k-means on empirical weight distributions and
  then symmetry-folding with the sign mask.
- NVIDIA's NVFP4 announcement:
  <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
  — format spec, accuracy numbers vs FP8, and the comparison to MXFP4.
