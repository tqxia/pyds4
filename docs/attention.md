# DeepSeek V4 Flash — Attention Architecture

A walkthrough of the attention system, bridging from vanilla self-attention
through MHA → MQA → MLA → the full three-path architecture.

All activation shapes use `S` for sequence length, `B` for batch size, `H`
for head count, `D` for embedding/hidden dimension. Weight (parameter)
shapes are given as `(in_features, out_features)` per GGUF convention.

## 1. The KV-cache problem

In a standard Transformer, every generated token appends one K and one V
vector per head to the cache. The cache size is:

```
2 × n_layers × seq_len × n_heads × head_dim × bytes_per_elem
```

For DeepSeek V4 Flash: `2 × 43 × S × 64 × 512 × 2 = 5.6 MB/token`.
At 1M context that's **5.6 TB** of KV cache — not viable on any single GPU.

## 2. MQA (Multi-Query Attention)

MQA shares one KV head across all Q heads, dropping `n_heads_kv` to 1:

```
5.6 MB × (1 / 64) ≈ 89 KB/token  →  ~89 GB at 1M context
```

Better, but still not enough for the 1M-context target.

## 3. MLA (Multi-head Latent Attention)

DeepSeek V4 Flash takes this further. Instead of projecting directly to
`n_heads × head_dim` for K and V, it projects to a **low-rank latent space**
and expands from there. The KV is *never* materialized in per-head form in
the cache — only the compressed latent is stored.

Throughout this section, `x` has shape `(S, 4096)` — a sequence of token
embeddings after HC pre-weighting. (A batch dimension `B` can be prepended
to every activation shape below; we omit it for clarity.)

### 3.1 Q projection (low-rank)

The Q path uses a two-step projection to reduce parameter count:

```
x:            (S, 4096)                          — input sequence

qr  = x @ attn_q_a                               (S, 4096) @ (4096, 1024) → (S,  1024)
qr  = RMSNorm(qr, attn_q_a_norm)                 (S, 1024)                 — learned gain
q   = qr @ attn_q_b                               (S, 1024) @ (1024, 32768) → (S, 32768)
q   = reshape(q, (S, 64, 512))                   (S, 64, 512)              — 64 heads, 512 dims
q   = head_rms_norm(q)                           (S, 64, 512)              — per-head, no weight
```

Weight shapes:
```
attn_q_a:        (4096, 1024)                    Q8_0
attn_q_a_norm:   (1024,)                         F32
attn_q_b:        (1024, 32768)                   Q8_0     — 64 heads × 512 head_dim
```

Without the low-rank split:
`4096 × 32768 = 134.2M` params.

With it:
`4096 × 1024 + 1024 × 32768 = 4.2M + 33.6M = 37.7M` — about **3.5× smaller**.

### 3.2 KV projection (the key insight)

```
kv = x @ attn_kv                                 (S, 4096) @ (4096, 512) → (S, 512)
kv = RMSNorm(kv, attn_kv_a_norm)                 (S, 512)                  — learned gain
kv = kv.unsqueeze(1)                             (S, 1, 512)               — 1 KV "head"
```

Weight shapes:
```
attn_kv:          (4096, 512)                    Q8_0
attn_kv_a_norm:   (512,)                         F32
```

That's it. **512 dims, 1 projection, no head expansion.** K and V are the
**same** vector — the architecture treats the single latent as both key
and value. Each of the 64 Q heads scores against this same 512-d latent
during attention.

| Approach  | K per token    | V per token    | Total KV / token |
|-----------|----------------|----------------|------------------|
| MHA       | 64 × 512 = 32K | 64 × 512 = 32K | 65,536           |
| MQA       | 1 × 512 = 512  | 1 × 512 = 512  | 1,024            |
| **MLA**   | **512**        | **same as K**  | **512**          |

MLA is **128× smaller** than MHA.

### 3.3 RoPE — only on the tail

RoPE is applied only to the **last 64** of the 512 head dims. The first 448
are left un-rotated (called "no-pe" in the code):

```
n_rot  = 64                                       — rotated dims per head
n_nope = 512 - 64 = 448                           — non-rotated dims per head

Split each head:
  q_nope  = q[:, :,  :448]          (S, 64, 448)
  q_rope  = q[:, :,  448:]          (S, 64,  64)
  kv_nope = kv[:, :, :448]          (S,  1, 448)
  kv_rope = kv[:, :,  448:]         (S,  1,  64)

RoPE rotation (last 64 dims, applied per head at its position):
  For each position pos_s (s = 0..S-1) and each pair (2i, 2i+1):
    θ = pos_s × freq_base^(-2i / 64)
    (x[2i], x[2i+1]) ← (x[2i]·cosθ - x[2i+1]·sinθ,
                         x[2i]·sinθ + x[2i+1]·cosθ)

q   = concat([q_nope,   q_rope_rot],  dim=-1)    (S, 64, 512)
kv  = concat([kv_nope,  kv_rope_rot], dim=-1)    (S,  1, 512)
```

The split lets the model separate *what* a token is (the 448-d content
projection) from *where* it is (the 64-d rotated tail). This matters for
the compressed/indexed paths (see §5), where position information must be
rotated differently at different compression ratios.

### 3.4 Attention compute

```
Q: (S, 64, 512)     — 64 heads, each attending independently
K: (S,  1, 512)     — 1 latent, broadcast to all 64 Q heads
V: (S,  1, 512)     — same as K

For head h in 0..63, over all query positions q and key positions k:

  scores[h, q, k] = dot(Q[q, h], K[k, 0]) / sqrt(512)

  sink[h] = learned per-head virtual-row logit (64 values, shape (64,) F32)
  augmented_scores = concat([scores, sink[h]], dim=-1)

weights = softmax(augmented_scores, dim=-1)      — causal mask applies to real keys
real_weights = weights[..., :S]                  — discard virtual sink weight

attn_out[h] = real_weights[h] @ V[:, 0, :]       — sink contributes no value vector
  attn_out: (S, 64, 512)
```

`sinks[h]` (`attn_sinks: (64,) F32`) is the logit for a virtual zero-valued
row. It participates in the softmax denominator, allowing a head to assign
probability mass to "no value," but it is not added to every real-key score.
Adding the same scalar to every score would cancel out under softmax.

With 4 tokens, each query gets one additional sink column:

```
augmented_scores[h] before softmax:

          tok0    tok1    tok2    tok3    sink
  tok0  [ s00     -inf    -inf    -inf    z_h ]
  tok1  [ s10     s11     -inf    -inf    z_h ]
  tok2  [ s20     s21     s22     -inf    z_h ]
  tok3  [ s30     s31     s32     s33     z_h ]

The sink column has no corresponding V row; only its denominator mass remains.
```

### 3.5 Inverse RoPE + grouped output projection

After attention, the inverse RoPE is applied (same rotation with negated sin),
then the output goes through a **grouped low-rank projection**:

```
attn_out:         (S, 64, 512)
after inv-RoPE:   (S, 64, 512)     — same shape

attn_flat = attn_out.reshape(S, 32768)          (S, 64×512)

Grouped projection (8 groups of 8 heads):
  chunk_size = 512 × 64 // 8 = 4096
  lora_o = 1024

  Group 0: heads  0- 7 → attn_flat[:,   0:4096] @ output_a[:,   0:1024] → (S, 1024)
  Group 1: heads  8-15 → attn_flat[:, 4096:8192] @ output_a[:, 1024:2048] → (S, 1024)
  ...
  Group 7: heads 56-63 → attn_flat[:, 28672:32768] @ output_a[:, 7168:8192] → (S, 1024)

Concat 8 groups → low: (S, 8192)
output = low @ attn_output_b                      (S, 8192) @ (8192, 4096) → (S, 4096)
```

Weight shapes:
```
attn_output_a:  (4096, 8192)    Q8_0   — 8 column-blocks of 1024 each, each block serves one head group
attn_output_b:  (8192, 4096)    Q8_0   — final linear back to embedding dim
```

`attn_output_a` is conceptually 8 separate `(4096→1024)` matrices stacked
side-by-side. The first 1024 columns serve heads 0-7, the next 1024 serve
heads 8-15, etc. `attn_output_b` collapses the concatenated 8192-d vector
back to 4096.

## 4. Where RoPE appears in the flow

RoPE is applied **three times** per layer — forward on Q, forward on K,
**inverse on the attention output**:

```
Step 1: Apply forward RoPE to Q  — (S, 64, 64) rotated tail of each head
Step 2: Apply forward RoPE to K  — (S,  1, 64) rotated tail
Step 3: [attention computed in rotated space, Q and K both rotated]
Step 4: Apply inverse RoPE to attention output — (S, 64, 64) tail
```

The K in the KV cache is stored *post-RoPE* (and FP8-quantized, with a
Hadamard transform applied to the rotated tail). Attention is computed in
the rotated space, then the output is rotated back before the grouped
output projection.

RoPE forward vs inverse:
```
forward:  (x0, x1) ← (x0·cosθ - x1·sinθ,  x0·sinθ + x1·cosθ)
inverse:  (x0, x1) ← (x0·cosθ - x1·(-sinθ), x0·(-sinθ) + x1·cosθ)
                   = (x0·cosθ + x1·sinθ, -x0·sinθ + x1·cosθ)
```

Same cos, negated sin — equivalent to rotating by `-θ`.

## 5. The three-path attention (per-layer)

What makes DeepSeek V4 Flash unique is that **each layer runs up to three
attention paths in parallel** on shared KV cache segments:

```
Layer   0, 1  (compress_ratio =   0):
  Raw sliding-window only (128 tokens)
  No compressor, no indexer
  These are the shallow layers — no long-range attention needed

Layers  2, 4, 6, ..., 42  (compress_ratio =   4):
  Path 1: Raw window (128 tokens)
  Path 2: HCA ratio-4 compressor (pooled view of the long past)
  Path 3: CSA indexer (scores all past compressor rows, takes top-512)

Layers  3, 5, 7, ..., 41  (compress_ratio = 128):
  Path 1: Raw window (128 tokens)
  Path 2: HCA ratio-128 compressor
  No indexer — dense-ish compressed attention only
```

DSv4 never runs CSA without HCA. Layers with `ratio=4` get all three paths;
layers with `ratio=128` get two; layers with `ratio=0` get one.

### 5.1 Raw sliding-window

The raw path attends causally to at most `n_swa=128` recent KV rows. After
tail RoPE, DS4 runs the 448-wide non-RoPE prefix through a block-scaled E4M3
round trip (seven independent 64-value blocks); the 64-wide rotated tail is
left in float. A learned sink logit contributes denominator mass but no value.

During prefill, query `t` can see keys `k <= t` satisfying `t - k < 128`.
Persistent ring-buffer storage for decode is separate from this prefill mask
and is scheduled for M19.

### 5.2 HCA compressor (Heavily Compressed Attention)

The compressor projects every normalized token into a KV lane and a score
lane, adds an absolute-phase embedding (APE) to the scores, then performs a
separate softmax across candidate tokens for every one of the 512 features.

For `ratio=4`, the projection width is `2 × head_dim = 1024`. Compressed row
zero pools four lane-1 candidates from tokens 0–3. Later row `c` pools eight
candidates: lane 0 from block `c-1` and lane 1 from block `c`. This overlapping
two-lane design preserves more local information than ordinary 4:1 pooling.

For `ratio=128`, the width is 512 and each row independently pools one
128-token block. Both layouts then apply learned RMSNorm, layer-specific tail
RoPE, and the same E4M3 round trip as raw KV.

Prefill also returns the CUDA-compatible frontier (`state_kv`, `state_score`)
needed by a later decode loop. Ratio-4 CUDA rebuilds its primary frontier lane
from the last four prompt projections; ratio-128 retains the incomplete tail.

### 5.3 CSA indexer (Compressed Sparse Attention)

Present only on `ratio==4` layers. The indexer scores every past compressor
row against the current Q (using its own Q projection, `indexer.attn_q_b`),
selects the top-512 rows, and includes them in the main attention call.

The indexer has its **own smaller compressor** for the K side:
- `indexer_compressor_*`: runs a ratio-4 compressor with `index_width = 2 × indexer_head_dim = 256`
- `indexer.proj`: projects the input to `indexer_n_head=64` scalar scores
- `indexer.attn_q_b`: Q projection for indexer scoring `(lora_q, 64×128)`

The selected rows are passed to `layer_attention_rows_one`, the same
attention kernel used for the raw path — just with different KV rows.

## 6. HC (Hyper-Connection) residual flow

Each attention sub-layer is wrapped by the HC system. The residual state
is **4 parallel streams** of shape `(S, 4, 4096)` — one 4096-d vector per
stream per token.

### HC pre (before attention)

```
Input: hc_state shape (S, 4, 4096)

1. Flatten: (S, 4, 4096) → (S, 16384)
2. RMSNorm (no learned weight) over the 16384-d flat vector
3. matvec @ hc_attn_fn: (S, 16384) @ (16384, 24) → (S, 24)
4. Partition the 24 control features (per token):
     [:, 0:4]   → pre_raw:   (S, 4)    — pre-gate logits
     [:, 4:8]   → post_raw:  (S, 4)    — post-gate logits
     [:, 8:24]  → comb_raw:  (S, 4, 4) — combine matrix logits

5. Pre-gates:
     pre_z  = pre_raw * scale[0] + base[0:4]          (S, 4)
     pre    = sigmoid(pre_z) + eps                      (S, 4)   ∈ (eps, 1+eps)

6. Post-gates:
     post_z = post_raw * scale[1] + base[4:8]          (S, 4)
     post   = 2 * sigmoid(post_z)                       (S, 4)   ∈ [0, 2]

7. Combine matrix (Sinkhorn):
     comb_z = comb_raw * scale[2] + base[8:24]         (S, 4, 4)
     → row_softmax(comb_z) + eps
     → column_normalize
     → 20 iterations alternating row/column normalization
     → comb: (S, 4, 4)      columns exactly 1.0, rows ≈1.0

8. Weighted sum (collapse HC streams to sub-layer input):
     attn_input = sum(pre[s,h] * hc_state[s,h])         (S, 4096)
```

### Attention forward

```
attn_input (S, 4096)  →  [MLA §3]  →  attn_output (S, 4096)
```

### HC post (after attention)

```
for each stream h in 0..3, each token s:
  new_hc[s, h] = attn_output[s] × post[s, h]           (S, 4, 4096)
               + sum_j(comb[s, h, j] × old_hc[s, j])

In vector form:
  mixed = bmm(comb, old_hc)                              (S, 4, 4096)
  new_hc = attn_output.unsqueeze(1) * post.unsqueeze(-1) + mixed
```

The 4 HC streams are like 4 parallel residual paths with a learned router.
The pre-gate decides how much each stream contributes to the sub-layer
input; the post-gate decides how much the sub-layer output contributes
back to each stream; the Sinkhorn matrix controls cross-stream mixing.

The same HC pattern repeats for the FFN sub-layer (with a different set of
`hc_ffn_*` weights), giving each layer two independent HC bundles.

## 7. Output head

At the top of the stack, the 4 HC streams are collapsed back to one:

```
hc_state: (S, 4, 4096)                            — output of last block

1. Flatten: (S, 4, 4096) → (S, 16384)
2. RMSNorm (no weight) over the 16384-d flat vector
3. matvec @ output_hc_fn: (S, 16384) @ (16384, 4) → (S, 4)
4. Gate: sigmoid(pre * scale + base) + eps  → (S, 4)
5. Weighted sum: x = sum(gate[s,h] × hc_state[s,h])  → (S, 4096)
6. RMSNorm with output_norm.weight: (S, 4096)
7. LM head: x @ output.weight  → (S, 4096) @ (4096, 129280) → (S, 129280)
```

Weight shapes:
```
output_hc_fn:     (16384, 4)     F16
output_hc_base:   (4,)           F32
output_hc_scale:  (1,)           F32
output_norm:      (4096,)        F32
output.weight:    (4096, 129280) Q8_0
```

## 8. Weight inventory

Every attention-related tensor and its shape:

| Tensor | Shape | Quant | Purpose |
|--------|-------|-------|---------|
| `attn_norm` | (4096,) | F32 | Pre-attention RMSNorm gain |
| `attn_q_a` | (4096, 1024) | Q8_0 | Q low-rank input projection |
| `attn_q_a_norm` | (1024,) | F32 | Q low-rank norm gain |
| `attn_q_b` | (1024, 32768) | Q8_0 | Q low-rank output projection |
| `attn_kv` | (4096, 512) | Q8_0 | KV latent projection (K==V) |
| `attn_kv_a_norm` | (512,) | F32 | KV latent norm gain |
| `attn_sinks` | (64,) | F32 | Per-head attention sink bias |
| `attn_output_a` | (4096, 8192) | Q8_0 | Grouped output projection (8×1024) |
| `attn_output_b` | (8192, 4096) | Q8_0 | Final output projection |
| `attn_compressor_kv` | (4096, comp_w) | F16 | HCA compressor KV projection |
| `attn_compressor_gate` | (4096, comp_w) | F16 | HCA compressor gate |
| `attn_compressor_ape` | (comp_w, ratio) | F16 | HCA compressor APE bias |
| `attn_compressor_norm` | (512,) | F32 | HCA compressor norm |
| `indexer.attn_q_b` | (1024, 8192) | F16 | Indexer Q projection for scoring |
| `indexer.proj` | (4096, 64) | F16 | Indexer score embedding |
| `indexer_compressor_kv` | (4096, 256) | F16 | Indexer's own compressor KV |
| `indexer_compressor_gate` | (4096, 256) | F16 | Indexer's own compressor gate |
| `indexer_compressor_ape` | (256, 4) | F16 | Indexer compressor APE bias |
| `indexer_compressor_norm` | (128,) | F32 | Indexer compressor norm |

## 9. What pyds4 implements vs what's pending

| Milestone | Status | What it covers |
|-----------|--------|----------------|
| M7 | ✓ done | All attention parameters declared |
| M8b | ✓ done | Initial dense causal MLA forward |
| M8d | ✓ done | HC pre/post with Sinkhorn wrapping attention |
| M9 | ✓ done | Raw 128-token prefill window, E4M3 KV, YaRN, ds4 tap parity |
| M10 | ✓ done | Ratio-4/128 HCA rows + CUDA-compatible decode frontier |
| M11 | open | CSA indexer scoring + top-K selection |
| M12 | open | Mixed attention: raw + compressor + indexer in one call |
