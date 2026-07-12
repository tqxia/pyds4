# pyds4 — milestone checklist

Living checklist of the 22-milestone plan. Status (`[x]` done / `[ ]` open),
**goal** (what the milestone proves), **artifact** (files that land), and
**oracle** (how we know it works). The plan source is
`~/.claude/plans/i-am-running-on-silly-dolphin.md`; this doc tracks execution
against it.

The headline deliverable is **M12** — full-model logit parity with `ds4
--dump-logits --cuda --nothink` on a 20-token prompt within max-abs-diff
`< 1e-2`. Phases A → D get us there; Phase E swaps in Triton kernels behind a
flag (parity must keep holding); Phase F is an optional inference engine on
top.

---

## Phase A — Read the file

Just enough plumbing to turn the 81 GB GGUF into typed data.

### M1. GGUF parser — **done** (`44d6ffd`)
- **Goal:** parse the file format ds4 reads — header, KV metadata, tensor desc
  table — without materializing tensor bytes (mmap-backed).
- **Artifact:** `pyds4/gguf.py`. CLI: `python -m pyds4.gguf inspect <file>`.
- **Oracle:** `tests/test_gguf.py` — tensor count + named shapes match what
  ds4 itself reports on the same file.

### M2. Config extraction — **done** (`d9059f9`)
- **Goal:** turn the GGUF metadata block into a typed `DS4Config` (n_layer,
  n_embd, n_head, head_dim, n_routed_experts, top_k, n_hc, vocab_size, …).
- **Artifact:** `pyds4/config.py`.
- **Oracle:** `tests/test_config.py` — values match the `ds4_engine_summary`
  banner ds4 prints at startup.

### M3. Tokenizer — **done** (`4867ae8`)
- **Goal:** byte-level BPE encoder matching ds4's tokenizer byte-for-byte —
  including the JoyAI pre-tokenizer (the 9-branch byte walker in
  `ds4.c::bpe_tokenize_text`) and full-width-bracket special tokens
  (`<｜DSML｜>`, `<｜begin▁of▁sentence｜>`, …).
- **Artifact:** `pyds4/tokenizer.py`; deep-dive notes in `docs/tokenizer.md`.
- **Oracle:** `tests/test_tokenizer.py::test_roundtrip_vs_ds4` against
  `./ds4 --dump-tokens -p "..."` for a corpus including DSML tokens, raw
  bytes, leading whitespace, and CJK.

---

## Phase B — Dequantize (NumPy CPU reference)

After Phase B any tensor in the GGUF can be turned into fp32 on CPU. These
three formats cover every quantized tensor in ds4flash.gguf; everything else
(F16, F32, BF16, I32) is read raw.

### M4. Q8_0 dequant — **done** (`7b74b66`)
- **Goal:** dequant the 32-element / 34-byte blocks ds4 uses for attention
  KV projections, embeddings, and norms. Fully vectorized NumPy.
- **Artifact:** `pyds4/quant.py::dequant_q8_0`. Plus `quantize_q8_0` for the
  round-trip test.
- **Oracle:** `scripts/q8_0_oracle.c` — verbatim `f16_to_f32` from ds4.c plus
  the `d * (float)int8` dequant formula. Captured 8192 elements of
  `blk.0.attn_kv.weight` into `tests/data/q8_0_*.bin`.
  `test_dequant_q8_0_bit_exact_vs_c_oracle` is **byte-equal** (zero diff).

### M5. IQ2_XXS dequant — **done** (`47590c6`)
- **Goal:** dequant the 256-element / 66-byte blocks used by routed gate/up
  experts. Two lookup tables (256-entry uint64 grid, 128-entry sign table)
  driven by 8-bit grid indices and 7-bit sign indices.
- **Artifact:** `pyds4/quant_tables.py` (tables ported verbatim from
  `ds4_iq2_tables_cuda.inc`) + `pyds4/quant.py::dequant_iq2_xxs`. Vectorized
  with one `(nb, 8, 4, 8)` fancy-index lookup for magnitudes and another for
  ±1 signs.
- **Oracle:** `scripts/iq2_xxs_oracle.c` — same per-element formula
  (`d * (2*ls_raw + 1) * 0.125 * sign * grid_byte`). Captured 65,536 elements
  of `blk.0.ffn_gate_exps.weight`.
  `test_dequant_iq2_xxs_bit_exact_vs_c_oracle` is **byte-equal**.

### M6. Q2_K dequant — **done** (`2784dc7`)
- **Goal:** dequant the 256-element / 84-byte super-blocks used by routed
  down experts. Two-level scaling: super-block fp16 `d` + `dmin`, plus 16
  sub-block 4-bit (scale, min) packed in `scales[16]`. Per-element quants
  are 2-bit packed into `qs[64]` at shifts 0/2/4/6 (one byte feeds four
  sub-blocks across two halves of the block).
- **Artifact:** `pyds4/quant.py::dequant_q2_k` (NumPy, same shape as
  M4/M5). Hoists `d_sc = d * sc` and `dmin_mn = dmin * mn` once per
  sub-block so the operation order matches the C oracle exactly.
- **Oracle:** `scripts/q2_k_oracle.c` — verbatim `f16_to_f32` plus the
  scalar dequant from `ds4.c::ds4_vec_dot_q2_K_q8_K` (non-NEON path).
  Captured 65,536 elements of `blk.0.ffn_down_exps.weight`.
  `test_dequant_q2_k_bit_exact_vs_c_oracle` is **byte-equal** (zero diff).
- **Why:** finishes Phase B. After M6 we can load every weight tensor in
  ds4flash.gguf and convert it to a PyTorch fp32 tensor.

---

## Phase C — Naive forward pass (PyTorch, bf16, no Triton, no sparsity)

End-to-end code path exists. **Logit parity is not yet expected** — after M9
the raw sliding window is real, while CSA + HCA are still absent.

### M7. Model skeleton — **done** (`bb41b5f`)
- **Goal:** `DS4Model.__init__` builds all layer parameters and loads weights
  via the Phase B dequant. No `forward()` yet.
- **Artifact:** `pyds4/model.py`, `pyds4/layers/{rms,attention,moe,hc}.py`.
  Every leaf is declared on `device='meta'` by default, so building the full
  43-layer 284 B-parameter graph allocates only shape metadata. Optional
  sub-modules (HCA Compressor, CSA Indexer, hash-routing table, learned
  router bias) are gated by per-layer `compress_ratio` and `n_hash_layer`,
  matching the conditional layout in `ds4.c::weights_validate_layout`.
  `build_name_map(cfg)` is a pure function returning a bijective map from
  GGUF tensor name to dotted model `state_dict` key — shape-preserving, no
  transposes. `DS4Model.load_weights(g, device, dtype, names=...)`
  materializes parameters off meta via the Phase B dequant kernels (Q8_0 /
  IQ2_XXS / Q2_K / raw F32/F16/I32); a full load is deferred to later
  phases since the 81 GB GGUF expands past any single-device budget.
- **Oracle:** `tests/test_model.py` (6 tests, all pass) — meta instantiation
  allocates nothing; the name map's targets are unique; `sum(p.numel())`
  across `state_dict` equals **284,334,567,511** — exactly the "logical
  parameters" number `ds4_engine_summary` prints for `ds4flash.gguf`; the
  GGUF↔model name map is bijective on all 1328 tensors with zero orphans
  on either side; every per-tensor shape matches (not just totals); a
  partial `load_weights` smoke test loads 43 `attn_norm` tensors to CPU
  fp32 and verifies plausible DS4 pre-norm gain statistics.

### M8. Forward pass, naive — **done**
- **M8a.** Token embedding in HC space, output-HC collapse, final RMSNorm,
  and LM head produce finite logits with the expected shape.
- **M8b.** Dense causal MLA implements Q/KV projection, tail RoPE, sink-aware
  softmax, inverse RoPE, and grouped output projection.
- **M8c.** MoE implements token-hash selection for layers 0–2, biased top-K
  thereafter, normalized route weights, lazy expert dequant, shared experts,
  and ds4-compatible SwiGLU.
- **M8d.** Hyper-connections carry `[seq, n_hc, d]` state through fp32
  Sinkhorn mixing around attention and FFN.
- **M8e.** The full 43-layer stack ran against the real 81 GB GGUF on a fixed
  20-token prompt. The 2026-07-11 validation loaded 1,199 non-expert tensors,
  streamed routed experts from mmap, completed the forward in **259.3 s**,
  produced finite `(20, 129280)` logits, and returned **18 distinct argmax
  tokens**. The opt-in oracle `PYDS4_RUN_M8E=1 pytest
  tests/test_model.py::test_forward_m8e_full_20_tokens -s` passed. Logit
  parity remains intentionally deferred to M12.

---

## Phase D — Real attention (logit parity goal)

Replace the naive attention stand-in with ds4's actual three-path attention.
Per-layer parity uses ds4's `DS4_METAL_GRAPH_DUMP_*` activation hooks.

### M9. Raw sliding-window attention — **done**
- **Goal:** implement the raw prefill path: layer-specific RoPE/YaRN, the
  block-scaled E4M3 round trip on the non-RoPE KV prefix, a causal 128-token
  mask, virtual-sink normalization, inverse RoPE, and grouped output.
- **Artifact:** `quantize_fp8_kv`, `raw_sliding_window_attention`, and the M9
  path in `Attention.forward`; persistent decode ring storage remains M19.
- **Oracle:** scalar translations of ds4's E4M3 and raw-row attention formulas
  are covered in `tests/test_attention.py`. A real layer-0 CUDA activation tap
  on a 13-token prompt replays with max-abs-diff **0.04263** and mean-abs-diff
  **0.00424** (`test_layer0_attention_matches_ds4_activation_tap`). A real
  129-token layer test crosses the `n_swa=128` boundary.

### M10. Compressor (HCA) prefill — **done**
- **Goal:** project full-prompt KV/score lanes, add phase-dependent APE, pool
  ratio-4 and ratio-128 blocks per feature, normalize/rotate/E4M3-round the
  compressed rows, and return the exact CUDA decode frontier plus per-token
  compressed-row counts. Mixed attention remains M12.
- **Artifact:** `Compressor.prefill`, `compressor_prefill_from_projected`, and
  `CompressorPrefillOutput` in `pyds4/layers/attention.py`.
- **Oracle:** scalar ratio-128 pooling/frontier coverage plus CUDA activation
  taps for both layouts. Layer 2 (ratio 4) matches compressed rows at max
  **0.0625**, mean **7.8e-05**; frontier KV/score max errors are **7.6e-05**
  and **3.1e-04**. Layer 3 (ratio 128) matches rows at max **1.03e-04**,
  mean **8.2e-07**; frontier masks are exact.

### M11. Indexer (CSA) prefill — **open**
- **Goal:** Compressed-Sparse Attention indexer — score every past compressor
  row against the current query and take top-K.
- **Oracle:** per-query top-K indices match `indexer_topk_*` output from ds4.

### M12. Mixed attention forward — **open** *(headline)*
- **Goal:** raw window + indexer-selected compressed rows in a single
  attention call. Full forward pass.
- **Oracle:** `tests/test_parity.py::test_logit_parity_short_prompt` —
  max-abs-diff vs `./ds4 --dump-logits --cuda --nothink` is **< 1e-2** on a
  fixed 20-token prompt. **This is the deliverable.**

---

## Phase E — Triton kernels (parity must hold after each swap)

Each kernel replaces a PyTorch op behind a feature flag. After every swap,
the M12 parity test must still pass. Roughly easiest-to-hardest:

### M13. Triton Q8_0 dequant-+-matmul — **open**
- **Goal:** simplest layout, biggest win — Q8_0 is used everywhere except
  routed experts.

### M14. Triton IQ2_XXS dequant-+-matmul — **open**
- **Goal:** stress lookup-table indexing in Triton.

### M15. Triton Q2_K dequant-+-matmul — **open**

### M16. Triton MoE expert dispatch — **open**
- **Goal:** grouped matmul over selected expert IDs. Structurally the hardest
  one — sparse, gather/scatter.

### M17. Triton compressor pool update — **open**
- **Goal:** streaming ratio-4 update.

### M18. Triton indexer scoring + top-K — **open**
- **Goal:** Triton matmul + radix top-K. WMMA variants are stretch.

---

## Phase F — Inference engine (optional, after Phase E)

### M19. KV cache management — **open**
- Ring buffer for the raw window, append-only frontier for the compressor.

### M20. Decode loop — **open**
- One token at a time after prefill.

### M21. Sampling — **open**
- Temperature, top-p, min-p — match ds4 defaults.

### M22. CLI — **open**
- `python -m pyds4 -p "..."` end-to-end.

---

## Explicit non-goals

- Metal / Mac code.
- HTTP server, CLI REPL, native agent, eval harness.
- Disk KV cache, tool-call DSML canonicalization.
- Long-context (> 4k) chunked prefill in Phase D — deferred to Phase F if we
  even get there.
- Multi-Token Prediction (MTP) speculative decoding.
- Directional steering.
