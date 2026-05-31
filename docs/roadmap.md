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

### M6. Q2_K dequant — **open**
- **Goal:** dequant the 256-element / 84-byte super-blocks used by routed
  down experts. Two-level scaling: super-block fp16 `d` + `dmin`, plus 16
  sub-block 4-bit scales packed in `scales[16]`. Per-element quants are
  2-bit packed into `qs[64]`.
- **Artifact:** `pyds4/quant.py::dequant_q2_k` (NumPy, same shape as M4/M5).
- **Oracle:** `scripts/q2_k_oracle.c` — port the scalar dequant from
  `ds4.c::ds4_vec_dot_q2_K_q8_K`. Capture a slice of
  `blk.0.ffn_down_exps.weight`. Bit-exact target.
- **Why:** finishes Phase B. After M6 we can load every weight tensor in
  ds4flash.gguf and convert it to a PyTorch fp32 tensor.

---

## Phase C — Naive forward pass (PyTorch, bf16, no Triton, no sparsity)

End-to-end code path exists. **Logit parity is not yet expected** — we're
using dense causal attention as a stand-in for CSA + HCA.

### M7. Model skeleton — **open**
- **Goal:** `DS4Model.__init__` builds all layer parameters and loads weights
  via the Phase B dequant. No `forward()` yet.
- **Artifact:** `pyds4/model.py`, `pyds4/layers/{rms,attention,moe,hc}.py`.
- **Oracle:** instantiation succeeds + total parameter count matches the
  `ds4_engine_summary` number.

### M8. Forward pass, naive — **open** (five sub-milestones)
- **M8a.** Token embedding (HC space) + final norm + LM head. Forward a
  1-token prompt; logits are garbage but **shapes must be right**.
- **M8b.** One transformer block with **dense causal attention** (Q/K/V
  projections, scaled dot-product, RoPE on the rotated tail).
- **M8c.** MoE FFN: router (top-K softmax/sigmoid), routed experts + shared
  experts, SwiGLU, weighted combine.
- **M8d.** Hyper-connection split/expand (mHC). Residual stream goes from
  `[seq, d]` to `[seq, n_hc, d]`; layers communicate via Sinkhorn-normalized
  mixing.
- **M8e.** Stack all layers. Forward a 20-token prompt — the run completes
  and produces a plausible distribution (top-K isn't garbage). **No parity
  yet.**

---

## Phase D — Real attention (logit parity goal)

Swap dense attention for ds4's actual three-path attention. Per-layer
parity is checked through a temporary `DS4T_TAP=<name>` instrumentation hook
in the ds4 reference (scratch-only, not committed upstream).

### M9. Raw sliding-window attention — **open**
- **Goal:** the "raw" KV path — last N tokens kept verbatim, attended with a
  sliding-window mask.
- **Oracle:** per-layer activation tap vs ds4.

### M10. Compressor (HCA) prefill — **open**
- **Goal:** Heavily-Compressed Attention. Build the ratio-4 pooled view of
  the long past from the full prompt; store frontier compressor states.
- **Oracle:** per-layer compressed-K / compressed-V tensors match ds4
  layer-by-layer.

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
