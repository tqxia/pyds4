# pyds4 — bootstrap notes for Claude Code

Educational re-implementation of [antirez/ds4](https://github.com/antirez/ds4) (the DeepSeek V4 Flash inference engine) in **PyTorch + Triton**, CUDA-only, targeting an NVIDIA GB10 box. The point is to learn the model by re-deriving it, not to beat ds4 on speed.

## Where things live

- **This repo** (`/home/tqxia/workspace/pyds4`): all new Python + Triton code, tests, scripts. Package name is `pyds4`.
- **Reference repo** (`/home/tqxia/workspace/ds4`): read-only oracle. Most useful files:
  - `ds4.h` — public engine API
  - `ds4_gpu.h` — line-by-line spec of every GPU kernel we need to provide an equivalent for
  - `ds4_cuda.cu` — CUDA reference for each kernel (~10k lines; grep, don't read top-to-bottom)
  - `ds4_iq2_tables_cuda.inc` — IQ2 dequant lookup tables (port verbatim)
  - `ds4.c` — model load, GGUF parse, CPU reference, sessions
- **Weights**: antirez GGUF `q2-imatrix` (~81 GB) at `/home/tqxia/workspace/ds4/ds4flash.gguf` (symlink). Same artifact ds4 uses, so we can do bit-level parity.
- **Plan file**: `~/.claude/plans/i-am-running-on-silly-dolphin.md` — 22 milestones across 6 phases. Note: layout section says `ds4t/` and `ds4-for-gb10/` — both are stale, use `pyds4/` and `pyds4/` (this dir) instead.

## Ground truth (oracle)

The ds4 binary in the reference repo, with debug flags, is the parity oracle:

```sh
./ds4 --dump-tokens -p "..."                                  # tokenizer parity
./ds4 --dump-logits /tmp/logits.json --cuda --nothink \
      --prompt-file prompt.txt                                # full-model logit parity
```

The headline deliverable (end of Phase D) is `tests/test_parity.py::test_logit_parity_short_prompt`: our model's logits on a fixed 20-token prompt within max-abs-diff `< 1e-2` of ds4's `--dump-logits --cuda --nothink` output.

## Working philosophy

Build step by step. Each milestone is a small (~150 lines) addition with a runnable test and a parity oracle. Don't move to milestone N+1 until N's test passes. No large code dumps.

## Architecture, in 30 seconds

DeepSeek V4 Flash is an MoE model with **two novel attention paths** per layer running in parallel, plus a raw sliding window:

- **CSA** (Compressed Sparse Attention) — an *indexer* scores all past tokens and picks top-K to attend to. ds4 symbols: `ds4_gpu_indexer_*`, `attention_indexed_mixed_*`.
- **HCA** (Heavily Compressed Attention) — a *compressor* maintains a ratio-4 pooled view of the long past. Symbols: `ds4_gpu_compressor_*`, `attention_*_mixed_*`.
- **mHC** (Manifold-Constrained Hyper-Connections) — residual stream split into `n_hc` parallel streams, Sinkhorn-mixed each layer. Symbols: `hc_split_*`, `hc_expand_*`, `embed_token_hc_kernel`.
- **MoE FFN** — top-K routed experts + shared experts, SwiGLU. Symbols: `router_select_*`, `routed_moe_*`, `swiglu_kernel`.
- **Quantization** — routed experts at IQ2_XXS (up/gate) + Q2_K (down); other blocks Q8_0/F16. Indexer uses an FP4 Hadamard transform.

## Out of scope

Metal / Mac, HTTP server, CLI REPL, native agent, disk KV cache, tool-call canonicalization, long-context chunked prefill in Phase D, MTP, directional steering.

## Hardware

NVIDIA GB10 (aarch64, CUDA 13.0, 121 GB unified RAM, 812 GB disk). Build ds4 reference binary with `make cuda-spark` in `/home/tqxia/workspace/ds4`.
