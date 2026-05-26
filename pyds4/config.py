"""DS4Config — parsed from GGUF metadata, validates against ds4's fixed layout.

ds4 hardcodes the DeepSeek-V4-Flash layout as enums in ds4.c (DS4_N_LAYER=43,
DS4_N_EMBD=4096, …; see ds4.c lines 87-116). The GGUF carries the same numbers
as metadata. We read from the GGUF (so the code stays declarative) and then
cross-check against ds4's pinned constants — any drift means either the file
has been re-quantized to a different layout, or our key names are wrong.

We expose three categories of field:

- **Shape**: layer count, embedding/head/expert dimensions. Drive every tensor.
- **Hyperparams**: epsilons, RoPE base, YaRN scaling. Match numeric kernels.
- **Per-layer arrays**: `compress_ratios` and `swiglu_clamp_exp`. Stored in
  GGUF as arrays; we materialize them into Python lists since they're small.

`compress_ratios` is the per-layer CSA/HCA control: see CLAUDE.md and the
`ds4_layer_compress_ratio()` function in ds4.c (layers 0-1 dense, then 4 and
128 alternating). The GGUF array length is `n_layer + 1` (the trailing entry
is for the output head).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import gguf


# ds4.c pins these as enum values (lines 87-116). We assert each field
# against the corresponding entry here in DS4Config.validate(), so a mismatch
# between the GGUF and the inference engine fails loud at load time.
DS4_EXPECTED: dict[str, Any] = {
    "n_layer":            43,
    "n_embd":             4096,
    "vocab_size":         129280,
    "n_head":             64,
    "n_head_kv":          1,
    "head_dim":           512,
    "value_dim":          512,
    "n_rot":              64,
    "out_group":          8,
    "lora_q":             1024,
    "lora_o":             1024,
    "n_expert":           256,
    "n_expert_used":      6,
    "n_expert_shared":    1,
    "ff_exp":             2048,
    "n_hash_layer":       3,
    "n_swa":              128,
    "indexer_n_head":     64,
    "indexer_head_dim":   128,
    "indexer_top_k":      512,
    "n_hc":               4,
    "sinkhorn_iters":     20,
}


@dataclass(frozen=True)
class DS4Config:
    """Architectural configuration for DeepSeek V4 Flash.

    Built with `DS4Config.from_gguf(file)`. All fields are required; we do not
    apply defaults, because ds4 itself refuses to load if a key is missing.
    """

    # --- shape ---------------------------------------------------------------
    arch: str                       # "deepseek4"
    n_layer: int                    # number of transformer blocks = 43
    n_embd: int                     # hidden dim of one HC stream = 4096
    vocab_size: int                 # 129280
    max_position: int               # training context length = 1_048_576

    # Attention.
    n_head: int                     # Q heads per layer = 64
    n_head_kv: int                  # KV heads (MLA-style sharing) = 1
    head_dim: int                   # K dim per head = 512
    value_dim: int                  # V dim per head = 512
    n_rot: int                      # rotary-applied tail per head = 64
    out_group: int                  # attention output low-rank groups = 8
    lora_q: int                     # Q-side low-rank projection rank = 1024
    lora_o: int                     # output low-rank projection rank = 1024
    n_swa: int                      # raw sliding-window size = 128

    # Indexer (CSA path).
    indexer_n_head: int             # 64
    indexer_head_dim: int           # 128
    indexer_top_k: int              # 512 -- never lower; algorithmic, not perf

    # Hyper-connections (mHC).
    n_hc: int                       # 4 parallel residual streams per layer
    sinkhorn_iters: int             # 20 iterations for the HC mixing matrix
    hc_eps: float                   # 1e-6

    # MoE FFN.
    n_expert: int                   # 256 routed experts
    n_expert_used: int              # top-K = 6
    n_expert_shared: int            # 1 always-on expert
    ff_exp: int                     # expert intermediate size = 2048
    expert_weights_scale: float     # 1.5
    expert_weights_norm: bool       # True (renormalize top-K weights)
    expert_gating_func: int         # 4 (router activation enum)
    n_hash_layer: int               # 3 (hash-based routing layer count)

    # Numeric epsilons / norms.
    rms_eps: float                  # 1e-6

    # RoPE / YaRN scaling for the main attention path.
    rope_freq_base: float           # 10000.0
    yarn_factor: float              # 16.0
    yarn_beta_fast: float           # 32.0
    yarn_beta_slow: float           # 1.0
    yarn_orig_ctx: int              # 65536

    # Separate RoPE base used by the compressor (HCA) projections.
    compress_rope_freq_base: float  # 160000.0

    # Per-layer arrays (length n_layer or n_layer+1).
    # compress_ratios[i] is the ratio for layer i: 0 = dense, 4 = HCA ratio-4,
    # 128 = HCA ratio-128. The GGUF carries n_layer+1 entries (the trailing
    # one corresponds to the output head; ds4.c::ds4_layer_compress_ratio
    # alternates 4/128 after layer 1).
    compress_ratios: tuple[int, ...] = field(default_factory=tuple)
    # Per-layer SwiGLU clamping exponent (ds4.c::DS4_SWIGLU_CLAMP_EXP=10.0
    # is the C default but the GGUF can override per layer).
    swiglu_clamp_exp: tuple[float, ...] = field(default_factory=tuple)

    # --- factory -------------------------------------------------------------

    @classmethod
    def from_gguf(cls, g: gguf.GGUFFile) -> "DS4Config":
        """Pull every required key out of `g.kv` and build a config.

        Raises `KeyError` with the offending key name if the GGUF is missing
        something. ds4 dies in the same situation; we mirror that bluntness.
        """

        def need(key: str) -> Any:
            kv = g.kv.get(key)
            if kv is None:
                raise KeyError(f"GGUF missing required metadata key: {key!r}")
            return kv.value

        def need_array(key: str) -> list[Any]:
            return g.array(key)

        return cls(
            arch=need("general.architecture"),
            n_layer=need("deepseek4.block_count"),
            n_embd=need("deepseek4.embedding_length"),
            vocab_size=need("deepseek4.vocab_size"),
            max_position=need("deepseek4.context_length"),
            n_head=need("deepseek4.attention.head_count"),
            n_head_kv=need("deepseek4.attention.head_count_kv"),
            head_dim=need("deepseek4.attention.key_length"),
            value_dim=need("deepseek4.attention.value_length"),
            n_rot=need("deepseek4.rope.dimension_count"),
            out_group=need("deepseek4.attention.output_group_count"),
            lora_q=need("deepseek4.attention.q_lora_rank"),
            lora_o=need("deepseek4.attention.output_lora_rank"),
            n_swa=need("deepseek4.attention.sliding_window"),
            indexer_n_head=need("deepseek4.attention.indexer.head_count"),
            indexer_head_dim=need("deepseek4.attention.indexer.key_length"),
            indexer_top_k=need("deepseek4.attention.indexer.top_k"),
            n_hc=need("deepseek4.hyper_connection.count"),
            sinkhorn_iters=need("deepseek4.hyper_connection.sinkhorn_iterations"),
            hc_eps=need("deepseek4.hyper_connection.epsilon"),
            n_expert=need("deepseek4.expert_count"),
            n_expert_used=need("deepseek4.expert_used_count"),
            n_expert_shared=need("deepseek4.expert_shared_count"),
            ff_exp=need("deepseek4.expert_feed_forward_length"),
            expert_weights_scale=need("deepseek4.expert_weights_scale"),
            expert_weights_norm=bool(need("deepseek4.expert_weights_norm")),
            expert_gating_func=need("deepseek4.expert_gating_func"),
            n_hash_layer=need("deepseek4.hash_layer_count"),
            rms_eps=need("deepseek4.attention.layer_norm_rms_epsilon"),
            rope_freq_base=need("deepseek4.rope.freq_base"),
            yarn_factor=need("deepseek4.rope.scaling.factor"),
            yarn_beta_fast=need("deepseek4.rope.scaling.yarn_beta_fast"),
            yarn_beta_slow=need("deepseek4.rope.scaling.yarn_beta_slow"),
            yarn_orig_ctx=need("deepseek4.rope.scaling.original_context_length"),
            compress_rope_freq_base=need("deepseek4.attention.compress_rope_freq_base"),
            compress_ratios=tuple(need_array("deepseek4.attention.compress_ratios")),
            swiglu_clamp_exp=tuple(need_array("deepseek4.swiglu_clamp_exp")),
        )

    # --- validation ----------------------------------------------------------

    def validate(self) -> None:
        """Cross-check fields against ds4's pinned model layout.

        Raises ValueError on the first mismatch. Use after `from_gguf` to fail
        loudly if the GGUF doesn't describe DeepSeek-V4-Flash exactly as ds4
        expects it.
        """
        if self.arch != "deepseek4":
            raise ValueError(f"unsupported architecture: {self.arch!r}")
        for key, expected in DS4_EXPECTED.items():
            actual = getattr(self, key)
            if actual != expected:
                raise ValueError(
                    f"DS4Config.{key} = {actual}, expected {expected} "
                    "(see ds4.c::DS4_N_* enum)"
                )
        if len(self.compress_ratios) not in (self.n_layer, self.n_layer + 1):
            raise ValueError(
                f"compress_ratios has {len(self.compress_ratios)} entries; "
                f"expected n_layer={self.n_layer} or n_layer+1"
            )
        if len(self.swiglu_clamp_exp) != self.n_layer:
            raise ValueError(
                f"swiglu_clamp_exp has {len(self.swiglu_clamp_exp)} entries; "
                f"expected n_layer={self.n_layer}"
            )

    # --- derived helpers -----------------------------------------------------

    def layer_compress_ratio(self, il: int) -> int:
        """Per-layer compression ratio. Mirrors ds4.c::ds4_layer_compress_ratio.

        Returns 0 for dense attention (layers 0-1), 4 or 128 for HCA layers.
        We prefer the GGUF-provided array (authoritative) over the C heuristic.
        """
        if not 0 <= il < self.n_layer:
            raise IndexError(il)
        return int(self.compress_ratios[il])

    # --- summary -------------------------------------------------------------

    def summary(self) -> str:
        """Multi-line summary, formatted to be visually comparable to
        `ds4.c::model_summary` startup output."""
        lines = [
            f"arch:      {self.arch}",
            f"layers:    {self.n_layer}",
            f"embedding: {self.n_embd}",
            f"vocab:     {self.vocab_size}",
            f"context:   {self.max_position}",
            f"attention: heads={self.n_head} kv_heads={self.n_head_kv} "
            f"head_dim={self.head_dim} swa={self.n_swa}",
            f"indexer:   heads={self.indexer_n_head} "
            f"head_dim={self.indexer_head_dim} top_k={self.indexer_top_k}",
            f"experts:   routed={self.n_expert} used={self.n_expert_used} "
            f"shared={self.n_expert_shared} ff={self.ff_exp}",
            f"hc:        n={self.n_hc} sinkhorn_iters={self.sinkhorn_iters} "
            f"eps={self.hc_eps:g}",
            f"rope:      base={self.rope_freq_base:g} yarn_factor={self.yarn_factor:g} "
            f"orig_ctx={self.yarn_orig_ctx}",
            f"compress:  ratios={list(self.compress_ratios)[:6]}... "
            f"(len={len(self.compress_ratios)})",
        ]
        return "\n".join(lines)


# ---- CLI: `python -m pyds4.config inspect <gguf>` ---------------------------


def main(argv: list[str] | None = None) -> int:
    import sys

    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2 or argv[0] != "inspect":
        print("usage: python -m pyds4.config inspect <gguf-path>", file=sys.stderr)
        return 2
    with gguf.parse(argv[1]) as g:
        c = DS4Config.from_gguf(g)
    c.validate()
    print(c.summary())
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
