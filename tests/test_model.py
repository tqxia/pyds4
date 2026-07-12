"""M7 tests — model skeleton instantiates with the right parameter shape.

The headline oracle is the number `ds4_engine_summary` prints as
"logical parameters": `sum(t.n_elements for t in gguf.tensors.values())`
(see `ds4.c::model_summary`, line 1270). On `ds4flash.gguf` that's
**284,334,567,511** elements. The model's `state_dict()` must sum to the
same value, with each leaf shape matching the corresponding GGUF tensor
exactly.

We also verify the GGUF↔model name map is bijective — every GGUF tensor
points at exactly one model state_dict key, and vice versa. A missing
direction silently hides parameters from the count.

Instantiation runs on `device='meta'`, so the whole 284 B-parameter graph
costs essentially nothing (only shape metadata). A small subset of weights
(the F32 norm gains) is loaded onto CPU as a smoke test for the
`load_weights` mechanism.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyds4 import gguf  # noqa: E402
from pyds4.config import DS4Config  # noqa: E402
from pyds4.model import DS4Model, _to_logical_layout, build_name_map  # noqa: E402


GGUF_PATH = Path("/home/tqxia/workspace/ds4/ds4flash.gguf")

# Number printed by `ds4 --info` / `ds4_engine_summary` for this file. Hard-coded
# so the test fails loudly if a future GGUF re-quantization quietly changes
# the tensor inventory.
DS4FLASH_LOGICAL_PARAMS = 284_334_567_511


gguf_available = pytest.mark.skipif(
    not GGUF_PATH.exists(),
    reason=f"ds4 GGUF not available at {GGUF_PATH}",
)


# ---------------------------------------------------------------------------
# Cheap tests that don't need the 81 GB file.
# ---------------------------------------------------------------------------


def test_instantiate_meta_smoke() -> None:
    """Building on `device='meta'` should be instant and allocate nothing."""
    # Hand-built minimal config matching ds4's pinned layout (DS4_EXPECTED).
    cfg = _synthetic_cfg()
    model = DS4Model(cfg, device="meta")
    # Every leaf must report device='meta' so we know nothing was allocated.
    for name, p in model.state_dict().items():
        assert p.device.type == "meta", f"{name} on device {p.device}, expected meta"


def test_name_map_no_duplicates() -> None:
    """Every model state_dict key appears at most once on the RHS of the map."""
    cfg = _synthetic_cfg()
    name_map = build_name_map(cfg)
    targets = list(name_map.values())
    assert len(targets) == len(set(targets)), "duplicate model target in name map"


def test_gguf_storage_converts_to_logical_matrix_layout() -> None:
    """GGUF dimension 0 is contiguous; torch's last dimension is contiguous."""
    stored = np.array([0, 1, 10, 11, 20, 21], dtype=np.float32)
    logical = _to_logical_layout(stored, (2, 3))
    expected = np.array([[0, 10, 20], [1, 11, 21]], dtype=np.float32)
    np.testing.assert_array_equal(logical, expected)
    assert logical.flags.c_contiguous

    # Expert is outermost in GGUF and remains the last logical dimension.
    stored_3d = np.arange(12, dtype=np.float32)
    logical_3d = _to_logical_layout(stored_3d, (2, 3, 2))
    assert logical_3d.shape == (2, 3, 2)
    expected_expert0 = np.array([[0, 2, 4], [1, 3, 5]], dtype=np.float32)
    np.testing.assert_array_equal(logical_3d[:, :, 0], expected_expert0)


# ---------------------------------------------------------------------------
# Real-GGUF tests.
# ---------------------------------------------------------------------------


@gguf_available
def test_param_count_matches_ds4flash_summary() -> None:
    """Model state_dict sum-of-numel matches ds4's reported logical params."""
    with gguf.parse(GGUF_PATH) as g:
        cfg = DS4Config.from_gguf(g)
        cfg.validate()
        model = DS4Model(cfg, device="meta")
        n_model = sum(v.numel() for v in model.state_dict().values())
        n_gguf = sum(t.n_elements for t in g.tensors.values())

    assert n_model == n_gguf, (
        f"param-count mismatch: model={n_model:,} gguf={n_gguf:,} "
        f"(diff={n_model - n_gguf:+,})"
    )
    assert n_model == DS4FLASH_LOGICAL_PARAMS, (
        f"got {n_model:,}, expected hard-coded ds4flash total "
        f"{DS4FLASH_LOGICAL_PARAMS:,}"
    )


@gguf_available
def test_name_map_bijective_against_gguf() -> None:
    """Every GGUF tensor has a model leaf, every model leaf has a GGUF source."""
    with gguf.parse(GGUF_PATH) as g:
        cfg = DS4Config.from_gguf(g)
        model = DS4Model(cfg, device="meta")
        gnames = set(g.tensors.keys())
        mnames = set(model.state_dict().keys())
        name_map = build_name_map(cfg)

    mapped_gguf = set(name_map.keys())
    mapped_model = set(name_map.values())

    assert gnames == mapped_gguf, (
        f"GGUF↔map mismatch on the source side: "
        f"unmapped GGUF tensors={sorted(gnames - mapped_gguf)[:5]}, "
        f"phantom map keys={sorted(mapped_gguf - gnames)[:5]}"
    )
    assert mnames == mapped_model, (
        f"GGUF↔map mismatch on the target side: "
        f"unmapped model leaves={sorted(mnames - mapped_model)[:5]}, "
        f"phantom map targets={sorted(mapped_model - mnames)[:5]}"
    )


@gguf_available
def test_per_tensor_shapes_match() -> None:
    """Every named GGUF tensor lands at a model leaf with the identical shape."""
    with gguf.parse(GGUF_PATH) as g:
        cfg = DS4Config.from_gguf(g)
        model = DS4Model(cfg, device="meta")
        sd = model.state_dict()
        name_map = build_name_map(cfg)
        mismatches = []
        for gname, mname in name_map.items():
            gshape = tuple(g.tensors[gname].shape)
            mshape = tuple(sd[mname].shape)
            if gshape != mshape:
                mismatches.append((gname, gshape, mname, mshape))

    assert not mismatches, (
        f"shape mismatches ({len(mismatches)}): "
        + ", ".join(f"{g}{gs} != {m}{ms}" for g, gs, m, ms in mismatches[:5])
    )


@gguf_available
def test_load_weights_partial_smoke() -> None:
    """load_weights() materializes a subset of parameters on CPU and copies bytes in.

    Loads only the 43 `attn_norm.weight` tensors (one per layer, 4096 fp32 each).
    Verifies the leaves leave meta device and look like plausible RMSNorm gains
    (DeepSeek-V4 pre-norm gains are small, ~0.02-0.5, but always finite and
    non-degenerate).
    """
    with gguf.parse(GGUF_PATH) as g:
        cfg = DS4Config.from_gguf(g)
        model = DS4Model(cfg, device="meta")
        name_map = build_name_map(cfg)
        targets = [n for n in name_map if n.endswith(".attn_norm.weight")]
        assert len(targets) == cfg.n_layer
        model.load_weights(g, device="cpu", dtype=torch.float32, names=targets)

    norm0 = model.blocks[0].attn.norm
    assert norm0.device.type == "cpu"
    assert norm0.dtype == torch.float32
    assert norm0.shape == (cfg.n_embd,)
    assert torch.isfinite(norm0).all()
    # Trained pre-RMSNorm gains in DS4 are tiny but non-zero. Loose bound just
    # rules out a degenerate load (all zeros, all garbage f16-as-f32 patterns).
    assert 0.0 < norm0.std().item() < 1.0
    assert 0.0 < norm0.abs().mean().item() < 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_cfg() -> DS4Config:
    """A DS4Config with the ds4-pinned values, no GGUF required.

    Used by tests that just exercise instantiation / map structure. Values
    come from DS4_EXPECTED in `pyds4.config`.
    """
    return DS4Config(
        arch="deepseek4",
        n_layer=43,
        n_embd=4096,
        vocab_size=129280,
        max_position=1_048_576,
        n_head=64,
        n_head_kv=1,
        head_dim=512,
        value_dim=512,
        n_rot=64,
        out_group=8,
        lora_q=1024,
        lora_o=1024,
        n_swa=128,
        indexer_n_head=64,
        indexer_head_dim=128,
        indexer_top_k=512,
        n_hc=4,
        sinkhorn_iters=20,
        hc_eps=1e-6,
        n_expert=256,
        n_expert_used=6,
        n_expert_shared=1,
        ff_exp=2048,
        expert_weights_scale=1.5,
        expert_weights_norm=True,
        expert_gating_func=4,
        n_hash_layer=3,
        rms_eps=1e-6,
        rope_freq_base=10000.0,
        yarn_factor=16.0,
        yarn_beta_fast=32.0,
        yarn_beta_slow=1.0,
        yarn_orig_ctx=65536,
        compress_rope_freq_base=160000.0,
        compress_ratios=(0, 0) + tuple(4 if i % 2 == 0 else 128 for i in range(41)) + (0,),
        swiglu_clamp_exp=tuple(10.0 for _ in range(43)),
    )


# ---------------------------------------------------------------------------
# M8 forward pass tests
# ---------------------------------------------------------------------------


def _weight_list_block0_non_expert() -> list[str]:
    """GGUF tensor names for block 0: everything except the 3D expert tensors."""
    return [
        # HC bundles
        "blk.0.hc_attn_fn.weight", "blk.0.hc_attn_base.weight", "blk.0.hc_attn_scale.weight",
        "blk.0.hc_ffn_fn.weight", "blk.0.hc_ffn_base.weight", "blk.0.hc_ffn_scale.weight",
        # Attention
        "blk.0.attn_norm.weight",
        "blk.0.attn_q_a.weight", "blk.0.attn_q_a_norm.weight", "blk.0.attn_q_b.weight",
        "blk.0.attn_kv.weight", "blk.0.attn_kv_a_norm.weight",
        "blk.0.attn_sinks.weight",
        "blk.0.attn_output_a.weight", "blk.0.attn_output_b.weight",
        # MoE (non-expert)
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate_inp.weight",
        "blk.0.ffn_gate_shexp.weight", "blk.0.ffn_up_shexp.weight", "blk.0.ffn_down_shexp.weight",
        "blk.0.ffn_gate_tid2eid.weight",
    ]


def _weight_list_top_level() -> list[str]:
    return [
        "token_embd.weight",
        "output_hc_fn.weight", "output_hc_base.weight", "output_hc_scale.weight",
        "output_norm.weight",
        "output.weight",
    ]


@gguf_available
def test_forward_m8a_embed_output_head() -> None:
    """M8a: token embedding + output head — shapes and finiteness check.

    Loads top-level tensors (token_embd + output_*). Blocks are emptied
    so the forward goes straight from embed to output head. Logits are
    garbage but shapes must be right.
    """
    import torch.nn as nn

    with gguf.parse(GGUF_PATH) as g:
        cfg = DS4Config.from_gguf(g)
        model = DS4Model(cfg, device="meta")
        model.load_weights(g, device="cuda", dtype=torch.bfloat16, names=_weight_list_top_level())

    # Remove blocks so forward skips to output head (M8a scope)
    model.blocks = nn.ModuleList([])

    model.eval()
    tokens = torch.tensor([0], device="cuda")

    with torch.no_grad():
        logits = model(tokens)

    assert logits.shape == (1, cfg.vocab_size)
    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(logits).all()
    assert logits.std().item() > 0.0, "logits are constant"


@gguf_available
def test_forward_m8d_hc_roundtrip() -> None:
    """M8d: HC pre/post roundtrip — shapes + Sinkhorn doubly-stochastic check.

    Loads one HC bundle, tests pre/post with random input. The comb matrix
    must be ~doubly-stochastic after Sinkhorn.
    """
    with gguf.parse(GGUF_PATH) as g:
        cfg = DS4Config.from_gguf(g)
        model = DS4Model(cfg, device="meta")
        model.load_weights(g, device="cuda", dtype=torch.bfloat16, names=[
            "blk.0.hc_attn_fn.weight",
            "blk.0.hc_attn_base.weight",
            "blk.0.hc_attn_scale.weight",
        ])

    hc_residual = torch.randn(2, cfg.n_hc, cfg.n_embd, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        x, post, comb = model.blocks[0].hc_attn.forward_pre(
            hc_residual, cfg.rms_eps, cfg.hc_eps, cfg.sinkhorn_iters,
        )

    assert x.shape == (2, cfg.n_embd)
    assert post.shape == (2, cfg.n_hc)
    assert comb.shape == (2, cfg.n_hc, cfg.n_hc)

    # comb must be approximately doubly-stochastic (fp32 after Sinkhorn).
    # ds4's Sinkhorn ends on col-norm, so columns are exactly 1.0 and rows
    # are ~1.0 (within a few percent for near-diagonal matrices).
    row_sums = comb.sum(dim=-1)  # (2, 4)
    col_sums = comb.sum(dim=-2)  # (2, 4)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-5), \
        f"col sums deviate: max_err={(col_sums - 1).abs().max().item():.1e}"
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.1), \
        f"row sums deviate: max_err={(row_sums - 1).abs().max().item():.1e}"

    # Forward post: recombine
    block_out = torch.randn(2, cfg.n_embd, device="cuda", dtype=torch.bfloat16)
    new_hc = model.blocks[0].hc_attn.forward_post(block_out, hc_residual, post, comb)
    assert new_hc.shape == (2, cfg.n_hc, cfg.n_embd)
    assert torch.isfinite(new_hc).all()


@gguf_available
def test_forward_m9_attention_one_block_crosses_window() -> None:
    """M9 integration: a real layer runs across the 128-token SWA cutoff."""
    with gguf.parse(GGUF_PATH) as g:
        cfg = DS4Config.from_gguf(g)
        model = DS4Model(cfg, device="meta")
        # Load block 0 attention weights
        attn_names = [
            "blk.0.attn_norm.weight",
            "blk.0.attn_q_a.weight", "blk.0.attn_q_a_norm.weight",
            "blk.0.attn_q_b.weight",
            "blk.0.attn_kv.weight", "blk.0.attn_kv_a_norm.weight",
            "blk.0.attn_sinks.weight",
            "blk.0.attn_output_a.weight", "blk.0.attn_output_b.weight",
        ]
        model.load_weights(g, device="cuda", dtype=torch.bfloat16, names=attn_names)

    from pyds4.layers.rope import precompute_rope_freqs
    inv_freq = precompute_rope_freqs(cfg.n_rot, cfg.rope_freq_base, "cuda")

    seq = cfg.n_swa + 1
    x = torch.randn(seq, cfg.n_embd, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(seq, device="cuda")

    with torch.no_grad():
        out = model.blocks[0].attn.forward(
            x, positions, inv_freq,
            cfg.n_head, cfg.head_dim, cfg.n_rot, cfg.out_group, cfg.lora_o,
        )

    assert out.shape == (seq, cfg.n_embd)
    assert torch.isfinite(out).all()
    assert out.std().item() > 0.0


@gguf_available
def test_forward_m8c_moe_one_block() -> None:
    """M8c: MoE FFN forward — shapes correct, output finite.

    Uses random expert_load_fn since the 3D tensors are too large to load.
    """
    with gguf.parse(GGUF_PATH) as g:
        cfg = DS4Config.from_gguf(g)
        model = DS4Model(cfg, device="meta")
        ffn_names = [
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_gate_inp.weight",
            "blk.0.ffn_gate_shexp.weight", "blk.0.ffn_up_shexp.weight", "blk.0.ffn_down_shexp.weight",
        ]
        model.load_weights(g, device="cuda", dtype=torch.bfloat16, names=ffn_names)

        def expert_load(e_id: int):
            gate = torch.randn(cfg.n_embd, cfg.ff_exp, device="cuda", dtype=torch.bfloat16) * 0.01
            up = torch.randn(cfg.n_embd, cfg.ff_exp, device="cuda", dtype=torch.bfloat16) * 0.01
            down = torch.randn(cfg.ff_exp, cfg.n_embd, device="cuda", dtype=torch.bfloat16) * 0.01
            return gate, up, down

    x = torch.randn(3, cfg.n_embd, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        out = model.blocks[0].ffn.forward(
            x, cfg.expert_weights_scale, float(cfg.swiglu_clamp_exp[0]),
            expert_load_fn=expert_load,
        )

    assert out.shape == (3, cfg.n_embd)
    assert torch.isfinite(out).all()
    assert out.std().item() > 0.0


@pytest.mark.skipif(
    os.environ.get("PYDS4_RUN_M8E") != "1",
    reason="set PYDS4_RUN_M8E=1 to run the expensive full-model M8 test",
)
@gguf_available
def test_forward_m8e_full_20_tokens() -> None:
    """M8e: full 43-layer forward with 20-token prompt.

    Loads all non-expert weights (~20 GB bf16 GPU). Expert 3D tensors are
    streamed via lazy dequant from the GGUF mmap, one expert at a time.
    On GB10 the verified full forward takes about 4.5 minutes after weight
    loading, so this remains opt-in.

    Run manually with:
      PYDS4_RUN_M8E=1 pytest tests/test_model.py::test_forward_m8e_full_20_tokens -s
    """
    with gguf.parse(GGUF_PATH) as g:
        cfg = DS4Config.from_gguf(g)
        model = DS4Model(cfg, device="meta")

        # Load all non-expert weights
        name_map = model.gguf_name_map()
        non_expert = sorted(
            n for n in name_map
            if not any(kw in n for kw in ("gate_exps", "up_exps", "down_exps"))
        )
        print(f"Loading {len(non_expert)} non-expert tensors (~20 GB bf16)...")
        model.load_weights(g, device="cuda", dtype=torch.bfloat16, names=non_expert)

        # Build per-layer expert loaders from GGUF
        from pyds4.model import _make_expert_loader
        loaders: dict[int, callable] = {}
        for il in range(cfg.n_layer):
            prefix = f"blk.{il}"
            loaders[il] = _make_expert_loader(
                g, prefix, cfg.n_expert, cfg.n_embd, cfg.ff_exp,
                device="cuda", dtype=torch.bfloat16, cache_size=1,
            )

        def expert_load_fn(layer_idx: int, expert_id: int):
            return loaders[layer_idx](expert_id)

        model.eval()
        # Fixed 20-token prompt assembled from the tokenizer parity corpus.
        tokens = torch.tensor([
            19923, 14, 2058, 16, 262, 688, 1840, 40374, 11, 680,
            2161, 58810, 1393, 3465, 52735, 4042, 2605, 6895, 1883, 30594,
        ], device="cuda", dtype=torch.long)

        import time
        t0 = time.time()
        with torch.no_grad():
            logits = model(tokens, expert_load_fn=expert_load_fn)
        elapsed = time.time() - t0
        print(f"Forward completed in {elapsed:.1f}s")

    assert logits.shape == (20, cfg.vocab_size)
    assert torch.isfinite(logits).all()

    # Plausibility: distribution shouldn't be degenerate
    probs = torch.softmax(logits, dim=-1)
    top_prob = probs.max(dim=-1).values
    assert (top_prob < 0.999).all(), "distribution too peaked"

    # At least 2 distinct argmax tokens
    argmax = probs.argmax(dim=-1)
    print(f"Distinct argmax tokens: {argmax.unique().shape[0]}")
    assert argmax.unique().shape[0] >= 2
