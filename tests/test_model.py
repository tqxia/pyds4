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

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pyds4 import gguf  # noqa: E402
from pyds4.config import DS4Config  # noqa: E402
from pyds4.model import DS4Model, build_name_map  # noqa: E402


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
