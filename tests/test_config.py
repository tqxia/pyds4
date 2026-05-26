"""M2 parity checks for DS4Config.

We parse the real ds4flash.gguf and compare every field against the layout
constants pinned in `ds4.c` lines 87-116 (the `DS4_N_*` enum). Those constants
are ds4's own internal contract: if the GGUF doesn't match them, ds4 itself
refuses to load. So matching them is a strong correctness signal for M2.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyds4 import config, gguf


GGUF_PATH = Path("/home/tqxia/workspace/ds4/ds4flash.gguf")

pytestmark = pytest.mark.skipif(
    not GGUF_PATH.exists(),
    reason=f"ds4 GGUF not available at {GGUF_PATH}",
)


@pytest.fixture(scope="module")
def cfg() -> config.DS4Config:
    with gguf.parse(GGUF_PATH) as g:
        c = config.DS4Config.from_gguf(g)
    c.validate()
    return c


def test_matches_ds4_constants(cfg: config.DS4Config) -> None:
    """Every integer field must match the DS4_N_* enum in ds4.c."""
    for key, expected in config.DS4_EXPECTED.items():
        assert getattr(cfg, key) == expected, (
            f"DS4Config.{key} = {getattr(cfg, key)}, expected {expected}"
        )


def test_arch_and_context(cfg: config.DS4Config) -> None:
    assert cfg.arch == "deepseek4"
    # 1M training context. The model card and CLAUDE.md both quote this.
    assert cfg.max_position == 1_048_576


def test_floats_and_epsilons(cfg: config.DS4Config) -> None:
    """Float constants line up with ds4.c::DS4_* defines (~lines 53-62)."""
    assert cfg.rms_eps == pytest.approx(1e-6, rel=1e-3)
    assert cfg.hc_eps == pytest.approx(1e-6, rel=1e-3)
    assert cfg.rope_freq_base == pytest.approx(10000.0)
    assert cfg.compress_rope_freq_base == pytest.approx(160000.0)
    assert cfg.yarn_factor == pytest.approx(16.0)
    assert cfg.yarn_beta_fast == pytest.approx(32.0)
    assert cfg.yarn_beta_slow == pytest.approx(1.0)
    assert cfg.yarn_orig_ctx == 65536
    assert cfg.expert_weights_scale == pytest.approx(1.5)
    assert cfg.expert_weights_norm is True


def test_compress_ratios_pattern(cfg: config.DS4Config) -> None:
    """Per-layer compress ratios must match ds4.c::ds4_layer_compress_ratio.

    Layers 0-1 are dense (ratio=0). After that, even layers use ratio 4 and odd
    layers use ratio 128 (HCA ratio-128 has no indexer; ratio-4 does).
    """
    # GGUF carries n_layer + 1 entries; trailing one is for the output head.
    assert len(cfg.compress_ratios) == cfg.n_layer + 1

    assert cfg.layer_compress_ratio(0) == 0
    assert cfg.layer_compress_ratio(1) == 0
    for il in range(2, cfg.n_layer):
        expected = 4 if (il & 1) == 0 else 128
        assert cfg.layer_compress_ratio(il) == expected, (
            f"layer {il}: got {cfg.layer_compress_ratio(il)}, "
            f"expected {expected}"
        )


def test_swiglu_clamp_per_layer(cfg: config.DS4Config) -> None:
    """SwiGLU clamp exponent: ds4 has 10.0 as the C default for all layers."""
    assert len(cfg.swiglu_clamp_exp) == cfg.n_layer
    for il, v in enumerate(cfg.swiglu_clamp_exp):
        assert v == pytest.approx(10.0), f"layer {il}: clamp={v}"


def test_validate_rejects_wrong_layer_count() -> None:
    """If a field disagrees with ds4's layout, validate() must complain."""
    with gguf.parse(GGUF_PATH) as g:
        c = config.DS4Config.from_gguf(g)
    bad = dataclass_replace(c, n_layer=42)
    with pytest.raises(ValueError, match="n_layer"):
        bad.validate()


def dataclass_replace(c, **kwargs):
    from dataclasses import replace
    return replace(c, **kwargs)


def test_summary_runs(cfg: config.DS4Config) -> None:
    text = cfg.summary()
    # Spot-check a few load-bearing values appear.
    assert "deepseek4" in text
    assert "layers:    43" in text
    assert "indexer:   heads=64 head_dim=128 top_k=512" in text
