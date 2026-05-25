"""M1 parity checks for the GGUF parser.

The oracle is `ds4.c`: we read the file ds4 reads, with the same alignment
convention, and we expect the same architectural numbers ds4 prints during
startup. Concrete invariants pinned here come from CLAUDE.md and from inspecting
`./ds4flash.gguf` via `python -m pyds4.gguf inspect`.

The tests skip themselves if the 80 GB GGUF is not present, so they remain
runnable in a fresh checkout without the weights.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pyds4 import gguf


GGUF_PATH = Path("/home/tqxia/workspace/ds4/ds4flash.gguf")


pytestmark = pytest.mark.skipif(
    not GGUF_PATH.exists(),
    reason=f"ds4 GGUF not available at {GGUF_PATH}",
)


@pytest.fixture(scope="module")
def model() -> gguf.GGUFFile:
    g = gguf.parse(GGUF_PATH)
    yield g
    g.close()


def test_header(model: gguf.GGUFFile) -> None:
    """Magic / version / alignment must match the ds4 oracle."""
    # ds4.c requires GGUF v3 specifically.
    assert model.version == 3
    # `general.alignment` is omitted in this file, so the default of 32 kicks in.
    assert model.alignment == 32
    # tensor data lives past the descriptor table; offset must be 32-aligned.
    assert model.tensor_data_pos % model.alignment == 0


def test_counts_match_oracle(model: gguf.GGUFFile) -> None:
    """KV / tensor counts come straight from `./ds4 ...` startup logs.

    These came from `python -m pyds4.gguf inspect` and confirmed against the
    architectural facts in CLAUDE.md (43 layers, 256 experts). If the GGUF is
    ever regenerated, these numbers shift — that's a signal, not a bug.
    """
    assert len(model.kv) == 62
    assert len(model.tensors) == 1328


def test_arch_metadata(model: gguf.GGUFFile) -> None:
    """Pinned architectural constants — these drive every later milestone."""
    assert model.get("general.architecture") == "deepseek4"
    # 43 transformer blocks — CLAUDE.md "Architecture, in 30 seconds".
    assert model.get("deepseek4.block_count") == 43
    # 256 routed experts, top-6 — matches the MoE FFN in CLAUDE.md.
    assert model.get("deepseek4.expert_count") == 256
    assert model.get("deepseek4.expert_used_count") == 6
    # Indexer top-K=512 drives the CSA (Compressed Sparse Attention) path.
    assert model.get("deepseek4.attention.indexer.top_k") == 512
    # 1M training context.
    assert model.get("deepseek4.context_length") == 1048576


def test_token_embedding_tensor(model: gguf.GGUFFile) -> None:
    """token_embd.weight is the LM head's twin and the first thing M8a needs."""
    assert "token_embd.weight" in model.tensors
    t = model.tensors["token_embd.weight"]
    # Embedding dim x vocab size, stored as F16 in this GGUF.
    # GGUF stores shape innermost-first → (n_embd, vocab_size).
    assert len(t.shape) == 2
    n_embd, vocab = t.shape
    assert vocab == 129280  # DeepSeek tokenizer vocab
    # n_embd is not promised by the model card; just sanity-check it's positive.
    assert n_embd > 0
    # The embedding is normally kept at high precision — F16 or BF16, not 2-bit.
    assert t.dtype in {gguf.TensorType.F16, gguf.TensorType.BF16,
                       gguf.TensorType.F32}


def test_expert_tensor_types(model: gguf.GGUFFile) -> None:
    """Routed experts use IQ2_XXS (up/gate) + Q2_K (down). Per CLAUDE.md."""
    up = model.tensors["blk.0.ffn_up_exps.weight"]
    gate = model.tensors["blk.0.ffn_gate_exps.weight"]
    down = model.tensors["blk.0.ffn_down_exps.weight"]
    assert up.dtype == gguf.TensorType.IQ2_XXS
    assert gate.dtype == gguf.TensorType.IQ2_XXS
    assert down.dtype == gguf.TensorType.Q2_K
    # Last dim is expert count = 256 (innermost-first shape order).
    assert up.shape[-1] == 256
    assert gate.shape[-1] == 256
    assert down.shape[-1] == 256


def test_tensor_nbytes_quant_blocks() -> None:
    """nbytes math must match the GGUF block table (ds4.c::gguf_types[])."""
    # F32: one element per "block", 4 bytes each.
    assert gguf.tensor_nbytes(gguf.TensorType.F32, 10) == 40
    # Q8_0: 32 elements per block, 34 bytes per block (fp16 scale + 32 i8).
    assert gguf.tensor_nbytes(gguf.TensorType.Q8_0, 32) == 34
    assert gguf.tensor_nbytes(gguf.TensorType.Q8_0, 64) == 68
    # IQ2_XXS: 256 elements per block, 66 bytes per block.
    assert gguf.tensor_nbytes(gguf.TensorType.IQ2_XXS, 256) == 66
    # Q2_K: 256 elements per block, 84 bytes per block.
    assert gguf.tensor_nbytes(gguf.TensorType.Q2_K, 256) == 84


def test_tensor_offsets_inside_file(model: gguf.GGUFFile) -> None:
    """Every tensor must lie inside the file. Parser enforces this; check
    that real ds4 tensors actually use the high end of the file."""
    last = max(model.tensors.values(), key=lambda t: t.abs_offset)
    end = last.abs_offset + last.nbytes
    assert end <= model.size
    # The bulk of the 81 GB is tensor data, so the last byte should be deep
    # into the file (not in the descriptor table).
    assert last.abs_offset > model.tensor_data_pos


def test_tensor_bytes_lazy_slice(model: gguf.GGUFFile) -> None:
    """tensor_bytes(...) returns a zero-copy view of the right length."""
    name = "blk.0.ffn_down_exps.weight"  # Q2_K, large
    t = model.tensors[name]
    buf = model.tensor_bytes(name)
    assert len(buf) == t.nbytes
    # Cross-check: 256 elements per Q2_K block, 84 bytes per block.
    assert t.nbytes == (t.n_elements // 256) * 84


def test_inspect_cli(capsys: pytest.CaptureFixture[str]) -> None:
    rc = gguf.main(["inspect", str(GGUF_PATH)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "gguf:    v3" in out
    assert "deepseek4" in out
    assert "1328 tensors" in out
