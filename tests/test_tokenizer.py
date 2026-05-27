"""M3 parity tests for the tokenizer.

Oracle: 20 prompts captured from `./ds4 --dump-tokens` with a leading BOS
(see scripts/capture_tokenizer_oracle.py). The BOS prefix forces ds4 into the
`is_rendered_chat_prompt` path, which BPEs text and matches our 7 special
tokens by literal substring — same path `Tokenizer.encode()` implements here.

The oracle file lives at tests/data/tokenizer_oracle.json and was generated
against ds4flash.gguf. Both the oracle and the live ds4 binary are required
for these tests; we skip cleanly if either is missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyds4 import gguf, tokenizer

GGUF_PATH = Path("/home/tqxia/workspace/ds4/ds4flash.gguf")
ORACLE_PATH = Path(__file__).parent / "data" / "tokenizer_oracle.json"


pytestmark = pytest.mark.skipif(
    not (GGUF_PATH.exists() and ORACLE_PATH.exists()),
    reason="GGUF and/or captured tokenizer oracle not available",
)


@pytest.fixture(scope="module")
def tok() -> tokenizer.Tokenizer:
    with gguf.parse(GGUF_PATH) as g:
        return tokenizer.Tokenizer.from_gguf(g)


@pytest.fixture(scope="module")
def oracle() -> dict[str, list[int]]:
    return json.loads(ORACLE_PATH.read_text(encoding="utf-8"))


def test_vocab_loaded(tok: tokenizer.Tokenizer) -> None:
    assert tok.n_vocab == 129280
    # Special-token IDs match the values printed by ds4 with --dump-tokens.
    # We don't hard-code their numeric ids here — the byte-equal parity test
    # below will catch any drift in a single failed assertion.
    assert isinstance(tok.bos_id, int) and tok.bos_id >= 0
    assert tok.eos_id != tok.bos_id


@pytest.mark.parametrize(
    "text", json.loads(ORACLE_PATH.read_text(encoding="utf-8")).keys()
)
def test_encode_parity(tok: tokenizer.Tokenizer, oracle: dict[str, list[int]],
                       text: str) -> None:
    """Byte-equal token IDs against the ds4 oracle."""
    assert tok.encode(text) == oracle[text]


def test_encode_raw_skips_specials(tok: tokenizer.Tokenizer) -> None:
    """`encode_raw` BPEs the literal text — no special-token interpretation.

    The DSML literal "｜DSML｜" routes to a single special ID via `encode`, but
    `encode_raw` must split it into ordinary BPE tokens (the literal bytes are
    not in the special-token list at this entry point).
    """
    raw_ids = tok.encode_raw("｜DSML｜")
    cooked_ids = tok.encode("｜DSML｜")
    assert cooked_ids == [tok.dsml_id]
    assert raw_ids != cooked_ids
    # The raw path also has at least one token.
    assert len(raw_ids) >= 1


def test_special_ids_distinct(tok: tokenizer.Tokenizer) -> None:
    ids = [
        tok.bos_id, tok.eos_id, tok.user_id, tok.assistant_id,
        tok.think_start_id, tok.think_end_id, tok.dsml_id,
    ]
    assert len(set(ids)) == len(ids)


def test_byte_codepoint_map_is_bijective() -> None:
    """The 256 raw bytes map to 256 distinct codepoints (the GPT-2 invariant).
    Catches off-by-one mistakes in the printable-byte range table."""
    bt = tokenizer._BYTE_TO_CP
    assert len(bt) == 256
    assert len(set(bt)) == 256
    # The "unprintable" bytes land in 0x100..0x143 (68 entries).
    high = [c for c in bt if c >= 256]
    assert len(high) == 68
    assert sorted(high) == list(range(0x100, 0x100 + 68))
