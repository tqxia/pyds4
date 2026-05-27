"""GPT-2 byte-level BPE tokenizer with ds4's JoyAI pre-tokenization.

DeepSeek V4 Flash uses the standard GPT-2 byte-level BPE: arbitrary bytes are
losslessly mapped to printable Unicode codepoints, merges run on the resulting
string, and the final symbols are looked up in the vocab.

The trick is the pre-tokenizer. The GGUF declares `tokenizer.ggml.pre =
"joyai-llm"`, a hand-rolled split that emits pieces shaped to play well with
the published merges. The shape of those pieces matters: different splits
produce different merge sequences even when the raw text is identical.

We mirror `bpe_tokenize_text()` in ds4.c (lines 15041-15108) verbatim, byte
for byte. The pre-tokenizer operates on bytes (not codepoints), so that
multi-byte UTF-8 chars are consumed by `next_utf8_char` rather than indexed
character-by-character.

Encode-only API for M3. Decode and chat-template handling come later.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import gguf


# --- GPT-2 byte ↔ codepoint mapping ------------------------------------------
#
# The standard GPT-2 byte encoder: 188 "printable" bytes map to themselves;
# the other 68 bytes (0..32, 127..160, 173) are pushed up to U+0100..U+0143.
# This keeps the BPE vocabulary text-friendly while still being 1:1 with raw
# bytes. Mirrors ds4.c::gpt2_byte_to_codepoint (line 14797).
def _build_byte_codepoint_maps() -> tuple[list[int], dict[int, int]]:
    printable = (
        list(range(33, 127))      # 33..126 (94 codepoints)
        + list(range(161, 173))   # 161..172 (12)
        + list(range(174, 256))   # 174..255 (82)
    )
    bs: list[int] = list(printable)
    cs: list[int] = list(printable)
    n = 0
    for b in range(256):
        if b not in printable:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    # byte → codepoint
    byte_to_cp = [0] * 256
    for b, c in zip(bs, cs):
        byte_to_cp[b] = c
    # codepoint → byte (needed for decode later; cheap to build now)
    cp_to_byte = {c: b for b, c in zip(bs, cs)}
    return byte_to_cp, cp_to_byte


_BYTE_TO_CP, _CP_TO_BYTE = _build_byte_codepoint_maps()


def _byte_encode(piece: bytes) -> str:
    """Map raw bytes → GPT-2 printable-codepoint string."""
    return "".join(chr(_BYTE_TO_CP[b]) for b in piece)


# --- JoyAI pre-tokenizer (operates on bytes) ---------------------------------


def _utf8_len(first_byte: int) -> int:
    if first_byte < 0x80:
        return 1
    if (first_byte & 0xe0) == 0xc0:
        return 2
    if (first_byte & 0xf0) == 0xe0:
        return 3
    if (first_byte & 0xf8) == 0xf0:
        return 4
    return 1


def _next_utf8_char(text: bytes, pos: int) -> int:
    n = _utf8_len(text[pos])
    if pos + n > len(text):
        n = 1
    return pos + n


def _ascii_alpha(c: int) -> bool:
    return (0x41 <= c <= 0x5a) or (0x61 <= c <= 0x7a)


def _ascii_digit(c: int) -> bool:
    return 0x30 <= c <= 0x39


def _ascii_space(c: int) -> bool:
    # ' ', '\t', '\n', '\r', '\v', '\f'
    return c in (0x20, 0x09, 0x0a, 0x0d, 0x0b, 0x0c)


def _ascii_newline(c: int) -> bool:
    return c in (0x0a, 0x0d)


def _joyai_ascii_punct_symbol(c: int) -> bool:
    """Matches ds4.c::joyai_ascii_punct_symbol — the four ASCII punct ranges."""
    return (
        (0x21 <= c <= 0x2f) or  # ! " # $ % & ' ( ) * + , - . /
        (0x3a <= c <= 0x40) or  # : ; < = > ? @
        (0x5b <= c <= 0x60) or  # [ \ ] ^ _ `
        (0x7b <= c <= 0x7e)     # { | } ~
    )


def _joyai_letter_like_at(text: bytes, pos: int) -> bool:
    """ASCII alpha OR any non-ASCII byte. Mirrors ds4.c::joyai_letter_like_at.

    Non-ASCII bytes are treated as letters here; CJK/hiragana/katakana ranges
    are isolated *before* this rule fires by the dedicated branch above.
    """
    b = text[pos]
    if b < 128:
        return _ascii_alpha(b)
    return True


def _joyai_consume_letters(text: bytes, pos: int) -> int:
    n = len(text)
    while pos < n and _joyai_letter_like_at(text, pos):
        pos = _next_utf8_char(text, pos)
    return pos


def _utf8_decode_codepoint(text: bytes, pos: int) -> int:
    """Decode the codepoint starting at pos (no validation; trust the input)."""
    c0 = text[pos]
    n = _utf8_len(c0)
    if pos + n > len(text):
        n = 1
    if n == 1:
        return c0
    if n == 2:
        return ((c0 & 0x1f) << 6) | (text[pos + 1] & 0x3f)
    if n == 3:
        return (
            ((c0 & 0x0f) << 12)
            | ((text[pos + 1] & 0x3f) << 6)
            | (text[pos + 2] & 0x3f)
        )
    return (
        ((c0 & 0x07) << 18)
        | ((text[pos + 1] & 0x3f) << 12)
        | ((text[pos + 2] & 0x3f) << 6)
        | (text[pos + 3] & 0x3f)
    )


def _utf8_is_cjk_hira_kata(cp: int) -> bool:
    return (
        (0x4e00 <= cp <= 0x9fa5)  # CJK ideographs
        or (0x3040 <= cp <= 0x309f)  # Hiragana
        or (0x30a0 <= cp <= 0x30ff)  # Katakana
    )


def _joyai_cjk_at(text: bytes, pos: int) -> bool:
    if text[pos] < 128:
        return False
    cp = _utf8_decode_codepoint(text, pos)
    return _utf8_is_cjk_hira_kata(cp)


def _pretokenize(text: bytes) -> list[bytes]:
    """JoyAI pre-tokenization. Yields raw byte pieces.

    Direct port of ds4.c::bpe_tokenize_text (lines 15041-15108). Branch order
    is load-bearing — re-ordering breaks token stream parity.
    """
    n = len(text)
    pieces: list[bytes] = []
    pos = 0
    while pos < n:
        start = pos
        c = text[pos]

        # 1. up to 3 ASCII digits.
        if _ascii_digit(c):
            ndigits = 0
            while pos < n and _ascii_digit(text[pos]) and ndigits < 3:
                pos += 1
                ndigits += 1

        # 2. CJK / Hiragana / Katakana run.
        elif _joyai_cjk_at(text, pos):
            while True:
                pos = _next_utf8_char(text, pos)
                if pos >= n or not _joyai_cjk_at(text, pos):
                    break

        # 3. punct followed by ASCII alpha: take the punct + alpha run.
        elif (
            _joyai_ascii_punct_symbol(c)
            and pos + 1 < n
            and _ascii_alpha(text[pos + 1])
        ):
            pos += 1
            while pos < n and _ascii_alpha(text[pos]):
                pos += 1

        # 4. letter-like run.
        elif _joyai_letter_like_at(text, pos):
            pos = _joyai_consume_letters(text, pos)

        # 5. one non-newline non-punct byte followed by letter-like. This is
        # what lets a single leading space attach to the following word
        # ("    int" → "   ", " int").
        elif (
            not _ascii_newline(c)
            and not _joyai_ascii_punct_symbol(c)
            and pos + 1 < n
            and _joyai_letter_like_at(text, pos + 1)
        ):
            pos += 1
            pos = _joyai_consume_letters(text, pos)

        # 6. " " followed by punct: consume the space + punct run + trailing
        # newlines (so " {\n" stays one piece).
        elif (
            c == 0x20
            and pos + 1 < n
            and _joyai_ascii_punct_symbol(text[pos + 1])
        ):
            pos += 1
            while pos < n and _joyai_ascii_punct_symbol(text[pos]):
                pos += 1
            while pos < n and _ascii_newline(text[pos]):
                pos += 1

        # 7. bare punct run (+ trailing newlines).
        elif _joyai_ascii_punct_symbol(c):
            while pos < n and _joyai_ascii_punct_symbol(text[pos]):
                pos += 1
            while pos < n and _ascii_newline(text[pos]):
                pos += 1

        # 8. whitespace run. Newlines snap to end-of-last-newline; otherwise
        # a single trailing space is reserved for the next word piece.
        elif _ascii_space(c):
            p = pos
            last_newline_end = 0
            while p < n and _ascii_space(text[p]):
                if _ascii_newline(text[p]):
                    last_newline_end = p + 1
                p += 1
            if last_newline_end:
                pos = last_newline_end
            elif (
                p < n
                and p > pos + 1
                and (
                    _joyai_letter_like_at(text, p)
                    or _joyai_ascii_punct_symbol(text[p])
                )
            ):
                pos = p - 1
            else:
                pos = p

        # 9. fallback: skip one UTF-8 char.
        else:
            pos = _next_utf8_char(text, pos)

        if pos == start:
            pos = _next_utf8_char(text, pos)

        pieces.append(text[start:pos])
    return pieces


# --- Tokenizer ---------------------------------------------------------------


# Specials are matched literally during the rendered-chat scan. Order doesn't
# matter — none of these are prefixes of another — but we keep ds4's list.
# Mirrors `special_token_at()` in ds4.c (line 15200).
_SPECIAL_TEXTS = (
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<think>",
    "</think>",
    "｜DSML｜",
)


@dataclass
class Tokenizer:
    """Loaded vocab + merges + special-token IDs."""

    vocab: list[str]                       # id → token string (codepoint-encoded)
    token_to_id: dict[str, int]
    merge_rank: dict[tuple[str, str], int]  # (left, right) → rank (lower wins)
    bos_id: int
    eos_id: int
    user_id: int
    assistant_id: int
    think_start_id: int
    think_end_id: int
    dsml_id: int

    # --- factory ------------------------------------------------------------

    @classmethod
    def from_gguf(cls, g: gguf.GGUFFile) -> "Tokenizer":
        tokens = g.array("tokenizer.ggml.tokens")
        merges = g.array("tokenizer.ggml.merges")
        token_to_id = {t: i for i, t in enumerate(tokens)}

        merge_rank: dict[tuple[str, str], int] = {}
        for rank, m in enumerate(merges):
            # GGUF merge entries are "left right" with a single space sep.
            left, _, right = m.partition(" ")
            merge_rank[(left, right)] = rank

        def need(text: str) -> int:
            tok = token_to_id.get(text)
            if tok is None:
                raise KeyError(f"required tokenizer entry missing: {text!r}")
            return tok

        return cls(
            vocab=list(tokens),
            token_to_id=token_to_id,
            merge_rank=merge_rank,
            bos_id=need("<｜begin▁of▁sentence｜>"),
            eos_id=need("<｜end▁of▁sentence｜>"),
            user_id=need("<｜User｜>"),
            assistant_id=need("<｜Assistant｜>"),
            think_start_id=need("<think>"),
            think_end_id=need("</think>"),
            dsml_id=need("｜DSML｜"),
        )

    @property
    def n_vocab(self) -> int:
        return len(self.vocab)

    # --- BPE core -----------------------------------------------------------

    def _bpe_piece(self, piece: bytes) -> list[int]:
        """Apply byte-level BPE to one pre-tokenized piece, emit token IDs.

        Mirrors ds4.c::bpe_emit_piece (line 14866). Symbols start as one-
        codepoint strings and get merged greedily by lowest merge rank.
        """
        # Each codepoint after byte-encoding becomes one "symbol".
        symbols: list[str] = [chr(_BYTE_TO_CP[b]) for b in piece]
        if not symbols:
            return []

        rank = self.merge_rank
        while len(symbols) >= 2:
            best_i = -1
            best_rank = -1
            for i in range(len(symbols) - 1):
                r = rank.get((symbols[i], symbols[i + 1]))
                if r is not None and (best_i < 0 or r < best_rank):
                    best_rank = r
                    best_i = i
            if best_i < 0:
                break
            symbols[best_i : best_i + 2] = [symbols[best_i] + symbols[best_i + 1]]

        # Look each merged symbol up in the vocab; fall back to per-codepoint
        # bytes if the merged form isn't a known token (rare for byte-level BPE,
        # since every single byte has a vocab entry).
        ids: list[int] = []
        for sym in symbols:
            tok = self.token_to_id.get(sym)
            if tok is not None:
                ids.append(tok)
            else:
                for ch in sym:
                    tok2 = self.token_to_id.get(ch)
                    if tok2 is not None:
                        ids.append(tok2)
        return ids

    # --- public encode ------------------------------------------------------

    def encode_raw(self, text: str) -> list[int]:
        """Pure JoyAI BPE on `text` — no special-token interpretation.

        A literal `<｜DSML｜>` in `text` is BPEd as ordinary characters here.
        Mirrors ds4.c::bpe_tokenize_text.
        """
        out: list[int] = []
        data = text.encode("utf-8")
        for piece in _pretokenize(data):
            out.extend(self._bpe_piece(piece))
        return out

    def encode(self, text: str) -> list[int]:
        """Encode text the way ds4's rendered-chat path does: scan for special
        tokens (BOS, User, Assistant, think markers, DSML) and emit them by ID
        literally; BPE everything in between.

        Mirrors ds4.c::tokenize_rendered_chat_vocab (line 15234).
        """
        specials: list[tuple[str, int]] = [
            ("<｜begin▁of▁sentence｜>", self.bos_id),
            ("<｜end▁of▁sentence｜>", self.eos_id),
            ("<｜User｜>", self.user_id),
            ("<｜Assistant｜>", self.assistant_id),
            ("<think>", self.think_start_id),
            ("</think>", self.think_end_id),
            ("｜DSML｜", self.dsml_id),
        ]
        # Scan in code points (not bytes) for simpler index math; the C version
        # walks byte-by-byte but the match semantics are identical.
        out: list[int] = []
        span_start = 0
        i = 0
        while i < len(text):
            matched = None
            for s, tok in specials:
                if text.startswith(s, i):
                    matched = (s, tok)
                    break
            if matched is not None:
                if i > span_start:
                    out.extend(self.encode_raw(text[span_start:i]))
                out.append(matched[1])
                i += len(matched[0])
                span_start = i
            else:
                i += 1
        if span_start < len(text):
            out.extend(self.encode_raw(text[span_start:]))
        return out
