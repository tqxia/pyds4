#!/usr/bin/env python3
"""Capture ds4 --dump-tokens output as the M3 tokenizer oracle.

Each test string is fed to `./ds4 --dump-tokens` prefixed with BOS, which
takes the `is_rendered_chat_prompt` path in ds4_cli.c (line 334) — that path
just BPEs the text verbatim and inserts specials by literal match. The first
token in the output is always BOS=0 and is stripped from the saved oracle.

Run from the repo root: scripts/capture_tokenizer_oracle.py
Output: tests/data/tokenizer_oracle.json
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

DS4_BIN = Path("/home/tqxia/workspace/ds4/ds4")
DS4_CWD = Path("/home/tqxia/workspace/ds4")
BOS = "<｜begin▁of▁sentence｜>"

# A spread of cases that have historically broken naive BPE/pretokenizer
# implementations. We avoid newlines in args (shell-fragile) and use \t to
# keep things one-line-per-case in the source.
CASES: list[str] = [
    "Hello, world.",
    "Hello",
    "   int main(void) {",
    "self.method()",
    "def foo(x):",
    "12345",                       # digit run >3, exercises 3-digit cap
    "123 456 7890",                # space-separated digit runs
    "...!!! ???",                  # punctuation runs
    "café",                        # unicode latin accent
    "你好世界",                    # CJK ideographs
    "こんにちは",                  # hiragana
    "ｱｲｳｴｵ",                       # half-width katakana
    "a\tb\tc",                     # tabs
    "foo bar baz",                 # normal words with single spaces
    "    spaced",                  # leading runs of spaces
    "key=value",                   # punctuation in the middle of letters
    "snake_case_var",              # underscore
    "｜DSML｜",                    # the DSML literal (special token)
    "<think>think</think>",        # think markers (specials)
    "<｜User｜>hi<｜Assistant｜>",   # multiple specials wrapping text
]


def run_ds4(prompt: str) -> list[int]:
    """Invoke ./ds4 --dump-tokens and parse the first line `[id, id, ...]`."""
    r = subprocess.run(
        [str(DS4_BIN), "--dump-tokens", "-p", BOS + prompt],
        cwd=str(DS4_CWD),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    first = r.stdout.splitlines()[0].strip()
    m = re.fullmatch(r"\[([^\]]*)\]", first)
    if not m:
        raise RuntimeError(f"Unexpected ds4 stdout: {r.stdout!r}")
    ids = [int(x) for x in m.group(1).split(",")] if m.group(1).strip() else []
    if ids[0] != 0:
        raise RuntimeError(f"Expected BOS=0 at index 0, got {ids[:5]}")
    return ids[1:]  # strip the BOS we prepended


def main() -> None:
    out: dict[str, list[int]] = {}
    for s in CASES:
        ids = run_ds4(s)
        out[s] = ids
        print(f"{len(ids):4d} tokens  {s!r}")
    dest = Path(__file__).resolve().parent.parent / "tests" / "data" / "tokenizer_oracle.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
