#!/usr/bin/env python3
"""Capture an IQ2_XXS dequant oracle for the M5 parity test.

Picks a slice of `blk.0.ffn_gate_exps.weight` (an IQ2_XXS tensor — these are
the routed gate/up matrices). We grab the first 256 blocks (256 * 256 = 65536
elements, 256 KiB fp32 output) and pipe the raw bytes through a C oracle
(`scripts/iq2_xxs_oracle.c`) that re-implements ds4.c's tables and dequant
formula verbatim.

Saves:
  tests/data/iq2_xxs_input.bin     raw IQ2_XXS bytes (256 * 66 = 16896 bytes)
  tests/data/iq2_xxs_expected.bin  fp32 reference (256 * 256 * 4 = 256 KiB)
  tests/data/iq2_xxs_meta.json     tensor name + n_blocks for documentation

Run from the repo root: scripts/capture_iq2_xxs_oracle.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from pyds4 import gguf

REPO = Path(__file__).resolve().parent.parent
GGUF_PATH = Path("/home/tqxia/workspace/ds4/ds4flash.gguf")
TENSOR_NAME = "blk.0.ffn_gate_exps.weight"
N_BLOCKS = 256                       # 65536 elements, 256 KiB fp32 output


def build_oracle(src: Path, dst: Path) -> None:
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        raise RuntimeError("no C compiler on PATH (need cc or gcc)")
    subprocess.run([cc, "-O2", "-std=c11", "-o", str(dst), str(src)], check=True)


def main() -> None:
    data_dir = REPO / "tests" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    input_bin = data_dir / "iq2_xxs_input.bin"
    expected_bin = data_dir / "iq2_xxs_expected.bin"
    meta_json = data_dir / "iq2_xxs_meta.json"

    oracle_src = REPO / "scripts" / "iq2_xxs_oracle.c"
    oracle_bin = REPO / "scripts" / "iq2_xxs_oracle"
    build_oracle(oracle_src, oracle_bin)

    with gguf.parse(GGUF_PATH) as g:
        t = g.tensors[TENSOR_NAME]
        if t.dtype != 16:  # IQ2_XXS
            raise RuntimeError(f"{TENSOR_NAME} is dtype {t.dtype}, expected IQ2_XXS=16")
        raw = bytes(g.tensor_bytes(TENSOR_NAME)[: N_BLOCKS * 66])

    input_bin.write_bytes(raw)
    subprocess.run(
        [str(oracle_bin), str(input_bin), str(expected_bin), str(N_BLOCKS)],
        check=True,
    )
    meta_json.write_text(json.dumps(
        {"tensor": TENSOR_NAME, "n_blocks": N_BLOCKS, "block_elems": 256,
         "block_bytes": 66, "n_elements": N_BLOCKS * 256},
        indent=2,
    ), encoding="utf-8")
    print(f"wrote {input_bin}     ({input_bin.stat().st_size} bytes)")
    print(f"wrote {expected_bin}  ({expected_bin.stat().st_size} bytes)")
    print(f"wrote {meta_json}")


if __name__ == "__main__":
    main()
