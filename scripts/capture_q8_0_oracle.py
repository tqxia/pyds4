#!/usr/bin/env python3
"""Capture a Q8_0 dequant oracle for the M4 parity test.

Picks a slice of `blk.0.attn_kv.weight` from ds4flash.gguf (a Q8_0 tensor,
first 256 blocks = 8192 elements). Saves:

  tests/data/q8_0_input.bin     raw quant bytes (256 * 34 = 8704 bytes)
  tests/data/q8_0_expected.bin  fp32 dequanted reference (256 * 32 * 4 = 32 KiB)
  tests/data/q8_0_meta.json     tensor name + n_blocks for documentation

We pipe the raw bytes through a tiny C program (`scripts/q8_0_oracle.c`) that
reimplements ds4.c::f16_to_f32 plus the dequant formula. The C oracle is the
parity reference; our NumPy `dequant_q8_0` must produce bit-identical fp32
bytes against the same input.

Run from the repo root: scripts/capture_q8_0_oracle.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from pyds4 import gguf

REPO = Path(__file__).resolve().parent.parent
GGUF_PATH = Path("/home/tqxia/workspace/ds4/ds4flash.gguf")
TENSOR_NAME = "blk.0.attn_kv.weight"
N_BLOCKS = 256                       # 8192 elements, 32 KiB fp32 output


def build_oracle(src: Path, dst: Path) -> None:
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        raise RuntimeError("no C compiler on PATH (need cc or gcc)")
    subprocess.run([cc, "-O2", "-std=c11", "-o", str(dst), str(src)], check=True)


def main() -> None:
    data_dir = REPO / "tests" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    input_bin = data_dir / "q8_0_input.bin"
    expected_bin = data_dir / "q8_0_expected.bin"
    meta_json = data_dir / "q8_0_meta.json"

    oracle_src = REPO / "scripts" / "q8_0_oracle.c"
    oracle_bin = REPO / "scripts" / "q8_0_oracle"
    build_oracle(oracle_src, oracle_bin)

    with gguf.parse(GGUF_PATH) as g:
        t = g.tensors[TENSOR_NAME]
        if t.dtype != 8:  # Q8_0
            raise RuntimeError(f"{TENSOR_NAME} is dtype {t.dtype}, expected Q8_0=8")
        raw = bytes(g.tensor_bytes(TENSOR_NAME)[: N_BLOCKS * 34])

    input_bin.write_bytes(raw)
    subprocess.run(
        [str(oracle_bin), str(input_bin), str(expected_bin), str(N_BLOCKS)],
        check=True,
    )
    meta_json.write_text(json.dumps(
        {"tensor": TENSOR_NAME, "n_blocks": N_BLOCKS, "block_elems": 32,
         "block_bytes": 34, "n_elements": N_BLOCKS * 32},
        indent=2,
    ), encoding="utf-8")
    print(f"wrote {input_bin}     ({input_bin.stat().st_size} bytes)")
    print(f"wrote {expected_bin}  ({expected_bin.stat().st_size} bytes)")
    print(f"wrote {meta_json}")


if __name__ == "__main__":
    main()
