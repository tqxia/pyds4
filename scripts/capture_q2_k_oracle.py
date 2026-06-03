#!/usr/bin/env python3
"""Capture a Q2_K dequant oracle for the M6 parity test.

Picks a slice of `blk.0.ffn_down_exps.weight` (a Q2_K tensor — these are the
routed `W_down` expert matrices). We grab the first 256 blocks (256 * 256 =
65536 elements, 256 KiB fp32 output) and pipe the raw bytes through a C oracle
(`scripts/q2_k_oracle.c`) that re-implements ds4.c's f16_to_f32 + the Q2_K
dequant formula verbatim.

Saves:
  tests/data/q2_k_input.bin     raw Q2_K bytes (256 * 84 = 21504 bytes)
  tests/data/q2_k_expected.bin  fp32 reference (256 * 256 * 4 = 256 KiB)
  tests/data/q2_k_meta.json     tensor name + n_blocks for documentation

Run from the repo root: scripts/capture_q2_k_oracle.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from pyds4 import gguf

REPO = Path(__file__).resolve().parent.parent
GGUF_PATH = Path("/home/tqxia/workspace/ds4/ds4flash.gguf")
TENSOR_NAME = "blk.0.ffn_down_exps.weight"
N_BLOCKS = 256                       # 65536 elements, 256 KiB fp32 output


def build_oracle(src: Path, dst: Path) -> None:
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        raise RuntimeError("no C compiler on PATH (need cc or gcc)")
    subprocess.run([cc, "-O2", "-std=c11", "-o", str(dst), str(src)], check=True)


def main() -> None:
    data_dir = REPO / "tests" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    input_bin = data_dir / "q2_k_input.bin"
    expected_bin = data_dir / "q2_k_expected.bin"
    meta_json = data_dir / "q2_k_meta.json"

    oracle_src = REPO / "scripts" / "q2_k_oracle.c"
    oracle_bin = REPO / "scripts" / "q2_k_oracle"
    build_oracle(oracle_src, oracle_bin)

    with gguf.parse(GGUF_PATH) as g:
        t = g.tensors[TENSOR_NAME]
        if t.dtype != 10:  # Q2_K
            raise RuntimeError(f"{TENSOR_NAME} is dtype {t.dtype}, expected Q2_K=10")
        raw = bytes(g.tensor_bytes(TENSOR_NAME)[: N_BLOCKS * 84])

    input_bin.write_bytes(raw)
    subprocess.run(
        [str(oracle_bin), str(input_bin), str(expected_bin), str(N_BLOCKS)],
        check=True,
    )
    meta_json.write_text(json.dumps(
        {"tensor": TENSOR_NAME, "n_blocks": N_BLOCKS, "block_elems": 256,
         "block_bytes": 84, "n_elements": N_BLOCKS * 256},
        indent=2,
    ), encoding="utf-8")
    print(f"wrote {input_bin}     ({input_bin.stat().st_size} bytes)")
    print(f"wrote {expected_bin}  ({expected_bin.stat().st_size} bytes)")
    print(f"wrote {meta_json}")


if __name__ == "__main__":
    main()
