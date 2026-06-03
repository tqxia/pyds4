"""DS4Model — top-level skeleton + GGUF weight loader.

M7 deliverable: instantiation succeeds + parameter count matches the number
ds4's `model_summary` reports. The number is just
`sum(t.n_elements for t in gguf.tensors.values())`, since ds4 counts every
tensor (including the I32 hash routing tables) as a logical parameter.

`DS4Model.__init__` defaults to `device='meta'`, so building the full
43-layer 284B-parameter graph allocates only shape metadata. To actually
populate weights, call `model.load_weights(gguf_file, device=..., dtype=...)`
— this materializes parameters on a real device and copies dequanted bytes
in. For the 81 GB GGUF that's >280 GB of data even at fp16, so most callers
will want to load a subset (e.g. for testing) or stream weights to GPU as
needed in later milestones.

Layer-presence rules (see ds4.c, line 2408 onward):

  - compress_ratio == 0 (layers 0, 1): main attention only, no compressor,
    no indexer.
  - compress_ratio == 4 (layers 2, 4, ..., 42): main attention + compressor
    + indexer (CSA path).
  - compress_ratio == 128 (layers 3, 5, ..., 41): main attention +
    compressor only (HCA, no CSA).
  - layer < n_hash_layer (= 3): hash-routed FFN — carries `ffn_gate_tid2eid`
    but NOT `exp_probs_b`. Layer 2 is the boundary and carries both gate_inp
    AND the hash table.
  - layer >= n_hash_layer: learned routing — carries `exp_probs_b.bias`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from pyds4 import gguf
from pyds4.config import DS4Config
from pyds4.layers import (
    Attention,
    HyperConnections,
    MoEFFN,
    OutputHC,
    RMSNorm,
)
from pyds4.quant import dequant_iq2_xxs, dequant_q2_k, dequant_q8_0


# ---------------------------------------------------------------------------
# Module composition
# ---------------------------------------------------------------------------


class DS4Block(nn.Module):
    """One transformer layer. Sub-modules depend on layer index (see file docstring)."""

    def __init__(
        self,
        cfg: DS4Config,
        layer_idx: int,
        *,
        device: torch.device | str | None = "meta",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.ratio = cfg.compress_ratios[layer_idx]

        self.hc_attn = HyperConnections(cfg.n_embd, cfg.n_hc, device=device)
        self.attn = Attention(cfg, self.ratio, device=device)
        self.hc_ffn = HyperConnections(cfg.n_embd, cfg.n_hc, device=device)
        self.ffn = MoEFFN(
            cfg,
            is_hash_routed=(layer_idx < cfg.n_hash_layer),
            has_exp_bias=(layer_idx >= cfg.n_hash_layer),
            device=device,
        )


class DS4Model(nn.Module):
    """The whole network. M7 = parameters only; forward() lands in M8."""

    def __init__(
        self,
        cfg: DS4Config,
        *,
        device: torch.device | str | None = "meta",
    ) -> None:
        super().__init__()
        self.cfg = cfg

        # Token embedding stored as (n_embd, vocab) per the GGUF tensor desc.
        self.token_embd = nn.Parameter(
            torch.empty(cfg.n_embd, cfg.vocab_size, device=device, dtype=torch.float16),
            requires_grad=False,
        )

        self.blocks = nn.ModuleList(
            [DS4Block(cfg, i, device=device) for i in range(cfg.n_layer)]
        )

        self.output_norm = RMSNorm(cfg.n_embd, cfg.rms_eps, device=device)
        # LM head — same shape as token_embd but Q8_0 in the GGUF.
        self.output = nn.Parameter(
            torch.empty(cfg.n_embd, cfg.vocab_size, device=device, dtype=torch.float32),
            requires_grad=False,
        )
        self.output_hc = OutputHC(cfg.n_embd, cfg.n_hc, device=device)

    # -----------------------------------------------------------------------
    # Weight loading
    # -----------------------------------------------------------------------

    def load_weights(
        self,
        g: gguf.GGUFFile,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        names: list[str] | None = None,
    ) -> None:
        """Materialize parameters on `device` and load (possibly a subset of) GGUF tensors.

        `names`: optional iterable of GGUF tensor names to load. Passing `None`
        means "every tensor in the map". Tensors not in `names` are left on
        their current device (meta if untouched).

        `dtype` is the **target** dtype for float parameters. The I32 hash
        routing buffers are always loaded as int32.

        Memory note: a full fp32 load is ~1.1 TB; bf16 is ~568 GB. Most callers
        should pass `names=` with a small subset. Streaming the routed-expert
        tensors directly to GPU is a Phase E concern.
        """
        name_map = self.gguf_name_map()
        if names is None:
            wanted = list(name_map.keys())
        else:
            wanted = list(names)

        state_paths = dict(self._named_param_paths())  # path → (parent, attr, dtype_kind)

        for gname in wanted:
            mkey = name_map.get(gname)
            if mkey is None:
                raise KeyError(f"GGUF tensor {gname!r} not in model name map")
            parent, attr, kind = state_paths[mkey]
            tensor_desc = g.tensors[gname]
            arr = _dequant_tensor(g, tensor_desc)
            t = torch.from_numpy(arr).reshape(tuple(tensor_desc.shape))
            if kind == "int":
                t = t.to(device=device, dtype=torch.int32)
            else:
                t = t.to(device=device, dtype=dtype)
            # Materialize via assign so we don't need pre-allocated storage.
            existing = getattr(parent, attr)
            if isinstance(existing, nn.Parameter):
                setattr(parent, attr, nn.Parameter(t, requires_grad=False))
            else:
                setattr(parent, attr, t)

    # -----------------------------------------------------------------------
    # Name map: GGUF tensor name -> dotted model state_dict key
    # -----------------------------------------------------------------------

    def gguf_name_map(self) -> dict[str, str]:
        """Bidirectional-ready GGUF→model name map. Computed from cfg, deterministic."""
        return build_name_map(self.cfg)

    # -----------------------------------------------------------------------
    # Internal: enumerate every loadable leaf and its host module
    # -----------------------------------------------------------------------

    def _named_param_paths(self) -> dict[str, tuple[nn.Module, str, str]]:
        """Map dotted state_dict key → (parent_module, attr_name, kind).

        kind is "float" for nn.Parameter, "int" for I32 buffers (the hash tables).
        """
        out: dict[str, tuple[nn.Module, str, str]] = {}
        for path, parent, attr, kind in _walk_leaves(self):
            out[path] = (parent, attr, kind)
        return out


# ---------------------------------------------------------------------------
# Name map builder — pure function of cfg.
# ---------------------------------------------------------------------------


def build_name_map(cfg: DS4Config) -> dict[str, str]:
    """Return {gguf_tensor_name: model_state_dict_key}.

    The mapping is shape-preserving: every GGUF tensor lands in a leaf with
    exactly the same number of elements. The model's `_walk_leaves` is the
    inverse — every loadable leaf has exactly one GGUF source.
    """
    m: dict[str, str] = {}

    # --- top level ---------------------------------------------------------
    m["token_embd.weight"] = "token_embd"
    m["output_norm.weight"] = "output_norm.weight"
    m["output.weight"] = "output"
    m["output_hc_fn.weight"] = "output_hc.fn"
    m["output_hc_base.weight"] = "output_hc.base"
    m["output_hc_scale.weight"] = "output_hc.scale"

    # --- per-layer ---------------------------------------------------------
    for il in range(cfg.n_layer):
        ratio = cfg.compress_ratios[il]
        p = f"blocks.{il}"
        b = f"blk.{il}"

        # HC (attn) and HC (ffn) bundles.
        m[f"{b}.hc_attn_fn.weight"]    = f"{p}.hc_attn.fn"
        m[f"{b}.hc_attn_base.weight"]  = f"{p}.hc_attn.base"
        m[f"{b}.hc_attn_scale.weight"] = f"{p}.hc_attn.scale"
        m[f"{b}.hc_ffn_fn.weight"]     = f"{p}.hc_ffn.fn"
        m[f"{b}.hc_ffn_base.weight"]   = f"{p}.hc_ffn.base"
        m[f"{b}.hc_ffn_scale.weight"]  = f"{p}.hc_ffn.scale"

        # Main attention (always present).
        a = f"{p}.attn"
        m[f"{b}.attn_norm.weight"]      = f"{a}.norm"
        m[f"{b}.attn_q_a.weight"]       = f"{a}.q_a"
        m[f"{b}.attn_q_a_norm.weight"]  = f"{a}.q_a_norm"
        m[f"{b}.attn_q_b.weight"]       = f"{a}.q_b"
        m[f"{b}.attn_kv.weight"]        = f"{a}.kv"
        m[f"{b}.attn_kv_a_norm.weight"] = f"{a}.kv_a_norm"
        m[f"{b}.attn_sinks.weight"]     = f"{a}.sinks"
        m[f"{b}.attn_output_a.weight"]  = f"{a}.output_a"
        m[f"{b}.attn_output_b.weight"]  = f"{a}.output_b"

        # Compressor (HCA path).
        if ratio != 0:
            c = f"{a}.compressor"
            m[f"{b}.attn_compressor_norm.weight"] = f"{c}.norm"
            m[f"{b}.attn_compressor_kv.weight"]   = f"{c}.kv"
            m[f"{b}.attn_compressor_gate.weight"] = f"{c}.gate"
            m[f"{b}.attn_compressor_ape.weight"]  = f"{c}.ape"

        # Indexer (CSA path) — only present on ratio-4 layers.
        if ratio == 4:
            ix = f"{a}.indexer"
            m[f"{b}.indexer.attn_q_b.weight"]          = f"{ix}.attn_q_b"
            m[f"{b}.indexer.proj.weight"]              = f"{ix}.proj"
            m[f"{b}.indexer_compressor_norm.weight"]   = f"{ix}.compressor.norm"
            m[f"{b}.indexer_compressor_kv.weight"]     = f"{ix}.compressor.kv"
            m[f"{b}.indexer_compressor_gate.weight"]   = f"{ix}.compressor.gate"
            m[f"{b}.indexer_compressor_ape.weight"]    = f"{ix}.compressor.ape"

        # MoE FFN.
        f = f"{p}.ffn"
        m[f"{b}.ffn_norm.weight"]      = f"{f}.norm"
        m[f"{b}.ffn_gate_inp.weight"]  = f"{f}.gate_inp"
        m[f"{b}.ffn_gate_exps.weight"] = f"{f}.gate_exps"
        m[f"{b}.ffn_up_exps.weight"]   = f"{f}.up_exps"
        m[f"{b}.ffn_down_exps.weight"] = f"{f}.down_exps"
        m[f"{b}.ffn_gate_shexp.weight"] = f"{f}.gate_shexp"
        m[f"{b}.ffn_up_shexp.weight"]   = f"{f}.up_shexp"
        m[f"{b}.ffn_down_shexp.weight"] = f"{f}.down_shexp"
        if il < cfg.n_hash_layer:
            m[f"{b}.ffn_gate_tid2eid.weight"] = f"{f}.tid2eid"
        else:
            m[f"{b}.exp_probs_b.bias"] = f"{f}.exp_probs_b"

    return m


# ---------------------------------------------------------------------------
# Internal: enumerate leaves (parameters + I32 buffers) and dequant dispatch
# ---------------------------------------------------------------------------


def _walk_leaves(module: nn.Module):
    """Yield (dotted_path, parent_module, attr_name, kind) for every loadable leaf.

    "Loadable leaf" = nn.Parameter (float) or persistent int32 buffer
    (currently only `MoEFFN.tid2eid`). The dotted_path matches what
    `state_dict()` would produce.
    """
    for name, sub in module.named_modules():
        for p_name, p in sub.named_parameters(recurse=False):
            path = f"{name}.{p_name}" if name else p_name
            yield path, sub, p_name, "float"
        for b_name, b in sub.named_buffers(recurse=False):
            if b is None:
                continue
            path = f"{name}.{b_name}" if name else b_name
            yield path, sub, b_name, "int"


def _dequant_tensor(g: gguf.GGUFFile, t: gguf.Tensor) -> np.ndarray:
    """Read `t`'s bytes from the GGUF and decode to a 1-D numpy array.

    Routed through the Phase B dequant kernels by dtype. The output is always
    1-D with `t.n_elements` entries; the caller reshapes to `t.shape`.
    """
    buf = g.tensor_bytes(t.name)
    n = t.n_elements
    dt = t.dtype
    if dt == int(gguf.TensorType.F32):
        return np.frombuffer(bytes(buf), dtype=np.float32, count=n).copy()
    if dt == int(gguf.TensorType.F16):
        return np.frombuffer(bytes(buf), dtype=np.float16, count=n).astype(np.float32)
    if dt == int(gguf.TensorType.Q8_0):
        return dequant_q8_0(buf, n)
    if dt == int(gguf.TensorType.Q2_K):
        return dequant_q2_k(buf, n)
    if dt == int(gguf.TensorType.IQ2_XXS):
        return dequant_iq2_xxs(buf, n)
    if dt == int(gguf.TensorType.I32):
        return np.frombuffer(bytes(buf), dtype=np.int32, count=n).copy()
    raise NotImplementedError(f"dequant for dtype {dt} ({t.dtype_name()}) not implemented")
