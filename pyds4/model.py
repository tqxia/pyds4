"""DS4Model — top-level skeleton + GGUF weight loader + naive forward pass.

M7: `DS4Model.__init__` declares all parameters on `device='meta'`.
M8: `DS4Model.forward()` runs the end-to-end pipeline using dense causal
    attention (no CSA/HCA compressor/indexer), bf16 compute, PyTorch ops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

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
from pyds4.layers.rms import rms_norm_weight
from pyds4.layers.rope import precompute_rope_freqs
from pyds4.quant import dequant_iq2_xxs, dequant_q2_k, dequant_q8_0


# ---------------------------------------------------------------------------
# Module composition
# ---------------------------------------------------------------------------


class DS4Block(nn.Module):
    """One transformer layer: HC pre → Attn → HC post → HC pre → FFN → HC post."""

    def __init__(
        self,
        cfg: DS4Config,
        layer_idx: int,
        *,
        device: torch.device | str | None = "meta",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.cfg = cfg
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

    def forward(
        self,
        hc_state: torch.Tensor,
        positions: torch.Tensor,
        inv_freq: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        expert_load_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """hc_state: (seq, n_hc, n_embd), positions: (seq,). Returns (seq, n_hc, n_embd)."""
        cfg = self.cfg

        # ---- Attention sub-layer ----
        x_attn_in, post_attn, comb_attn = self.hc_attn.forward_pre(
            hc_state, cfg.rms_eps, cfg.hc_eps, cfg.sinkhorn_iters,
        )
        x_attn_norm = rms_norm_weight(x_attn_in, self.attn.norm, cfg.rms_eps)
        attn_out = self.attn.forward(
            x_attn_norm, positions, inv_freq,
            cfg.n_head, cfg.head_dim, cfg.n_rot, cfg.out_group, cfg.lora_o,
        )
        hc_state = self.hc_attn.forward_post(attn_out, hc_state, post_attn, comb_attn)

        # ---- FFN sub-layer ----
        x_ffn_in, post_ffn, comb_ffn = self.hc_ffn.forward_pre(
            hc_state, cfg.rms_eps, cfg.hc_eps, cfg.sinkhorn_iters,
        )
        x_ffn_norm = rms_norm_weight(x_ffn_in, self.ffn.norm, cfg.rms_eps)
        ffn_out = self.ffn.forward(
            x_ffn_norm, cfg.expert_weights_scale,
            float(cfg.swiglu_clamp_exp[self.layer_idx]),
            token_ids=token_ids,
            expert_load_fn=(lambda eid: expert_load_fn(self.layer_idx, eid))
            if expert_load_fn is not None else None,
        )
        hc_state = self.hc_ffn.forward_post(ffn_out, hc_state, post_ffn, comb_ffn)

        return hc_state


class DS4Model(nn.Module):
    """The whole network."""

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

    # -------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------

    def forward(
        self,
        tokens: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        expert_load_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Full forward pass.

        tokens: (seq,) int64 token IDs
        positions: (seq,) or None (defaults to arange(seq))
        expert_load_fn: callable(layer_idx, expert_id) -> (gate_w, up_w, down_w)
            for lazy expert dequant. If None, experts must be pre-loaded as
            nn.Parameters.
        Returns: (seq, vocab_size) — logits
        """
        seq = tokens.shape[0]
        device = tokens.device
        cfg = self.cfg

        if positions is None:
            positions = torch.arange(seq, device=device)

        # RoPE frequencies (same across layers; compressed layers differ but
        # ds4 applies a freq_scale, handled in M9+).
        inv_freq = precompute_rope_freqs(cfg.n_rot, cfg.rope_freq_base, device)

        # Token embedding: token_embd has shape (n_embd, vocab_size).
        # embed[:, tok] gives (n_embd, seq) → transpose to (seq, n_embd).
        x = self.token_embd[:, tokens].T  # (seq, n_embd)

        # Expand to n_hc identical streams
        hc_state = x.unsqueeze(1).expand(-1, cfg.n_hc, -1)  # (seq, n_hc, n_embd)

        # Process each block
        for block in self.blocks:
            hc_state = block(
                hc_state,
                positions,
                inv_freq,
                token_ids=tokens,
                expert_load_fn=expert_load_fn,
            )

        # Output HC collapse: (seq, n_hc, n_embd) → (seq, n_embd)
        x = self.output_hc.forward(hc_state, cfg.rms_eps, cfg.hc_eps)

        # Final RMSNorm
        x = rms_norm_weight(x, self.output_norm.weight, cfg.rms_eps)

        # LM head: (seq, n_embd) @ (n_embd, vocab_size) → (seq, vocab_size)
        logits = torch.matmul(x, self.output.to(x.dtype))
        return logits

    # -------------------------------------------------------------------
    # Weight loading
    # -------------------------------------------------------------------

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
            arr = _to_logical_layout(arr, tensor_desc.shape)
            t = torch.from_numpy(arr)
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

    # -------------------------------------------------------------------
    # Name map: GGUF tensor name -> dotted model state_dict key
    # -------------------------------------------------------------------

    def gguf_name_map(self) -> dict[str, str]:
        """Bidirectional-ready GGUF→model name map. Computed from cfg, deterministic."""
        return build_name_map(self.cfg)

    # -------------------------------------------------------------------
    # Internal: enumerate every loadable leaf and its host module
    # -------------------------------------------------------------------

    def _named_param_paths(self) -> dict[str, tuple[nn.Module, str, str]]:
        """Map dotted state_dict key → (parent_module, attr_name, kind).

        kind is "float" for nn.Parameter, "int" for I32 buffers (the hash tables).
        """
        out: dict[str, tuple[nn.Module, str, str]] = {}
        for path, parent, attr, kind in _walk_leaves(self):
            out[path] = (parent, attr, kind)
        return out


# ---------------------------------------------------------------------------
# Expert lazy loader (for M8e: keeps expert 3D tensors in GGUF mmap)
# ---------------------------------------------------------------------------

# Dequant block constants keyed by TensorType.
_BLOCK_ELEMS = {8: 32, 10: 256, 16: 256}     # Q8_0, Q2_K, IQ2_XXS
_BLOCK_BYTES = {8: 34, 10: 84, 16: 66}


def _make_expert_loader(
    g: gguf.GGUFFile,
    prefix: str,
    n_expert: int,
    n_embd: int,
    ff_exp: int,
    device: torch.device,
    dtype: torch.dtype,
    cache_size: int = 128,
) -> Callable:
    """Return `fn(expert_id, slot)` that loads (gate, up, down) for one expert.

    Expert tensors are 3-D: gate=(n_embd, ff_exp, n_expert), same for up,
    down=(ff_exp, n_embd, n_expert). Byte layout puts the expert axis outermost
    so each expert's bytes are contiguous. We dequant directly from the GGUF
    mmap without materializing the full tensor.

    `cache` is a simple most-recently-used dict of size `cache_size`.
    """
    gate_t = g.tensors[f"{prefix}.ffn_gate_exps.weight"]
    up_t = g.tensors[f"{prefix}.ffn_up_exps.weight"]
    down_t = g.tensors[f"{prefix}.ffn_down_exps.weight"]

    gate_dtype = gate_t.dtype
    up_dtype = up_t.dtype
    down_dtype = down_t.dtype

    n_expert_t = gate_t.shape[-1] if len(gate_t.shape) == 3 else 1
    if n_expert_t != n_expert:
        raise ValueError(f"expected {n_expert} experts, got {n_expert_t}")

    # Elements per expert = product of leading dims
    gate_elts_per_exp = n_embd * ff_exp
    up_elts_per_exp = n_embd * ff_exp
    down_elts_per_exp = ff_exp * n_embd

    # Bytes per expert
    gate_blks = (gate_elts_per_exp + _BLOCK_ELEMS[gate_dtype] - 1) // _BLOCK_ELEMS[gate_dtype]
    up_blks = (up_elts_per_exp + _BLOCK_ELEMS[up_dtype] - 1) // _BLOCK_ELEMS[up_dtype]
    down_blks = (down_elts_per_exp + _BLOCK_ELEMS[down_dtype] - 1) // _BLOCK_ELEMS[down_dtype]
    gate_bytes_per_exp = gate_blks * _BLOCK_BYTES[gate_dtype]
    up_bytes_per_exp = up_blks * _BLOCK_BYTES[up_dtype]
    down_bytes_per_exp = down_blks * _BLOCK_BYTES[down_dtype]

    gate_name = f"{prefix}.ffn_gate_exps.weight"
    up_name = f"{prefix}.ffn_up_exps.weight"
    down_name = f"{prefix}.ffn_down_exps.weight"

    _cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    # Track insert order for FIFO eviction
    _evict: list[int] = []

    def load(expert_id: int):
        nonlocal _evict

        if expert_id in _cache:
            return _cache[expert_id]

        def _dequant_and_shape(buf, elts, qt, logical_shape):
            if qt == 16:
                arr = dequant_iq2_xxs(buf, elts)
            elif qt == 10:
                arr = dequant_q2_k(buf, elts)
            else:
                raise RuntimeError(f"unsupported dtype {qt}")
            arr = _to_logical_layout(arr, logical_shape)
            return torch.from_numpy(arr).to(device=device, dtype=dtype)

        # Slice and dequant each expert
        gate_full = g.tensor_bytes(gate_name)
        goff = expert_id * gate_bytes_per_exp
        gate_w = _dequant_and_shape(
            gate_full[goff : goff + gate_bytes_per_exp],
            gate_elts_per_exp, gate_dtype, (n_embd, ff_exp),
        )
        gate_full.release()

        up_full = g.tensor_bytes(up_name)
        uoff = expert_id * up_bytes_per_exp
        up_w = _dequant_and_shape(
            up_full[uoff : uoff + up_bytes_per_exp],
            up_elts_per_exp, up_dtype, (n_embd, ff_exp),
        )
        up_full.release()

        down_full = g.tensor_bytes(down_name)
        doff = expert_id * down_bytes_per_exp
        down_w = _dequant_and_shape(
            down_full[doff : doff + down_bytes_per_exp],
            down_elts_per_exp, down_dtype, (ff_exp, n_embd),
        )
        down_full.release()

        # Evict oldest if cache full
        while len(_evict) >= cache_size:
            oldest = _evict.pop(0)
            del _cache[oldest]
        _cache[expert_id] = (gate_w, up_w, down_w)
        _evict.append(expert_id)
        return _cache[expert_id]

    return load


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


def _to_logical_layout(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Convert GGUF innermost-first storage to a C-contiguous logical tensor.

    GGUF dimension 0 is contiguous. The model exposes those dimensions in
    logical order, while NumPy and PyTorch make the last dimension contiguous.
    Reverse the storage shape, then reverse the axes so matmul sees the same
    rows and columns as ds4.
    """
    if len(shape) <= 1:
        return arr.reshape(shape)
    storage_shape = tuple(reversed(shape))
    axes = tuple(reversed(range(len(shape))))
    return arr.reshape(storage_shape).transpose(axes).copy()
