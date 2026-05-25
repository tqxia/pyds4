"""GGUF v3 parser (read-only, mmap-backed).

A GGUF file is the binary container used by llama.cpp and friends. ds4 ships
weights as one GGUF, and `ds4.c` parses it in place (the bytes never leave the
file-backed mapping). We mirror that approach here in Python:

    +-------------------------------------------------------------+
    |  header        : magic ("GGUF"), u32 version, u64 n_tensors,|
    |                  u64 n_kv                                   |
    +-------------------------------------------------------------+
    |  metadata KV[] : (string key, u32 type, value)              |
    |                  value layout depends on type; for arrays   |
    |                  we record (item_type, len, data_pos) and   |
    |                  skip past without materializing            |
    +-------------------------------------------------------------+
    |  tensor desc[] : (string name, u32 ndim, u64 dim[ndim],     |
    |                   u32 dtype, u64 rel_offset)                |
    +-------------------------------------------------------------+
    |  padding to `general.alignment` (default 32)                |
    +-------------------------------------------------------------+
    |  tensor data ... (each tensor at tensor_data_pos+rel_offset)|
    +-------------------------------------------------------------+

All offsets in the tensor desc are *relative* to `tensor_data_pos`. We resolve
them to absolute file offsets after the desc table is read.

The 81 GB ds4 GGUF means we **must not** eagerly load tensor bytes. mmap +
record offsets is the only viable approach. Tensor *data* is fetched lazily by
`GGUFFile.tensor_bytes(name)`; later milestones (Q8_0 / IQ2 / Q2_K dequant) do
the real work on the returned memoryview.

Reference: ds4.c lines ~845-1260 (parse_metadata / parse_tensors / model_open).
"""

from __future__ import annotations

import enum
import mmap
import os
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# "GGUF" little-endian. Identical to DS4_GGUF_MAGIC in ds4.c.
GGUF_MAGIC = 0x46554747

# We support GGUF v3 only, same restriction ds4 imposes.
GGUF_VERSION = 3


class ValueType(enum.IntEnum):
    """Type tags for metadata values. Wire values match the GGUF spec."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


# Fixed-size scalar value types: (struct format, size in bytes).
# ARRAY and STRING are variable-sized and handled separately.
_SCALAR_FMT: dict[int, tuple[str, int]] = {
    ValueType.UINT8: ("<B", 1),
    ValueType.INT8: ("<b", 1),
    ValueType.UINT16: ("<H", 2),
    ValueType.INT16: ("<h", 2),
    ValueType.UINT32: ("<I", 4),
    ValueType.INT32: ("<i", 4),
    ValueType.FLOAT32: ("<f", 4),
    ValueType.BOOL: ("<B", 1),  # decoded as bool downstream
    ValueType.UINT64: ("<Q", 8),
    ValueType.INT64: ("<q", 8),
    ValueType.FLOAT64: ("<d", 8),
}


class TensorType(enum.IntEnum):
    """GGUF tensor element/quant types.

    Subset matching `gguf_types[]` in ds4.c. We list every type the ds4 model
    can contain so the parser doesn't reject unknown weights, but only the four
    types ds4 actually uses (F32/F16/Q8_0/Q2_K/IQ2_XXS) get dequantization
    implementations in later milestones.
    """

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30


# (block_elems, block_bytes) for every TensorType.
# A "block" is the smallest dequant unit; raw types like F32 have block_elems=1.
# Quantized types pack many elements into a single header+payload struct, e.g.
# Q8_0 = 32 elems in 34 bytes (one fp16 scale + 32 int8 values).
# Matches the table in ds4.c::gguf_types[].
_TENSOR_BLOCK: dict[int, tuple[int, int]] = {
    TensorType.F32: (1, 4),
    TensorType.F16: (1, 2),
    TensorType.Q4_0: (32, 18),
    TensorType.Q4_1: (32, 20),
    TensorType.Q5_0: (32, 22),
    TensorType.Q5_1: (32, 24),
    TensorType.Q8_0: (32, 34),
    TensorType.Q8_1: (32, 40),
    TensorType.Q2_K: (256, 84),
    TensorType.Q3_K: (256, 110),
    TensorType.Q4_K: (256, 144),
    TensorType.Q5_K: (256, 176),
    TensorType.Q6_K: (256, 210),
    TensorType.Q8_K: (256, 292),
    TensorType.IQ2_XXS: (256, 66),
    TensorType.IQ2_XS: (256, 74),
    TensorType.IQ3_XXS: (256, 98),
    TensorType.IQ1_S: (256, 110),
    TensorType.IQ4_NL: (256, 50),
    TensorType.IQ3_S: (256, 110),
    TensorType.IQ2_S: (256, 82),
    TensorType.IQ4_XS: (256, 136),
    TensorType.I8: (1, 1),
    TensorType.I16: (1, 2),
    TensorType.I32: (1, 4),
    TensorType.I64: (1, 8),
    TensorType.F64: (1, 8),
    TensorType.IQ1_M: (256, 56),
    TensorType.BF16: (1, 2),
}


def tensor_nbytes(dtype: int, n_elements: int) -> int:
    """Return on-disk byte count for `n_elements` of `dtype`.

    Quantized tensors are stored as `ceil(n / block_elems)` blocks. n_elements
    in the ds4 GGUF is always a multiple of block_elems but we do not assume so.
    """
    if dtype not in _TENSOR_BLOCK:
        raise ValueError(f"unknown GGUF tensor type {dtype}")
    block_elems, block_bytes = _TENSOR_BLOCK[dtype]
    blocks = (n_elements + block_elems - 1) // block_elems
    return blocks * block_bytes


@dataclass(frozen=True)
class ArrayRef:
    """A metadata array left in the file (not eagerly decoded).

    The ds4 GGUF carries some arrays that are O(vocab_size) — eagerly building
    Python lists for those would be slow and pointless when callers only need a
    handful. Materialize with `GGUFFile.array(...)` when actually wanted.
    """

    item_type: int
    length: int
    data_offset: int  # absolute file offset where item data starts


@dataclass(frozen=True)
class KV:
    """One metadata entry. `value` may be a primitive, str, or ArrayRef."""

    key: str
    type: int  # ValueType
    value: Any


@dataclass(frozen=True)
class Tensor:
    """One tensor descriptor. Bytes live at [abs_offset, abs_offset+nbytes)."""

    name: str
    dtype: int  # TensorType
    shape: tuple[int, ...]
    abs_offset: int
    nbytes: int

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def dtype_name(self) -> str:
        try:
            return TensorType(self.dtype).name.lower()
        except ValueError:
            return f"type{self.dtype}"


# ---- low-level byte cursor ---------------------------------------------------
#
# We hand-roll a tiny cursor over the mmap rather than reach for `struct.unpack`
# at every call: GGUF v3 only uses fixed little-endian widths and length-prefixed
# strings, so an explicit cursor is shorter than a per-field struct format.
# It also mirrors `ds4_cursor` in ds4.c, which is helpful when debugging parity.


class _Cursor:
    __slots__ = ("buf", "pos")

    def __init__(self, buf: memoryview | mmap.mmap, pos: int = 0):
        self.buf = buf
        self.pos = pos

    def u32(self) -> int:
        v = struct.unpack_from("<I", self.buf, self.pos)[0]
        self.pos += 4
        return v

    def u64(self) -> int:
        v = struct.unpack_from("<Q", self.buf, self.pos)[0]
        self.pos += 8
        return v

    def read(self, fmt: str, size: int) -> Any:
        v = struct.unpack_from(fmt, self.buf, self.pos)[0]
        self.pos += size
        return v

    def string(self) -> str:
        """GGUF string = u64 length + raw UTF-8 bytes (no NUL terminator)."""
        n = self.u64()
        s = bytes(self.buf[self.pos : self.pos + n]).decode("utf-8", errors="replace")
        self.pos += n
        return s

    def skip(self, n: int) -> None:
        self.pos += n


def _read_scalar(cur: _Cursor, vtype: int) -> Any:
    fmt, size = _SCALAR_FMT[vtype]
    v = cur.read(fmt, size)
    if vtype == ValueType.BOOL:
        return bool(v)
    return v


def _read_value(cur: _Cursor, vtype: int, depth: int = 0) -> Any:
    """Decode a metadata value at the cursor.

    Scalars and strings are materialized in Python. Arrays return an `ArrayRef`
    that points back into the mmap and *the cursor is advanced past the array
    bytes*, so the parser can keep walking the KV table without us decoding
    huge tokenizer lists.
    """
    if depth > 8:
        raise ValueError("metadata array nesting is too deep")

    if vtype == ValueType.STRING:
        return cur.string()

    if vtype == ValueType.ARRAY:
        item_type = cur.u32()
        n = cur.u64()
        data_off = cur.pos
        if item_type in _SCALAR_FMT:
            cur.skip(_SCALAR_FMT[item_type][1] * n)
        elif item_type == ValueType.STRING:
            # Variable-length items: walk and skip each one.
            for _ in range(n):
                slen = cur.u64()
                cur.skip(slen)
        elif item_type == ValueType.ARRAY:
            for _ in range(n):
                _read_value(cur, ValueType.ARRAY, depth + 1)
        else:
            raise ValueError(f"unsupported array item type {item_type}")
        return ArrayRef(item_type=item_type, length=n, data_offset=data_off)

    if vtype in _SCALAR_FMT:
        return _read_scalar(cur, vtype)

    raise ValueError(f"unknown metadata value type {vtype}")


# ---- public parser -----------------------------------------------------------


@dataclass
class GGUFFile:
    """Parsed view of a GGUF file. Holds the mmap; tensor bytes are lazy."""

    path: Path
    version: int
    alignment: int
    kv: dict[str, KV]
    tensors: dict[str, Tensor]
    tensor_data_pos: int
    size: int
    _mm: mmap.mmap = field(repr=False)
    _fd: int = field(repr=False)

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            try:
                os.close(self._fd)
            except OSError:
                pass

    def __enter__(self) -> "GGUFFile":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # --- accessors ------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        kv = self.kv.get(key)
        return default if kv is None else kv.value

    def array(self, key: str) -> list[Any]:
        """Materialize an array metadata value into a Python list.

        Costs O(length); use only when you actually want the data (vocab, etc).
        """
        kv = self.kv.get(key)
        if kv is None:
            raise KeyError(key)
        if not isinstance(kv.value, ArrayRef):
            raise TypeError(f"{key!r} is not an array (type={kv.type})")
        ref = kv.value
        cur = _Cursor(memoryview(self._mm), ref.data_offset)
        items: list[Any] = []
        if ref.item_type in _SCALAR_FMT:
            for _ in range(ref.length):
                items.append(_read_scalar(cur, ref.item_type))
        elif ref.item_type == ValueType.STRING:
            for _ in range(ref.length):
                items.append(cur.string())
        else:
            raise NotImplementedError(
                f"nested arrays not yet supported (item_type={ref.item_type})"
            )
        return items

    def tensor_bytes(self, name: str) -> memoryview:
        """Return a zero-copy view of the raw quantized bytes of `name`.

        The slice covers exactly `tensor.nbytes` and lives inside the mmap, so
        the caller must not retain it past `close()`. Used by dequant code in
        later milestones; not needed for M1.
        """
        t = self.tensors[name]
        return memoryview(self._mm)[t.abs_offset : t.abs_offset + t.nbytes]

    def total_tensor_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors.values())

    def total_elements(self) -> int:
        return sum(t.n_elements for t in self.tensors.values())


def _align_up(value: int, alignment: int) -> int:
    rem = value % alignment
    return value if rem == 0 else value + (alignment - rem)


def parse(path: str | os.PathLike) -> GGUFFile:
    """Open a GGUF file and parse header + metadata + tensor desc table.

    Does not touch the tensor data region (which can be tens of GB). The
    returned `GGUFFile` keeps the mmap alive; close it (or use as a context
    manager) when you're done.
    """
    p = Path(path)
    fd = os.open(p, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        if size < 32:
            raise ValueError(f"{p}: file too small to be GGUF")
        mm = mmap.mmap(fd, size, prot=mmap.PROT_READ)
    except BaseException:
        os.close(fd)
        raise

    try:
        cur = _Cursor(mm, 0)
        magic = cur.u32()
        if magic != GGUF_MAGIC:
            raise ValueError(f"{p}: not a GGUF file (magic={magic:#x})")
        version = cur.u32()
        if version != GGUF_VERSION:
            raise ValueError(f"{p}: only GGUF v3 supported (got v{version})")
        n_tensors = cur.u64()
        n_kv = cur.u64()

        # First pass: walk the KV table. The default alignment is 32 unless the
        # file overrides via the "general.alignment" key; ds4.c reads alignment
        # inline during the metadata walk to handle that case, and we do the
        # same.
        alignment = 32
        kv_map: dict[str, KV] = {}
        for _ in range(n_kv):
            key = cur.string()
            vtype = cur.u32()
            value = _read_value(cur, vtype)
            kv_map[key] = KV(key=key, type=vtype, value=value)
            if key == "general.alignment" and isinstance(value, int) and value > 0:
                alignment = value

        # Second pass: tensor descriptor table. Offsets are *relative* to the
        # tensor data region; we resolve them once tensor_data_pos is known.
        raw_tensors: list[tuple[str, int, tuple[int, ...], int, int]] = []
        for _ in range(n_tensors):
            name = cur.string()
            ndim = cur.u32()
            if not 1 <= ndim <= 8:
                raise ValueError(f"tensor {name!r}: bad ndim={ndim}")
            # GGUF stores dims in "natural" order (innermost-first, i.e. the
            # convention used by llama.cpp). We preserve that here; callers
            # that prefer torch-style (outer-first) ordering should reverse.
            shape = tuple(cur.u64() for _ in range(ndim))
            dtype = cur.u32()
            rel_off = cur.u64()
            n_elem = 1
            for d in shape:
                n_elem *= d
            nbytes = tensor_nbytes(dtype, n_elem)
            raw_tensors.append((name, dtype, shape, rel_off, nbytes))

        tensor_data_pos = _align_up(cur.pos, alignment)

        tensors: dict[str, Tensor] = {}
        for name, dtype, shape, rel_off, nbytes in raw_tensors:
            abs_off = tensor_data_pos + rel_off
            if nbytes != 0 and (abs_off > size or nbytes > size - abs_off):
                raise ValueError(f"tensor {name!r} extends past end of file")
            tensors[name] = Tensor(
                name=name,
                dtype=dtype,
                shape=shape,
                abs_offset=abs_off,
                nbytes=nbytes,
            )

        return GGUFFile(
            path=p,
            version=version,
            alignment=alignment,
            kv=kv_map,
            tensors=tensors,
            tensor_data_pos=tensor_data_pos,
            size=size,
            _mm=mm,
            _fd=fd,
        )
    except BaseException:
        mm.close()
        os.close(fd)
        raise


# ---- CLI: `python -m pyds4.gguf inspect <path>` ------------------------------


def _fmt_bytes(n: int) -> str:
    gib = 1024.0 ** 3
    return f"{n / gib:.2f} GiB" if n >= gib else f"{n / (1024.0 ** 2):.2f} MiB"


def _inspect(path: str, *, n_samples: int = 20) -> int:
    with parse(path) as g:
        print(f"file:    {g.path}")
        print(f"size:    {_fmt_bytes(g.size)} ({g.size} bytes)")
        print(f"gguf:    v{g.version}, {len(g.kv)} metadata keys, "
              f"{len(g.tensors)} tensors, alignment={g.alignment}")
        print(f"tensor data starts at offset {g.tensor_data_pos}")
        print(f"total tensor bytes: {_fmt_bytes(g.total_tensor_bytes())}")
        print(f"total elements:     {g.total_elements():,}")

        # Selected high-signal metadata keys (matches ds4.c::model_summary).
        for key in [
            "general.name",
            "general.architecture",
            "deepseek4.block_count",
            "deepseek4.context_length",
            "deepseek4.attention.head_count",
            "deepseek4.attention.head_count_kv",
            "deepseek4.attention.key_length",
            "deepseek4.attention.sliding_window",
            "deepseek4.attention.indexer.head_count",
            "deepseek4.attention.indexer.key_length",
            "deepseek4.attention.indexer.top_k",
            "deepseek4.expert_count",
            "deepseek4.expert_used_count",
        ]:
            if key in g.kv:
                v = g.kv[key].value
                if isinstance(v, ArrayRef):
                    v = f"<array len={v.length} type={v.item_type}>"
                print(f"  {key} = {v}")

        # Sample of tensor descriptors.
        names = list(g.tensors)
        sample = names[:n_samples]
        print(f"\nfirst {len(sample)} tensors:")
        for name in sample:
            t = g.tensors[name]
            print(f"  {t.dtype_name:7s} {list(t.shape)!s:25s} {name}")

        # Tensor-type histogram is useful for sanity-checking quant coverage.
        hist: dict[str, int] = {}
        for t in g.tensors.values():
            hist[t.dtype_name] = hist.get(t.dtype_name, 0) + 1
        print("\ntensor dtype histogram:")
        for k, c in sorted(hist.items(), key=lambda kv: -kv[1]):
            print(f"  {k:7s} {c}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 1 or argv[0] in {"-h", "--help"}:
        print("usage: python -m pyds4.gguf inspect <path>", file=sys.stderr)
        return 2
    cmd = argv[0]
    if cmd == "inspect":
        if len(argv) != 2:
            print("usage: python -m pyds4.gguf inspect <path>", file=sys.stderr)
            return 2
        return _inspect(argv[1])
    print(f"unknown subcommand: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
