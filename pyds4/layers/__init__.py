"""DS4 layer modules.

M7 scope: parameter declarations only. No forward methods — those land in M8.
Every leaf tensor on a layer is an `nn.Parameter` whose shape matches the
corresponding GGUF tensor *exactly* (no transposes). This makes the
GGUF→model name map an identity on shape and lets the M7 test check
parameter-count parity by summing `numel()` over `model.state_dict()`.

Modules default to `device='meta'` so instantiation is essentially free
(only shape metadata is allocated). The weight loader in `pyds4.model`
materializes parameters from GGUF tensor bytes when called.
"""

from pyds4.layers.attention import Attention, Compressor, Indexer
from pyds4.layers.hc import HyperConnections, OutputHC
from pyds4.layers.moe import MoEFFN
from pyds4.layers.rms import RMSNorm

__all__ = [
    "Attention",
    "Compressor",
    "HyperConnections",
    "Indexer",
    "MoEFFN",
    "OutputHC",
    "RMSNorm",
]
