"""DS4 layer modules.

M7 scope: parameter declarations only.
M8 scope: forward methods on every module.
"""

from pyds4.layers.attention import Attention, Compressor, Indexer
from pyds4.layers.hc import HyperConnections, OutputHC
from pyds4.layers.moe import MoEFFN
from pyds4.layers.rms import RMSNorm, rms_norm_no_weight, rms_norm_weight
from pyds4.layers.rope import (
    precompute_layer_rope_freqs,
    precompute_rope_freqs,
    rope_forward,
    rope_inverse,
)

__all__ = [
    "Attention",
    "Compressor",
    "HyperConnections",
    "Indexer",
    "MoEFFN",
    "OutputHC",
    "RMSNorm",
    "precompute_rope_freqs",
    "precompute_layer_rope_freqs",
    "rms_norm_no_weight",
    "rms_norm_weight",
    "rope_forward",
    "rope_inverse",
]
