"""
HSG-DET neural network modules.

Importing this package registers SparseGlobalBlock and SparseGlobalBlockGated
into the Ultralytics nn.modules namespace automatically.
"""
from .sparse_global import (
    SparseGlobalBlock,
    SparseGlobalBlockGated,
    _register_into_ultralytics,
)

# Register on import so any subsequent YOLO(yaml) call can resolve these names
_register_into_ultralytics()

__all__ = [
    "SparseGlobalBlock",
    "SparseGlobalBlockGated",
]
