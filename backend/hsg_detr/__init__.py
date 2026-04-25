"""
HSG-DETR package.

Importing ``hsg_detr.nn`` registers SparseGlobalTokenBlock and related
blocks into the Ultralytics namespace so YAML model definitions can use them.
"""

from .nn import register as _register

# Auto-register on package import (idempotent)
_register()
