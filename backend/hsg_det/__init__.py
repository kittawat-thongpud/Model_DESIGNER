"""
HSG-DET — Hybrid Sparse-Global Detection

Custom Ultralytics-compatible model implementation.
Import this package to register SparseGlobalBlock into the Ultralytics
module registry before constructing any YOLO model from hsg_det configs.

Usage:
    import hsg_det  # registers modules
    from ultralytics import YOLO
    model = YOLO("hsg_det/configs/hsg_det_m.yaml")
"""
from . import nn  # noqa: F401 — triggers module registration as side effect

__version__ = "0.1.0"
__all__ = ["nn"]
