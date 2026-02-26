"""
Pydantic schemas for built (compiled) model records.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import datetime


class LayerInfo(BaseModel):
    """Describes a single layer in the built model."""
    index: int
    layer_type: str
    params: dict = Field(default_factory=dict)


class BuildRecord(BaseModel):
    """Full record of a built (code-generated) model."""
    build_id: str
    model_id: str
    model_name: str
    class_name: str
    code: str
    layers: list[LayerInfo] = Field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0
    created_at: str = ""


class BuildSummary(BaseModel):
    """Lightweight listing for the Models page."""
    build_id: str
    model_id: str
    model_name: str
    class_name: str
    layer_count: int = 0
    layer_types: list[str] = Field(default_factory=list)
    created_at: str = ""
