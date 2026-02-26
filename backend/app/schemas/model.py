"""
Schemas for Ultralytics-native Model Designer.

A model is a YAML architecture definition with:
  - Parameters: nc, scales, kpt_shape
  - Backbone: list of [from, repeats, module, args] layers
  - Head: list of [from, repeats, module, args] layers
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any


class LayerDef(BaseModel):
    """Single layer in backbone or head: [from, repeats, module, args]."""
    from_: int | list[int] = Field(-1, alias="from")
    repeats: int = 1
    module: str
    args: list[Any] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class ScaleDef(BaseModel):
    """Compound scaling factors: [depth_mult, width_mult, max_channels]."""
    depth: float = 1.0
    width: float = 1.0
    max_channels: int = 1024


class ModelYAML(BaseModel):
    """Complete Ultralytics model YAML definition."""
    nc: int = 80
    scales: dict[str, list[float]] = Field(default_factory=dict)
    kpt_shape: list[int] | None = None
    backbone: list[LayerDef] = Field(default_factory=list)
    head: list[LayerDef] = Field(default_factory=list)


class ModelRecord(BaseModel):
    """Persisted model record with metadata + YAML definition."""
    model_id: str
    name: str = "Untitled"
    description: str = ""
    task: str = "detect"
    yaml_def: ModelYAML = Field(default_factory=ModelYAML)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ModelSummary(BaseModel):
    """Lightweight model listing."""
    model_id: str
    name: str
    task: str
    layer_count: int = 0
    input_shape: list[int] | None = None
    params: int | None = None
    gradients: int | None = None
    flops: float | None = None
    created_at: datetime
    updated_at: datetime


class SaveModelRequest(BaseModel):
    """Request to save a model."""
    name: str = "Untitled"
    description: str = ""
    task: str = "detect"
    yaml_def: dict[str, Any] = Field(default_factory=dict)


class ExportRequest(BaseModel):
    """Request to export a model."""
    model_id: str
    format: str = "yaml"
    weight_id: str | None = None
    imgsz: int = 640
    scale: str | None = None
