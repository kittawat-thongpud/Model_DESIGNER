"""
Pydantic schemas for model graph representation.
This is the core data contract between frontend and backend.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Literal
from datetime import datetime


class Position(BaseModel):
    x: float = 0.0
    y: float = 0.0


class GlobalVariable(BaseModel):
    """A typed global configuration variable that blocks can reference via $name."""
    id: str
    name: str
    type: Literal["bool", "float", "int", "str", "selector"]
    value: bool | int | float | str = ""
    options: list[str] = Field(default_factory=list)  # for 'selector' type
    description: str = ""


class NodeSchema(BaseModel):
    model_config = {"populate_by_name": True}
    id: str
    type: str
    position: Position = Field(default_factory=Position)
    params: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    enabled_by_global: str | None = Field(default=None, alias="enabledByGlobal")
    package_id: str | None = Field(default=None, alias="packageId") # For type="Package" nodes


class EdgeSchema(BaseModel):
    source: str
    target: str
    source_handle: str | None = None
    target_handle: str | None = None


class ModelMeta(BaseModel):
    name: str = "Untitled"
    version: str = "1.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    description: str = ""


class ModelGraph(BaseModel):
    """Complete model graph — the JSON contract between frontend and backend."""
    id: str | None = None
    meta: ModelMeta = Field(default_factory=ModelMeta)
    nodes: list[NodeSchema] = Field(default_factory=list)
    edges: list[EdgeSchema] = Field(default_factory=list)
    globals: list[GlobalVariable] = Field(default_factory=list)


class ModelSummary(BaseModel):
    """Lightweight model listing info."""
    id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    node_count: int
    edge_count: int


class BuildRequest(BaseModel):
    model_id: str


class BuildResponse(BaseModel):
    model_id: str
    code: str
    class_name: str


class TrainRequest(BaseModel):
    model_config = {"extra": "allow"}
    model_id: str
    dataset: str = "mnist"
    epochs: int = 5
    learning_rate: float = 0.001
    batch_size: int = 64
    weight_record_enabled: bool = False
    weight_record_layers: list[str] = Field(default_factory=list)
    weight_record_frequency: int = 5


class TrainStatusResponse(BaseModel):
    job_id: str
    status: str  # "running", "completed", "failed", "stopped"
    epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    train_accuracy: float = 0.0
    val_loss: float | None = None
    val_accuracy: float | None = None
    message: str = ""


class ValidateRequest(BaseModel):
    model_id: str
    dataset: str = "mnist"
    batch_size: int = 64


class ValidateResponse(BaseModel):
    model_id: str
    accuracy: float
    loss: float
    num_samples: int
    per_class_accuracy: dict[str, float] = Field(default_factory=dict)


class DatasetInfo(BaseModel):
    name: str
    display_name: str
    input_shape: list[int]
    num_classes: int
    train_size: int
    test_size: int
    classes: list[str]
    task_type: Literal["classification", "detection"] = "classification"
    task_type: Literal["classification", "detection"] = "classification"


class PackageParameter(BaseModel):
    """parameter exposed by a package (derived from global variable)"""
    name: str
    type: Literal["bool", "float", "int", "str", "selector"]
    default: Any = None
    description: str = ""
    options: list[str] = Field(default_factory=list)


class ModelPackage(BaseModel):
    """A reusable model component (sub-graph)"""
    id: str
    name: str
    description: str = ""
    nodes: list[NodeSchema] = Field(default_factory=list)
    edges: list[EdgeSchema] = Field(default_factory=list)
    globals: list[GlobalVariable] = Field(default_factory=list)
    exposed_params: list[PackageParameter] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CreatePackageRequest(BaseModel):
    """Request body for creating a package"""
    graph: ModelGraph
    name: str
    description: str = ""
    exposed_globals: list[str] = Field(default_factory=list)


class DetectionBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label_id: int
    label_name: str | None = None


class PredictRequest(BaseModel):
    model_id: str
    weight_id: str | None = None  # If None, use latest or model-linked
    image_base64: str


class PredictResponse(BaseModel):
    model_id: str
    task_type: str
    # For classification
    class_id: int | None = None
    class_name: str | None = None
    confidence: float | None = None
    # For detection
    boxes: list[DetectionBox] | None = None

