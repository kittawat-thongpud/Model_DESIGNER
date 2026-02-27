"""
Pydantic schemas for trained weight records.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import datetime


class WeightRecord(BaseModel):
    """Full weight metadata — persisted alongside .pt files."""
    weight_id: str
    model_id: str
    model_name: str = "Untitled"
    job_id: str | None = None
    dataset: str = ""
    epochs_trained: int = 0
    final_accuracy: float | None = None
    final_loss: float | None = None
    file_size_bytes: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class WeightSummary(BaseModel):
    """Lightweight weight listing."""
    weight_id: str
    model_id: str
    model_name: str
    dataset: str
    epochs_trained: int
    final_accuracy: float | None = None
    file_size_bytes: int = 0
    created_at: datetime


# ── Request schemas (moved from weight_controller.py) ─────────────────────────

class ExtractRequest(BaseModel):
    """Extract specific node weights from a weight file."""
    node_ids: list[str] = Field(..., description="Node IDs whose weights to extract")


class TransferRequest(BaseModel):
    """Transfer weights from a source weight to a target weight."""
    source_weight_id: str
    node_id_map: dict[str, str] | None = Field(
        default=None,
        description="Optional rename mapping {source_node_id: target_node_id}",
    )


class AutoMapRequest(BaseModel):
    """Auto-map layers between source and target weights."""
    source_weight_id: str = Field(..., description="Source weight to map FROM")


class ApplyMapRequest(BaseModel):
    """Apply a manual layer mapping for weight transfer."""
    source_weight_id: str
    mapping: list[dict] = Field(..., description="List of {src_key, tgt_key} pairs")
    freeze_node_ids: list[str] = Field(default_factory=list, description="Node IDs to freeze after transfer")


class CreateEmptyRequest(BaseModel):
    """Create an empty (randomly-initialized) weight from a model architecture."""
    model_id: str = Field(default="", description="Model to instantiate (custom model ID or empty for YOLO official)")
    name: str = Field(default="", description="Optional display name for the weight")
    model_scale: str | None = Field(default=None, description="Scale variant: n, s, m, l, x")
    yolo_model: str | None = Field(default=None, description="Official YOLO model key e.g. 'yolov8n', 'yolov8s' (overrides model_id)")
    use_pretrained: bool = Field(default=True, description="If yolo_model set: load pretrained COCO weights (True) or random init (False)")
