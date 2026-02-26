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
