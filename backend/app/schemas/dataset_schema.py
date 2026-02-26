"""
Pydantic schemas for dataset management.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any


class DatasetInfo(BaseModel):
    """Dataset metadata returned by dataset plugins."""
    name: str
    display_name: str = ""
    task_type: str = "classification"
    input_shape: list[int] = Field(default_factory=lambda: [3, 224, 224])
    num_classes: int = 0
    class_names: list[str] = Field(default_factory=list)
    train_size: int = 0
    test_size: int = 0
    val_size: int = 0
    available: bool = False


class SplitConfig(BaseModel):
    """Configuration for dataset train/val/test split redistribution."""
    seed: int = Field(42, ge=0)
    train_to_val: int = Field(0, ge=0, le=100)
    train_to_test: int = Field(0, ge=0, le=100)
    test_to_train: int = Field(0, ge=0, le=100)
    test_to_val: int = Field(0, ge=0, le=100)
    val_to_train: int = Field(0, ge=0, le=100)
    val_to_test: int = Field(0, ge=0, le=100)

    def model_post_init(self, __context):
        if self.train_to_val + self.train_to_test > 100:
            raise ValueError("train_to_val + train_to_test cannot exceed 100")
        if self.test_to_train + self.test_to_val > 100:
            raise ValueError("test_to_train + test_to_val cannot exceed 100")
        if self.val_to_train + self.val_to_test > 100:
            raise ValueError("val_to_train + val_to_test cannot exceed 100")


class UpdatePartitionMethod(BaseModel):
    """Update the partition method for a dataset."""
    method: str = Field(..., description="Partition method: random, stratified, round_robin, iterative")


class CreatePartition(BaseModel):
    """Create a new named partition within a dataset."""
    name: str = Field(..., min_length=1, max_length=100)
    percent: int = Field(..., ge=1, le=99)


class SplitPartitionItem(BaseModel):
    """A single child partition in a split operation."""
    name: str = Field(..., min_length=1, max_length=100)
    percent: int = Field(..., ge=1, le=99)


class SplitPartitionBody(BaseModel):
    """Body for splitting an existing partition into sub-partitions."""
    children: list[SplitPartitionItem] = Field(..., min_length=2)
