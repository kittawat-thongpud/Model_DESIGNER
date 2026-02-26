"""
Schemas for Module Designer — custom nn.Module blocks.

Users create custom PyTorch modules that get registered into the
Ultralytics namespace so they can be used in model YAML configs.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any


class ModuleArg(BaseModel):
    """Argument definition for a custom module."""
    name: str
    type: str = "int"          # int, float, bool, str, list
    default: Any = None
    description: str = ""


class ModuleRecord(BaseModel):
    """Persisted custom module definition."""
    module_id: str
    name: str                               # class name, e.g. "CustomBlock"
    code: str                               # full Python source code
    args: list[ModuleArg] = Field(default_factory=list)
    category: str = "custom"                # custom, backbone, head, etc.
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ModuleSummary(BaseModel):
    """Lightweight module listing."""
    module_id: str
    name: str
    category: str
    description: str
    arg_count: int = 0
    created_at: datetime


class SaveModuleRequest(BaseModel):
    """Request to create/update a custom module."""
    name: str
    code: str
    args: list[ModuleArg] = Field(default_factory=list)
    category: str = "custom"
    description: str = ""


class BuiltinModuleInfo(BaseModel):
    """Info about a built-in Ultralytics module."""
    name: str
    category: str              # basic, composite, head, torch_nn, custom
    args: list[ModuleArg] = Field(default_factory=list)
    description: str = ""
    source: str = "ultralytics"  # ultralytics, torch.nn, torchvision.ops, custom
