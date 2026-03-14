"""
Safe JSON normalization helpers for MCP responses.
Handles NaN, Inf, non-serializable types before sending over MCP.
"""
from __future__ import annotations
import math
from typing import Any


def safe_value(v: Any) -> Any:
    """Recursively normalize a value to be JSON-safe."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, dict):
        return {k: safe_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [safe_value(i) for i in v]
    # Convert non-primitive types to string
    if not isinstance(v, (str, int, bool, type(None))):
        return str(v)
    return v


def safe_dict(record: dict[str, Any]) -> dict[str, Any]:
    return {k: safe_value(v) for k, v in record.items()}


def safe_list(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [safe_dict(r) for r in records]


def ok(data: Any, **extra: Any) -> dict[str, Any]:
    """Wrap a successful MCP tool result."""
    result: dict[str, Any] = {"ok": True}
    result.update(extra)
    if isinstance(data, dict):
        result.update(safe_dict(data))
    elif isinstance(data, list):
        result["items"] = safe_list(data)
    else:
        result["data"] = safe_value(data)
    return result


def err(message: str, code: str = "error") -> dict[str, Any]:
    """Wrap an error MCP tool result."""
    return {"ok": False, "error": code, "message": message}
