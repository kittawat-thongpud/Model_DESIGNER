"""
Field filtering helpers for MCP responses.
Controls which keys are returned in summary vs. detail mode.
"""
from __future__ import annotations
from typing import Any


_SUMMARY_FIELDS: dict[str, set[str]] = {
    "model": {"model_id", "name", "task", "description", "created_at", "updated_at", "params", "gradients", "flops"},
    "dataset": {"name", "task", "num_classes", "class_names", "splits", "status", "total_images"},
    "job": {
        "job_id", "model_id", "model_name", "model_scale", "task",
        "status", "epoch", "total_epochs", "dataset_name",
        "best_mAP50", "best_mAP50_95", "best_fitness",
        "message", "created_at", "started_at", "completed_at",
    },
    "weight": {
        "weight_id", "model_id", "model_name", "model_scale",
        "dataset_name", "dataset", "epochs_trained", "total_epochs",
        "final_accuracy", "final_loss", "file_size_bytes",
        "job_id", "created_at",
    },
    "benchmark": {
        "benchmark_id", "weight_id", "dataset", "split",
        "mAP50", "mAP50_95", "precision", "recall",
        "latency_ms", "params", "flops", "created_at",
    },
    "log": {"timestamp", "level", "message"},
}

_STRIP_ALWAYS: set[str] = {
    "yaml_def", "_graph", "confusion_matrix", "per_class_metrics",
    "history", "training_runs", "edits",
}


def apply_view(
    record: dict[str, Any],
    domain: str,
    view: str = "summary",
    extra_fields: list[str] | None = None,
    strip_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Return a filtered copy of *record* for the given *view* and *domain*.

    - ``view="summary"`` keeps only the curated key set for the domain.
    - ``view="detail"`` strips only known-heavy blobs unless *strip_fields* override.
    - ``extra_fields`` add extra keys on top of the summary set.
    - ``strip_fields`` remove specific keys regardless of view.
    """
    out = dict(record)

    additional = set(extra_fields or [])
    forced_strip = set(strip_fields or [])

    if view == "summary":
        allowed = (_SUMMARY_FIELDS.get(domain, set()) | additional) - forced_strip
        out = {k: v for k, v in out.items() if k in allowed}
    else:
        strip = (_STRIP_ALWAYS - additional) | forced_strip
        out = {k: v for k, v in out.items() if k not in strip}

    return out


def apply_list_view(
    records: list[dict[str, Any]],
    domain: str,
    view: str = "summary",
    extra_fields: list[str] | None = None,
    strip_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    return [apply_view(r, domain, view, extra_fields, strip_fields) for r in records]


def paginate(items: list[Any], limit: int | None = None, offset: int = 0) -> list[Any]:
    sliced = items[offset:]
    if limit is not None and limit > 0:
        sliced = sliced[:limit]
    return sliced
