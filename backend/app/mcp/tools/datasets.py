"""
MCP tools — Dataset management.
Wraps dataset_registry and dataset plugin logic.
"""
from __future__ import annotations
from typing import Any

from ..filters import paginate
from ..serializers import ok, err


def list_datasets(
    view: str = "summary",
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """List all available datasets.

    Args:
        view: "summary" (default) or "detail".
        limit: Max items to return.
        offset: Items to skip.
    """
    try:
        from ...services.dataset_registry import get_all_datasets
        records = [r.dict() if hasattr(r, "dict") else dict(r) for r in get_all_datasets()]
        records = paginate(records, limit=limit, offset=offset)

        if view == "summary":
            summary_keys = {"name", "task", "num_classes", "class_names", "splits", "status", "total_images"}
            records = [{k: v for k, v in r.items() if k in summary_keys} for r in records]

        return {"ok": True, "count": len(records), "items": records}
    except Exception as e:
        return err(str(e), "list_datasets_failed")


def get_dataset(name: str, view: str = "summary") -> dict[str, Any]:
    """Get dataset info by name.

    Args:
        name: Dataset name (e.g. "coco128", "idd").
        view: "summary" (default) or "detail".
    """
    try:
        from ...services.dataset_registry import get_dataset_info as _get_info
        ds = _get_info(name)
        if not ds:
            return err(f"Dataset not found: {name}", "not_found")

        record = ds.dict() if hasattr(ds, "dict") else dict(ds)
        if view == "summary":
            summary_keys = {"name", "task", "num_classes", "class_names", "splits", "status", "total_images"}
            record = {k: v for k, v in record.items() if k in summary_keys}

        return ok(record)
    except Exception as e:
        return err(str(e), "get_dataset_failed")


def preview_dataset(name: str, count: int = 4) -> dict[str, Any]:
    """Get sample entries from a dataset (labels and image metadata only, no base64).

    Args:
        name: Dataset name.
        count: Number of samples to return (default 4, max 16).
    """
    try:
        from ...services.dataset_registry import get_dataset_info as _get_info
        from ...plugins.loader import get_dataset_plugin
        count = min(count, 16)

        ds = _get_info(name)
        if not ds:
            return err(f"Dataset not found: {name}", "not_found")

        plugin = get_dataset_plugin(name.lower())
        if not plugin:
            return err(f"No plugin for dataset: {name}", "no_plugin")

        dataset = plugin.load_train()
        if dataset is None:
            return err(f"Cannot load train split for: {name}", "load_failed")

        class_names = ds.class_names if hasattr(ds, "class_names") else []
        samples = []
        for i in range(min(count, len(dataset))):
            _, label = dataset[i]
            samples.append({
                "index": i,
                "label": label,
                "class_name": (
                    class_names[label]
                    if isinstance(label, int) and label < len(class_names)
                    else str(label)
                ),
            })

        return {"ok": True, "dataset": name, "count": len(samples), "samples": samples}
    except Exception as e:
        return err(str(e), "preview_dataset_failed")
