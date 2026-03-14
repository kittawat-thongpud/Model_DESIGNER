"""
MCP tools — Benchmark management.
Wraps benchmark_controller helpers and benchmark storage.
"""
from __future__ import annotations
import json
from typing import Any

from ..filters import paginate
from ..serializers import ok, err


def _benchmark_dir():
    from ...config import DATA_DIR
    d = DATA_DIR / "benchmarks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_benchmark_datasets(weight_id: str | None = None) -> dict[str, Any]:
    """List available datasets for benchmarking.

    Args:
        weight_id: Optional weight to filter compatible datasets.
    """
    try:
        from ...services import weight_storage as _ws
        from ...controllers.benchmark_controller import _list_available_datasets

        weight_meta = None
        if weight_id:
            weight_meta = _ws.load_weight_meta(weight_id)
        result = _list_available_datasets(weight_meta)
        return {"ok": True, "count": len(result), "items": result}
    except Exception as e:
        return err(str(e), "list_benchmark_datasets_failed")


def run_benchmark(
    weight_id: str,
    dataset: str,
    split: str = "val",
    conf: float = 0.001,
    iou: float = 0.6,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "",
) -> dict[str, Any]:
    """Run benchmark validation for a weight against a dataset.

    Returns mAP, per-class metrics, latency, params, and FLOPs.

    Args:
        weight_id: Target weight ID.
        dataset: Dataset name (e.g. "coco128", "idd").
        split: Dataset split to evaluate (val, test, train).
        conf: Confidence threshold.
        iou: IoU threshold.
        imgsz: Inference image size.
        batch: Batch size.
        device: Device string ("", "cpu", "0", "cuda:0").
    """
    try:
        from ...controllers.benchmark_controller import BenchmarkRequest, _run_benchmark
        req = BenchmarkRequest(
            weight_id=weight_id,
            dataset=dataset,
            split=split,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            batch=batch,
            device=device,
        )
        result = _run_benchmark(req)
        return {"ok": True, **result}
    except ValueError as e:
        return err(str(e), "benchmark_invalid")
    except Exception as e:
        return err(str(e), "run_benchmark_failed")


def list_benchmarks(
    weight_id: str | None = None,
    view: str = "summary",
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """List past benchmark results.

    Args:
        weight_id: Filter by weight ID.
        view: "summary" returns key metrics only; "detail" returns full record.
        limit: Max items to return (default 20).
        offset: Items to skip.
    """
    try:
        bdir = _benchmark_dir()
        results = []
        _summary_keys = {
            "benchmark_id", "weight_id", "dataset", "split",
            "mAP50", "mAP50_95", "precision", "recall",
            "latency_ms", "params", "flops", "created_at",
        }
        for path in sorted(bdir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text())
                if weight_id and data.get("weight_id") != weight_id:
                    continue
                if view == "summary":
                    data = {k: v for k, v in data.items() if k in _summary_keys}
                results.append(data)
            except Exception:
                pass

        results = paginate(results, limit=limit, offset=offset)
        return {"ok": True, "count": len(results), "items": results}
    except Exception as e:
        return err(str(e), "list_benchmarks_failed")


def get_benchmark(benchmark_id: str, view: str = "detail") -> dict[str, Any]:
    """Get a specific benchmark result by ID.

    Args:
        benchmark_id: Benchmark ID.
        view: "summary" or "detail" (default).
    """
    try:
        path = _benchmark_dir() / f"{benchmark_id}.json"
        if not path.exists():
            return err(f"Benchmark not found: {benchmark_id}", "not_found")
        data = json.loads(path.read_text())
        if view == "summary":
            summary_keys = {
                "benchmark_id", "weight_id", "dataset", "split",
                "mAP50", "mAP50_95", "precision", "recall",
                "latency_ms", "params", "flops", "created_at",
            }
            data = {k: v for k, v in data.items() if k in summary_keys}
        return ok(data)
    except Exception as e:
        return err(str(e), "get_benchmark_failed")
