#!/usr/bin/env python3
"""
dump_training_state.py
======================
Pull all training state (jobs, weights, benchmarks, models, datasets)
from the local backend storage and save as JSON.

This produces the same data as calling MCP tools (list_jobs, get_job,
list_weights, get_weight, list_benchmarks, etc.) but runs directly against
the on-disk storage so it works even when the HTTP/MCP server is down.

Usage:
    cd /home/rase/kittawat_ws/Model_DESIGNER
    python scripts/dump_training_state.py [--output ./training_state]

Output files:
    jobs.json            – summary list of all jobs
    jobs_detail.json     – full record for every job
    jobs_history.json    – per-epoch training history for every job
    weights.json         – summary list of all weights
    weights_detail.json  – full metadata for every weight
    weights_info.json    – param count / GFLOPs for every weight
    benchmarks.json      – summary list of all benchmarks
    benchmarks_detail.json – full results for every benchmark
    models.json          – summary list of all custom models
    datasets.json        – summary list of all registered datasets
    top_performers.json  – ranked table of best jobs by mAP
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure backend is importable
# ---------------------------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parent.parent / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _safe_json(obj: Any) -> Any:
    """Return a JSON-safe copy (NaN/Inf → None, Path → str)."""
    return json.loads(json.dumps(obj, default=lambda o: str(o) if hasattr(o, "__str__") else None))


def dump_jobs(store, out_dir: Path) -> dict:
    """Dump all jobs + detail + history."""
    print("[jobs] listing …")
    jobs = store.list_jobs()
    (out_dir / "jobs.json").write_text(json.dumps(_safe_json({"count": len(jobs), "items": jobs}), indent=2))

    detail = []
    history = {}
    for j in jobs:
        jid = j.get("job_id")
        if not jid:
            continue
        rec = store.load_job(jid)
        if rec:
            detail.append(rec)
        try:
            hist = store.get_job_history(jid)
            if hist:
                history[jid] = hist
        except Exception:
            pass  # history may not exist

    (out_dir / "jobs_detail.json").write_text(json.dumps(_safe_json(detail), indent=2))
    (out_dir / "jobs_history.json").write_text(json.dumps(_safe_json(history), indent=2))
    print(f"[jobs] {len(jobs)} jobs, {len(detail)} detail, {len(history)} histories")
    return {"jobs": jobs, "detail": detail, "history": history}


def dump_weights(wstore, out_dir: Path) -> dict:
    """Dump all weights + detail + lineage."""
    print("[weights] listing …")
    weights = wstore.list_weights()
    (out_dir / "weights.json").write_text(json.dumps(_safe_json({"count": len(weights), "items": weights}), indent=2))

    detail = []
    lineages = {}
    for w in weights:
        wid = w.get("weight_id")
        if not wid:
            continue
        rec = wstore.load_weight_meta(wid)
        if rec:
            detail.append(rec)
        try:
            lin = wstore.get_lineage(wid)
            if lin:
                lineages[wid] = lin
        except Exception:
            pass

    (out_dir / "weights_detail.json").write_text(json.dumps(_safe_json(detail), indent=2))
    (out_dir / "weights_lineages.json").write_text(json.dumps(_safe_json(lineages), indent=2))
    print(f"[weights] {len(weights)} weights, {len(detail)} detail, {len(lineages)} lineages")
    return {"weights": weights, "detail": detail, "lineages": lineages}


def dump_benchmarks(bdir: Path, out_dir: Path) -> dict:
    """Dump all benchmark results from DATA_DIR/benchmarks/*.json."""
    print("[benchmarks] listing …")
    benches = []
    if bdir.exists():
        for path in sorted(bdir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text())
                benches.append(data)
            except Exception:
                pass

    (out_dir / "benchmarks.json").write_text(
        json.dumps(_safe_json({"count": len(benches), "items": benches}), indent=2)
    )
    print(f"[benchmarks] {len(benches)} results")
    return {"benchmarks": benches, "detail": benches}


def dump_models(mstore, out_dir: Path) -> dict:
    """Dump all custom model definitions."""
    print("[models] listing …")
    models = mstore.list_models()
    (out_dir / "models.json").write_text(json.dumps(_safe_json({"count": len(models), "items": models}), indent=2))
    print(f"[models] {len(models)} models")
    return {"models": models}


def dump_datasets(out_dir: Path) -> dict:
    """Dump all registered datasets."""
    print("[datasets] listing …")
    from app.services import dataset_registry
    datasets = dataset_registry.get_all_datasets()
    (out_dir / "datasets.json").write_text(json.dumps(_safe_json({"count": len(datasets), "items": [d.__dict__ if hasattr(d, "__dict__") else d for d in datasets]}), indent=2))
    print(f"[datasets] {len(datasets)} datasets")
    return {"datasets": datasets}


def build_top_performers(jobs_data: list[dict], out_dir: Path) -> None:
    """Build a ranked table of best completed jobs by mAP50-95."""
    completed = [
        j for j in jobs_data
        if j.get("status") == "completed" and j.get("best_mAP50_95") is not None
    ]
    ranked = sorted(completed, key=lambda x: x.get("best_mAP50_95", 0), reverse=True)

    table = []
    for r in ranked:
        table.append({
            "job_id": r.get("job_id"),
            "model_name": r.get("model_name"),
            "model_id": r.get("model_id"),
            "model_scale": r.get("model_scale"),
            "dataset": r.get("dataset_name"),
            "epochs": f"{r.get('epoch')}/{r.get('total_epochs')}",
            "best_mAP50": round(r.get("best_mAP50", 0), 5),
            "best_mAP50_95": round(r.get("best_mAP50_95", 0), 5),
            "weight_id": r.get("weight_id"),
            "total_time_h": round(r.get("total_time", 0) / 3600, 1) if r.get("total_time") else None,
            "created_at": r.get("created_at"),
        })

    (out_dir / "top_performers.json").write_text(json.dumps(table, indent=2))
    print(f"[top_performers] {len(table)} completed jobs ranked")


def main():
    parser = argparse.ArgumentParser(description="Dump all training state to JSON")
    parser.add_argument("--output", default="./training_state", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dumping training state to: {out_dir.resolve()}\n")

    try:
        from app.services import job_storage, model_storage, weight_storage
        from app.config import DATA_DIR
    except Exception as e:
        print(f"ERROR: cannot import backend services: {e}")
        print("Make sure you run this script from the repo root with venv activated.")
        traceback.print_exc()
        sys.exit(1)

    # Jobs
    jobs_result = dump_jobs(job_storage, out_dir)

    # Weights (use correct function names)
    weights_result = dump_weights(weight_storage, out_dir)

    # Benchmarks (read directly from benchmarks dir)
    bench_result = dump_benchmarks(DATA_DIR / "benchmarks", out_dir)

    # Models
    models_result = dump_models(model_storage, out_dir)

    # Datasets
    datasets_result = dump_datasets(out_dir)

    # Top performers
    build_top_performers(jobs_result.get("detail", []), out_dir)

    # Metadata
    meta = {
        "dumped_at": datetime.utcnow().isoformat() + "Z",
        "server": "local",
        "counts": {
            "jobs": len(jobs_result.get("jobs", [])),
            "weights": len(weights_result.get("weights", [])),
            "benchmarks": len(bench_result.get("benchmarks", [])),
            "models": len(models_result.get("models", [])),
            "datasets": len(datasets_result.get("datasets", [])),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\nDone. All files written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
