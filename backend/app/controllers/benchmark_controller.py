"""
Benchmark Controller — Validate a weight against a dataset and return
confusion matrix, per-class mAP, latency, params, FLOPs.
"""
from __future__ import annotations
import asyncio
import json
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import DATA_DIR
from ..services import weight_storage, model_storage
from .. import logging_service as logger

router = APIRouter(prefix="/api/benchmark", tags=["Benchmark"])

BENCHMARK_DIR = DATA_DIR / "benchmarks"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)


# ── Schema ────────────────────────────────────────────────────────────────────

class BenchmarkRequest(BaseModel):
    weight_id: str
    dataset: str                    # dataset name (resolves to DATA_DIR/datasets/{name}/data.yaml)
    split: str = "val"              # train | val | test
    conf: float = 0.001
    iou: float = 0.6
    imgsz: int = 640
    batch: int = 16
    device: str = ""                # "" = auto


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_dataset_yaml(dataset: str, weight_meta: dict | None) -> Path | None:
    """
    Resolve dataset YAML path from multiple fallback locations:
    1. If dataset looks like a path and exists → use directly
    2. DATA_DIR/datasets/{dataset}/data.yaml
    3. DATA_DIR/datasets/{dataset}/{dataset}.yaml
    4. Job dir linked to this weight: DATA_DIR/jobs/{job_id}/data.yaml
    5. Any job dir that references this dataset name
    """
    datasets_dir = DATA_DIR / "datasets"
    jobs_dir = DATA_DIR / "jobs"

    # 1. Absolute/relative path given directly
    p = Path(dataset)
    if p.exists() and p.suffix in (".yaml", ".yml"):
        return p

    # 2 & 3. Standard dataset dir
    for fname in ("data.yaml", f"{dataset}.yaml"):
        candidate = datasets_dir / dataset / fname
        if candidate.exists():
            return candidate

    # 4. Job linked to this weight
    if weight_meta:
        job_id = weight_meta.get("job_id") or weight_meta.get("source_job_id")
        if job_id:
            job_yaml = jobs_dir / job_id / "data.yaml"
            if job_yaml.exists():
                return job_yaml

    # 5. Scan all job dirs for one referencing this dataset
    if jobs_dir.exists():
        for job_dir in jobs_dir.iterdir():
            job_yaml = job_dir / "data.yaml"
            if job_yaml.exists():
                try:
                    content = job_yaml.read_text()
                    if dataset.lower() in content.lower():
                        return job_yaml
                except Exception:
                    pass

    return None


def _list_available_datasets(weight_meta: dict | None = None) -> list[dict]:
    """Return all dataset YAMLs available for benchmarking."""
    import yaml as _yaml

    datasets_dir = DATA_DIR / "datasets"
    jobs_dir = DATA_DIR / "jobs"
    results: list[dict] = []
    seen: set[str] = set()

    def _add(label: str, yaml_path: Path, source: str):
        key = label.lower()
        if key in seen:
            return
        seen.add(key)
        nc = None
        try:
            data = _yaml.safe_load(yaml_path.read_text())
            nc = data.get("nc")
        except Exception:
            pass
        results.append({"label": label, "value": label, "yaml_path": str(yaml_path), "nc": nc, "source": source})

    # Datasets dir
    if datasets_dir.exists():
        for ds_dir in sorted(datasets_dir.iterdir()):
            for fname in ("data.yaml", f"{ds_dir.name}.yaml"):
                candidate = ds_dir / fname
                if candidate.exists():
                    _add(ds_dir.name, candidate, "dataset")
                    break

    # Job dirs — only include entries where we can resolve a real dataset name.
    # Jobs that used a custom partition (config.data points back into /jobs/)
    # are skipped unless the job record has an explicit dataset_name set.
    if jobs_dir.exists():
        from ..services import job_storage as _job_storage
        import re as _re
        job_id_first = (weight_meta or {}).get("job_id") or (weight_meta or {}).get("source_job_id")
        job_dirs = sorted(jobs_dir.iterdir(), key=lambda d: (d.name != job_id_first, d.name))
        for job_dir in job_dirs:
            job_yaml = job_dir / "data.yaml"
            if not job_yaml.exists():
                continue
            job_rec = _job_storage.load_job(job_dir.name)
            ds_name = (job_rec or {}).get("dataset_name", "")
            if not ds_name:
                # Try to extract dataset name from config.data path
                raw_data = (job_rec or {}).get("config", {}).get("data", "")
                if raw_data:
                    m = _re.search(r'/datasets/([^/]+)/', raw_data.replace("\\", "/"))
                    if m:
                        ds_name = m.group(1)
            if not ds_name:
                # Fallback: read the job data.yaml and extract dataset name from 'path' field
                try:
                    job_yaml_data = _yaml.safe_load(job_yaml.read_text())
                    yaml_path_field = str(job_yaml_data.get("path", "")).replace("\\", "/")
                    if yaml_path_field:
                        m = _re.search(r'/datasets/([^/]+)$', yaml_path_field)
                        if m:
                            ds_name = m.group(1)
                except Exception:
                    pass
            # Skip entries we cannot name — avoids "job:xxxx" appearing in the list
            if not ds_name:
                continue
            _add(ds_name, job_yaml, "job")

    return results


def _run_benchmark(req: BenchmarkRequest) -> dict:
    """Blocking — runs in threadpool."""
    import sys
    import torch
    from ultralytics import YOLO
    from pathlib import Path as _Path

    # Ensure backend/ is in sys.path so custom packages (e.g. hsg_det) are importable
    _backend_dir = str(_Path(__file__).resolve().parents[2])
    if _backend_dir not in sys.path:
        sys.path.insert(0, _backend_dir)
    # Register all arch plugin custom modules before loading
    try:
        from ..plugins.loader import all_arch_plugins
        for _ap in all_arch_plugins():
            try:
                _ap.register_modules()
            except Exception:
                pass
    except Exception:
        pass

    meta = weight_storage.load_weight_meta(req.weight_id)
    if not meta:
        raise ValueError(f"Weight '{req.weight_id}' not found")

    pt_path = weight_storage.weight_pt_path(req.weight_id)
    if not pt_path.exists():
        raise ValueError(f"Weight file missing: {req.weight_id}")

    # Resolve dataset yaml — try multiple locations in priority order
    data_yaml = _resolve_dataset_yaml(req.dataset, meta)
    if data_yaml is None:
        raise ValueError(
            f"Dataset YAML not found for '{req.dataset}'. "
            "Use GET /api/benchmark/datasets to see available options."
        )

    try:
        model = YOLO(str(pt_path))
    except (KeyError, Exception) as e:
        raise ValueError(
            f"Cannot load weight file for benchmarking: {e}. "
            "This weight may have been created with an older version. "
            "Try re-creating the empty weight to regenerate the file."
        )

    # Resolve device
    device = req.device
    if not device:
        device = "0" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    val_results = model.val(
        data=str(data_yaml),
        split=req.split,
        conf=req.conf,
        iou=req.iou,
        imgsz=req.imgsz,
        batch=req.batch,
        device=device,
        verbose=False,
        plots=True,
        save_json=False,
    )
    elapsed_s = time.time() - t0

    # ── Extract metrics ──────────────────────────────────────────────────────
    mp = val_results.box    # detect metrics proxy

    # Per-class metrics
    per_class: list[dict] = []
    names = model.names or {}
    if hasattr(mp, "ap_class_index") and mp.ap_class_index is not None:
        for i, cls_id in enumerate(mp.ap_class_index.tolist()):
            per_class.append({
                "class_id": int(cls_id),
                "class_name": names.get(int(cls_id), str(cls_id)),
                "ap50": round(float(mp.ap50[i]), 4) if mp.ap50 is not None else None,
                "ap50_95": round(float(mp.ap[i]), 4) if mp.ap is not None else None,
                "precision": round(float(mp.p[i]), 4) if mp.p is not None else None,
                "recall": round(float(mp.r[i]), 4) if mp.r is not None else None,
                "f1": round(float(mp.f1[i]), 4) if hasattr(mp, "f1") and mp.f1 is not None else None,
            })

    # Confusion matrix
    confusion_data = None
    try:
        cm = val_results.confusion_matrix
        if cm is not None:
            matrix = cm.matrix
            confusion_data = {
                "matrix": matrix.tolist(),
                "names": [names.get(i, str(i)) for i in range(len(names))] + ["background"],
            }
    except Exception:
        pass

    # Speed
    speed = getattr(val_results, "speed", {}) or {}

    # Model info (params / FLOPs)
    # model.info(verbose=False) returns (layers, params, gradients, GFLOPs)
    params = None
    flops_gflops = None
    try:
        info = model.info(verbose=False, detailed=False)
        if isinstance(info, (list, tuple)):
            if len(info) >= 4:
                params = int(info[1])          # index 1 = params
                flops_gflops = float(info[3])  # index 3 = GFLOPs (already GFLOPs)
            elif len(info) >= 2:
                params = int(info[0])
                flops_gflops = float(info[1])
    except Exception:
        pass

    # Fallback: try val_results speed-derived model info if available
    if params is None:
        try:
            from ultralytics.utils.torch_utils import get_flops, get_num_params
            params = get_num_params(model.model)
            flops_gflops = get_flops(model.model, imgsz=req.imgsz)
        except Exception:
            pass

    result = {
        "benchmark_id": uuid.uuid4().hex[:12],
        "weight_id": req.weight_id,
        "dataset": req.dataset,
        "split": req.split,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(elapsed_s, 1),
        # Overall metrics
        "mAP50": round(float(mp.map50), 4) if mp.map50 is not None else None,
        "mAP50_95": round(float(mp.map), 4) if mp.map is not None else None,
        "precision": round(float(mp.mp), 4) if mp.mp is not None else None,
        "recall": round(float(mp.mr), 4) if mp.mr is not None else None,
        "fitness": round(float(val_results.fitness), 4) if hasattr(val_results, "fitness") else None,
        # Per-class
        "per_class": per_class,
        # Confusion matrix
        "confusion_matrix": confusion_data,
        # Latency
        "preprocess_ms": round(speed.get("preprocess", 0), 2),
        "inference_ms": round(speed.get("inference", 0), 2),
        "postprocess_ms": round(speed.get("postprocess", 0), 2),
        # Model info
        "params": params,
        "flops_gflops": round(flops_gflops, 3) if flops_gflops else None,
        # Config
        "conf": req.conf,
        "iou": req.iou,
        "imgsz": req.imgsz,
    }

    # Save result to disk
    out_path = BENCHMARK_DIR / f"{result['benchmark_id']}.json"
    out_path.write_text(json.dumps(result, indent=2))

    return result


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/datasets", summary="List available datasets for benchmarking")
async def list_benchmark_datasets(weight_id: str | None = None):
    """Return all dataset YAMLs available for benchmarking (from datasets dir + job dirs)."""
    weight_meta = None
    if weight_id:
        weight_meta = weight_storage.load_weight_meta(weight_id)
    return _list_available_datasets(weight_meta)


@router.post("/run", summary="Run benchmark validation for a weight")
async def run_benchmark(req: BenchmarkRequest):
    """Run val() against a dataset and return full benchmark results."""
    try:
        result = await asyncio.to_thread(_run_benchmark, req)
        logger.log("system", "INFO", "Benchmark complete", {
            "weight_id": req.weight_id,
            "dataset": req.dataset,
            "mAP50": result.get("mAP50"),
        })
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Benchmark failed: {e}")


@router.get("/history", summary="List past benchmark results")
async def list_benchmarks(weight_id: str | None = None, limit: int = 20):
    """Return recent benchmark results, optionally filtered by weight_id."""
    results = []
    for path in sorted(BENCHMARK_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text())
            if weight_id and data.get("weight_id") != weight_id:
                continue
            results.append(data)
            if len(results) >= limit:
                break
        except Exception:
            pass
    return results


@router.get("/{benchmark_id}", summary="Get a specific benchmark result")
async def get_benchmark(benchmark_id: str):
    """Return a specific benchmark result by ID."""
    path = BENCHMARK_DIR / f"{benchmark_id}.json"
    if not path.exists():
        raise HTTPException(404, f"Benchmark '{benchmark_id}' not found")
    return json.loads(path.read_text())


@router.delete("/{benchmark_id}", summary="Delete a benchmark result")
async def delete_benchmark(benchmark_id: str):
    path = BENCHMARK_DIR / f"{benchmark_id}.json"
    if not path.exists():
        raise HTTPException(404, f"Benchmark '{benchmark_id}' not found")
    path.unlink()
    return {"message": f"Benchmark '{benchmark_id}' deleted"}
