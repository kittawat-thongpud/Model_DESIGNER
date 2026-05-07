"""
Benchmark Controller — Validate a weight against a dataset and return
confusion matrix, per-class mAP, latency, params, FLOPs.
"""
from __future__ import annotations
import asyncio
import json
import re
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import DATA_DIR
from ..services import weight_storage, model_storage
from ..services.config_service import get_benchmark_config
from .. import logging_service as logger

router = APIRouter(prefix="/api/benchmark", tags=["Benchmark"])

BENCHMARK_DIR = DATA_DIR / "benchmarks"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
_BENCHMARK_DEFAULTS = get_benchmark_config().get("defaults", {})
_BENCHMARK_API_DEFAULTS = get_benchmark_config().get("api_defaults", {})


# ── Schema ────────────────────────────────────────────────────────────────────

class BenchmarkRequest(BaseModel):
    weight_id: str
    dataset: str                    # dataset name (resolves to DATA_DIR/datasets/{name}/data.yaml)
    split: str = str(_BENCHMARK_DEFAULTS.get("split", "val"))              # train | val | test
    conf: float = float(_BENCHMARK_DEFAULTS.get("conf", 0.001))
    iou: float = float(_BENCHMARK_DEFAULTS.get("iou", 0.6))
    imgsz: int = int(_BENCHMARK_DEFAULTS.get("imgsz", 640))
    batch: int = int(_BENCHMARK_DEFAULTS.get("batch", 16))
    device: str = ""                # "" = auto


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rewrite_yaml_paths(yaml_path: Path) -> Path:
    """Ensure a data.yaml points to paths valid on the current machine.

    Strategy (in priority order):
    1. All paths already exist → return as-is.
    2. Dataset name can be resolved under current DATA_DIR/datasets/<name> →
       regenerate data.yaml from scratch using generate_data_yaml (best: fresh
       paths, no stale .cache references).
    3. Fallback: line-by-line remap of absolute paths via /datasets/<name> segment.
    """
    import tempfile
    import yaml as _yaml

    try:
        content = yaml_path.read_text()
        data = _yaml.safe_load(content)
    except Exception:
        return yaml_path

    fields = ["path", "train", "val", "test"]

    def _remap_path_obj(p: Path) -> Path:
        if p.exists():
            return p
        parts = p.parts
        for i, part in enumerate(parts):
            if part == "datasets" and i + 1 < len(parts):
                for candidate in (
                    DATA_DIR / Path(*parts[i:]),
                    DATA_DIR / "datasets" / Path(*parts[i + 1:]),
                ):
                    if candidate.exists():
                        return candidate
        for i in range(1, len(parts)):
            candidate = DATA_DIR / Path(*parts[i:])
            if candidate.exists():
                return candidate
        return p

    def _split_path_for_value(value: str) -> Path:
        raw = value.split("#", 1)[0].strip()
        p = Path(raw)
        if p.is_absolute():
            return _remap_path_obj(p)
        root = _remap_path_obj(Path(str(data.get("path") or ".")))
        return _remap_path_obj(root / p)

    def _txt_has_stale_paths(txt_path: Path) -> bool:
        if not txt_path.exists() or txt_path.suffix.lower() != ".txt":
            return False
        try:
            for raw in txt_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                p = Path(line)
                return p.is_absolute() and not p.exists() and _remap_path_obj(p).exists()
        except Exception:
            return False
        return False

    def _remap_txt_file(txt_path: Path) -> tuple[Path, bool]:
        txt_path = _remap_path_obj(txt_path)
        if not txt_path.exists() or txt_path.suffix.lower() != ".txt":
            return txt_path, False
        changed = False
        lines_out: list[str] = []
        try:
            for raw in txt_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                p = Path(line)
                remapped = _remap_path_obj(p) if p.is_absolute() else p
                if str(remapped) != line:
                    changed = True
                lines_out.append(str(remapped))
        except Exception:
            return txt_path, False
        if not changed:
            return txt_path, False
        tmp_txt = tempfile.NamedTemporaryFile(
            mode="w", suffix=f"_{txt_path.stem}.txt", delete=False, dir=str(BENCHMARK_DIR)
        )
        tmp_txt.write("\n".join(lines_out) + "\n")
        tmp_txt.flush()
        tmp_txt.close()
        return Path(tmp_txt.name), True

    needs_fix = False
    for field in fields:
        val = data.get(field)
        if val and isinstance(val, str):
            p = Path(val.split("#")[0].strip())
            if p.is_absolute() and not p.exists():
                needs_fix = True
                break
            if field != "path" and _txt_has_stale_paths(_split_path_for_value(val)):
                needs_fix = True
                break

    if not needs_fix:
        return yaml_path

    # ── Strategy 2: rebuild yaml with current paths, preserve class info ─────
    # Extract dataset name from the 'path' field (last component)
    ds_path_str = str(data.get("path", "")).split("#")[0].strip()
    if ds_path_str:
        ds_name = Path(ds_path_str).name  # e.g. "coco", "coco128"
        local_ds_path = DATA_DIR / "datasets" / ds_name
        if local_ds_path.exists():
            try:
                # Build a minimal but correct data.yaml with current paths,
                # preserving nc/names from the original file.
                orig_names = data.get("names") or {}
                orig_nc = data.get("nc") or (len(orig_names) if orig_names else 0)
                # Auto-detect split dirs on local dataset
                def _find_split(ds: Path, split: str) -> str:
                    for cand in (
                        f"images/{split}2017", f"{split}2017",
                        f"images/{split}", split,
                    ):
                        if (ds / cand).exists():
                            return cand
                    return f"images/{split}"
                train_dir = _find_split(local_ds_path, "train")
                val_dir   = _find_split(local_ds_path, "val")
                if isinstance(orig_names, dict):
                    names_block = "\n".join(f"  {k}: {v}" for k, v in orig_names.items())
                else:
                    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(orig_names))
                lines_out = [
                    f"# Ultralytics data.yaml — {ds_name} (paths remapped by Model DESIGNER)",
                    f"",
                    f"path: {local_ds_path}",
                    f"train: {train_dir}",
                    f"val: {val_dir}",
                    f"",
                    f"nc: {orig_nc}",
                    f"names:",
                    names_block,
                    f"",
                ]
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix="_data.yaml", delete=False,
                    dir=str(BENCHMARK_DIR)
                )
                tmp.write("\n".join(lines_out))
                tmp.flush()
                tmp.close()
                return Path(tmp.name)
            except Exception:
                pass  # fall through to line-by-line remap

    # ── Strategy 3: line-by-line remap ────────────────────────────────────────
    def _remap(val: str) -> str:
        comment = ""
        if "#" in val:
            val, comment = val.split("#", 1)
            val = val.strip()
            comment = " #" + comment
        p = Path(val)
        remapped = _remap_path_obj(p) if p.is_absolute() else p
        return str(remapped) + comment

    new_lines = []
    for line in content.splitlines():
        stripped = line.strip()
        for field in fields:
            if stripped.startswith(f"{field}:"):
                after_colon = line.split(":", 1)[1].strip()
                if after_colon:
                    remapped = _remap(after_colon) if after_colon[0] == "/" else after_colon
                    if field != "path":
                        txt_path = _split_path_for_value(remapped)
                        remapped_txt, changed = _remap_txt_file(txt_path)
                        if changed:
                            remapped = str(remapped_txt)
                    line = f"{field}: {remapped}"
                break
        new_lines.append(line)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix="_data.yaml", delete=False, dir=str(BENCHMARK_DIR)
    )
    tmp.write("\n".join(new_lines))
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


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

    # 6. Plugin-based: generate data.yaml on-the-fly for installed datasets
    try:
        from ..plugins.loader import get_dataset_plugin, discover_plugins
        from ..services.dataset_yaml import generate_data_yaml
        discover_plugins()
        plugin = get_dataset_plugin(dataset.lower())
        if plugin and plugin.is_available():
            cached_yaml = datasets_dir / dataset / "data.yaml"
            cached_yaml.parent.mkdir(parents=True, exist_ok=True)
            cached_yaml.write_text(generate_data_yaml(dataset))
            return cached_yaml
    except Exception:
        pass

    return None


def _list_available_datasets(weight_meta: dict | None = None) -> list[dict]:
    """Return all dataset YAMLs available for benchmarking."""
    import tempfile
    import yaml as _yaml

    datasets_dir = DATA_DIR / "datasets"
    jobs_dir = DATA_DIR / "jobs"
    results: list[dict] = []
    seen: set[str] = set()

    def _add(label: str, yaml_path: Path, source: str, nc_override: int | None = None):
        key = label.lower()
        if key in seen:
            return
        seen.add(key)
        nc = nc_override
        if nc is None:
            try:
                data = _yaml.safe_load(yaml_path.read_text())
                nc = data.get("nc")
            except Exception:
                pass
        results.append({"label": label, "value": label, "yaml_path": str(yaml_path), "nc": nc, "source": source})

    # ── Installed plugins (same source as training jobs) ──────────────────────
    # Generate data.yaml on-the-fly for each available plugin and cache it in
    # the dataset dir so benchmark can reference it.
    try:
        from ..plugins.loader import all_dataset_plugins, discover_plugins
        from ..services.dataset_yaml import generate_data_yaml
        discover_plugins()
        for plugin in all_dataset_plugins():
            try:
                if not plugin.is_available():
                    continue
                ds_name = plugin.name
                ds_dir = datasets_dir / ds_name
                cached_yaml = ds_dir / "data.yaml"
                # Re-generate if missing or stale (older than 1 day)
                import time as _time
                needs_regen = (
                    not cached_yaml.exists()
                    or (_time.time() - cached_yaml.stat().st_mtime) > 86400
                )
                if needs_regen:
                    yaml_content = generate_data_yaml(ds_name)
                    cached_yaml.parent.mkdir(parents=True, exist_ok=True)
                    cached_yaml.write_text(yaml_content)
                _add(ds_name, cached_yaml, "dataset", nc_override=plugin.num_classes)
            except Exception:
                pass
    except Exception:
        pass

    # ── Datasets dir (static data.yaml files, e.g. COCO downloaded manually) ─
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


def _update_benchmark_failed(benchmark_id: str, error: str) -> None:
    """Update an in-progress benchmark record to failed status."""
    out_path = BENCHMARK_DIR / f"{benchmark_id}.json"
    try:
        data = json.loads(out_path.read_text()) if out_path.exists() else {"benchmark_id": benchmark_id}
        data["status"] = "failed"
        data["error"] = error
        data["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        out_path.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def _collect_hsg_decoder_alpha(model) -> list[dict]:
    """Collect RTDETRDecoderSGB alpha values from a loaded Ultralytics model."""
    root = getattr(model, "model", model)
    modules = getattr(root, "named_modules", None)
    if not callable(modules):
        return []

    values: list[dict] = []
    for name, module in modules():
        if module.__class__.__name__ != "RTDETRDecoderSGB":
            continue
        ensure = getattr(module, "_ensure_runtime_attrs", None)
        if callable(ensure):
            try:
                ensure()
            except Exception:
                pass
        alpha = getattr(module, "alpha", None)
        if alpha is None:
            values.append({"module": name, "alpha": None})
            continue
        try:
            values.append({"module": name, "alpha": float(alpha.detach().cpu().reshape(-1)[0])})
        except Exception:
            values.append({"module": name, "alpha": None})
    return values


def _isolated_weight_benchmark_error(meta: dict) -> str | None:
    """Return a blocking reason for isolated checkpoints unsupported here."""
    return None


def _scale_from_rtdetrv2_meta(meta: dict) -> str:
    scale = str(meta.get("model_scale") or "").strip().lower()
    if scale in {"s", "m", "l", "x"}:
        return scale
    arch = str(meta.get("arch_plugin") or meta.get("model_arch") or "").lower()
    if arch.startswith("rtdetrv2_"):
        scale = arch.rsplit("_", 1)[-1]
        if scale in {"s", "m", "l", "x"}:
            return scale
    return "s"


_RTDETRV2_FLOPS_G = {"s": 60.0, "m": 100.0, "l": 136.0, "x": 259.0}
_DINO_DETECTOR_CKPT_URL = "https://huggingface.co/wdy413/DINO/resolve/main/checkpoint0011_4scale.pth"
_COCO_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]
_COCO_ID_TO_INDEX = {cat_id: idx for idx, cat_id in enumerate(_COCO_CATEGORY_IDS)}
_COCO_INDEX_TO_ID = {idx: cat_id for idx, cat_id in enumerate(_COCO_CATEGORY_IDS)}


def _scale_from_dino_meta(meta: dict) -> str:
    scale = str(meta.get("model_scale") or "").strip().lower()
    if scale in {"vits16", "vits8", "vitb16", "vitb8", "resnet50"}:
        return scale
    arch = str(meta.get("arch_plugin") or meta.get("model_arch") or "").lower()
    if arch.startswith("dino_"):
        scale = arch.split("dino_", 1)[1]
        if scale in {"vits16", "vits8", "vitb16", "vitb8", "resnet50"}:
            return scale
    model_name = str(meta.get("model_name") or "").lower()
    for candidate in ("vits16", "vits8", "vitb16", "vitb8", "resnet50"):
        if candidate in model_name:
            return candidate
    return "vits16"


def _profile_dino_runtime(
    *,
    root: Path,
    scale: str,
    pt_path: Path,
    device: str,
    imgsz: int,
    env: dict,
) -> dict[str, float | int | None | str]:
    """Measure DINO backbone forward latency/params/FLOPs in a clean subprocess."""
    import subprocess
    import sys

    script = r"""
import json
import sys
import time

import torch

scale, ckpt_path, device, imgsz_s = sys.argv[1:5]
imgsz = int(imgsz_s)

if scale == "resnet50":
    import torchvision.models as tv_models
    model = tv_models.resnet50(weights=None)
    model.fc = torch.nn.Identity()
else:
    import vision_transformer as vits
    arch, patch = {
        "vits16": ("vit_small", 16),
        "vits8": ("vit_small", 8),
        "vitb16": ("vit_base", 16),
        "vitb8": ("vit_base", 8),
    }.get(scale, ("vit_small", 16))
    model = vits.__dict__[arch](patch_size=patch, num_classes=0)

raw = torch.load(ckpt_path, map_location="cpu")
states = []
if isinstance(raw, dict):
    for key in ("teacher", "student", "model", "state_dict"):
        val = raw.get(key)
        if isinstance(val, dict):
            states.append((key, val))
states.append(("root", raw if isinstance(raw, dict) else {}))

def strip_state(sd):
    out = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        nk = str(k)
        for prefix in (
            "module.backbone.",
            "backbone.",
            "module.encoder_q.",
            "encoder_q.",
            "module.",
        ):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
                break
        out[nk] = v
    return out

model_keys = set(model.state_dict().keys())
best_name, best_state, best_matches = None, {}, -1
for name, sd in states:
    stripped = strip_state(sd)
    matches = sum(1 for k in stripped if k in model_keys)
    if matches > best_matches:
        best_name, best_state, best_matches = name, stripped, matches

if best_state:
    model.load_state_dict(best_state, strict=False)

flops_g = None
try:
    model.eval()
    with torch.no_grad(), torch.profiler.profile(with_flops=True, activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        _ = model(torch.rand(1, 3, imgsz, imgsz))
    total_flops = sum((getattr(evt, "flops", 0) or 0) for evt in prof.key_averages())
    flops_g = total_flops / 1e9 if total_flops else None
except Exception:
    flops_g = None

model = model.to(device).eval()
x = torch.rand(1, 3, imgsz, imgsz, device=device)

def sync():
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()

with torch.no_grad():
    for _ in range(5):
        y = model(x)
    sync()
    n = 30
    t0 = time.perf_counter()
    for _ in range(n):
        y = model(x)
    sync()
    t1 = time.perf_counter()

if isinstance(y, (list, tuple)):
    y0 = y[0]
elif isinstance(y, dict):
    y0 = next((v for v in y.values() if torch.is_tensor(v)), None)
else:
    y0 = y
if torch.is_tensor(y0):
    yd = y0.detach().float()
    output_shape = list(yd.shape)
    feature_dim = int(yd.shape[-1]) if yd.ndim > 0 else 1
    output_mean = float(yd.mean().item())
    output_std = float(yd.std(unbiased=False).item())
    embedding_norm = float(yd.flatten(1).norm(dim=1).mean().item()) if yd.ndim > 1 else float(yd.norm().item())
else:
    output_shape = None
    feature_dim = None
    output_mean = None
    output_std = None
    embedding_norm = None

params = sum(p.numel() for p in model.parameters())
print(json.dumps({
    "inference_ms": (t1 - t0) * 1000.0 / n,
    "params": params,
    "flops_gflops": flops_g,
    "loaded_state": best_name,
    "matched_keys": best_matches,
    "output_shape": output_shape,
    "feature_dim": feature_dim,
    "output_mean": output_mean,
    "output_std": output_std,
    "embedding_norm": embedding_norm,
}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script, scale, str(pt_path), str(device), str(imgsz)],
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        return {
            "inference_ms": None,
            "params": None,
            "flops_gflops": None,
            "error": (proc.stdout or "")[-2000:],
        }
    try:
        return json.loads((proc.stdout or "").strip().splitlines()[-1])
    except Exception:
        return {
            "inference_ms": None,
            "params": None,
            "flops_gflops": None,
            "error": (proc.stdout or "")[-2000:],
        }


def _run_dino_benchmark(
    req: BenchmarkRequest,
    benchmark_id: str,
    meta: dict,
    pt_path: Path,
) -> dict:
    """Benchmark a DINO self-supervised backbone checkpoint.

    DINO checkpoints are feature extractors rather than object detectors, so
    this path reports model/runtime information and leaves detection metrics
    empty instead of forcing the checkpoint through Ultralytics val().
    """
    import os
    import torch
    from dino.installer import ensure_installed

    if int(meta.get("key_count") or 0) <= 0:
        raise ValueError(
            f"{meta.get('model_name') or req.weight_id} is an empty DINO profile placeholder, "
            "not a trained/pretrained checkpoint. Create it with pretrained enabled or run training first."
        )

    root = ensure_installed()
    scale = _scale_from_dino_meta(meta)
    device = req.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str) and device.isdigit():
        device = f"cuda:{device}"

    env = os.environ.copy()
    backend_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join([str(root), backend_root, env.get("PYTHONPATH", "")])
    env["PYTHONFAULTHANDLER"] = "1"

    t0 = time.time()
    runtime = _profile_dino_runtime(
        root=root,
        scale=scale,
        pt_path=pt_path,
        device=device,
        imgsz=req.imgsz,
        env=env,
    )
    elapsed_s = time.time() - t0
    if runtime.get("error") and runtime.get("inference_ms") is None:
        raise ValueError(f"DINO benchmark failed: {runtime['error']}")

    result = {
        "benchmark_id": benchmark_id,
        "weight_id": req.weight_id,
        "dataset": req.dataset,
        "split": req.split,
        "status": "completed",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(elapsed_s, 1),
        "mAP50": None,
        "mAP50_95": None,
        "precision": None,
        "recall": None,
        "fitness": None,
        "per_class": [],
        "confusion_matrix": None,
        "preprocess_ms": 0.0,
        "inference_ms": round(runtime.get("inference_ms"), 2) if runtime.get("inference_ms") is not None else None,
        "postprocess_ms": 0.0,
        "params": int(runtime.get("params") or meta.get("param_count") or 0) or None,
        "flops_gflops": round(float(runtime["flops_gflops"]), 3) if runtime.get("flops_gflops") is not None else None,
        "hsg_decoder_alpha": [],
        "conf": req.conf,
        "iou": req.iou,
        "imgsz": req.imgsz,
        "source_type": "dino",
        "benchmark_type": "backbone",
        "loaded_state": runtime.get("loaded_state"),
        "matched_keys": runtime.get("matched_keys"),
        "feature_dim": runtime.get("feature_dim"),
        "output_shape": runtime.get("output_shape"),
        "embedding_norm": round(float(runtime["embedding_norm"]), 4) if runtime.get("embedding_norm") is not None else None,
        "output_mean": round(float(runtime["output_mean"]), 6) if runtime.get("output_mean") is not None else None,
        "output_std": round(float(runtime["output_std"]), 6) if runtime.get("output_std") is not None else None,
        "backbone_metrics": {
            "feature_dim": runtime.get("feature_dim"),
            "output_shape": runtime.get("output_shape"),
            "embedding_norm": round(float(runtime["embedding_norm"]), 4) if runtime.get("embedding_norm") is not None else None,
            "output_mean": round(float(runtime["output_mean"]), 6) if runtime.get("output_mean") is not None else None,
            "output_std": round(float(runtime["output_std"]), 6) if runtime.get("output_std") is not None else None,
            "matched_keys": runtime.get("matched_keys"),
            "loaded_state": runtime.get("loaded_state"),
        },
    }
    (BENCHMARK_DIR / f"{benchmark_id}.json").write_text(json.dumps(result, indent=2))
    return result


def _dino_detector_root() -> Path:
    from dino.detector_installer import ensure_installed

    return ensure_installed(build_ops=True)


def _ensure_dino_detector_checkpoint(meta: dict, pt_path: Path) -> Path:
    """Return a DINO detector checkpoint, downloading the official R50-4scale ckpt if needed."""
    import torch
    from torch.hub import download_url_to_file
    from ..config import DATA_DIR

    try:
        state = torch.load(pt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and isinstance(state.get("model"), dict):
            return pt_path
    except Exception:
        pass

    cache_dir = DATA_DIR / "weights" / "_downloads" / "dino_detector"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / "checkpoint0011_4scale.pth"
    if not cached.exists() or cached.stat().st_size < 100_000_000:
        download_url_to_file(_DINO_DETECTOR_CKPT_URL, str(cached), progress=True)
    return cached


def _prepare_dino_detector_coco(data_yaml: Path, out_dir: Path) -> tuple[Path, Path]:
    """Create a COCO2017-shaped directory for IDEA DINO detector eval/train."""
    import os
    import yaml as _yaml
    from PIL import Image

    data = _yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root = Path(str(data.get("path") or "."))
    if not root.is_absolute():
        root = data_yaml.parent / root
    raw_names = data.get("names") or {}
    names = {int(k): str(v) for k, v in raw_names.items()} if isinstance(raw_names, dict) else {
        i: str(v) for i, v in enumerate(raw_names)
    }

    def _images(split: str) -> list[Path]:
        raw = data.get(split) or data.get("val") or data.get("train")
        p = Path(str(raw))
        if p.suffix == ".txt":
            txt = p if p.is_absolute() else root / p
            return [Path(x.strip()) for x in txt.read_text(encoding="utf-8").splitlines() if x.strip()]
        base = p if p.is_absolute() else root / p
        return sorted(
            x for x in base.rglob("*")
            if x.is_file() and x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )

    def _label_for_image(image_path: Path) -> Path:
        parts = list(image_path.parts)
        for i, part in enumerate(parts):
            if part == "images":
                parts[i] = "labels"
                return Path(*parts).with_suffix(".txt")
        return image_path.with_suffix(".txt")

    def _export(split: str, target_name: str) -> None:
        img_dir = out_dir / target_name
        ann_dir = out_dir / "annotations"
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        images = []
        annotations = []
        ann_id = 1
        for img_id, src in enumerate(_images(split), start=1):
            dst_name = f"{img_id:012d}{src.suffix.lower() or '.jpg'}"
            dst = img_dir / dst_name
            if not dst.exists():
                try:
                    os.symlink(src.resolve(), dst)
                except OSError:
                    import shutil

                    shutil.copy2(src, dst)
            with Image.open(src) as im:
                width, height = im.size
            images.append({"id": img_id, "file_name": dst_name, "width": width, "height": height})
            label_path = _label_for_image(src)
            if not label_path.exists():
                continue
            for raw in label_path.read_text(encoding="utf-8").splitlines():
                cols = raw.strip().split()
                if len(cols) < 5:
                    continue
                cls_idx = int(float(cols[0]))
                cat_id = _COCO_INDEX_TO_ID.get(cls_idx, cls_idx)
                xc, yc, bw, bh = [float(v) for v in cols[1:5]]
                box_w = bw * width
                box_h = bh * height
                x = (xc * width) - box_w / 2
                y = (yc * height) - box_h / 2
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cat_id),
                    "bbox": [max(0.0, x), max(0.0, y), max(0.0, box_w), max(0.0, box_h)],
                    "area": max(0.0, box_w) * max(0.0, box_h),
                    "iscrowd": 0,
                })
                ann_id += 1
        categories = [
            {"id": _COCO_INDEX_TO_ID.get(i, i), "name": names.get(i, str(i))}
            for i in sorted(names)
        ]
        (ann_dir / f"instances_{target_name}.json").write_text(json.dumps({
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }), encoding="utf-8")

    _export("train", "train2017")
    _export("val", "val2017")
    return out_dir, out_dir / "annotations" / "instances_val2017.json"


def _parse_dino_detector_log(log_path: Path) -> dict[str, float | None]:
    if not log_path.exists():
        return {"mAP50_95": None, "mAP50": None, "recall": None}
    last = {}
    for raw in log_path.read_text(encoding="utf-8").splitlines():
        try:
            last = json.loads(raw)
        except json.JSONDecodeError:
            continue
    values = last.get("test_coco_eval_bbox") or []
    return {
        "mAP50_95": float(values[0]) if len(values) > 0 else None,
        "mAP50": float(values[1]) if len(values) > 1 else None,
        "recall": float(values[8]) if len(values) > 8 else None,
    }


def _load_dino_confusion(results_path: Path, names: dict[int, str], conf: float = 0.001) -> dict | None:
    if not results_path.exists():
        return None
    try:
        import torch
        import numpy as _np
        from torchvision.ops import box_iou

        data = torch.load(results_path, map_location="cpu", weights_only=False)
    except Exception:
        return None

    gt_list = data.get("gt_info") or []
    pred_list = data.get("res_info") or []
    n = len(names)
    matrix = _np.zeros((n + 1, n + 1), dtype=int)

    def _cxcywh_to_xyxy(boxes):
        if boxes.numel() == 0:
            return boxes.reshape(0, 4)
        x, y, w, h = boxes.unbind(-1)
        return torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=-1)

    for gt, pred in zip(gt_list, pred_list):
        gt = gt.float()
        pred = pred.float()
        gt_boxes = _cxcywh_to_xyxy(gt[:, :4]) if gt.numel() else torch.empty((0, 4))
        gt_labels = [int(x.item()) for x in gt[:, 4]] if gt.numel() else []
        pred = pred[pred[:, 4] >= float(conf)] if pred.numel() else pred.reshape(0, 6)
        pred_boxes = _cxcywh_to_xyxy(pred[:, :4]) if pred.numel() else torch.empty((0, 4))
        pred_scores = pred[:, 4] if pred.numel() else torch.empty((0,))
        pred_labels = [int(x.item()) for x in pred[:, 5]] if pred.numel() else []
        used_preds: set[int] = set()
        ious = box_iou(gt_boxes, pred_boxes) if len(gt_boxes) and len(pred_boxes) else torch.empty((len(gt_boxes), len(pred_boxes)))
        for gi, gt_label in enumerate(gt_labels):
            gt_idx = _COCO_ID_TO_INDEX.get(gt_label, gt_label)
            pred_idx = n
            if ious.shape[1]:
                order = torch.argsort(pred_scores, descending=True).tolist()
                best = next((pi for pi in order if pi not in used_preds and float(ious[gi, pi]) >= 0.5), None)
                if best is not None:
                    used_preds.add(best)
                    pred_idx = _COCO_ID_TO_INDEX.get(pred_labels[best], pred_labels[best])
            if 0 <= gt_idx < n and 0 <= pred_idx <= n:
                matrix[gt_idx, pred_idx] += 1
        for pi, pred_label in enumerate(pred_labels):
            if pi in used_preds:
                continue
            pred_idx = _COCO_ID_TO_INDEX.get(pred_label, pred_label)
            if 0 <= pred_idx < n:
                matrix[n, pred_idx] += 1

    return {"matrix": matrix.tolist(), "names": [names.get(i, str(i)) for i in range(n)] + ["background"]}


def _run_dino_detector_benchmark(
    req: BenchmarkRequest,
    benchmark_id: str,
    meta: dict,
    pt_path: Path,
    data_yaml: Path,
) -> dict:
    """Run IDEA-Research/DINO detector evaluation for real COCO detection metrics."""
    import os
    import subprocess
    import sys
    import yaml as _yaml

    root = _dino_detector_root()
    ckpt = _ensure_dino_detector_checkpoint(meta, pt_path)
    out_dir = BENCHMARK_DIR / benchmark_id / "dino_detector"
    coco_dir, _ = _prepare_dino_detector_coco(data_yaml, out_dir / "coco")
    run_dir = out_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    device = req.device or ("cuda:0" if __import__("torch").cuda.is_available() else "cpu")
    if isinstance(device, str) and device.isdigit():
        device = f"cuda:{device}"

    env = os.environ.copy()
    backend_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join([str(root), backend_root, env.get("PYTHONPATH", "")])
    env["PYTHONFAULTHANDLER"] = "1"
    env.setdefault("RANK", "0")
    env.setdefault("WORLD_SIZE", "1")
    env.setdefault("LOCAL_RANK", "0")
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    try:
        port_offset = int(benchmark_id[:4], 16) % 1000
    except ValueError:
        port_offset = sum(ord(ch) for ch in benchmark_id[:8]) % 1000
    env.setdefault("MASTER_PORT", str(29600 + port_offset))

    cmd = [
        sys.executable,
        "main.py",
        "-c",
        "config/DINO/DINO_4scale.py",
        "--coco_path",
        str(coco_dir),
        "--output_dir",
        str(run_dir),
        "--eval",
        "--save_results",
        "--resume",
        str(ckpt),
        "--device",
        str(device),
        "--num_workers",
        "0",
        "--options",
        f"batch_size={max(1, int(req.batch or 1))}",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(root), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed_s = time.time() - t0
    (out_dir / "stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    if proc.returncode != 0:
        raise ValueError(f"DINO detector benchmark failed with exit code {proc.returncode}: {(proc.stdout or '')[-2000:]}")

    stats = _parse_dino_detector_log(run_dir / "log.txt")
    per_class, mean_precision = _load_rtdetrv2_per_class(run_dir / "eval.pth", data_yaml)
    try:
        raw_names = (_yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}).get("names") or {}
        names = {int(k): str(v) for k, v in raw_names.items()} if isinstance(raw_names, dict) else {
            i: str(v) for i, v in enumerate(raw_names)
        }
    except Exception:
        names = {}
    confusion = _load_dino_confusion(run_dir / "results-0.pkl", names, req.conf)

    result = {
        "benchmark_id": benchmark_id,
        "weight_id": req.weight_id,
        "dataset": req.dataset,
        "split": req.split,
        "status": "completed",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(elapsed_s, 1),
        "mAP50": round(stats["mAP50"], 4) if stats["mAP50"] is not None else None,
        "mAP50_95": round(stats["mAP50_95"], 4) if stats["mAP50_95"] is not None else None,
        "precision": round(mean_precision, 4) if mean_precision is not None else None,
        "recall": round(stats["recall"], 4) if stats["recall"] is not None else None,
        "fitness": None,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "preprocess_ms": None,
        "inference_ms": None,
        "postprocess_ms": None,
        "params": int(meta.get("param_count") or 0) or None,
        "flops_gflops": None,
        "hsg_decoder_alpha": [],
        "conf": req.conf,
        "iou": req.iou,
        "imgsz": req.imgsz,
        "source_type": "dino",
        "benchmark_type": "detector",
        "stdout_log": str(out_dir / "stdout.log"),
    }
    (BENCHMARK_DIR / f"{benchmark_id}.json").write_text(json.dumps(result, indent=2))
    return result


def _parse_rtdetrv2_coco_stats(output: str) -> dict[str, float | None]:
    values: list[float] = []
    for line in output.splitlines():
        if "Average Precision" in line or "Average Recall" in line:
            try:
                values.append(float(line.rsplit("=", 1)[-1].strip()))
            except ValueError:
                pass
    return {
        "mAP50_95": values[0] if len(values) > 0 else None,
        "mAP50": values[1] if len(values) > 1 else None,
        "mAP75": values[2] if len(values) > 2 else None,
        "recall": values[8] if len(values) > 8 else None,
    }


def _parse_rtdetrv2_latency(output: str) -> float | None:
    matches = re.findall(r"Total time: .*?\(([-+]?\d*\.?\d+)\s+s\s*/\s+it\)", output)
    if not matches:
        return None
    try:
        return float(matches[-1]) * 1000.0
    except ValueError:
        return None


def _load_rtdetrv2_per_class(eval_path: Path, data_yaml: Path) -> tuple[list[dict], float | None]:
    if not eval_path.exists():
        return [], None
    try:
        import torch
        import yaml as _yaml
        import numpy as _np

        eval_data = torch.load(eval_path, map_location="cpu", weights_only=False)
    except Exception:
        return [], None

    precision = eval_data.get("precision")
    recall = eval_data.get("recall")
    if precision is None:
        return [], None

    try:
        names_raw = (_yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}).get("names") or {}
        if isinstance(names_raw, list):
            names = {i: str(n) for i, n in enumerate(names_raw)}
        else:
            names = {int(k): str(v) for k, v in names_raw.items()}
    except Exception:
        names = {}

    p = _np.asarray(precision)
    r = _np.asarray(recall) if recall is not None else None
    # precision shape: [TxRxKxAxM], recall shape: [TxKxAxM]
    if p.ndim != 5:
        return [], None
    area_idx = 0
    maxdet_idx = p.shape[4] - 1
    ap_all = p[:, :, :, area_idx, maxdet_idx]
    ap50_all = p[0, :, :, area_idx, maxdet_idx]
    class_count = ap_all.shape[2]
    per_class: list[dict] = []
    precisions_for_mean: list[float] = []
    for cls_id in range(class_count):
        ap_vals = ap_all[:, :, cls_id]
        ap_vals = ap_vals[ap_vals > -1]
        ap50_vals = ap50_all[:, cls_id]
        ap50_vals = ap50_vals[ap50_vals > -1]
        ap = float(ap_vals.mean()) if ap_vals.size else None
        ap50 = float(ap50_vals.mean()) if ap50_vals.size else None
        rec = None
        if r is not None and r.ndim == 4 and cls_id < r.shape[1]:
            r_vals = r[:, cls_id, area_idx, maxdet_idx]
            r_vals = r_vals[r_vals > -1]
            rec = float(r_vals.mean()) if r_vals.size else None
        # COCO eval does not expose final per-class precision directly in the
        # same way Ultralytics does; this is mean precision over IoU/recall.
        prec = ap
        if prec is not None:
            precisions_for_mean.append(prec)
        per_class.append({
            "class_id": cls_id,
            "class_name": names.get(cls_id, str(cls_id)),
            "ap50": round(ap50, 4) if ap50 is not None else None,
            "ap50_95": round(ap, 4) if ap is not None else None,
            "precision": round(prec, 4) if prec is not None else None,
            "recall": round(rec, 4) if rec is not None else None,
            "f1": None,
        })
    mean_precision = sum(precisions_for_mean) / len(precisions_for_mean) if precisions_for_mean else None
    return per_class, mean_precision


def _profile_rtdetrv2_runtime(
    *,
    root: Path,
    upstream_config: Path,
    pt_path: Path,
    device: str,
    imgsz: int,
    env: dict,
) -> dict[str, float | None]:
    """Measure pure forward/postprocess latency in a clean upstream subprocess."""
    import subprocess
    import sys

    script = r"""
import json
import sys
import time
import torch
from src.core import YAMLConfig

config, ckpt_path, device, imgsz_s = sys.argv[1:5]
imgsz = int(imgsz_s)
cfg = YAMLConfig(config)
model = cfg.model
postprocessor = cfg.postprocessor
state = torch.load(ckpt_path, map_location="cpu")
if isinstance(state, dict):
    if "ema" in state and isinstance(state["ema"], dict) and "module" in state["ema"]:
        state = state["ema"]["module"]
    elif "model" in state:
        state = state["model"]
if isinstance(state, dict):
    model.load_state_dict(state, strict=False)
model = model.to(device).eval()
try:
    postprocessor = postprocessor.to(device).eval()
except Exception:
    pass
x = torch.rand(1, 3, imgsz, imgsz, device=device)
sizes = torch.tensor([[imgsz, imgsz]], device=device)
with torch.no_grad():
    for _ in range(3):
        y = model(x)
        _ = postprocessor(y, sizes)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        y = model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(n):
        _ = postprocessor(y, sizes)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t2 = time.perf_counter()
params = sum(p.numel() for p in model.parameters())
print(json.dumps({"inference_ms": (t1 - t0) * 1000.0 / n, "postprocess_ms": (t2 - t1) * 1000.0 / n, "params": params}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script, str(upstream_config), str(pt_path), str(device), str(imgsz)],
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        return {"inference_ms": None, "postprocess_ms": None, "params": None}
    try:
        return json.loads((proc.stdout or "").strip().splitlines()[-1])
    except Exception:
        return {"inference_ms": None, "postprocess_ms": None, "params": None}


def _run_rtdetrv2_benchmark(
    req: BenchmarkRequest,
    benchmark_id: str,
    meta: dict,
    pt_path: Path,
    data_yaml: Path,
) -> dict:
    """Run upstream RT-DETRv2 validation and return a Model Designer benchmark record."""
    import os
    import subprocess
    import sys
    import yaml as _yaml
    from ..services.rtdetrv2_trainer import _prepare_coco_dataset, _SCALE_TO_CONFIG
    from rtdetrv2.installer import ensure_installed

    scale = _scale_from_rtdetrv2_meta(meta)
    spec = _SCALE_TO_CONFIG.get(scale) or _SCALE_TO_CONFIG["s"]
    root = ensure_installed()
    upstream_config = root / spec["config"]
    if not upstream_config.exists():
        raise ValueError(f"RT-DETRv2 upstream config not found: {upstream_config}")

    out_dir = BENCHMARK_DIR / benchmark_id
    out_dir.mkdir(parents=True, exist_ok=True)
    export_dir = out_dir / "rtdetrv2_dataset"
    # Upstream RT-DETR's config instantiates train/val dataloaders even for
    # --test-only.  For benchmarking, avoid scanning/exporting the full train
    # split when the requested split is val/test; point both dataloaders at the
    # selected split.  This mirrors Ultralytics model.val(split=...) behavior.
    try:
        data_for_split = _yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
        selected = data_for_split.get(req.split) or data_for_split.get("val") or data_for_split.get("train")
        if selected is not None:
            data_for_split["train"] = selected
            data_for_split["val"] = selected
            bench_yaml = out_dir / "data_selected_split.yaml"
            bench_yaml.write_text(_yaml.safe_dump(data_for_split, sort_keys=False), encoding="utf-8")
            data_yaml = bench_yaml
    except Exception:
        pass
    image_root, train_json, val_json, nc = _prepare_coco_dataset(
        benchmark_id,
        data_yaml,
        export_dir,
    )

    workers = 0
    batch = max(1, int(req.batch or 1))
    device = req.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    if isinstance(device, str) and device.isdigit():
        device = f"cuda:{device}"
    updates = [
        f"num_classes={nc}",
        "remap_mscoco_category=False",
        f"val_dataloader.dataset.img_folder='{image_root}'",
        f"val_dataloader.dataset.ann_file='{val_json}'",
        f"val_dataloader.total_batch_size={batch}",
        f"val_dataloader.num_workers={workers}",
        f"train_dataloader.dataset.img_folder='{image_root}'",
        f"train_dataloader.dataset.ann_file='{train_json}'",
        f"train_dataloader.total_batch_size={batch}",
        f"train_dataloader.num_workers={workers}",
    ]

    cmd = [
        sys.executable,
        "tools/train.py",
        "-c",
        str(upstream_config),
        "--test-only",
        "--resume",
        str(pt_path),
        "--output-dir",
        str(out_dir / "run"),
        "-u",
        *updates,
    ]
    if device:
        cmd.extend(["--device", str(device)])

    env = os.environ.copy()
    backend_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join([str(root), backend_root, env.get("PYTHONPATH", "")])
    env["PYTHONFAULTHANDLER"] = "1"

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed_s = time.time() - t0
    (out_dir / "stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    if proc.returncode != 0:
        raise ValueError(f"RT-DETRv2 benchmark failed with exit code {proc.returncode}: {(proc.stdout or '')[-2000:]}")

    stats = _parse_rtdetrv2_coco_stats(proc.stdout or "")
    per_class, mean_precision = _load_rtdetrv2_per_class(out_dir / "run" / "eval.pth", data_yaml)
    eval_iter_ms = _parse_rtdetrv2_latency(proc.stdout or "")
    runtime = _profile_rtdetrv2_runtime(
        root=root,
        upstream_config=upstream_config,
        pt_path=pt_path,
        device=device,
        imgsz=req.imgsz,
        env=env,
    )
    result = {
        "benchmark_id": benchmark_id,
        "weight_id": req.weight_id,
        "dataset": req.dataset,
        "split": req.split,
        "status": "completed",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(elapsed_s, 1),
        "mAP50": round(stats["mAP50"], 4) if stats["mAP50"] is not None else None,
        "mAP50_95": round(stats["mAP50_95"], 4) if stats["mAP50_95"] is not None else None,
        "mAP75": round(stats["mAP75"], 4) if stats["mAP75"] is not None else None,
        "precision": round(mean_precision, 4) if mean_precision is not None else None,
        "recall": round(stats["recall"], 4) if stats["recall"] is not None else None,
        "fitness": None,
        "per_class": per_class,
        "confusion_matrix": None,
        "preprocess_ms": None,
        "inference_ms": round(runtime.get("inference_ms"), 2) if runtime.get("inference_ms") is not None else None,
        "postprocess_ms": round(runtime.get("postprocess_ms"), 2) if runtime.get("postprocess_ms") is not None else None,
        "eval_iter_ms": round(eval_iter_ms, 2) if eval_iter_ms is not None else None,
        "params": int(runtime.get("params") or meta.get("param_count") or 0) or None,
        "flops_gflops": _RTDETRV2_FLOPS_G.get(scale),
        "hsg_decoder_alpha": [],
        "conf": req.conf,
        "iou": req.iou,
        "imgsz": req.imgsz,
        "source_type": "rtdetrv2",
        "stdout_log": str(out_dir / "stdout.log"),
    }
    (BENCHMARK_DIR / f"{benchmark_id}.json").write_text(json.dumps(result, indent=2))
    return result


def _run_benchmark(req: BenchmarkRequest, benchmark_id: str | None = None) -> dict:
    """Blocking — runs in threadpool."""
    import sys
    import torch
    from ultralytics import YOLO
    from pathlib import Path as _Path

    # Generate benchmark_id early and write running status for persistence
    if benchmark_id is None:
        benchmark_id = uuid.uuid4().hex[:12]
    out_path = BENCHMARK_DIR / f"{benchmark_id}.json"
    out_path.write_text(json.dumps({
        "benchmark_id": benchmark_id,
        "weight_id": req.weight_id,
        "dataset": req.dataset,
        "split": req.split,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "conf": req.conf,
        "iou": req.iou,
        "imgsz": req.imgsz,
    }, indent=2))

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
    isolated_error = _isolated_weight_benchmark_error(meta)
    if isolated_error:
        raise ValueError(isolated_error)

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
    # Rewrite absolute paths in data.yaml that may point to another machine
    data_yaml = _rewrite_yaml_paths(data_yaml)

    if str(meta.get("source_type") or "").lower() == "dino":
        scale = _scale_from_dino_meta(meta)
        if scale == "resnet50" or str(meta.get("benchmark_type") or meta.get("task") or "").lower() == "detector":
            return _run_dino_detector_benchmark(req, benchmark_id, meta, pt_path, data_yaml)
        raise ValueError(
            f"{meta.get('model_name') or req.weight_id} is a facebookresearch/DINO self-supervised backbone checkpoint. "
            "COCO mAP, per-class precision/recall, and confusion matrix require a DINO detector checkpoint "
            "(IDEA-Research/DINO, e.g. dino_resnet50). Re-create the weight as DINO ResNet-50 detector with pretrained enabled."
        )

    if str(meta.get("source_type") or "").lower() == "rtdetrv2":
        if int(meta.get("key_count") or 0) <= 0:
            raise ValueError(
                f"{meta.get('model_name') or req.weight_id} is an empty RT-DETRv2 profile placeholder, "
                "not a trained/pretrained checkpoint. Create it with pretrained enabled or run training first."
            )
        return _run_rtdetrv2_benchmark(req, benchmark_id, meta, pt_path, data_yaml)

    # Delete stale .cache files — they embed absolute paths from the machine that
    # built them, so they will be invalid on a different machine.  Ultralytics
    # will rebuild them automatically on first use.
    try:
        import yaml as _yaml
        _yaml_data = _yaml.safe_load(data_yaml.read_text())
        _ds_path = _yaml_data.get("path", "")
        if _ds_path:
            for _sub in ("labels", "images", "."):
                _cache_dir = Path(_ds_path) / _sub if _sub != "." else Path(_ds_path)
                for _cf in _cache_dir.glob("*.cache"):
                    try:
                        _cf.unlink()
                    except Exception:
                        pass
    except Exception:
        pass

    try:
        model = YOLO(str(pt_path))
    except (KeyError, Exception) as e:
        raise ValueError(
            f"Cannot load weight file for benchmarking: {e}. "
            "This weight may have been created with an older version. "
            "Try re-creating the empty weight to regenerate the file."
        )
    hsg_decoder_alpha = _collect_hsg_decoder_alpha(model)

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

    # Resolve class names: dataset data.yaml names take priority over model.names.
    # This ensures correct labels when using empty weights or weights trained on a
    # different dataset's class set (e.g. empty-weight evaluation on idd returns
    # numeric ids from model.names but the dataset has proper class names in data.yaml).
    import yaml as _yaml_cls
    _dataset_names: dict[int, str] = {}
    try:
        _yaml_cls_data = _yaml_cls.safe_load(data_yaml.read_text())
        _raw_names = _yaml_cls_data.get("names") or {}
        if isinstance(_raw_names, list):
            _dataset_names = {i: str(n) for i, n in enumerate(_raw_names)}
        elif isinstance(_raw_names, dict):
            _dataset_names = {int(k): str(v) for k, v in _raw_names.items()}
    except Exception:
        pass
    # Merge: dataset names override model names for any matching class id
    names: dict[int, str] = {**(model.names or {}), **_dataset_names}

    # Per-class metrics
    per_class: list[dict] = []
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
            # nc is matrix dimension - 1 (last row/col is background).
            # Use matrix shape as the authoritative nc so labels always match
            # the matrix even when model.names count differs from dataset nc.
            _cm_nc = matrix.shape[0] - 1
            confusion_data = {
                "matrix": matrix.tolist(),
                "names": [names.get(i, str(i)) for i in range(_cm_nc)] + ["background"],
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
        "benchmark_id": benchmark_id,
        "weight_id": req.weight_id,
        "dataset": req.dataset,
        "split": req.split,
        "status": "completed",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
        "hsg_decoder_alpha": hsg_decoder_alpha,
        # Config
        "conf": req.conf,
        "iou": req.iou,
        "imgsz": req.imgsz,
    }

    # Save result to disk (replaces running status)
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
    from ..services.task_queue import enqueue, complete, cancel, TaskType

    benchmark_id = uuid.uuid4().hex[:12]
    task_id, admitted, admission_msg = enqueue(
        TaskType.BENCHMARK,
        ref_id=benchmark_id,
        payload={"weight_id": req.weight_id, "dataset": req.dataset},
        gpu_device=req.device if req.device.startswith("cuda") else None,
    )
    if not admitted:
        cancel(task_id)
        raise HTTPException(
            409,
            f"GPU is busy: {admission_msg}",
        )

    _task_error: str | None = None
    try:
        result = await asyncio.to_thread(_run_benchmark, req, benchmark_id)
        logger.log("system", "INFO", "Benchmark complete", {
            "weight_id": req.weight_id,
            "dataset": req.dataset,
            "mAP50": result.get("mAP50"),
        })
        return result
    except ValueError as e:
        _task_error = str(e)
        _update_benchmark_failed(benchmark_id, _task_error)
        raise HTTPException(400, str(e))
    except Exception as e:
        _task_error = str(e)
        _update_benchmark_failed(benchmark_id, _task_error)
        raise HTTPException(500, f"Benchmark failed: {e}")
    finally:
        complete(task_id, error=_task_error)


@router.get("/history", summary="List past benchmark results")
async def list_benchmarks(weight_id: str | None = None, limit: int = int(_BENCHMARK_API_DEFAULTS.get("history_limit", 20))):
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
