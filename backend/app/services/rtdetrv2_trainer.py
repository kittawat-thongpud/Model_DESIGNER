"""Isolated RT-DETRv2 trainer integration.

This runs upstream ``lyuwenyu/RT-DETR`` in a subprocess while keeping Model
Designer's job records, logs, SSE events, queue ownership, and dataset
selection behavior.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ..config import DATASETS_DIR, JOBS_DIR, WEIGHTS_DIR
from ..constants import train_channel
from . import dataset_registry, dataset_yaml, event_bus, job_storage


_SCALE_TO_CONFIG = {
    "s": {
        "config": "configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml",
        "checkpoint": "https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth",
    },
    "m": {
        "config": "configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml",
        "checkpoint": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_m_7x_coco_ema.pth",
    },
    "l": {
        "config": "configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml",
        "checkpoint": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth",
    },
    "x": {
        "config": "configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml",
        "checkpoint": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r101vd_6x_coco_from_paddle.pth",
    },
}


def _publish(job_id: str, job: dict) -> None:
    event_bus.publish_sync(train_channel(job_id), {"type": "status", **job})


def _set_job(job_id: str, **updates: Any) -> dict | None:
    job = job_storage.load_job(job_id)
    if not job:
        return None
    job.update(updates)
    job_storage.save_job(job)
    _publish(job_id, job)
    return job


def _read_data_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve_split_paths(data: dict[str, Any], split: str) -> list[Path]:
    root = Path(str(data.get("path") or "."))
    raw = data.get(split) or data.get("train")
    if raw is None:
        return []
    raw_path = Path(str(raw))
    if raw_path.suffix == ".txt":
        txt = raw_path if raw_path.is_absolute() else root / raw_path
        return [
            Path(line.strip())
            for line in txt.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    base = raw_path if raw_path.is_absolute() else root / raw_path
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt"


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:
        import cv2

        im = cv2.imread(str(path))
        if im is None:
            raise RuntimeError(f"Could not read image size: {path}")
        h, w = im.shape[:2]
        return w, h


def _normalise_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    if isinstance(names, list):
        return [str(x) for x in names]
    return []


def _export_split_to_coco(
    image_paths: list[Path],
    *,
    out_json: Path,
    image_root: Path,
) -> None:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1

    for img_id, image_path in enumerate(image_paths, start=1):
        width, height = _image_size(image_path)
        try:
            file_name = str(image_path.relative_to(image_root))
        except ValueError:
            file_name = image_path.name
        images.append({"id": img_id, "file_name": file_name, "width": width, "height": height})

        label_path = _label_path_for_image(image_path)
        if not label_path.exists():
            continue
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            cols = raw.strip().split()
            if len(cols) < 5:
                continue
            cls_id = int(float(cols[0]))
            xc, yc, bw, bh = [float(v) for v in cols[1:5]]
            box_w = bw * width
            box_h = bh * height
            x = (xc * width) - box_w / 2
            y = (yc * height) - box_h / 2
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [x, y, box_w, box_h],
                    "area": max(box_w, 0.0) * max(box_h, 0.0),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"images": images, "annotations": annotations}), encoding="utf-8")


def _prepare_coco_dataset(job_id: str, data_yaml_path: Path, out_dir: Path) -> tuple[Path, Path, Path, int]:
    data = _read_data_yaml(data_yaml_path)
    names = _normalise_names(data.get("names"))
    nc = int(data.get("nc") or len(names) or 1)
    image_root = Path(str(data.get("path") or DATASETS_DIR)).resolve()
    categories = [{"id": i, "name": names[i] if i < len(names) else f"class{i}"} for i in range(nc)]

    train_paths = _resolve_split_paths(data, "train")
    val_paths = _resolve_split_paths(data, "val") or train_paths
    if not train_paths:
        raise RuntimeError(f"No train images found from {data_yaml_path}")
    if not val_paths:
        raise RuntimeError(f"No val images found from {data_yaml_path}")

    ann_dir = out_dir / "annotations"
    train_json = ann_dir / "instances_train.json"
    val_json = ann_dir / "instances_val.json"

    _export_split_to_coco(train_paths, out_json=train_json, image_root=image_root)
    _export_split_to_coco(val_paths, out_json=val_json, image_root=image_root)

    for p in (train_json, val_json):
        payload = json.loads(p.read_text(encoding="utf-8"))
        payload["categories"] = categories
        p.write_text(json.dumps(payload), encoding="utf-8")

    job_storage.append_job_log(
        job_id,
        "INFO",
        f"RT-DETRv2 COCO export: train={len(train_paths)} images, val={len(val_paths)} images, classes={nc}",
        {"train_json": str(train_json), "val_json": str(val_json), "image_root": str(image_root)},
    )
    return image_root, train_json, val_json, nc


def _checkpoint_candidates(out_dir: Path) -> list[Path]:
    return [
        out_dir / "best.pth",
        out_dir / "last.pth",
        out_dir / "checkpoint.pth",
        *sorted(out_dir.rglob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True),
    ]


def _save_weight(job_id: str, out_dir: Path) -> str | None:
    checkpoint = next((p for p in _checkpoint_candidates(out_dir) if p.exists()), None)
    if checkpoint is None:
        return None
    from . import weight_storage

    job = job_storage.load_job(job_id) or {}
    weight_id = uuid.uuid4().hex[:12]
    dest_dir = WEIGHTS_DIR / weight_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, dest_dir / "weight.pt")
    cfg = job.get("config", {})
    weight_storage.save_weight_meta(
        model_id=job.get("model_id", ""),
        model_name=job.get("model_name", "RT-DETRv2"),
        model_scale=job.get("model_scale", ""),
        job_id=job_id,
        dataset=cfg.get("dataset_name") or cfg.get("data", ""),
        epochs_trained=job.get("total_epochs", 0),
        final_accuracy=None,
        final_loss=None,
        weight_id=weight_id,
        parent_weight_id=None,
        total_time=None,
        device=str(cfg.get("device", "")),
    )
    meta = weight_storage.load_weight_meta(weight_id)
    if meta:
        meta.update(
            {
                "arch_plugin": cfg.get("model_arch"),
                "model_arch": cfg.get("model_arch"),
                "source_type": "rtdetrv2",
                "train_args": {
                    "model_arch": cfg.get("model_arch"),
                    "data": cfg.get("data"),
                    "epochs": cfg.get("epochs"),
                    "batch": cfg.get("batch"),
                },
            }
        )
        weight_storage._store.save(weight_id, meta)
    return weight_id


def _resolve_checkpoint(value: str) -> Path | None:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"false", "0", "none", "true"}:
        return None
    p = Path(raw)
    if p.exists():
        return p
    try:
        from . import weight_storage

        candidate = weight_storage.weight_pt_path(raw)
        if candidate.exists():
            return candidate
    except Exception:
        return None
    return None


def _append_output_log(job_id: str, line: str) -> None:
    text = line.strip()
    if not text:
        return
    job_storage.append_job_log(job_id, "INFO", text)
    m = re.search(r"(?:epoch|Epoch)\D+(\d+)", text)
    if m:
        try:
            epoch = int(m.group(1))
            _set_job(job_id, epoch=epoch, message=f"Training epoch {epoch}")
        except Exception:
            pass


def run_worker(payload: dict[str, Any]) -> None:
    """Run one RT-DETRv2 training job inside the existing training child."""
    from rtdetrv2.installer import ensure_installed

    job_id = str(payload["job_id"])
    config = dict(payload.get("config") or {})
    partition_configs = list(payload.get("partition_configs") or [])
    model_scale = str(payload.get("model_scale") or "").lower() or "s"
    if model_scale not in _SCALE_TO_CONFIG:
        model_scale = "s"

    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    _set_job(
        job_id,
        status="running",
        started_at=datetime.utcnow().isoformat() + "Z",
        message="Preparing RT-DETRv2 training...",
    )

    def _log(msg: str) -> None:
        job_storage.append_job_log(job_id, "INFO", msg)

    try:
        from ..plugins.loader import discover_plugins

        discover_plugins()
    except Exception as exc:
        job_storage.append_job_log(job_id, "WARNING", f"RT-DETRv2 plugin discovery warning: {exc}")

    root = ensure_installed(log_fn=_log)
    spec = _SCALE_TO_CONFIG[model_scale]
    upstream_config = root / spec["config"]
    if not upstream_config.exists():
        raise RuntimeError(f"RT-DETRv2 upstream config not found: {upstream_config}")

    data_arg = str(config.get("data") or "")
    if not dataset_registry.is_image_dataset(data_arg):
        raise RuntimeError(f"RT-DETRv2 requires a registered image dataset, got: {data_arg!r}")

    data_yaml_path = job_dir / "data.yaml"
    dataset_yaml.write_data_yaml(
        data_arg,
        data_yaml_path,
        partition_configs=partition_configs if partition_configs else None,
    )
    config["data"] = str(data_yaml_path)
    config["dataset_name"] = data_arg
    job = job_storage.load_job(job_id)
    if job:
        job["config"] = config.copy()
        job_storage.save_job(job)

    export_dir = job_dir / "rtdetrv2_dataset"
    image_root, train_json, val_json, nc = _prepare_coco_dataset(job_id, data_yaml_path, export_dir)

    out_dir = job_dir / "runs" / "rtdetrv2"
    out_dir.mkdir(parents=True, exist_ok=True)

    batch = int(config["batch"] if config.get("batch") is not None else config.get("batch_size", 16))
    workers = int(config["workers"] if config.get("workers") is not None else 4)
    epochs = int(config["epochs"] if config.get("epochs") is not None else 100)
    use_amp = bool(config.get("amp", True))
    use_pretrained = bool(config.get("use_yolo_pretrained", True))
    resume = str(config.get("resume") or "").strip()
    resume_ckpt = _resolve_checkpoint(resume)
    pretrained_ckpt = _resolve_checkpoint(str(config.get("pretrained") or ""))

    updates = [
        f"num_classes={nc}",
        "remap_mscoco_category=False",
        f"epoches={epochs}",
        f"train_dataloader.dataset.img_folder='{image_root}'",
        f"train_dataloader.dataset.ann_file='{train_json}'",
        f"train_dataloader.total_batch_size={batch}",
        f"train_dataloader.num_workers={workers}",
        f"val_dataloader.dataset.img_folder='{image_root}'",
        f"val_dataloader.dataset.ann_file='{val_json}'",
        f"val_dataloader.total_batch_size={batch}",
        f"val_dataloader.num_workers={workers}",
    ]

    cmd = [
        sys.executable,
        "tools/train.py",
        "-c",
        str(upstream_config),
        "--output-dir",
        str(out_dir),
        "-u",
        *updates,
    ]
    if use_amp:
        cmd.append("--use-amp")
    if config.get("device"):
        cmd.extend(["--device", str(config["device"])])
    if config.get("seed") is not None:
        cmd.extend(["--seed", str(config["seed"])])
    if resume_ckpt is not None:
        cmd.extend(["--resume", str(resume_ckpt)])
    elif pretrained_ckpt is not None:
        cmd.extend(["--tuning", str(pretrained_ckpt)])
    elif use_pretrained:
        cmd.extend(["--tuning", str(spec["checkpoint"])])

    env = os.environ.copy()
    backend_root = str(Path(__file__).resolve().parents[2])
    pythonpath = [str(root), backend_root]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env["MD_JOB_ID"] = job_id
    env["PYTHONFAULTHANDLER"] = "1"

    job_storage.append_job_log(
        job_id,
        "INFO",
        f"Launching RT-DETRv2 training: {' '.join(cmd)}",
        {"cwd": str(root), "output_dir": str(out_dir)},
    )

    started = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout or []:
        _append_output_log(job_id, line)
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"RT-DETRv2 training failed with exit code {returncode}")

    weight_id = _save_weight(job_id, out_dir)
    _set_job(
        job_id,
        status="completed",
        message="Training complete",
        weight_id=weight_id,
        completed_at=datetime.utcnow().isoformat() + "Z",
    )
    job_storage.append_job_log(
        job_id,
        "INFO",
        f"RT-DETRv2 training completed in {time.time() - started:.1f}s. Weight: {weight_id}",
    )
