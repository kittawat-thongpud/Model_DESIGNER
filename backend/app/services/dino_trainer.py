"""Isolated facebookresearch/DINO trainer integration.

This runs upstream DINO in a subprocess while preserving Model Designer job
records, logs, SSE events, queue ownership, and dataset selection behavior.
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


_SCALE_TO_SPEC = {
    "vits16": {"arch": "vit_small", "patch_size": 16, "label": "ViT-S/16"},
    "vits8": {"arch": "vit_small", "patch_size": 8, "label": "ViT-S/8"},
    "vitb16": {"arch": "vit_base", "patch_size": 16, "label": "ViT-B/16"},
    "vitb8": {"arch": "vit_base", "patch_size": 8, "label": "ViT-B/8"},
    "resnet50": {"arch": "resnet50", "patch_size": 16, "label": "ResNet-50"},
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
    if isinstance(raw, list):
        paths: list[Path] = []
        for item in raw:
            clone = dict(data)
            clone[split] = item
            paths.extend(_resolve_split_paths(clone, split))
        return paths
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


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _prepare_imagefolder(
    job_id: str,
    data_yaml_path: Path,
    out_dir: Path,
    max_images: int | None = None,
) -> tuple[Path, int]:
    data = _read_data_yaml(data_yaml_path)
    image_paths = _resolve_split_paths(data, "train")
    if not image_paths:
        raise RuntimeError(f"No train images found from {data_yaml_path}")
    if max_images and max_images > 0:
        image_paths = image_paths[:max_images]

    class_dir = out_dir / "all"
    class_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for idx, image_path in enumerate(image_paths):
        suffix = image_path.suffix.lower() or ".jpg"
        name = f"{idx:08d}_{image_path.stem}{suffix}"
        while name in seen:
            name = f"{idx:08d}_{uuid.uuid4().hex[:8]}{suffix}"
        seen.add(name)
        _safe_link_or_copy(image_path.resolve(), class_dir / name)

    job_storage.append_job_log(
        job_id,
        "INFO",
        f"DINO ImageFolder export: train={len(image_paths)} images, class_folder=all",
        {"imagefolder": str(out_dir), "source_data": str(data_yaml_path)},
    )
    return out_dir, len(image_paths)


def _checkpoint_candidates(out_dir: Path) -> list[Path]:
    return [
        out_dir / "checkpoint.pth",
        *sorted(out_dir.glob("checkpoint*.pth"), key=lambda p: p.stat().st_mtime, reverse=True),
        *sorted(out_dir.rglob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True),
    ]


def _read_last_log_stats(out_dir: Path) -> dict[str, Any]:
    log_path = out_dir / "log.txt"
    if not log_path.exists():
        return {}
    last: dict[str, Any] = {}
    for raw in log_path.read_text(encoding="utf-8").splitlines():
        try:
            last = json.loads(raw)
        except json.JSONDecodeError:
            continue
    return last


def _save_weight(job_id: str, out_dir: Path, total_time: float | None) -> str | None:
    checkpoint = next((p for p in _checkpoint_candidates(out_dir) if p.exists()), None)
    if checkpoint is None:
        return None
    from . import weight_storage

    job = job_storage.load_job(job_id) or {}
    cfg = job.get("config", {})
    stats = _read_last_log_stats(out_dir)
    final_loss = stats.get("train_loss")
    weight_id = uuid.uuid4().hex[:12]
    dest_dir = WEIGHTS_DIR / weight_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, dest_dir / "weight.pt")
    weight_storage.save_weight_meta(
        model_id=job.get("model_id", "arch:dino"),
        model_name=job.get("model_name", "DINO"),
        model_scale=job.get("model_scale", ""),
        job_id=job_id,
        dataset=cfg.get("dataset_name") or cfg.get("data", ""),
        epochs_trained=job.get("total_epochs", 0),
        final_accuracy=None,
        final_loss=float(final_loss) if isinstance(final_loss, (int, float)) else None,
        weight_id=weight_id,
        parent_weight_id=None,
        total_time=total_time,
        device=str(cfg.get("device", "")),
    )
    meta = weight_storage.load_weight_meta(weight_id)
    if meta:
        meta.update(
            {
                "arch_plugin": cfg.get("model_arch"),
                "model_arch": cfg.get("model_arch"),
                "source_type": "dino",
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
    if not raw or raw.lower() in {"false", "0", "none"}:
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


def _append_history_from_log(job_id: str, out_dir: Path) -> None:
    log_path = out_dir / "log.txt"
    if not log_path.exists():
        return
    job = job_storage.load_job(job_id)
    if not job:
        return
    history = list(job.get("history") or [])
    for raw in log_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        epoch = int(row.get("epoch", len(history)))
        history.append(
            {
                "epoch": epoch,
                "loss": row.get("train_loss"),
                "lr": row.get("train_lr"),
                "weight_decay": row.get("train_wd"),
                "timestamp": time.time(),
                "metrics": row,
            }
        )
    job["history"] = history
    job_storage.save_job(job)
    _publish(job_id, job)


def _append_output_log(job_id: str, line: str) -> None:
    text = line.strip()
    if not text:
        return
    job_storage.append_job_log(job_id, "INFO", text)
    m = re.search(r"Epoch:\s*\[(\d+)/(\d+)\]", text)
    if m:
        epoch = int(m.group(1))
        total = int(m.group(2))
        _set_job(job_id, epoch=epoch, total_epochs=total, message=f"DINO epoch {epoch}/{total}")


def run_worker(payload: dict[str, Any]) -> None:
    """Run one DINO training job inside the existing training child."""
    from dino.installer import ensure_installed

    job_id = str(payload["job_id"])
    config = dict(payload.get("config") or {})
    partition_configs = list(payload.get("partition_configs") or [])
    model_scale = str(payload.get("model_scale") or "").lower() or "vits16"
    if model_scale not in _SCALE_TO_SPEC:
        model_arch = str(config.get("model_arch") or "")
        prefix = "dino_"
        if model_arch.startswith(prefix) and model_arch[len(prefix):] in _SCALE_TO_SPEC:
            model_scale = model_arch[len(prefix):]
        else:
            model_scale = "vits16"

    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    _set_job(
        job_id,
        status="running",
        started_at=datetime.utcnow().isoformat() + "Z",
        message="Preparing DINO training...",
    )

    def _log(msg: str) -> None:
        job_storage.append_job_log(job_id, "INFO", msg)

    root = ensure_installed(log_fn=_log)
    if not (root / "main_dino.py").exists():
        raise RuntimeError(f"DINO upstream trainer not found: {root / 'main_dino.py'}")

    data_arg = str(config.get("data") or "")
    if not dataset_registry.is_image_dataset(data_arg):
        raise RuntimeError(f"DINO requires a registered image dataset, got: {data_arg!r}")

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

    max_images = int(config.get("dino_max_images") or config.get("max_images") or 0) or None
    imagefolder, image_count = _prepare_imagefolder(
        job_id,
        data_yaml_path,
        job_dir / "dino_imagefolder",
        max_images=max_images,
    )
    if image_count < int(config.get("batch", 1)):
        job_storage.append_job_log(
            job_id,
            "WARNING",
            f"DINO train image count ({image_count}) is smaller than batch ({config.get('batch')}); upstream uses drop_last=True.",
        )

    out_dir = job_dir / "runs" / "dino"
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = _SCALE_TO_SPEC[model_scale]
    batch = int(config["batch"] if config.get("batch") is not None else config.get("batch_size", 64))
    workers = int(config["workers"] if config.get("workers") is not None else 4)
    epochs = int(config["epochs"] if config.get("epochs") is not None else 100)
    use_fp16 = bool(config.get("amp", True))
    save_period = int(config["save_period"] if config.get("save_period") is not None else 20)
    warmup_epochs = int(float(config.get("warmup_epochs", 10)))
    lr = float(config.get("lr0", config.get("lr", 0.0005)))
    min_lr = float(config.get("min_lr", 1e-6))
    weight_decay = float(config.get("weight_decay", 0.04))
    seed = int(config.get("seed", 0))
    local_crops_number = int(config.get("local_crops_number", config.get("dino_local_crops_number", 8)))

    resume_ckpt = _resolve_checkpoint(str(config.get("resume") or ""))
    if resume_ckpt is None:
        resume_ckpt = _resolve_checkpoint(str(config.get("pretrained") or ""))
    if resume_ckpt is not None:
        shutil.copy2(resume_ckpt, out_dir / "checkpoint.pth")
        job_storage.append_job_log(job_id, "INFO", f"DINO resume checkpoint staged: {resume_ckpt}")

    cmd = [
        sys.executable,
        "main_dino.py",
        "--arch",
        str(spec["arch"]),
        "--patch_size",
        str(spec["patch_size"]),
        "--data_path",
        str(imagefolder),
        "--output_dir",
        str(out_dir),
        "--epochs",
        str(epochs),
        "--batch_size_per_gpu",
        str(batch),
        "--num_workers",
        str(workers),
        "--seed",
        str(seed),
        "--use_fp16",
        "true" if use_fp16 else "false",
        "--saveckp_freq",
        str(save_period),
        "--warmup_epochs",
        str(max(0, min(warmup_epochs, epochs))),
        "--lr",
        str(lr),
        "--min_lr",
        str(min_lr),
        "--weight_decay",
        str(weight_decay),
        "--local_crops_number",
        str(local_crops_number),
    ]

    env = os.environ.copy()
    backend_root = str(Path(__file__).resolve().parents[2])
    pythonpath = [str(root), backend_root]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env["MD_JOB_ID"] = job_id
    env["PYTHONFAULTHANDLER"] = "1"
    env.setdefault("RANK", "0")
    env.setdefault("WORLD_SIZE", "1")
    env.setdefault("LOCAL_RANK", "0")
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", str(29500 + (int(job_id[:4], 16) % 1000)))
    device = str(config.get("device") or "").strip()
    if device and device.lower() != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = device.split(",")[0]

    job_storage.append_job_log(
        job_id,
        "INFO",
        f"Launching DINO training: {' '.join(cmd)}",
        {"cwd": str(root), "output_dir": str(out_dir), "variant": spec["label"]},
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
        raise RuntimeError(f"DINO training failed with exit code {returncode}")

    elapsed = time.time() - started
    _append_history_from_log(job_id, out_dir)
    weight_id = _save_weight(job_id, out_dir, elapsed)
    _set_job(
        job_id,
        status="completed",
        epoch=epochs,
        message="Training complete",
        weight_id=weight_id,
        completed_at=datetime.utcnow().isoformat() + "Z",
    )
    job_storage.append_job_log(
        job_id,
        "INFO",
        f"DINO training completed in {elapsed:.1f}s. Weight: {weight_id}",
    )
