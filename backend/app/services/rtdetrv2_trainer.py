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
from ..constants import job_channel, train_channel
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
    """Publish to both train_channel (SSE progress) and job_channel (status)."""
    event_bus.publish_sync(train_channel(job_id), {"type": "status", **job})
    event_bus.publish_sync(job_channel(job_id), {
        "type": "job_update",
        "job_id": job_id,
        "status": job.get("status", "running"),
        "epoch": job.get("epoch", 0),
        "total_epochs": job.get("total_epochs", 0),
        "message": job.get("message", ""),
        "best_fitness": job.get("best_fitness"),
    })


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


def _remap_dataset_path(path: Path) -> Path:
    """Map stale absolute /.../datasets/<name>/... paths to this workspace."""
    if path.exists():
        return path
    parts = list(path.parts)
    for i, part in enumerate(parts):
        if part == "datasets" and i + 1 < len(parts):
            candidate = DATASETS_DIR / Path(*parts[i + 1:])
            if candidate.exists():
                return candidate
    return path


def _resolve_split_paths(data: dict[str, Any], split: str) -> list[Path]:
    root = _remap_dataset_path(Path(str(data.get("path") or ".")))
    raw = data.get(split) or data.get("train")
    if raw is None:
        return []
    raw_path = Path(str(raw))
    if raw_path.suffix == ".txt":
        txt = _remap_dataset_path(raw_path if raw_path.is_absolute() else root / raw_path)
        return [
            _remap_dataset_path(Path(line.strip()))
            for line in txt.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    base = _remap_dataset_path(raw_path if raw_path.is_absolute() else root / raw_path)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            return _remap_dataset_path(Path(*parts).with_suffix(".txt"))
    return _remap_dataset_path(image_path.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt")


def _image_size(path: Path) -> tuple[int, int]:
    path = _remap_dataset_path(path)
    pil_error = None
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception as exc:
        pil_error = exc

    try:
        import cv2
        im = cv2.imread(str(path))
        if im is None:
            exists = path.exists()
            size = path.stat().st_size if exists else None
            raise RuntimeError(
                f"Could not read image size: {path} "
                f"(exists={exists}, bytes={size}, pil_error={pil_error})"
            )
        h, w = im.shape[:2]
        return w, h
    except RuntimeError:
        raise
    except Exception as exc:
        exists = path.exists()
        size = path.stat().st_size if exists else None
        raise RuntimeError(
            f"Could not read image size: {path} "
            f"(exists={exists}, bytes={size}, pil_error={pil_error}, cv2_error={exc})"
        )


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
) -> int:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1
    skipped = 0

    for img_id, image_path in enumerate(image_paths, start=1):
        image_path = _remap_dataset_path(image_path)
        try:
            width, height = _image_size(image_path)
        except RuntimeError:
            skipped += 1
            continue
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
    return skipped


def _prepare_coco_dataset(job_id: str, data_yaml_path: Path, out_dir: Path) -> tuple[Path, Path, Path, int]:
    data = _read_data_yaml(data_yaml_path)
    names = _normalise_names(data.get("names"))
    nc = int(data.get("nc") or len(names) or 1)
    image_root = _remap_dataset_path(Path(str(data.get("path") or DATASETS_DIR))).resolve()
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

    skipped_train = _export_split_to_coco(train_paths, out_json=train_json, image_root=image_root)
    skipped_val = _export_split_to_coco(val_paths, out_json=val_json, image_root=image_root)

    for p in (train_json, val_json):
        payload = json.loads(p.read_text(encoding="utf-8"))
        if not payload.get("images"):
            raise RuntimeError(f"No readable images exported for RT-DETRv2 from {p}")
        payload["categories"] = categories
        p.write_text(json.dumps(payload), encoding="utf-8")

    if skipped_train or skipped_val:
        job_storage.append_job_log(
            job_id,
            "WARNING",
            f"RT-DETRv2 COCO export skipped unreadable images: train={skipped_train}, val={skipped_val}",
        )

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


def _append_output_log(job_id: str, line: str, state: dict[str, Any]) -> None:
    """Parse RT-DETRv2 subprocess stdout and update job state.

    Recognised patterns from ``det_engine.py`` / ``det_solver.py``:
    - ``Epoch: [N]``  → training epoch started
    - ``Averaged stats: ...loss: X.XXXX ...``  → train / test loss
    - ``best_stat: {...}``  → best eval metric so far
    - ``Average Precision  (AP) @[ IoU=0.50:0.95 ...`` → COCO eval mAP
    """
    text = line.strip()
    if not text:
        return
    job_storage.append_job_log(job_id, "INFO", text)

    # ── Epoch header / batch progress: "Epoch: [N]  [batch/total]" ──────────
    m = re.match(r"Epoch:\s*\[(\d+)\]\s*\[(\d+)/(\d+)\]", text)
    if m:
        epoch_0 = int(m.group(1))
        epoch_1 = epoch_0 + 1
        batch = int(m.group(2))
        total_batches = int(m.group(3))
        total_epochs = state.get("total_epochs", 0)
        state["current_epoch"] = epoch_0

        # Extract loss from same line: "loss: X.XXXX (Y.YYYY)"
        loss_m = re.search(r"loss:\s*([\d.]+)", text)
        loss_val = float(loss_m.group(1)) if loss_m else None

        pct = round(batch / max(total_batches, 1) * 100, 1)
        _set_job(
            job_id,
            epoch=epoch_1,
            message=f"Epoch {epoch_1}/{total_epochs} [{batch}/{total_batches}]",
        )
        # Emit train-phase PROGRESS for SSE (rate-limited: only every 100 batches)
        if batch % 100 == 0:
            losses: dict[str, Any] = {}
            # Extract individual losses from the line
            for lk in ("loss_bbox", "loss_vfl", "loss_giou"):
                lm = re.search(rf"{lk}:\s*([\d.]+)", text)
                if lm:
                    key_map = {"loss_bbox": "box", "loss_vfl": "cls", "loss_giou": "dfl"}
                    losses[key_map[lk]] = round(float(lm.group(1)), 4)
            job_storage.append_job_log(
                job_id,
                "PROGRESS",
                f"Epoch {epoch_1}/{total_epochs} | {pct}% | Batch {batch}/{total_batches}",
                {
                    "type": "progress",
                    "phase": "train",
                    "epoch": f"{epoch_1}/{total_epochs}",
                    "batch": f"{batch}/{total_batches}",
                    "percent": pct,
                    "losses": losses,
                },
            )
        return
    # Epoch header without batch (just "Epoch: [N]")
    m2 = re.search(r"Epoch:\s*\[(\d+)\]", text)
    if m2:
        epoch_0 = int(m2.group(1))
        state["current_epoch"] = epoch_0
        return

    # ── Averaged stats (train or test) ──────────────────────────────────────
    if "Averaged stats:" in text:
        # Extract loss value:  "loss: 1.2345 (2.3456)"
        loss_m = re.search(r"loss:\s*([\d.]+)", text)
        if loss_m:
            loss_val = float(loss_m.group(1))
            # Determine if this is train or test stats
            if "Test:" in text or state.get("_last_header") == "test":
                state["val_loss"] = loss_val
                state["_last_header"] = None
            else:
                state["train_loss"] = loss_val
                # After train averaged stats, next eval block is test
                state["_last_header"] = "test"
        return

    # Detect "Test:" header (precedes eval averaged stats)
    if text.startswith("Test:"):
        state["_last_header"] = "test"
        return

    # ── COCO AP line: " Average Precision  (AP) @[ IoU=0.50:0.95 ..." ──────
    ap_m = re.search(
        r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50:0\.95.*?=\s*([\d.]+)",
        text,
    )
    if ap_m:
        state["mAP50_95"] = float(ap_m.group(1))
        return
    ap50_m = re.search(
        r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50\s.*?=\s*([\d.]+)",
        text,
    )
    if ap50_m:
        state["mAP50"] = float(ap50_m.group(1))
        return

    # ── best_stat line ──────────────────────────────────────────────────────
    if text.startswith("best_stat:"):
        state["_last_header"] = None  # reset
        return


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
        total_epochs=int(config.get("epochs") or 100),
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
    # Clamp workers: RT-DETRv2 runs as a nested subprocess where high num_workers
    # can exhaust /dev/shm (RuntimeError: received 0 items of ancdata).
    try:
        shm = shutil.disk_usage("/dev/shm")
        if shm.total < 2 * 1024 ** 3:  # < 2 GB shared memory
            workers = 0
            _log(f"Small /dev/shm ({shm.total // (1024**2)} MB) — setting num_workers=0")
    except Exception:
        pass
    workers = min(workers, 4)

    # Free GPU memory from any previous training in the parent process
    # and auto-clamp batch based on available VRAM to prevent OOM.
    try:
        import torch
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            free_vram_gb = (torch.cuda.get_device_properties(0).total_memory
                           - torch.cuda.memory_reserved(0)) / (1024 ** 3)
            _log(f"GPU: {torch.cuda.get_device_name(0)}, free VRAM: {free_vram_gb:.1f} GB")
            # RT-DETRv2 R18 (scale s) needs ~2 GB/batch at 640px
            # R50/R101 need ~3-4 GB/batch
            per_batch_gb = {"s": 1.5, "m": 2.5, "l": 3.0, "x": 3.5}.get(model_scale, 2.0)
            model_overhead_gb = {"s": 2.0, "m": 4.0, "l": 5.0, "x": 6.0}.get(model_scale, 3.0)
            safe_vram = free_vram_gb - model_overhead_gb - 1.0  # 1 GB safety margin
            max_batch = max(1, int(safe_vram / per_batch_gb))
            if batch > max_batch:
                _log(f"Auto-clamping batch {batch} → {max_batch} (available {free_vram_gb:.1f} GB, "
                     f"model overhead ~{model_overhead_gb} GB, ~{per_batch_gb} GB/sample)")
                batch = max_batch
    except Exception as e:
        _log(f"GPU memory check skipped: {e}")

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
    env["PYTHONUNBUFFERED"] = "1"  # force line-buffered stdout for real-time log parsing

    job_storage.append_job_log(
        job_id,
        "INFO",
        f"Launching RT-DETRv2 training: {' '.join(cmd)}",
        {"cwd": str(root), "output_dir": str(out_dir)},
    )

    started = time.time()
    log_txt = out_dir / "log.txt"
    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # State shared across stdout parsing
    parse_state: dict[str, Any] = {
        "total_epochs": epochs,
        "current_epoch": 0,
        "train_loss": None,
        "val_loss": None,
        "mAP50": None,
        "mAP50_95": None,
        "_log_txt_offset": 0,  # byte offset into log.txt
    }

    def _poll_log_txt() -> None:
        """Read new lines from upstream log.txt and push epoch history.

        Writes to ``<job_dir>/extended_metrics.jsonl`` using the same field
        names that ``job_storage.get_job_history()`` expects, and emits
        ``PROGRESS`` log entries so ``stream_controller`` can relay them
        to the frontend via SSE.
        """
        if not log_txt.exists():
            return
        try:
            with log_txt.open("r", encoding="utf-8") as f:
                f.seek(parse_state["_log_txt_offset"])
                new_lines = f.readlines()
                parse_state["_log_txt_offset"] = f.tell()
        except Exception:
            return
        for raw in new_lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            epoch = entry.get("epoch", parse_state["current_epoch"])
            # RT-DETRv2 uses 0-based epochs; Model Designer uses 1-based
            epoch_1 = epoch + 1
            elapsed = time.time() - started

            # ── Extract COCO eval AP (list of 12 standard COCO metrics) ──
            test_coco = entry.get("test_coco_eval_bbox")
            map50_95 = None
            map50 = None
            if isinstance(test_coco, list) and len(test_coco) >= 2:
                map50_95 = float(test_coco[0])
                map50 = float(test_coco[1])
                parse_state["mAP50_95"] = map50_95
                parse_state["mAP50"] = map50

            # ── Build extended_metrics.jsonl entry ──────────────────────
            # Field names MUST match what job_storage.get_job_history() reads:
            #   train_box_loss, train_cls_loss, train_dfl_loss,
            #   map50, map (=mAP50-95), map75, precision, recall, lr, etc.
            train_loss = entry.get("train_loss")
            train_loss_vfl = entry.get("train_loss_vfl")
            train_loss_bbox = entry.get("train_loss_bbox")
            train_loss_giou = entry.get("train_loss_giou")

            epoch_data: dict[str, Any] = {
                "epoch": epoch_1,
                "timestamp": time.time(),
                # Training losses — map RT-DETRv2 → Ultralytics-style names
                "train_box_loss": train_loss_bbox or train_loss,
                "train_cls_loss": train_loss_vfl,
                "train_dfl_loss": train_loss_giou,
                # Validation metrics (COCO AP)
                "map50": map50,
                "map": map50_95,  # job_storage maps "map" → mAP50_95
                # Learning rate
                "lr": entry.get("train_lr"),
                # Validation time
                "val_time_s": None,
            }
            # Remove None values
            epoch_data = {k: v for k, v in epoch_data.items() if v is not None}

            # Write to <job_dir>/extended_metrics.jsonl (same location as Ultralytics)
            ext_metrics_path = JOBS_DIR / job_id / "extended_metrics.jsonl"
            try:
                with ext_metrics_path.open("a") as mf:
                    mf.write(json.dumps(epoch_data) + "\n")
            except Exception:
                pass

            # ── Emit PROGRESS log → log.jsonl → SSE via stream_controller ──
            progress_data: dict[str, Any] = {
                "type": "progress",
                "phase": "validation_done",
                "epoch": f"{epoch_1}/{epochs}",
                "total_epochs": epochs,
                "batch": "0/0",
                "percent": 100,
                "losses": {
                    "box": round(train_loss_bbox, 4) if train_loss_bbox else None,
                    "cls": round(train_loss_vfl, 4) if train_loss_vfl else None,
                    "dfl": round(train_loss_giou, 4) if train_loss_giou else None,
                },
                "val_map50": round(map50, 4) if map50 is not None else None,
                "val_map": round(map50_95, 4) if map50_95 is not None else None,
                "total_elapsed_s": round(elapsed, 1),
            }
            # Compute timing estimates
            if epoch_1 > 0:
                avg_epoch_s = elapsed / epoch_1
                eta_s = avg_epoch_s * (epochs - epoch_1)
                progress_data["avg_epoch_s"] = round(avg_epoch_s, 1)
                progress_data["eta_s"] = round(eta_s, 0)

            job_storage.append_job_log(
                job_id,
                "PROGRESS",
                f"Epoch {epoch_1}/{epochs}"
                + (f" | mAP50={map50:.4f}" if map50 is not None else "")
                + (f" | mAP50-95={map50_95:.4f}" if map50_95 is not None else ""),
                progress_data,
            )

            # Also publish epoch event to train_channel for real-time chart update
            epoch_event: dict[str, Any] = {
                "type": "epoch",
                "epoch": epoch_1,
                "box_loss": train_loss_bbox or train_loss or 0,
                "cls_loss": train_loss_vfl or 0,
                "dfl_loss": train_loss_giou or 0,
                "mAP50": map50,
                "mAP50_95": map50_95,
                "lr": entry.get("train_lr", 0),
                "epoch_time": 0,
            }
            event_bus.publish_sync(train_channel(job_id), epoch_event)

            # ── Update job record ──────────────────────────────────────
            updates: dict[str, Any] = {
                "epoch": epoch_1,
                "message": f"Epoch {epoch_1}/{epochs}",
            }
            if parse_state.get("mAP50_95") is not None:
                updates["best_mAP50_95"] = parse_state["mAP50_95"]
            if parse_state.get("mAP50") is not None:
                updates["best_mAP50"] = parse_state["mAP50"]
            _set_job(job_id, **updates)

    for line in proc.stdout or []:
        _append_output_log(job_id, line, parse_state)
        # Periodically check log.txt for detailed metrics
        _poll_log_txt()

    # Final poll after process ends
    _poll_log_txt()

    returncode = proc.wait()
    elapsed = time.time() - started

    if returncode != 0:
        _set_job(
            job_id,
            status="failed",
            message=f"RT-DETRv2 training failed (exit code {returncode})",
            completed_at=datetime.utcnow().isoformat() + "Z",
        )
        raise RuntimeError(f"RT-DETRv2 training failed with exit code {returncode}")

    weight_id = _save_weight(job_id, out_dir)
    final_msg = f"Training complete ({elapsed:.0f}s)"
    if parse_state.get("mAP50_95") is not None:
        final_msg += f" — mAP50-95: {parse_state['mAP50_95']:.4f}"

    _set_job(
        job_id,
        status="completed",
        message=final_msg,
        weight_id=weight_id,
        completed_at=datetime.utcnow().isoformat() + "Z",
        best_mAP50_95=parse_state.get("mAP50_95"),
        best_mAP50=parse_state.get("mAP50"),
    )
    job_storage.append_job_log(
        job_id,
        "INFO",
        f"RT-DETRv2 training completed in {elapsed:.1f}s. Weight: {weight_id}",
    )
