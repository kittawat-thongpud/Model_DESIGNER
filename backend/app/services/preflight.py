"""
Preflight validator — checks dataset, weight, and config validity before
admitting a training job to the execution queue.

Returns a structured result so the caller can surface specific failure
reasons to the user rather than letting them manifest mid-training.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config_service import get_training_config


@dataclass
class PreflightResult:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
        }


def validate_train_request(
    *,
    model_id: str,
    yaml_path: str,
    dataset_name: str,
    config: dict[str, Any],
    weight_id: str | None = None,
    partition_configs: list[dict] | None = None,
) -> PreflightResult:
    """
    Run all preflight checks for a training job request.

    Checks (in order):
    1. Model YAML exists (for custom models)
    2. Dataset is available and has expected structure
    3. Weight file exists (if resume/pretrained weight_id given)
    4. Config sanity (epochs, batch, imgsz, etc.)
    5. Disk space estimate
    """
    from ..config import DATASETS_DIR, JOBS_DIR, DATA_DIR
    from ..services import weight_storage

    result = PreflightResult()
    training_config = get_training_config()
    preflight_config = training_config.get("preflight", {})
    imgsz_warn_threshold = int(preflight_config.get("imgsz_warn_threshold", 4096))
    disk_fail_gb = float(preflight_config.get("disk_fail_gb", 2.0))
    disk_warn_gb = float(preflight_config.get("disk_warn_gb", 5.0))

    # ── 1. Model YAML ──────────────────────────────────────────────────────────
    if yaml_path and not model_id.startswith("yolo:"):
        p = Path(yaml_path)
        if not p.exists():
            result.fail(f"Model YAML not found: {yaml_path}")
        elif p.stat().st_size == 0:
            result.fail(f"Model YAML is empty: {yaml_path}")

    # ── 2. Dataset availability ────────────────────────────────────────────────
    ds_dir = DATASETS_DIR / dataset_name
    data_yaml = ds_dir / "data.yaml"

    if not ds_dir.exists():
        result.fail(f"Dataset directory not found: {dataset_name}")
    elif not data_yaml.exists():
        result.fail(
            f"Dataset '{dataset_name}' is missing data.yaml. "
            "Run a dataset scan or conversion first."
        )
    else:
        # Check expected image dirs exist
        images_dir = ds_dir / "images"
        if not images_dir.exists():
            result.fail(
                f"Dataset '{dataset_name}' has no images/ directory. "
                "Dataset may be incomplete."
            )
        else:
            # Warn if train or val split is empty
            for split in ("train", "val"):
                split_dir = images_dir / split
                if not split_dir.exists():
                    result.warn(
                        f"Dataset '{dataset_name}' is missing images/{split}/ directory."
                    )
                else:
                    image_count = sum(
                        1 for f in split_dir.rglob("*")
                        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                    )
                    result.info[f"dataset_{split}_images"] = image_count
                    if image_count == 0:
                        result.warn(
                            f"Dataset '{dataset_name}' images/{split}/ contains no images."
                        )

        # Read nc from data.yaml for info
        try:
            import yaml as _yaml
            _d = _yaml.safe_load(data_yaml.read_text())
            result.info["dataset_nc"] = _d.get("nc")
            result.info["dataset_names"] = _d.get("names")
        except Exception:
            pass

    # ── 3. Weight file (resume or pretrained) ─────────────────────────────────
    if weight_id:
        pt_path = weight_storage.weight_pt_path(weight_id)
        if not pt_path.exists():
            result.fail(
                f"Weight file not found for weight_id='{weight_id}'. "
                "Re-create or re-download the weight before training."
            )
        else:
            result.info["weight_size_mb"] = round(pt_path.stat().st_size / 1e6, 1)

    # ── 4. Config sanity ──────────────────────────────────────────────────────
    epochs = config.get("epochs", 0)
    if not isinstance(epochs, int) or epochs <= 0:
        result.fail(f"epochs must be a positive integer, got: {epochs!r}")

    batch = config.get("batch", 0)
    if not isinstance(batch, int) or batch <= 0:
        result.fail(f"batch must be a positive integer, got: {batch!r}")

    imgsz = config.get("imgsz", 640)
    if not isinstance(imgsz, int) or imgsz < 32:
        result.fail(f"imgsz must be >= 32, got: {imgsz!r}")
    elif imgsz > imgsz_warn_threshold:
        result.warn(f"imgsz={imgsz} is very large and may cause OOM errors.")

    workers = config.get("workers")
    if workers is not None and (not isinstance(workers, int) or workers < 0):
        result.fail(f"workers must be a non-negative integer, got: {workers!r}")

    cache = config.get("cache")
    if cache not in (None, False, True, "auto", "ram", "disk", "none", "off", ""):
        result.warn(
            f"cache='{cache}' is not a recognized value. "
            "Supported values: 'auto', 'ram', 'disk', 'none'."
        )

    # ── 5. Disk space estimate (coarse) ──────────────────────────────────────
    try:
        import shutil
        free_gb = shutil.disk_usage(DATA_DIR).free / 1e9
        result.info["disk_free_gb"] = round(free_gb, 1)
        if free_gb < disk_fail_gb:
            result.fail(
                f"Insufficient disk space: {free_gb:.1f} GB free under DATA_DIR. "
                f"At least {disk_fail_gb:.1f} GB is required to start training."
            )
        elif free_gb < disk_warn_gb:
            result.warn(
                f"Low disk space: {free_gb:.1f} GB free under DATA_DIR. "
                "Training may fail if checkpoints or plots exhaust disk."
            )
    except Exception:
        pass

    return result
