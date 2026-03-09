from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = Path(os.environ.get("DATA_DIR", str(_BACKEND_ROOT / "data")))
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_CONFIG_PATH = _DATA_DIR / "config.json"

_DEFAULT_CONFIG: dict[str, Any] = {
    "app": {
        "name": "Model DESIGNER API",
        "version": "2.0.0",
        "cors_origins": [
            "http://localhost:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:5174",
        ],
    },
    "logging": {
        "level": "DEBUG",
        "retention_days": 30,
        "query_default_limit": 100,
        "query_default_days": 7,
    },
    "monitoring": {
        "worker_check_interval_s": 60,
        "worker_stop_timeout_s": 5.0,
        "training_setup_watchdog_timeout_s": 600,
        "training_setup_heartbeat_s": 30,
    },
    "queue": {
        "sqlite_path": "task_queue.db",
        "sqlite_timeout_s": 10.0,
        "cleanup_max_age_s": 604800.0,
        "concurrency_limits": {
            "training": 1,
            "benchmark": 1,
            "dataset_conversion": 2,
            "dataset_extraction": 2,
            "export": 2,
            "package_import": 1,
            "package_export": 1,
            "plot_generation": 2,
            "weight_transfer": 2,
        },
    },
    "training": {
        "defaults": {
            "data": "",
            "imgsz": 640,
            "batch": 16,
            "workers": 4,
            "epochs": 100,
            "patience": 100,
            "device": "",
            "seed": 0,
            "deterministic": True,
            "amp": True,
            "ema": True,
            "pin_memory": False,
            "close_mosaic": 10,
            "optimizer": "auto",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "cos_lr": False,
            "pretrained": "",
            "yolo_model": "",
            "use_yolo_pretrained": True,
            "freeze": 0,
            "resume": False,
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "pose": 12.0,
            "nbs": 64,
            "conf": None,
            "iou": 0.7,
            "max_det": 300,
            "agnostic_nms": False,
            "rect": False,
            "single_cls": False,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "bgr": 0.0,
            "mosaic": 1.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "erasing": 0.4,
            "crop_fraction": 1.0,
            "auto_augment": "",
            "save_period": -1,
            "val": True,
            "plots": True,
            "record_gradients": False,
            "gradient_interval": 1,
            "record_weights": False,
            "weight_interval": 1,
            "sample_per_class": 0,
            "overlap_mask": True,
            "mask_ratio": 4,
            "kobj": 1.0,
        },
        "api_defaults": {
            "append_additional_epochs": 50,
            "job_log_limit": 200,
        },
        "preflight": {
            "disk_fail_gb": 2.0,
            "disk_warn_gb": 5.0,
            "imgsz_warn_threshold": 4096,
        },
        "runtime": {
            "remote_fs_workers": 2,
            "remote_fs_ultralytics_threads": 1,
        },
    },
    "cache": {
        "sample_image_count": 200,
        "decompress_factor": 30,
        "ram_cache": {
            "windows_safety_factor": 3.0,
            "default_safety_factor": 1.5,
            "windows_max_gb": 16,
            "default_max_gb": 64,
        },
    },
    "platform": {
        "remote_fs_prefixes": [
            "/workspace",
            "/runpod-volume",
            "/mnt/nfs",
            "/mnt/smb",
        ],
        "remote_fs_types": [
            "nfs",
            "nfs4",
            "cifs",
            "smbfs",
            "sshfs",
            "fuse",
            "fuseblk",
            "overlay",
            "overlayfs",
            "tmpfs",
            "ramfs",
        ],
    },
    "benchmark": {
        "defaults": {
            "split": "val",
            "conf": 0.001,
            "iou": 0.6,
            "imgsz": 640,
            "batch": 16,
        },
    },
    "inference": {
        "defaults": {
            "conf": 0.25,
            "iou": 0.45,
            "imgsz": 640,
            "top_k": 5,
            "visualize_sgbg": False,
        },
        "limits": {
            "max_files_per_request": 32,
            "history_default_limit": 50,
            "history_max_entries": 200,
        },
    },
    "export": {
        "defaults": {
            "format": "onnx",
            "imgsz": 640,
            "device": "",
            "half": False,
            "simplify": True,
            "download_format": "onnx",
        },
    },
    "models": {
        "defaults": {
            "name": "Untitled",
            "task": "detect",
            "imported_name": "Imported Model",
            "export_format": "yaml",
            "export_imgsz": 640,
        },
        "validation": {
            "info_imgsz": 640,
        },
    },
    "packages": {
        "defaults": {
            "include_jobs": False,
            "rename_map": "{}",
        },
    },
    "dataset_samples": {
        "defaults": {
            "page_size": 50,
            "split": "train",
            "thumb_size": 128,
            "include_annotations": False,
            "detail_include_annotations": True,
        },
        "limits": {
            "max_page_size": 200,
            "encode_pool_workers": 4,
        },
    },
    "weight_snapshots": {
        "defaults": {
            "thumbnail_max_size": 48,
        },
    },
    "streaming": {
        "defaults": {
            "sse_timeout_s": 30.0,
            "system_log_timeout_s": 60.0,
            "job_log_timeout_s": 60.0,
            "tail_poll_interval_s": 0.5,
            "tail_wait_log_retries": 10,
            "tail_wait_log_retry_delay_s": 0.5,
        },
    },
}

_EFFECTIVE_CONFIG: dict[str, Any] | None = None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_config_file() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        return {}
    try:
        payload = json.loads(_CONFIG_PATH.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid config JSON at {_CONFIG_PATH}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Config file must contain a JSON object: {_CONFIG_PATH}")
    return payload


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    effective = deepcopy(config)

    log_level = os.environ.get("LOG_LEVEL")
    if log_level:
        effective.setdefault("logging", {})["level"] = log_level.upper()

    log_retention_days = os.environ.get("LOG_RETENTION_DAYS")
    if log_retention_days:
        effective.setdefault("logging", {})["retention_days"] = int(log_retention_days)

    if "CORS_ORIGINS" in os.environ:
        cors_raw = os.environ.get("CORS_ORIGINS", "")
        effective.setdefault("app", {})["cors_origins"] = [
            item.strip() for item in cors_raw.split(",") if item.strip()
        ]

    return effective


def reload_config() -> dict[str, Any]:
    global _EFFECTIVE_CONFIG
    file_config = _read_config_file()
    merged = _deep_merge(_DEFAULT_CONFIG, file_config)
    _EFFECTIVE_CONFIG = _apply_env_overrides(merged)
    return deepcopy(_EFFECTIVE_CONFIG)


def get_effective_config() -> dict[str, Any]:
    global _EFFECTIVE_CONFIG
    if _EFFECTIVE_CONFIG is None:
        return reload_config()
    return deepcopy(_EFFECTIVE_CONFIG)


def get_config_path() -> Path:
    return _CONFIG_PATH


def get_config_value(*keys: str, default: Any = None) -> Any:
    value: Any = get_effective_config()
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def get_app_config() -> dict[str, Any]:
    return get_effective_config().get("app", {})


def get_logging_config() -> dict[str, Any]:
    return get_effective_config().get("logging", {})


def get_monitoring_config() -> dict[str, Any]:
    return get_effective_config().get("monitoring", {})


def get_queue_config() -> dict[str, Any]:
    return get_effective_config().get("queue", {})


def get_training_config() -> dict[str, Any]:
    return get_effective_config().get("training", {})


def get_cache_config() -> dict[str, Any]:
    return get_effective_config().get("cache", {})


def get_platform_config() -> dict[str, Any]:
    return get_effective_config().get("platform", {})


def get_benchmark_config() -> dict[str, Any]:
    return get_effective_config().get("benchmark", {})


def get_inference_config() -> dict[str, Any]:
    return get_effective_config().get("inference", {})


def get_export_config() -> dict[str, Any]:
    return get_effective_config().get("export", {})


def get_model_config() -> dict[str, Any]:
    return get_effective_config().get("models", {})


def get_package_config() -> dict[str, Any]:
    return get_effective_config().get("packages", {})


def get_dataset_samples_config() -> dict[str, Any]:
    return get_effective_config().get("dataset_samples", {})


def get_weight_snapshots_config() -> dict[str, Any]:
    return get_effective_config().get("weight_snapshots", {})


def get_streaming_config() -> dict[str, Any]:
    return get_effective_config().get("streaming", {})
