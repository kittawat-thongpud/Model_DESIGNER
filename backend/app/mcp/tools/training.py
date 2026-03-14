"""
MCP tools — Training control.
Wraps ultra_trainer, train_controller, task_queue services.
"""
from __future__ import annotations
from typing import Any

from ...services import ultra_trainer, job_storage, model_storage
from ...services.config_service import get_model_config
from ..serializers import ok, err

_MODEL_DEFAULTS = get_model_config().get("defaults", {})


def start_training(
    model_id: str,
    config: dict,
    model_scale: str | None = None,
    partitions: list[dict] | None = None,
) -> dict[str, Any]:
    """Start a training job. Returns job_id immediately; training runs in background.

    Uses the same queue/admission logic as the REST API (1 GPU slot).

    Args:
        model_id: Model ID (use 'yolo:yolov8n' for official YOLO models).
        config: Training configuration dict (epochs, batch, imgsz, data, etc.).
        model_scale: Scale char (n, s, m, l, x) for custom models.
        partitions: List of partition split configs
                    [{"partition_id": "p_xxx", "train": true, "val": false}].
    """
    try:
        from ...services.preflight import validate_train_request

        if model_id.startswith("yolo:"):
            model_name = model_id.split(":")[1].upper()
            task = str(_MODEL_DEFAULTS.get("task", "detect"))
            yaml_path = ""
        else:
            record = model_storage.load_model(model_id)
            if not record:
                return err(f"Model not found: {model_id}", "not_found")
            yaml_path = model_storage.load_model_yaml_path(model_id)
            if not yaml_path:
                return err("No YAML file for this model", "missing_yaml")
            task = record.get("task", str(_MODEL_DEFAULTS.get("task", "detect")))
            model_name = record.get("name", "Untitled")

        partition_configs = partitions or []
        dataset_name = config.get("data", "")

        preflight = validate_train_request(
            model_id=model_id,
            yaml_path=str(yaml_path) if yaml_path else "",
            dataset_name=dataset_name,
            config=config,
            weight_id=config.get("pretrained") or None,
            partition_configs=partition_configs,
        )
        if not preflight.ok:
            return {
                "ok": False,
                "error": "preflight_failed",
                "message": "Preflight validation failed",
                "errors": preflight.errors,
                "warnings": preflight.warnings,
            }

        job_id = ultra_trainer.start_training(
            model_id=model_id,
            model_name=model_name,
            task=task,
            yaml_path=str(yaml_path) if yaml_path else "",
            config=config,
            partition_configs=partition_configs,
            model_scale=model_scale,
        )
        return {"ok": True, "job_id": job_id, "message": "Training started"}
    except Exception as e:
        return err(str(e), "start_training_failed")


def stop_training(job_id: str) -> dict[str, Any]:
    """Signal a running training job to stop.

    Args:
        job_id: Target job ID.
    """
    try:
        if not ultra_trainer.stop_training(job_id):
            return err(f"Job not found or not running: {job_id}", "not_running")
        return {"ok": True, "job_id": job_id, "message": "Stop signal sent"}
    except Exception as e:
        return err(str(e), "stop_training_failed")


def resume_training(job_id: str) -> dict[str, Any]:
    """Resume a stopped or failed training job from its last checkpoint.

    Args:
        job_id: Target job ID.
    """
    try:
        ultra_trainer.resume_training(job_id)
        return {"ok": True, "job_id": job_id, "message": "Resuming from last checkpoint"}
    except ValueError as e:
        return err(str(e), "resume_failed")
    except Exception as e:
        return err(str(e), "resume_training_failed")


def append_training(job_id: str, additional_epochs: int = 50) -> dict[str, Any]:
    """Append additional epochs to a completed or stopped training job.

    Args:
        job_id: Target job ID.
        additional_epochs: Number of extra epochs to train (default 50).
    """
    try:
        ultra_trainer.append_training(job_id, additional_epochs)
        return {
            "ok": True,
            "job_id": job_id,
            "message": f"Appending {additional_epochs} more epochs",
        }
    except ValueError as e:
        return err(str(e), "append_failed")
    except Exception as e:
        return err(str(e), "append_training_failed")


def get_training_queue() -> dict[str, Any]:
    """Get current training queue status including pending jobs and slot limits."""
    try:
        from ...services.task_queue import queue_status, TaskType
        result = queue_status(TaskType.TRAINING)
        return ok(result)
    except Exception as e:
        return err(str(e), "get_queue_failed")


def get_training_workers_health() -> dict[str, Any]:
    """Get health status of all active training worker threads."""
    try:
        result = ultra_trainer.get_worker_health()
        return ok(result)
    except Exception as e:
        return err(str(e), "get_workers_health_failed")
