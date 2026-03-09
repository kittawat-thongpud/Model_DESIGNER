"""
Pydantic schemas for Ultralytics training jobs.

Training config maps directly to Ultralytics model.train() kwargs.
Monitoring data comes from Ultralytics callbacks.
"""
from __future__ import annotations
import os
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any, Literal

from ..services.config_service import get_model_config, get_training_config

_TRAINING_DEFAULTS = get_training_config().get("defaults", {})
_MODEL_DEFAULTS = get_model_config().get("defaults", {})


def _training_default(key: str, default: Any) -> Any:
    return _TRAINING_DEFAULTS.get(key, default)


def _default_train_workers() -> int:
    raw = os.getenv("MODEL_DESIGNER_TRAIN_WORKERS", str(_training_default("workers", 4)))
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return int(_training_default("workers", 4))


class TrainConfig(BaseModel):
    """
    Ultralytics model.train() configuration.
    Every field maps 1:1 to a model.train() kwarg.
    See: https://docs.ultralytics.com/modes/train/#train-settings
    """

    # ── Data ──────────────────────────────────────────────────────────────────
    data: str = str(_training_default("data", ""))                     # path to data.yaml
    imgsz: int = int(_training_default("imgsz", 640))
    batch: int = int(_training_default("batch", 16))
    workers: int = Field(default_factory=_default_train_workers)

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = int(_training_default("epochs", 100))
    patience: int = int(_training_default("patience", 100))                # early stopping patience (0 = disabled)
    device: str = str(_training_default("device", ""))                   # "", "cpu", "0", "0,1", etc.
    seed: int = int(_training_default("seed", 0))
    deterministic: bool = bool(_training_default("deterministic", True))
    amp: bool = bool(_training_default("amp", True))                   # automatic mixed precision
    ema: bool = bool(_training_default("ema", True))                   # exponential moving average of model weights
    pin_memory: bool = bool(_training_default("pin_memory", False))           # pin dataloader memory (disable in Docker: small /dev/shm)
    close_mosaic: int = int(_training_default("close_mosaic", 10))             # disable mosaic in last N epochs

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer: Literal[
        "auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"
    ] = str(_training_default("optimizer", "auto"))
    lr0: float = float(_training_default("lr0", 0.01))                  # initial learning rate
    lrf: float = float(_training_default("lrf", 0.01))                  # final LR factor (final_lr = lr0 * lrf)
    momentum: float = float(_training_default("momentum", 0.937))
    weight_decay: float = float(_training_default("weight_decay", 0.0005))
    warmup_epochs: float = float(_training_default("warmup_epochs", 3.0))
    warmup_momentum: float = float(_training_default("warmup_momentum", 0.8))
    warmup_bias_lr: float = float(_training_default("warmup_bias_lr", 0.1))
    cos_lr: bool = bool(_training_default("cos_lr", False))               # cosine LR scheduler

    # ── Model ─────────────────────────────────────────────────────────────────
    pretrained: str = str(_training_default("pretrained", ""))               # weight path or weight_id
    yolo_model: Literal["", "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"] = str(_training_default("yolo_model", ""))  # official YOLO model
    use_yolo_pretrained: bool = bool(_training_default("use_yolo_pretrained", True))   # use YOLO pretrained weights (False = train from scratch)
    freeze: int | list[int] = _training_default("freeze", 0)        # freeze first N layers or specific indices
    resume: bool = bool(_training_default("resume", False))               # resume training from last checkpoint

    # ── Loss Weights ──────────────────────────────────────────────────────────
    box: float = float(_training_default("box", 7.5))                   # box loss weight
    cls: float = float(_training_default("cls", 0.5))                   # classification loss weight
    dfl: float = float(_training_default("dfl", 1.5))                   # distribution focal loss weight
    pose: float = float(_training_default("pose", 12.0))                 # pose loss gain (pose tasks)
    nbs: int = int(_training_default("nbs", 64))                      # nominal batch size

    # ── Inference / Validation ──────────────────────────────────────────────────
    conf: float | None = _training_default("conf", None)          # confidence threshold (None = auto)
    iou: float = float(_training_default("iou", 0.7))                   # NMS IoU threshold
    max_det: int = int(_training_default("max_det", 300))                 # max detections per image
    agnostic_nms: bool = bool(_training_default("agnostic_nms", False))         # class-agnostic NMS
    rect: bool = bool(_training_default("rect", False))                 # rectangular training
    single_cls: bool = bool(_training_default("single_cls", False))           # treat as single class

    # ── Augmentation ──────────────────────────────────────────────────────────
    hsv_h: float = float(_training_default("hsv_h", 0.015))
    hsv_s: float = float(_training_default("hsv_s", 0.7))
    hsv_v: float = float(_training_default("hsv_v", 0.4))
    degrees: float = float(_training_default("degrees", 0.0))
    translate: float = float(_training_default("translate", 0.1))
    scale: float = float(_training_default("scale", 0.5))
    shear: float = float(_training_default("shear", 0.0))
    perspective: float = float(_training_default("perspective", 0.0))
    flipud: float = float(_training_default("flipud", 0.0))
    fliplr: float = float(_training_default("fliplr", 0.5))
    bgr: float = float(_training_default("bgr", 0.0))
    mosaic: float = float(_training_default("mosaic", 1.0))
    mixup: float = float(_training_default("mixup", 0.0))
    copy_paste: float = float(_training_default("copy_paste", 0.0))
    erasing: float = float(_training_default("erasing", 0.4))
    crop_fraction: float = float(_training_default("crop_fraction", 1.0))
    auto_augment: str = str(_training_default("auto_augment", ""))             # auto augmentation policy (randaugment, autoaugment, augmix)

    # ── Saving ────────────────────────────────────────────────────────────────
    save_period: int = int(_training_default("save_period", -1))              # save checkpoint every N epochs (-1 = disabled)
    val: bool = bool(_training_default("val", True))                   # validate during training
    plots: bool = bool(_training_default("plots", True))                 # generate training plots

    # ── Monitoring (Model Designer specific) ──────────────────────────────────
    record_gradients: bool = bool(_training_default("record_gradients", False))     # record gradient statistics
    gradient_interval: int = int(_training_default("gradient_interval", 1))         # record gradients every N epochs
    record_weights: bool = bool(_training_default("record_weights", False))       # record weight statistics
    weight_interval: int = int(_training_default("weight_interval", 1))           # record weights every N epochs
    sample_per_class: int = int(_training_default("sample_per_class", 0))          # number of test samples to save per class (0 = disabled)

    # ── Segmentation / Pose specific ──────────────────────────────────────────
    overlap_mask: bool = bool(_training_default("overlap_mask", True))
    mask_ratio: int = int(_training_default("mask_ratio", 4))
    kobj: float = float(_training_default("kobj", 1.0))

    def to_train_kwargs(self) -> dict[str, Any]:
        """Convert to dict suitable for model.train(**kwargs)."""
        # Fields that are Model Designer-specific and NOT valid Ultralytics kwargs
        _INTERNAL_KEYS = {
            "record_gradients", "gradient_interval",
            "record_weights", "weight_interval",
            "sample_per_class",
        }
        # Note: yolo_model and use_yolo_pretrained are NOT in _INTERNAL_KEYS
        # because ultra_trainer needs them in the config dict
        d = self.model_dump(exclude_defaults=False)
        result = {}
        for k, v in d.items():
            if k in _INTERNAL_KEYS:
                continue
            if v is None or v == "":
                continue  # Let Ultralytics use its own defaults
            result[k] = v
        return result


class EpochMetrics(BaseModel):
    """Metrics from one training epoch.
    
    Data sources (priority order):
    1. extended_metrics.jsonl - comprehensive custom metrics (if exists)
    2. results.csv - Ultralytics standard metrics (fallback)
    3. record.json - legacy format (final fallback)
    """
    epoch: int
    timestamp: float | None = None  # Unix timestamp when epoch completed

    # Training losses (from trainer.loss_items)
    box_loss: float | None = None           # Train box loss
    cls_loss: float | None = None           # Train classification loss
    dfl_loss: float | None = None           # Train DFL loss

    # Validation losses (from metrics dict)
    val_box_loss: float | None = None       # Validation box loss
    val_cls_loss: float | None = None       # Validation classification loss
    val_dfl_loss: float | None = None       # Validation DFL loss

    # Basic validation metrics (from validator.metrics)
    precision: float | None = None          # Mean precision
    recall: float | None = None             # Mean recall
    mAP50: float | None = None              # mAP@0.5
    mAP50_95: float | None = None           # mAP@0.5:0.95
    mAP75: float | None = None              # mAP@0.75
    fitness: float | None = None            # Overall fitness score

    # Per-class metrics (from extended_metrics.jsonl)
    ap_per_class: list[float] | None = None         # AP per class
    ap50_per_class: list[float] | None = None       # AP@0.5 per class
    precision_per_class: list[float] | None = None  # Precision per class
    recall_per_class: list[float] | None = None     # Recall per class
    f1_per_class: list[float] | None = None         # F1 score per class

    # Extended validation metrics (from results.box) - legacy support
    all_ap: list[float] | None = None           # AP for all classes
    ap: list[float] | None = None               # AP per class (legacy)
    ap50: list[float] | None = None             # AP@0.5 per class (legacy)
    ap_class_index: list[int] | None = None     # Class indices
    class_result: list[list[float]] | None = None  # Per-class results
    f1: list[float] | None = None               # F1 score per class (legacy)
    f1_curve: Any | None = None                 # F1 curve data
    map: float | None = None                    # mAP@0.5:0.95 (legacy)
    map50: float | None = None                  # mAP@0.5 (legacy)
    map75: float | None = None                  # mAP@0.75 (legacy)
    maps: list[float] | None = None             # mAP at different IoU thresholds
    mean_results: list[float] | None = None     # Mean results
    mp: float | None = None                     # Mean precision (legacy)
    mr: float | None = None                     # Mean recall (legacy)
    p: list[float] | None = None                # Precision per class (legacy)
    p_curve: Any | None = None                  # Precision curve
    prec_values: Any | None = None              # Precision values
    px: Any | None = None                       # Specific precision metrics
    r: list[float] | None = None                # Recall per class (legacy)
    r_curve: Any | None = None                  # Recall curve

    # Inference latency metrics (ms) - from extended_metrics.jsonl
    inference_latency_ms: float | None = None       # Model forward pass time
    preprocess_latency_ms: float | None = None      # Preprocessing time
    postprocess_latency_ms: float | None = None     # Postprocessing time (NMS, etc.)
    total_latency_ms: float | None = None           # Total inference pipeline time

    # System info - from extended_metrics.jsonl
    device: str | None = None                   # Device used (cuda:0, cpu, etc.)
    ram_gb: float | None = None                 # RAM usage in GB
    gpu_mem_gb: float | None = None             # GPU memory allocated in GB
    gpu_mem_reserved_gb: float | None = None    # GPU memory reserved in GB

    # Training parameters
    lr: float | None = None                     # Learning rate
    epoch_time: float | None = None             # Time taken for epoch (seconds)
    val_time_s: float | None = None             # Validation time (seconds)
    
    # Legacy
    gpu_memory_mb: float | None = None          # Legacy GPU memory field


class JobRecord(BaseModel):
    """Full training job record — persisted to disk."""
    job_id: str
    model_id: str
    model_name: str = str(_MODEL_DEFAULTS.get("name", "Untitled"))
    task: str = str(_MODEL_DEFAULTS.get("task", "detect"))
    config: TrainConfig = Field(default_factory=TrainConfig)
    status: str = "pending"            # pending, running, completed, failed, stopped
    epoch: int = 0
    total_epochs: int = 0
    message: str = "Queued"
    history: list[EpochMetrics] = Field(default_factory=list)
    weight_id: str | None = None       # best.pt weight ID after training

    # Best metrics
    best_fitness: float | None = None
    best_mAP50: float | None = None
    best_mAP50_95: float | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class JobSummary(BaseModel):
    """Lightweight job listing."""
    job_id: str
    model_id: str
    model_name: str
    task: str
    status: str
    epoch: int
    total_epochs: int
    best_fitness: float | None = None
    created_at: datetime
    completed_at: datetime | None = None


class PartitionSplitConfig(BaseModel):
    """Configuration for which splits to use from a partition."""
    partition_id: str
    train: bool = False
    val: bool = False
    test: bool = False


class TrainRequest(BaseModel):
    """Request to start a training job."""
    model_id: str
    model_scale: str | None = None  # Scale char: n, s, m, l, x
    config: TrainConfig = Field(default_factory=TrainConfig)
    partitions: list[PartitionSplitConfig] = Field(default_factory=list)  # Partition split configuration


class JobLogEntry(BaseModel):
    """Single training log line."""
    timestamp: str
    level: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
