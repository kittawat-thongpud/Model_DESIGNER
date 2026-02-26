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


def _default_train_workers() -> int:
    """Default dataloader workers, overridable by env for memory tuning."""
    raw = os.getenv("MODEL_DESIGNER_TRAIN_WORKERS", "4")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 4


class TrainConfig(BaseModel):
    """
    Ultralytics model.train() configuration.
    Every field maps 1:1 to a model.train() kwarg.
    See: https://docs.ultralytics.com/modes/train/#train-settings
    """

    # ── Data ──────────────────────────────────────────────────────────────────
    data: str = ""                     # path to data.yaml
    imgsz: int = 640
    batch: int = 16
    workers: int = Field(default_factory=_default_train_workers)

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 100
    patience: int = 100                # early stopping patience (0 = disabled)
    device: str = ""                   # "", "cpu", "0", "0,1", etc.
    seed: int = 0
    deterministic: bool = True
    amp: bool = True                   # automatic mixed precision
    close_mosaic: int = 10             # disable mosaic in last N epochs

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer: Literal[
        "auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"
    ] = "auto"
    lr0: float = 0.01                  # initial learning rate
    lrf: float = 0.01                  # final LR factor (final_lr = lr0 * lrf)
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    cos_lr: bool = False               # cosine LR scheduler

    # ── Model ─────────────────────────────────────────────────────────────────
    pretrained: str = ""               # weight path or weight_id
    yolo_model: Literal["", "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"] = ""  # official YOLO model
    use_yolo_pretrained: bool = True   # use YOLO pretrained weights (False = train from scratch)
    freeze: int | list[int] = 0        # freeze first N layers or specific indices
    resume: bool = False               # resume training from last checkpoint

    # ── Loss Weights ──────────────────────────────────────────────────────────
    box: float = 7.5                   # box loss weight
    cls: float = 0.5                   # classification loss weight
    dfl: float = 1.5                   # distribution focal loss weight
    pose: float = 12.0                 # pose loss gain (pose tasks)
    nbs: int = 64                      # nominal batch size

    # ── Inference / Validation ──────────────────────────────────────────────────
    conf: float | None = None          # confidence threshold (None = auto)
    iou: float = 0.7                   # NMS IoU threshold
    max_det: int = 300                 # max detections per image
    agnostic_nms: bool = False         # class-agnostic NMS
    rect: bool = False                 # rectangular training
    single_cls: bool = False           # treat as single class

    # ── Augmentation ──────────────────────────────────────────────────────────
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    bgr: float = 0.0
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    erasing: float = 0.4
    crop_fraction: float = 1.0
    auto_augment: str = ""             # auto augmentation policy (randaugment, autoaugment, augmix)

    # ── Saving ────────────────────────────────────────────────────────────────
    save_period: int = -1              # save checkpoint every N epochs (-1 = disabled)
    val: bool = True                   # validate during training
    plots: bool = True                 # generate training plots

    # ── Monitoring (Model Designer specific) ──────────────────────────────────
    record_gradients: bool = False     # record gradient statistics
    gradient_interval: int = 1         # record gradients every N epochs
    record_weights: bool = False       # record weight statistics
    weight_interval: int = 1           # record weights every N epochs
    sample_per_class: int = 0          # number of test samples to save per class (0 = disabled)

    # ── Segmentation / Pose specific ──────────────────────────────────────────
    overlap_mask: bool = True
    mask_ratio: int = 4
    kobj: float = 1.0

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
    model_name: str = "Untitled"
    task: str = "detect"
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
