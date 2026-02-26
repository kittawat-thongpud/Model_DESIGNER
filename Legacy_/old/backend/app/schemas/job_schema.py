"""
Pydantic schemas for training job records.
Full training configuration with all PyTorch-native parameters.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any, Literal


class JobConfig(BaseModel):
    """
    Complete training configuration.
    Groups: Basic Training, Optimizer, Model, Augmentation, Loss, System.
    All fields have sensible defaults so existing callers still work.
    """

    # ── 1. Basic Training ─────────────────────────────────────────────────────
    dataset: str = "mnist"
    epochs: int = 5
    batch_size: int = 64
    imgsz: int = 0                     # 0 = use dataset native size
    device: str = "auto"               # "auto", "cpu", "cuda", "cuda:0", etc.
    workers: int = 2
    patience: int = 0                  # 0 = no early stopping
    val: bool = True
    seed: int = 0                      # 0 = random seed
    deterministic: bool = False

    # ── 2. Optimizer ──────────────────────────────────────────────────────────
    optimizer: Literal["Adam", "AdamW", "SGD"] = "Adam"
    lr0: float = 0.001                 # initial learning rate
    lrf: float = 0.01                  # final LR factor (final_lr = lr0 * lrf)
    momentum: float = 0.9             # SGD momentum / Adam beta1
    weight_decay: float = 0.0005
    warmup_epochs: int = 0             # 0 = no warmup
    warmup_momentum: float = 0.8       # starting momentum during warmup
    warmup_bias_lr: float = 0.1        # starting LR for bias during warmup
    cos_lr: bool = False               # use CosineAnnealingLR

    # ── 3. Model Structure ────────────────────────────────────────────────────
    pretrained: str = ""               # weight_id to load as starting point
    freeze: int = 0                    # freeze first N layers
    amp: bool = False                  # mixed precision (FP16)

    # ── 4. Augmentation (image datasets only) ─────────────────────────────────
    hsv_h: float = 0.015               # hue augmentation
    hsv_s: float = 0.7                 # saturation augmentation
    hsv_v: float = 0.4                 # brightness augmentation
    degrees: float = 0.0               # rotation ±degrees
    translate: float = 0.1             # translation ±fraction
    scale: float = 0.5                 # scale ±fraction
    shear: float = 0.0                 # shear ±degrees
    flipud: float = 0.0                # vertical flip probability
    fliplr: float = 0.5                # horizontal flip probability
    erasing: float = 0.0               # random erasing probability
    auto_augment: str = ""             # "", "randaugment", "autoaugment", "trivialaugmentwide"
    crop_fraction: float = 1.0         # 1.0 = no random crop

    # ── 5. Loss ───────────────────────────────────────────────────────────────
    cls_weight: float = 1.0            # classification loss weight

    # ── 6. System ─────────────────────────────────────────────────────────────
    save_period: int = 0               # save checkpoint every N epochs, 0 = only at end
    nbs: int = 64                      # nominal batch size for gradient accumulation
    global_overrides: dict[str, Any] = Field(default_factory=dict)  # global variable values

    # ── 7. Weight Recording ───────────────────────────────────────────────────
    weight_record_enabled: bool = False
    weight_record_layers: list[str] = Field(default_factory=list)  # empty = all layers
    weight_record_frequency: int = 5   # record every N epochs

    # Legacy alias
    @property
    def learning_rate(self) -> float:
        return self.lr0


class EpochMetrics(BaseModel):
    epoch: int

    # Losses
    train_loss: float
    train_cls_loss: float = 0.0
    val_loss: float | None = None
    val_cls_loss: float | None = None

    # Accuracy
    train_accuracy: float
    val_accuracy: float | None = None

    # Classification metrics (from validation)
    precision: float | None = None       # macro-averaged
    recall: float | None = None          # macro-averaged
    f1: float | None = None              # macro-averaged

    # System
    lr: float = 0.0                      # current learning rate
    epoch_time: float = 0.0              # seconds per epoch
    gpu_memory_mb: float | None = None   # peak GPU memory (MB)


class JobRecord(BaseModel):
    """Full training job record — persisted to disk."""
    job_id: str
    model_id: str
    model_name: str = "Untitled"
    config: JobConfig = Field(default_factory=JobConfig)
    status: str = "pending"  # pending, running, completed, failed, stopped
    epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    train_accuracy: float = 0.0
    val_loss: float | None = None
    val_accuracy: float | None = None
    best_val_loss: float | None = None
    best_val_accuracy: float | None = None
    message: str = "Queued"
    history: list[EpochMetrics] = Field(default_factory=list)
    weight_id: str | None = None

    # Post-training analysis
    confusion_matrix: list[list[int]] | None = None
    class_names: list[str] = Field(default_factory=list)
    per_class_metrics: list[dict] | None = None  # [{class, precision, recall, f1, support}]

    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class JobSummary(BaseModel):
    """Lightweight job listing."""
    job_id: str
    model_id: str
    model_name: str
    status: str
    epoch: int
    total_epochs: int
    train_loss: float
    train_accuracy: float
    val_accuracy: float | None = None
    created_at: datetime
    completed_at: datetime | None = None


class JobLogEntry(BaseModel):
    """Single training log line."""
    timestamp: str
    level: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
