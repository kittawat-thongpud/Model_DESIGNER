"""
Plugin base classes:
  - DatasetPlugin       : dataset sources (COCO, ImageNet, custom, …)
  - WeightSourcePlugin  : weight checkpoint parsers (Ultralytics, plain .pt, …)
  - ModelArchPlugin     : custom model architectures (HSG-DET, …) that plug into
                          the Ultralytics training pipeline via YAML + nn injection.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class DatasetPlugin(ABC):
    """Base class for dataset plugins."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def display_name(self) -> str: ...

    @property
    @abstractmethod
    def task_type(self) -> str: ...

    @property
    @abstractmethod
    def input_shape(self) -> list[int]: ...

    @property
    @abstractmethod
    def num_classes(self) -> int: ...

    @property
    @abstractmethod
    def class_names(self) -> list[str]: ...

    @property
    @abstractmethod
    def train_size(self) -> int: ...

    @property
    @abstractmethod
    def test_size(self) -> int: ...

    @property
    @abstractmethod
    def normalization(self) -> tuple[tuple, tuple]: ...

    @property
    def data_dirs(self) -> list[str]:
        return []

    def is_available(self) -> bool:
        return False

    @property
    def manual_download(self) -> bool:
        """True if dataset cannot be auto-downloaded and requires user upload."""
        return False

    @property
    def upload_instructions(self) -> str:
        """Instructions shown in the upload dialog for manual-download datasets."""
        return ""

    def download(self, state: dict) -> None:
        raise NotImplementedError

    def load_train(self, transform=None):
        raise NotImplementedError

    def load_test(self, transform=None):
        raise NotImplementedError

    def load_val(self, transform=None):
        """Optional validation split loader. Plugins can override."""
        return None

    def load_split(self, split: str = "train", transform=None):
        """Dispatch to load_train/load_test based on split name."""
        if split == "train":
            return self.load_train(transform=transform)
        elif split == "test":
            return self.load_test(transform=transform)
        elif split == "val":
            return self.load_val(transform=transform)
        return None

    def disk_size_bytes(self) -> int:
        """Return total disk size of dataset files. Default: sum of data_dirs."""
        from app.config import DATASETS_DIR
        total = 0
        for d in self.data_dirs:
            p = DATASETS_DIR / d
            if p.exists():
                for f in p.rglob("*"):
                    if f.is_file():
                        total += f.stat().st_size
        return total

    def scan_splits(self) -> dict:
        """Return split metadata: {split: {total, labeled}}. Default from train_size/test_size."""
        result = {}
        if self.train_size > 0:
            result["train"] = {"total": self.train_size, "labeled": self.train_size}
        if self.test_size > 0:
            result["test"] = {"total": self.test_size, "labeled": self.test_size}
        return result

    def codegen_load_code(self, var_name: str, split_train: bool, transform_var: str = "transform") -> str:
        return ""

    @property
    def val_size(self) -> int:
        return 0

    def to_info_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "task_type": self.task_type,
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "val_size": self.val_size,
            "available": self.is_available(),
        }


class WeightSourcePlugin(ABC):
    """Base class for weight source plugins (detect format, extract state_dict)."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def source_name(self) -> str:
        return self.name

    @property
    def display_name(self) -> str:
        return self.name

    @property
    def file_extensions(self) -> list[str]:
        return [".pt"]

    def can_parse(self, data: Any) -> bool:
        return False

    def extract_state_dict(self, data: Any) -> dict:
        raise NotImplementedError

    def get_layer_groups(self, sd: dict) -> list[dict]:
        return []

    @property
    def has_pretrained_catalog(self) -> bool:
        return False

    def list_pretrained(self) -> list[dict]:
        return []

    def download_pretrained(self, model_key: str) -> dict:
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "file_extensions": self.file_extensions,
            "has_pretrained_catalog": self.has_pretrained_catalog,
        }


class ModelArchPlugin(ABC):
    """Base class for custom model architecture plugins.

    An arch plugin packages together:
      - An Ultralytics-format YAML definition of the model.
      - A registration hook that injects any custom nn.Module subclasses
        into the Ultralytics module namespace before YAML parsing.
      - Optional: a pretrained weight key to warm-start from (e.g. "yolov8m").

    Example
    -------
    >>> plugin = MyArchPlugin()
    >>> plugin.register_modules()              # inject custom layers
    >>> model = YOLO(str(plugin.yaml_path()))  # parse architecture
    >>> model.train(data="data.yaml", ...)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique machine-readable identifier, e.g. 'hsg_det'."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for UI, e.g. 'HSG-DET (Medium)'."""
        ...

    @property
    def task_type(self) -> str:
        """Ultralytics task type: 'detect', 'classify', 'segment', 'pose'."""
        return "detect"

    @abstractmethod
    def yaml_path(self) -> "Path":  # type: ignore[name-defined]
        """Absolute path to the Ultralytics-format YAML file."""
        ...

    def register_modules(self) -> None:
        """Inject custom nn.Module classes into ultralytics.nn.modules.

        Default: no-op. Override to add custom layer types.
        Must be idempotent (safe to call multiple times).
        """

    def pretrain_key(self) -> str | None:
        """Optional: model key to warm-start backbone from (e.g. 'yolov8m').

        If not None, the training pipeline will attempt to transfer backbone
        weights from a pretrained Ultralytics checkpoint before training.
        """
        return None

    def warm_start(self, model, log_fn=None, model_scale: str | None = None) -> dict:
        """Optional: transfer pretrained weights into model before training.

        Called by the training pipeline after YOLO(yaml_path) is created
        and before model.train() is called. Default: no-op.

        Parameters
        ----------
        model : ultralytics.YOLO
            The freshly created model instance.
        log_fn : callable | None
            ``log_fn(msg: str)`` for job-log messages.

        Returns
        -------
        dict with at minimum ``{"transferred": int, "skipped": int}``.
        Empty dict means no warm-start was performed.
        """
        return {}

    @property
    def description(self) -> str:
        return ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "task_type": self.task_type,
            "yaml_path": str(self.yaml_path()),
            "pretrain_key": self.pretrain_key(),
            "description": self.description,
        }

