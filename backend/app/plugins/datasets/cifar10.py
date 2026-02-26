"""CIFAR-10 dataset plugin."""
from __future__ import annotations
from ..base import DatasetPlugin
from ..loader import register_dataset
from app.config import DATASETS_DIR


class CIFAR10Plugin(DatasetPlugin):
    @property
    def name(self) -> str:
        return "cifar10"

    @property
    def display_name(self) -> str:
        return "CIFAR-10"

    @property
    def task_type(self) -> str:
        return "classification"

    @property
    def input_shape(self) -> list[int]:
        return [3, 32, 32]

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def class_names(self) -> list[str]:
        return [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]

    @property
    def train_size(self) -> int:
        return 50000

    @property
    def test_size(self) -> int:
        return 10000

    @property
    def normalization(self) -> tuple[tuple, tuple]:
        return ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    @property
    def data_dirs(self) -> list[str]:
        return ["cifar-10-batches-py"]

    def is_available(self) -> bool:
        return (DATASETS_DIR / "cifar-10-batches-py").exists()

    def download(self, state: dict) -> None:
        from torchvision import datasets
        from app.utils.download import tracked_torchvision_download
        tracked_torchvision_download(datasets.CIFAR10, str(DATASETS_DIR), state)

    def codegen_load_code(self, var_name: str, split_train: bool, transform_var: str = "transform") -> str:
        flag = "True" if split_train else "False"
        return f'datasets.CIFAR10(DATASETS_DIR, train={flag}, download=True, transform={transform_var})'

    def load_train(self, transform=None):
        from torchvision import datasets
        return datasets.CIFAR10(str(DATASETS_DIR), train=True, download=True, transform=transform)

    def load_test(self, transform=None):
        from torchvision import datasets
        return datasets.CIFAR10(str(DATASETS_DIR), train=False, download=True, transform=transform)


register_dataset(CIFAR10Plugin())
