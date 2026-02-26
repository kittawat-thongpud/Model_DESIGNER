"""Fashion-MNIST dataset plugin."""
from __future__ import annotations
from ..base import DatasetPlugin
from ..loader import register_dataset
from app.config import DATASETS_DIR


class FashionMNISTPlugin(DatasetPlugin):
    @property
    def name(self) -> str:
        return "fashion_mnist"

    @property
    def display_name(self) -> str:
        return "Fashion-MNIST"

    @property
    def task_type(self) -> str:
        return "classification"

    @property
    def input_shape(self) -> list[int]:
        return [1, 28, 28]

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def class_names(self) -> list[str]:
        return [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ]

    @property
    def train_size(self) -> int:
        return 60000

    @property
    def test_size(self) -> int:
        return 10000

    @property
    def normalization(self) -> tuple[tuple, tuple]:
        return ((0.5,), (0.5,))

    @property
    def data_dirs(self) -> list[str]:
        return ["FashionMNIST"]

    def is_available(self) -> bool:
        return (DATASETS_DIR / "FashionMNIST").exists()

    def download(self, state: dict) -> None:
        from torchvision import datasets
        from app.utils.download import tracked_torchvision_download
        tracked_torchvision_download(datasets.FashionMNIST, str(DATASETS_DIR), state)

    def codegen_load_code(self, var_name: str, split_train: bool, transform_var: str = "transform") -> str:
        flag = "True" if split_train else "False"
        return f'datasets.FashionMNIST(DATASETS_DIR, train={flag}, download=True, transform={transform_var})'

    def load_train(self, transform=None):
        from torchvision import datasets
        return datasets.FashionMNIST(str(DATASETS_DIR), train=True, download=True, transform=transform)

    def load_test(self, transform=None):
        from torchvision import datasets
        return datasets.FashionMNIST(str(DATASETS_DIR), train=False, download=True, transform=transform)


register_dataset(FashionMNISTPlugin())
