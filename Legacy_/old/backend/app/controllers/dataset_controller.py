"""
Dataset Controller — list available datasets and preview samples.
Tagged as "Datasets" for ReDoc grouping.
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException

from ..schemas.model_schema import DatasetInfo

router = APIRouter(prefix="/api/datasets", tags=["Datasets"])

# ─── Built-in dataset registry ───────────────────────────────────────────────

DATASETS: dict[str, DatasetInfo] = {
    "mnist": DatasetInfo(
        name="mnist",
        display_name="MNIST Handwritten Digits",
        input_shape=[1, 28, 28],
        num_classes=10,
        train_size=60000,
        test_size=10000,
        classes=[str(i) for i in range(10)],
        task_type="classification",
    ),
    "cifar10": DatasetInfo(
        name="cifar10",
        display_name="CIFAR-10",
        input_shape=[3, 32, 32],
        num_classes=10,
        train_size=50000,
        test_size=10000,
        classes=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
        task_type="classification",
    ),
    "coco": DatasetInfo(
        name="coco",
        display_name="COCO 2017 (Detection)",
        input_shape=[3, 640, 640],
        num_classes=80,
        train_size=118287,
        test_size=5000,
        classes=["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"],
        task_type="detection",
    ),
}


@router.get("/", response_model=list[DatasetInfo], summary="List available datasets")
async def list_datasets():
    """Return info about all supported datasets."""
    return list(DATASETS.values())


@router.get("/{name}/info", response_model=DatasetInfo, summary="Get dataset metadata")
async def get_dataset_info(name: str):
    """Return metadata for a specific dataset."""
    ds = DATASETS.get(name.lower())
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    return ds


@router.get("/{name}/preview", summary="Preview dataset samples")
async def preview_dataset(name: str, count: int = 8):
    """Return a few sample images from the dataset as base64-encoded PNGs."""
    import base64
    import io

    ds = DATASETS.get(name.lower())
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")

    try:
        from torchvision import datasets, transforms
        from PIL import Image

        if name.lower() == "mnist":
            dataset = datasets.MNIST("./data", train=True, download=True)
        elif name.lower() == "cifar10":
            dataset = datasets.CIFAR10("./data", train=True, download=True)
        else:
            raise HTTPException(status_code=400, detail="Unsupported dataset")

        samples = []
        for i in range(min(count, len(dataset))):
            img, label = dataset[i]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            samples.append({
                "index": i,
                "label": label,
                "class_name": ds.classes[label] if label < len(ds.classes) else str(label),
                "image_base64": b64,
            })

        return {"dataset": name, "count": len(samples), "samples": samples}

    except ImportError:
        raise HTTPException(status_code=500, detail="torchvision not available")
