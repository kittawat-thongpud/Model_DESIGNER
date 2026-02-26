---
description: How to add a new dataset plugin to Model Designer
---

# Adding a New Dataset Plugin

This workflow guides you through creating a new dataset plugin for Model Designer.

## Overview

Dataset plugins extend Model Designer with new datasets. They handle:
- Dataset metadata (name, task type, classes)
- Download and extraction
- Loading train/val/test splits
- Converting to training format

## Step 1: Choose Dataset Type

Determine your dataset's task type:
- **Classification**: MNIST, CIFAR-10, Fashion-MNIST
- **Detection**: COCO, COCO128

## Step 2: Create Plugin File

Create a new file in `backend/app/plugins/datasets/`:

```bash
# Example: backend/app/plugins/datasets/my_dataset.py
```

## Step 3: Import Required Modules

```python
from __future__ import annotations
import json
import os
import threading
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image as PILImage

from app.config import DATASETS_DIR
from ..base import DatasetPlugin
from ..loader import register_dataset
```

## Step 4: Define Dataset Root and URLs

```python
_MY_DATASET_ROOT = DATASETS_DIR / "my_dataset"
_DOWNLOAD_URL = "https://example.com/my_dataset.zip"
```

## Step 5: Create Plugin Class

### For Classification Datasets

```python
class MyDatasetPlugin(DatasetPlugin):
    @property
    def name(self) -> str:
        return "my_dataset"
    
    @property
    def display_name(self) -> str:
        return "My Dataset (Classification)"
    
    @property
    def task_type(self) -> str:
        return "classification"
    
    @property
    def input_shape(self) -> list[int]:
        return [3, 224, 224]  # [channels, height, width]
    
    @property
    def num_classes(self) -> int:
        return 10
    
    @property
    def class_names(self) -> list[str]:
        return ["class0", "class1", ...]
    
    @property
    def train_size(self) -> int:
        return 50000
    
    @property
    def test_size(self) -> int:
        return 10000
    
    @property
    def normalization(self) -> tuple[tuple, tuple]:
        # (mean, std) for each channel
        return ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    @property
    def data_dirs(self) -> list[str]:
        return ["my_dataset"]
    
    def is_available(self) -> bool:
        return (_MY_DATASET_ROOT / "train").exists()
    
    def download(self, state: dict) -> None:
        from app.utils.download import download_and_extract
        
        state["message"] = "Downloading..."
        state["progress"] = 0
        
        download_and_extract(
            _DOWNLOAD_URL,
            str(DATASETS_DIR),
            state
        )
        
        state["message"] = "Complete"
        state["progress"] = 100
    
    def clear_data(self) -> list[str]:
        """Delete downloaded dataset files and return list of deleted directories."""
        import shutil
        
        deleted = []
        if _MY_DATASET_ROOT.exists():
            shutil.rmtree(_MY_DATASET_ROOT)
            deleted.append(str(_MY_DATASET_ROOT))
        
        return deleted
    
    def load_train(self, transform=None):
        from torchvision import datasets
        return datasets.ImageFolder(
            str(_MY_DATASET_ROOT / "train"),
            transform=transform
        )
    
    def load_test(self, transform=None):
        from torchvision import datasets
        return datasets.ImageFolder(
            str(_MY_DATASET_ROOT / "test"),
            transform=transform
        )
```

### For Detection Datasets

Detection datasets need additional components:

1. **Index Builder** (optional but recommended for large datasets):
```python
def _build_index(ann_path: Path, img_dir: Path, index_path: Path) -> list[dict]:
    """Build lightweight per-image index from annotations."""
    # Parse annotations and create index
    # Format: [{"file": "img.jpg", "w": 640, "h": 480, "anns": [...]}]
    pass
```

2. **Raw Dataset Class**:
```python
class MyRawDataset(Dataset):
    """Returns (PIL_image, [annotation_dicts])."""
    
    def __init__(self, img_dir: Path, index: list[dict], transform=None):
        self._img_dir = img_dir
        self._index = index
        self._transform = transform
    
    def __len__(self):
        return len(self._index)
    
    def __getitem__(self, idx):
        entry = self._index[idx]
        img_path = self._img_dir / entry["file"]
        img = PILImage.open(img_path).convert("RGB")
        
        if self._transform is not None:
            img = self._transform(img)
        
        return img, entry["anns"]
```

3. **Training Wrapper**:
```python
class _MyDetectionWrapper(Dataset):
    """Converts raw format to training format with normalized boxes."""
    
    def __init__(self, raw_ds, cat_id_to_idx: dict[int, int]):
        self._ds = raw_ds
        self._cat_map = cat_id_to_idx
    
    def __len__(self):
        return len(self._ds)
    
    def __getitem__(self, idx):
        img, anns = self._ds[idx]
        
        # Get image dimensions
        if isinstance(img, torch.Tensor):
            _, img_h, img_w = img.shape
        else:
            img_w, img_h = img.size
        
        boxes = []
        labels = []
        
        for ann in anns:
            # Convert bbox to normalized center format [cx, cy, w, h]
            x_min, y_min, bw, bh = ann["bbox"]  # pixel coords
            cx = (x_min + bw / 2) / img_w
            cy = (y_min + bh / 2) / img_h
            nw = bw / img_w
            nh = bh / img_h
            boxes.append([cx, cy, nw, nh])
            labels.append(self._cat_map[ann["category_id"]])
        
        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)
        
        return img, {"boxes": boxes_t, "labels": labels_t}
```

4. **Plugin Methods**:
```python
def category_id_to_name(self) -> dict[int, str]:
    return {i: name for i, name in enumerate(self.class_names)}

def scan_splits(self) -> dict[str, dict]:
    """Return split statistics."""
    return {
        "train": {"total": 1000, "labeled": 1000},
        "val": {"total": 100, "labeled": 100},
        "test": {"total": 0, "labeled": 0},
    }

def wrap_for_training(self, dataset) -> "_MyDetectionWrapper":
    """Wrap raw dataset for training."""
    cat_map = {i: i for i in range(self.num_classes)}
    return _MyDetectionWrapper(dataset, cat_map)
```

## Step 6: Add Required Methods

### For All Dataset Types

```python
def clear_data(self) -> list[str]:
    """Delete downloaded dataset files and return list of deleted directories.
    
    REQUIRED: Called by DELETE /api/datasets/{name}/data endpoint.
    """
    import shutil
    
    deleted = []
    if _MY_DATASET_ROOT.exists():
        shutil.rmtree(_MY_DATASET_ROOT)
        deleted.append(str(_MY_DATASET_ROOT))
        # Clear any cached data
        with self._index_lock:
            self._index = None
    
    return deleted
```

### For Detection Datasets Only

```python
@property
def category_id_map(self) -> dict[int, str]:
    """Map category_id → category name.
    
    REQUIRED: Used by dataset browser and sample viewer.
    """
    return {c["id"]: c["name"] for c in self._categories}
```

## Step 7: Download Implementation

**IMPORTANT**: Use correct `download_and_extract` signature:

```python
def download(self, state: dict) -> None:
    from app.utils.download import download_and_extract
    
    state["message"] = "Downloading dataset..."
    state["progress"] = 0
    
    # Correct signature: (url, extract_to, state, step_label, file_key=None)
    download_and_extract(
        _DOWNLOAD_URL,
        str(DATASETS_DIR),  # Extract destination
        state,
        "My Dataset"        # Step label for progress display
    )
    
    state["message"] = "Building index..."
    state["progress"] = 90
    
    # Build index if needed
    _build_index()
    
    state["message"] = "Complete"
    state["progress"] = 100
```

**Common Mistakes:**
- ❌ `download_and_extract(url, dest, state, extract_dir=...)` — No `extract_dir` parameter
- ✅ `download_and_extract(url, dest, state, "Label")` — Correct

## Step 8: Register Plugin

At the end of your file:

```python
register_dataset(MyDatasetPlugin())
```

## Step 9: Validation & Testing

### Pre-deployment Checklist

**Required Properties:**
- [ ] `name` — unique lowercase identifier
- [ ] `display_name` — human-readable name
- [ ] `task_type` — "classification" or "detection"
- [ ] `input_shape` — [channels, height, width]
- [ ] `num_classes` — number of classes
- [ ] `class_names` — list of class names
- [ ] `train_size` — number of training samples
- [ ] `test_size` — number of test samples
- [ ] `val_size` — number of validation samples (0 if none)
- [ ] `normalization` — ((mean...), (std...))
- [ ] `data_dirs` — list of directory names

**Required Methods:**
- [ ] `is_available()` — check if dataset is downloaded
- [ ] `download(state)` — download and extract dataset
- [ ] `clear_data()` — delete dataset files, return deleted dirs
- [ ] `load_train(transform)` — return training dataset
- [ ] `load_val(transform)` — return validation dataset (or None)
- [ ] `load_test(transform)` — return test dataset (or None)
- [ ] `scan_splits()` — return split statistics

**Detection-specific:**
- [ ] `category_id_map` property — map id → name
- [ ] `wrap_for_training(dataset)` — return training wrapper
- [ ] Training wrapper converts to normalized xywh format
- [ ] Boxes in format [cx, cy, w, h] normalized 0-1

**Code Quality:**
- [ ] No syntax errors
- [ ] All imports at top of file
- [ ] Thread-safe index caching (if applicable)
- [ ] `register_dataset()` called at end

### Testing Steps

1. **Import Test**
```bash
cd /path/to/Model_DESIGNER
PYTHONPATH=backend python3 -c "from app.plugins.datasets import my_dataset; print('OK')"
```

2. **Registration Test**
```bash
PYTHONPATH=backend python3 -c "
from app.plugins.loader import discover_plugins, get_dataset_plugin
discover_plugins()
p = get_dataset_plugin('my_dataset')
print(f'Plugin: {p.display_name}')
print(f'Classes: {p.num_classes}')
"
```

3. **Backend Test**
- Restart backend server
- Check `/api/datasets` endpoint
- Verify dataset appears in list

4. **UI Test**
- Open datasets page
- Check dataset card displays correctly
- Test download button
- Verify progress updates
- Check dataset browser after download

5. **Training Test**
- Create training job with dataset
- Verify data.yaml generation
- Check training starts successfully

## Examples

- **Classification**: `backend/app/plugins/datasets/mnist.py`
- **Detection (simple)**: `backend/app/plugins/datasets/coco128.py`
- **Detection (complex)**: `backend/app/plugins/datasets/coco.py`

## Common Patterns

### Lazy Loading with Index
```python
def _get_index(self) -> list[dict]:
    with self._index_lock:
        if self._index is not None:
            return self._index
    
    if _INDEX_PATH.exists():
        with open(_INDEX_PATH) as f:
            index = json.load(f)
    else:
        index = _build_index()
    
    with self._index_lock:
        self._index = index
    return index
```

### Thread-safe Initialization
```python
def __init__(self):
    self._categories = [...]  # Static data
    self._index: list[dict] | None = None
    self._index_lock = threading.Lock()
```

## Troubleshooting

### Error: 'MyPlugin' object has no attribute 'clear_data'
**Solution**: Add `clear_data()` method that returns `list[str]` of deleted directories.

### Error: download_and_extract() got an unexpected keyword argument 'extract_dir'
**Solution**: Use correct signature: `download_and_extract(url, dest, state, step_label)`

### Error: 'MyPlugin' object has no attribute 'category_id_map'
**Solution**: For detection datasets, add `@property category_id_map` that returns `dict[int, str]`.

### Dataset not appearing in UI
**Checks**:
1. File is in `backend/app/plugins/datasets/`
2. `register_dataset()` is called at end of file
3. No import errors (check backend logs)
4. Backend server restarted after adding plugin

### Download fails silently
**Checks**:
1. `state["message"]` and `state["progress"]` are updated
2. URL is accessible
3. `download_and_extract` parameters are correct
4. Exception handling doesn't swallow errors

### Training fails with "No such file or directory"
**Checks**:
1. `is_available()` returns correct status
2. Image/label paths in index are correct
3. Files actually exist on disk
4. Paths use absolute paths, not relative

## Quick Reference

### Minimal Classification Plugin
```python
from app.config import DATASETS_DIR
from ..base import DatasetPlugin
from ..loader import register_dataset

_ROOT = DATASETS_DIR / "my_dataset"

class MyPlugin(DatasetPlugin):
    @property
    def name(self) -> str: return "my_dataset"
    @property
    def display_name(self) -> str: return "My Dataset"
    @property
    def task_type(self) -> str: return "classification"
    @property
    def input_shape(self) -> list[int]: return [3, 224, 224]
    @property
    def num_classes(self) -> int: return 10
    @property
    def class_names(self) -> list[str]: return [...]
    @property
    def train_size(self) -> int: return 1000
    @property
    def test_size(self) -> int: return 100
    @property
    def normalization(self) -> tuple: return ((0.5,0.5,0.5), (0.5,0.5,0.5))
    @property
    def data_dirs(self) -> list[str]: return ["my_dataset"]
    
    def is_available(self) -> bool:
        return (_ROOT / "train").exists()
    
    def download(self, state: dict) -> None:
        from app.utils.download import download_and_extract
        download_and_extract(URL, str(DATASETS_DIR), state, "My Dataset")
    
    def clear_data(self) -> list[str]:
        import shutil
        if _ROOT.exists():
            shutil.rmtree(_ROOT)
            return [str(_ROOT)]
        return []
    
    def load_train(self, transform=None):
        from torchvision import datasets
        return datasets.ImageFolder(str(_ROOT / "train"), transform=transform)
    
    def load_test(self, transform=None):
        from torchvision import datasets
        return datasets.ImageFolder(str(_ROOT / "test"), transform=transform)

register_dataset(MyPlugin())
```
