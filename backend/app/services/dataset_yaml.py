"""
Dataset YAML Generation — bridges existing dataset plugins to Ultralytics
``data.yaml`` format for use with ``model.train(data="data.yaml")``.

Ultralytics data.yaml format:
    path: /abs/path/to/dataset
    train: images/train    (or train/)
    val: images/val
    test: images/test      (optional)
    nc: 80
    names: [person, bicycle, ...]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import DATASETS_DIR, SPLITS_DIR
from .dataset_registry import get_class_names, get_task_type


def _load_partition_cache(dataset_name: str) -> dict | None:
    """Load cached partition indices."""
    cache_path = SPLITS_DIR / f"{dataset_name.lower()}_partition_cache.json"
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text())
    except Exception:
        return None


def _find_images_dir(ds_path: Path, split_name: str) -> Path | None:
    """Find the images directory for a given split in a dataset."""
    candidates = [
        ds_path / 'images' / split_name,
        ds_path / 'images' / f'{split_name}2017',  # COCO format
        ds_path / split_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def generate_partition_txt_splits(
    dataset_name: str,
    partition_configs: list[dict[str, Any]],
) -> dict[str, Path]:
    """Generate TXT file lists for partition-based training.

    Writes absolute image paths into ``<dataset>/splits/`` files:
        train_<hash>.txt  — one absolute image path per line
        val_<hash>.txt

    Returns a dict mapping split name → Path to the txt file,
    e.g. {'train': Path('.../splits/train_abc123.txt'), 'val': Path(...)}.

    If a partition has no images for a split, that key is omitted.
    """
    import hashlib

    cache = _load_partition_cache(dataset_name)
    if not cache:
        raise RuntimeError(
            f"No partition cache found for dataset '{dataset_name}'. "
            "Please open the dataset and let the partition cache build first."
        )

    ds_path = DATASETS_DIR / dataset_name
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {ds_path}")

    # Stable hash of the partition config so we reuse existing txt files
    config_key = json.dumps(
        sorted(
            [
                {
                    "id": p["partition_id"],
                    "train": p.get("train", False),
                    "val": p.get("val", False),
                    "test": p.get("test", False),
                }
                for p in partition_configs
            ],
            key=lambda x: x["id"],
        ),
        sort_keys=True,
    )
    config_hash = hashlib.md5(config_key.encode()).hexdigest()[:8]

    splits_dir = ds_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Path] = {}

    for split_name in ("train", "val", "test"):
        # Which partitions contribute to this split?
        wants_split = any(p.get(split_name, False) for p in partition_configs)
        if not wants_split:
            continue

        split_cache = cache.get("splits", {}).get(split_name)
        if not split_cache:
            continue

        indices_map: dict[str, list[int]] = split_cache.get("indices", {})

        # Collect selected indices (preserving order, deduplicating)
        seen: set[int] = set()
        selected_indices: list[int] = []
        for p in partition_configs:
            if p.get(split_name, False):
                for idx in indices_map.get(p["partition_id"], []):
                    if idx not in seen:
                        seen.add(idx)
                        selected_indices.append(idx)

        if not selected_indices:
            continue

        # Find source images directory
        images_dir = _find_images_dir(ds_path, split_name)
        if not images_dir:
            continue

        image_files = sorted(images_dir.glob("*.*"))

        # Collect absolute paths for selected indices
        paths: list[str] = []
        for idx in selected_indices:
            if idx < len(image_files):
                paths.append(str(image_files[idx].resolve()))

        if not paths:
            continue

        txt_path = splits_dir / f"{split_name}_{config_hash}.txt"
        txt_path.write_text("\n".join(paths) + "\n")
        result[split_name] = txt_path

    return result


def generate_data_yaml(
    dataset_name: str,
    *,
    custom_path: str | Path | None = None,
    split_config: dict[str, Any] | None = None,
    partition_configs: list[dict[str, Any]] | None = None,
    txt_splits: dict[str, Path] | None = None,
) -> str:
    """Generate an Ultralytics-compatible ``data.yaml`` string.

    Parameters
    ----------
    dataset_name : str
        Registered dataset name (e.g. "coco", "coco128", "voc", "mnist").
    custom_path : str | Path | None
        Override the dataset root path. Defaults to ``DATASETS_DIR / dataset_name``.
    split_config : dict | None
        Optional overrides for split directory names (ignored when txt_splits given).
    partition_configs : list[dict] | None
        Kept for backward-compat / comment generation only.
    txt_splits : dict[str, Path] | None
        When provided, train/val/test point directly at these TXT file paths
        (absolute). This is the preferred partition approach — no copying needed.

    Returns
    -------
    str
        YAML content string.
    """
    if partition_configs is None:
        partition_configs = []

    ds_path = Path(custom_path) if custom_path else DATASETS_DIR / dataset_name

    class_names = get_class_names(dataset_name)
    task_type = get_task_type(dataset_name)
    nc = len(class_names) if class_names else 0

    # Format class names block
    if class_names:
        names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    else:
        names_block = "  0: class0"

    # ── Build partition info comment ──────────────────────────────────────────
    if partition_configs:
        partition_summary = []
        for p in partition_configs:
            splits = [s for s in ("train", "val", "test") if p.get(s)]
            partition_summary.append(f"{p['partition_id']}[{','.join(splits)}]")
        partition_info = f"partition configs: {', '.join(partition_summary)}"
    else:
        partition_info = "all partitions with all splits"

    lines = [
        f"# Ultralytics data.yaml — {dataset_name}",
        f"# Task: {task_type} | Classes: {nc}",
        f"# Generated by Model DESIGNER",
        f"# Selected {partition_info}",
        f"",
    ]

    if txt_splits:
        # ── TXT file list mode (preferred for partitions) ─────────────────────
        # path: dataset root (used to resolve relative label paths)
        lines.append(f"path: {ds_path}")
        if "train" in txt_splits:
            lines.append(f"train: {txt_splits['train']}")
        # Ultralytics requires 'val' key — use val if exists, else duplicate train
        if "val" in txt_splits:
            lines.append(f"val: {txt_splits['val']}")
        elif "train" in txt_splits:
            lines.append(f"val: {txt_splits['train']}  # No val split, using train for validation")
        if "test" in txt_splits:
            lines.append(f"test: {txt_splits['test']}")
    else:
        # ── Directory mode (no partitions / full dataset) ─────────────────────
        train_dir = "images/train"
        val_dir = "images/val"
        test_dir = "images/test"

        if split_config:
            train_dir = split_config.get("train", train_dir)
            val_dir = split_config.get("val", val_dir)
            test_dir = split_config.get("test", test_dir)

        # Auto-detect directory structure
        if ds_path.exists():
            if (ds_path / "images" / "train2017").exists():
                train_dir = "images/train2017"
                val_dir = "images/val2017" if (ds_path / "images" / "val2017").exists() else train_dir
            elif (ds_path / "train2017").exists():
                train_dir = "train2017"
                val_dir = "val2017" if (ds_path / "val2017").exists() else train_dir
            elif (ds_path / "images" / "train").exists():
                train_dir = "images/train"
                val_dir = "images/val" if (ds_path / "images" / "val").exists() else train_dir
                test_dir = "images/test"
            elif (ds_path / "train").exists():
                train_dir = "train"
                val_dir = "val" if (ds_path / "val").exists() else train_dir
                test_dir = "test"

        lines.append(f"path: {ds_path}")
        lines.append(f"train: {train_dir}")
        # Ultralytics requires 'val' key — use val if exists, else duplicate train
        if val_dir != train_dir:
            lines.append(f"val: {val_dir}")
        else:
            lines.append(f"val: {val_dir}  # No val split, using train for validation")
        if ds_path.exists() and (ds_path / test_dir).exists():
            lines.append(f"test: {test_dir}")

    lines.extend([
        f"",
        f"nc: {nc}",
        f"names:",
        names_block,
        f"",
    ])

    return "\n".join(lines)


def write_data_yaml(
    dataset_name: str,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    """Generate and write a ``data.yaml`` file to disk.

    When ``partition_configs`` is provided, generates TXT file lists inside
    ``<dataset>/splits/`` and points the YAML at those files.
    No data is copied or symlinked.

    Returns the Path to the written file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    partition_configs: list[dict[str, Any]] | None = kwargs.get("partition_configs")

    txt_splits: dict[str, Path] | None = None
    if partition_configs:
        txt_splits = generate_partition_txt_splits(dataset_name, partition_configs)
        # Remove partition_configs from kwargs so generate_data_yaml doesn't
        # try to use it for path resolution (we pass txt_splits instead)
        kwargs = {k: v for k, v in kwargs.items() if k != "partition_configs"}

    yaml_str = generate_data_yaml(dataset_name, txt_splits=txt_splits, **kwargs)
    out.write_text(yaml_str)
    return out
