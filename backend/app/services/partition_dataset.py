"""
Custom YOLO Dataset that dynamically filters samples based on partition configurations.
This avoids copying/symlinking dataset files to job directories.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ultralytics.data.dataset import YOLODataset

from ..config import DATASETS_DIR, SPLITS_DIR


def _load_partition_cache(dataset_name: str) -> dict | None:
    """Load cached partition indices."""
    cache_path = SPLITS_DIR / f"{dataset_name.lower()}_partition_cache.json"
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text())
    except Exception:
        return None


class PartitionYOLODataset(YOLODataset):
    """YOLO Dataset that filters samples based on partition configuration.
    
    This class extends YOLODataset to support dynamic filtering of samples
    based on partition indices, eliminating the need to copy dataset files.
    """
    
    def __init__(
        self,
        dataset_name: str,
        partition_configs: list[dict[str, Any]],
        split: str,
        *args,
        **kwargs
    ):
        """Initialize partition-filtered dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'coco')
            partition_configs: List of partition configurations
                [{'partition_id': 'p_xxx', 'train': True, 'val': False, 'test': True}, ...]
            split: Split name ('train', 'val', or 'test')
            *args, **kwargs: Additional arguments for YOLODataset
        """
        self.dataset_name = dataset_name
        self.partition_configs = partition_configs
        self.split_name = split
        self.partition_indices = None
        
        # Load partition cache to get indices
        cache = _load_partition_cache(dataset_name)
        if cache:
            split_cache = cache.get('splits', {}).get(split)
            if split_cache:
                indices_map = split_cache.get('indices', {})
                
                # Collect indices from selected partitions
                selected_indices = []
                for p in partition_configs:
                    if p.get(split, False):
                        partition_id = p['partition_id']
                        if partition_id in indices_map:
                            selected_indices.extend(indices_map[partition_id])
                
                if selected_indices:
                    self.partition_indices = set(selected_indices)
        
        # Initialize parent YOLODataset
        super().__init__(*args, **kwargs)
        
        # Filter samples based on partition indices
        if self.partition_indices is not None:
            self._filter_samples()
    
    def _filter_samples(self):
        """Filter dataset samples to only include partition indices."""
        if not hasattr(self, 'im_files') or not self.im_files:
            return
        
        # Create filtered lists
        filtered_im_files = []
        filtered_labels = []
        
        for idx, (im_file, label) in enumerate(zip(self.im_files, self.labels)):
            if idx in self.partition_indices:
                filtered_im_files.append(im_file)
                filtered_labels.append(label)
        
        # Update dataset attributes
        self.im_files = filtered_im_files
        self.labels = filtered_labels
        self.ni = len(self.im_files)


def create_partition_dataset_builder(
    dataset_name: str,
    partition_configs: list[dict[str, Any]]
):
    """Create a dataset builder function for use with Ultralytics trainer.
    
    Args:
        dataset_name: Name of the dataset
        partition_configs: Partition configurations
    
    Returns:
        A function that builds PartitionYOLODataset instances
    """
    def build_dataset(img_path, mode, batch_size, *args, **kwargs):
        """Build dataset with partition filtering.
        
        Args:
            img_path: Path to images directory
            mode: 'train' or 'val'
            batch_size: Batch size
            *args, **kwargs: Additional arguments
        """
        # Determine split name from mode
        split = 'train' if mode == 'train' else 'val'
        
        return PartitionYOLODataset(
            dataset_name=dataset_name,
            partition_configs=partition_configs,
            split=split,
            img_path=img_path,
            mode=mode,
            batch_size=batch_size,
            *args,
            **kwargs
        )
    
    return build_dataset
