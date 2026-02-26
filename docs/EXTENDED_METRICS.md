# Extended Metrics System

## Overview

Model Designer uses `extended_metrics.jsonl` to capture comprehensive training and validation metrics that go beyond Ultralytics' standard `results.csv`.

## File Structure

### Location
```
backend/data/jobs/{job_id}/extended_metrics.jsonl
```

### Format
JSONL (JSON Lines) - one JSON object per epoch, appended after each validation.

## Data Priority

When loading job history, the system uses this priority:

1. **extended_metrics.jsonl** (if exists) - Most comprehensive
2. **results.csv** (Ultralytics standard) - Fallback
3. **record.json** (legacy) - Final fallback

## Captured Metrics

### Training Losses
- `train_box_loss` - Bounding box regression loss
- `train_cls_loss` - Classification loss
- `train_dfl_loss` - Distribution Focal Loss

### Validation Losses
- `val_box_loss` - Validation bounding box loss
- `val_cls_loss` - Validation classification loss
- `val_dfl_loss` - Validation DFL loss

### Validation Metrics
- `map50` - mAP@0.5
- `map` - mAP@0.5:0.95
- `map75` - mAP@0.75
- `precision` - Mean precision
- `recall` - Mean recall
- `fitness` - Overall fitness score

### Per-Class Metrics
- `ap_per_class` - AP per class (list)
- `ap50_per_class` - AP@0.5 per class (list)
- `precision_per_class` - Precision per class (list)
- `recall_per_class` - Recall per class (list)
- `f1_per_class` - F1 score per class (list)

### Inference Latency (NEW)
- `inference_latency_ms` - Model forward pass time
- `preprocess_latency_ms` - Preprocessing time
- `postprocess_latency_ms` - Postprocessing time (NMS, etc.)
- `total_latency_ms` - Total inference pipeline time

### System Information
- `device` - Device used (e.g., "cuda:0", "cpu")
- `ram_gb` - RAM usage in GB
- `gpu_mem_gb` - GPU memory allocated in GB
- `gpu_mem_reserved_gb` - GPU memory reserved in GB

### Training Parameters
- `lr` - Learning rate at this epoch
- `val_time_s` - Validation time in seconds

### Metadata
- `epoch` - Epoch number
- `timestamp` - Unix timestamp when epoch completed

## Example Entry

```json
{
  "epoch": 13,
  "timestamp": 1708700000.123,
  "train_box_loss": 1.4490,
  "train_cls_loss": 2.1600,
  "train_dfl_loss": 1.5115,
  "val_box_loss": 1.3891,
  "val_cls_loss": 2.0044,
  "val_dfl_loss": 1.4634,
  "map50": 0.2644,
  "map": 0.1712,
  "map75": 0.1234,
  "precision": 0.4504,
  "recall": 0.2613,
  "fitness": 0.2156,
  "inference_latency_ms": 15.7,
  "preprocess_latency_ms": 2.3,
  "postprocess_latency_ms": 1.2,
  "total_latency_ms": 19.2,
  "device": "cuda:0",
  "ram_gb": 8.5,
  "gpu_mem_gb": 2.3,
  "gpu_mem_reserved_gb": 2.5,
  "lr": 0.00965,
  "val_time_s": 45.2
}
```

## Implementation

### Backend: Capture Metrics

**File:** `backend/app/services/custom_trainer.py`

```python
def validate(self):
    """Run validation and collect extended metrics."""
    # ... validation logic ...
    
    # Collect comprehensive metrics
    extended_metrics = self._extract_box_metrics(box_metrics)
    
    # Add train losses
    extended_metrics['train_box_loss'] = ...
    extended_metrics['train_cls_loss'] = ...
    
    # Add latency metrics
    extended_metrics['inference_latency_ms'] = ...
    
    # Save to extended_metrics.jsonl
    self._save_extended_metrics(extended_metrics)
```

### Backend: Load with Fallback

**File:** `backend/app/services/job_storage.py`

```python
def get_job_history(job_id: str) -> list[dict]:
    """Load history with priority fallback."""
    # 1. Try extended_metrics.jsonl
    if extended_metrics_path.exists():
        return parse_extended_metrics()
    
    # 2. Fallback to results.csv
    if results_path.exists():
        return parse_results_csv()
    
    # 3. Final fallback to record.json
    return legacy_history
```

### Frontend: Display Metrics

**API Endpoint:** `GET /api/jobs/{job_id}/history`

**Response:**
```json
{
  "history": [
    {
      "epoch": 1,
      "mAP50": 0.2644,
      "inference_latency_ms": 15.7,
      "train_box_loss": 1.4490,
      "val_box_loss": 1.3891,
      ...
    }
  ]
}
```

## Benefits

### vs results.csv

| Feature | results.csv | extended_metrics.jsonl |
|---------|-------------|------------------------|
| Train losses | ✅ | ✅ |
| Val metrics | ✅ | ✅ |
| Latency metrics | ❌ | ✅ |
| Per-class metrics | ❌ | ✅ |
| System info | ❌ | ✅ |
| Timestamp | ❌ | ✅ |
| Custom fields | ❌ | ✅ Unlimited |

### Advantages

1. **Non-intrusive** - Doesn't modify Ultralytics behavior
2. **Extensible** - Easy to add new metrics
3. **Complete** - Captures everything we need
4. **Fallback** - Gracefully falls back to results.csv
5. **Frontend-ready** - Direct API integration

## Adding New Metrics

To add a new custom metric:

1. **Capture in validate():**
   ```python
   extended_metrics['my_custom_metric'] = calculate_custom()
   ```

2. **Add to _save_extended_metrics():**
   ```python
   epoch_data = {
       ...
       "my_custom_metric": metrics.get('my_custom_metric'),
   }
   ```

3. **Update get_job_history():**
   ```python
   epoch_metrics = {
       ...
       "my_custom_metric": data.get("my_custom_metric"),
   }
   ```

4. **Update EpochMetrics schema:**
   ```python
   class EpochMetrics(BaseModel):
       ...
       my_custom_metric: float | None = None
   ```

That's it! The metric will be automatically captured, saved, and available to the frontend.
