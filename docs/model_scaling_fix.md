# Model Scaling Fix

## Problem

Model Designer was generating **fully expanded YAML** without `depth_multiple` and `width_multiple`, preventing Ultralytics from applying scale transformations correctly.

### Before (Broken)
```yaml
nc: 80
scales:
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]
  l: [1.00, 1.00, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 3, C2f, [256]]
```

**Issue:** No `depth_multiple` or `width_multiple` → Ultralytics can't scale the architecture.

### After (Fixed)
```yaml
nc: 80
depth_multiple: 1.0   # ← Added
width_multiple: 1.0   # ← Added
scales:
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]
  l: [1.00, 1.00, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 3, C2f, [256]]
```

**Fixed:** Ultralytics now recognizes multipliers and applies scaling correctly.

## How Ultralytics Scaling Works

### Structural Scaling (Not Runtime)
Ultralytics modifies the **architecture definition** before building the model:

1. **Parse YAML** → Read `depth_multiple` and `width_multiple`
2. **Scale Layers** → Multiply channels by `width_multiple`, repeats by `depth_multiple`
3. **Build Model** → Instantiate scaled architecture

### Scale Detection
Ultralytics detects scale from:
1. **Filename suffix**: `model_n.yaml` → scale 'n'
2. **Explicit scales dict**: If only one scale in `scales: {n: [0.33, 0.25, 1024]}`
3. **Multipliers**: `depth_multiple: 0.33, width_multiple: 0.25`

## Implementation

### 1. YAML Generation (`model_storage.py`)
```python
def _dict_to_yaml(yaml_def: dict) -> str:
    scales = yaml_def.get("scales", {})
    if scales:
        # Use 'l' scale as base (1.0, 1.0)
        base_scale = scales.get("l") or next(iter(scales.values()))
        depth_multiple = base_scale[0]
        width_multiple = base_scale[1]
        
        lines.append(f"depth_multiple: {depth_multiple}")
        lines.append(f"width_multiple: {width_multiple}")
        lines.append("scales:")
        for key, vals in scales.items():
            lines.append(f"  {key}: {vals}")
```

### 2. Temp File Naming (`yaml_utils.py`)
```python
def prepare_model_yaml(yaml_path, scale=None):
    # Include scale in filename so Ultralytics detects it
    suffix = f"_{scale}.yaml" if scale else ".yaml"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, text=True)
    # Result: /tmp/tmpXXX_n.yaml ✅
```

### 3. Training Flow
```
Frontend → model_scale: 'n'
    ↓
train_controller → ultra_trainer.start_training(model_scale='n')
    ↓
prepare_model_yaml(yaml_path, scale='n')
    ↓
Creates: /tmp/tmpXXX_n.yaml with depth_multiple/width_multiple
    ↓
YOLO loads and applies scale 'n' correctly ✅
```

## Verification

### Check YAML Output
```bash
cat /path/to/model.yaml
```

Should contain:
```yaml
depth_multiple: 1.0
width_multiple: 1.0
scales:
  n: [0.33, 0.25, 1024]
  ...
```

### Check Training Logs
```
Validated & Patched YAML: /tmp/tmpXXX_n.yaml (Scale: n)
tmpXXX_n summary: 220 layers, ~960,000 parameters  ← Correct for scale 'n'
```

### Expected Parameters by Scale
For HSG-DET model:
- **Scale n**: ~1M params (0.33 depth × 0.25 width)
- **Scale s**: ~4M params (0.33 depth × 0.50 width)
- **Scale m**: ~17M params (0.67 depth × 0.75 width)
- **Scale l**: ~46M params (1.00 depth × 1.00 width)
- **Scale x**: ~72M params (1.00 depth × 1.25 width)

## Important Notes

### Base Channels
Model Designer must store **base channel values** (for scale 'l' or 'm'), not scaled values.

Example:
- ✅ Store: `[256]` (base channel)
- ❌ Don't store: `[64]` (already scaled for 'n')

### Custom Modules
Custom modules like `SparseGlobalBlockGated` must support channel scaling:

```python
class SparseGlobalBlockGated(nn.Module):
    def __init__(self, c1, c2, ...):  # ✅ Accepts c1, c2
        # Ultralytics will pass scaled channels
```

If module hardcodes channels, scaling won't work:
```python
class BadModule(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d(256, 512, ...)  # ❌ Hardcoded
```

## Files Modified

1. `backend/app/services/model_storage.py` - Added `depth_multiple`/`width_multiple` to YAML output
2. `backend/app/utils/yaml_utils.py` - Added scale suffix to temp filename
3. `backend/app/controllers/train_controller.py` - Pass `model_scale` to trainer
4. `backend/app/services/ultra_trainer.py` - Use `model_scale` in `prepare_model_yaml`

## References

- [Ultralytics YOLOv8 Scaling](https://docs.ultralytics.com/models/yolov8/#supported-modes)
- [YAML Configuration Guide](https://docs.ultralytics.com/usage/cfg/)
