# Ultralytics CLI Mode Prevention

## Problem

When calling `model.train()` from Python, Ultralytics was entering CLI mode and trying to parse `sys.argv` as command-line arguments, causing errors like:

```
'sample_per_class' is not a valid YOLO argument.
Arguments received: ['yolo', ''].
```

This happened even though the code was calling Ultralytics from Python, not from CLI.

## Root Cause

Ultralytics has multiple entry points that check `sys.argv`:

1. **Import-time checks**: When importing `ultralytics`, it checks if it's being run as a CLI tool
2. **cfg.entrypoint()**: The `ultralytics.cfg.entrypoint()` function parses `sys.argv` and validates arguments
3. **BaseTrainer initialization**: Calls `get_cfg()` which may trigger CLI parsing
4. **model.train() call**: Internally checks `sys.argv` to determine if it's CLI or Python mode

Even with `sys.argv = []`, Ultralytics can still detect CLI patterns and try to parse arguments.

## Solution - Multi-Layer Protection

We implemented a **4-layer defense** to prevent CLI mode:

### Layer 1: Environment Variable (Global)
```python
# At module import time
os.environ['YOLO_CLI'] = '0'
```

**Files**: 
- `backend/app/main.py`
- `backend/app/services/ultra_trainer.py`
- `backend/app/services/custom_trainer.py`

### Layer 2: sys.argv Clearing (Multiple Points)
```python
# Clear sys.argv at critical points
sys.argv = []
```

**Locations**:
1. `backend/app/main.py` - On server startup
2. `backend/app/services/ultra_trainer.py` - At module import
3. `backend/app/services/ultra_trainer.py` - Before `model.train()`
4. `backend/app/services/custom_trainer.py` - In `__init__()`

### Layer 3: Invalid Key Filtering (Config Cleaning)
```python
# Remove invalid keys that trigger CLI validation
INVALID_KEYS = {'session', 'sample_per_class', 'record_gradients', 
                'gradient_interval', 'record_weights', 'weight_interval', 
                '_partition_configs', '_dataset_name'}

for key in INVALID_KEYS:
    cfg.pop(key, None)
```

**File**: `backend/app/services/custom_trainer.py`

### Layer 4: Entrypoint Monkey-Patching (Runtime)
```python
# Patch Ultralytics' entrypoint to prevent CLI parsing
from ultralytics import cfg
original_entrypoint = getattr(cfg, 'entrypoint', None)
cfg.entrypoint = lambda: None  # Replace with no-op

# ... run training ...

# Restore original
cfg.entrypoint = original_entrypoint
```

**File**: `backend/app/services/ultra_trainer.py`

## How It Works

### Complete Flow

```
1. Server starts → main.py sets sys.argv = [], YOLO_CLI=0
2. ultra_trainer.py imported → sets sys.argv = [] again
3. Training job starts → _training_worker() called
4. Before model.train():
   a. sys.argv = []
   b. YOLO_CLI=0
   c. Patch cfg.entrypoint to no-op
5. CustomDetectionTrainer.__init__:
   a. sys.argv = []
   b. YOLO_CLI=0
   c. Extract custom params
   d. Filter INVALID_KEYS from config
   e. Pass clean config to parent
6. model.train() executes → NO CLI mode detected
7. After training → restore cfg.entrypoint
```

## Why Each Layer Is Needed

### Why sys.argv = [] isn't enough?
Ultralytics checks `sys.argv` at multiple points during execution. Setting it once isn't enough because:
- Other libraries might modify it
- Ultralytics might cache the value
- Different modules might check it at different times

### Why YOLO_CLI=0 isn't enough?
The environment variable is checked at import time, but some code paths still check `sys.argv` directly.

### Why filtering INVALID_KEYS is needed?
Even if CLI mode is disabled, Ultralytics validates all config keys against a whitelist. Custom parameters and internal fields like `session` must be removed.

### Why monkey-patching entrypoint is needed?
The `cfg.entrypoint()` function is the main CLI entry point. Even if other protections work, this function can still be called internally and try to parse `sys.argv`. Replacing it with a no-op ensures it never runs.

## Files Modified

1. **backend/app/main.py**
   - Set `sys.argv = []` (changed from `['python']`)
   - Set `YOLO_CLI=0`

2. **backend/app/services/ultra_trainer.py**
   - Set `sys.argv = []` at module import
   - Set `sys.argv = []` before `model.train()`
   - Monkey-patch `cfg.entrypoint` during training
   - Restore `cfg.entrypoint` after training

3. **backend/app/services/custom_trainer.py**
   - Set `sys.argv = []` in `__init__()`
   - Set `YOLO_CLI=0` in `__init__()`
   - Filter `INVALID_KEYS` from config
   - Extract custom params before parent init

## Testing

To verify the fix works:

```bash
# Start a training job with custom parameters
# Should NOT see any CLI-related errors
# Should NOT see "Arguments received: ['yolo', '']"
```

## Prevention Checklist

Before deploying changes that interact with Ultralytics:

- [ ] Ensure `sys.argv = []` at entry points
- [ ] Set `YOLO_CLI=0` environment variable
- [ ] Filter custom params from config before passing to Ultralytics
- [ ] Don't pass unknown keys to `model.train()`
- [ ] Test with custom parameters like `sample_per_class`
- [ ] Check logs for CLI-related warnings

## Common Pitfalls

### ❌ Don't do this:
```python
# Passing custom params directly to train()
model.train(
    data="coco8.yaml",
    epochs=10,
    sample_per_class=5  # ❌ Will trigger validation error
)
```

### ✅ Do this instead:
```python
# Use custom trainer with params in _custom_params
class MyTrainer(CustomDetectionTrainer):
    _custom_params = {'sample_per_class': 5}

model.train(
    data="coco8.yaml",
    epochs=10,
    trainer=MyTrainer  # ✅ Custom params handled internally
)
```

## Related Issues

This fix resolves:
- ✅ `'sample_per_class' is not a valid YOLO argument`
- ✅ `'session' is not a valid YOLO argument`
- ✅ `Arguments received: ['yolo', '']`
- ✅ CLI mode detection in Python code
- ✅ Config validation errors for custom parameters

## References

- Ultralytics CLI detection: `ultralytics/cfg.py::entrypoint()`
- Config validation: `ultralytics/cfg.py::check_dict_alignment()`
- Trainer initialization: `ultralytics/engine/trainer.py::BaseTrainer.__init__()`
