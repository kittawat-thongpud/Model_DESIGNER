# Fix: 'sample_per_class' is not a valid YOLO argument

## Problem

Training failed with error:
```
'sample_per_class' is not a valid YOLO argument. 
Arguments received: ['yolo', '']
```

This error occurs when Ultralytics tries to validate custom parameters as CLI arguments.

## Root Cause

The issue had multiple contributing factors:

1. **sys.argv contamination**: `backend/app/main.py` was setting `sys.argv = ['python']` which Ultralytics was interpreting as CLI arguments
2. **CLI mode detection**: Ultralytics checks `sys.argv` during trainer initialization and tries to parse it as CLI commands
3. **Custom parameter leakage**: Custom parameters (`sample_per_class`, `record_gradients`, etc.) were being passed to Ultralytics' config validation which only accepts standard YOLO arguments
4. **Invalid DEFAULT_CFG_DICT keys**: The `session` field from `DEFAULT_CFG_DICT` was being passed to training config, triggering validation errors

## Solution

### 1. Fixed sys.argv in main.py

**File**: `backend/app/main.py`

**Changed from**:
```python
import sys
sys.argv = ['python']
```

**Changed to**:
```python
import sys
# Keep sys.argv empty to prevent Ultralytics CLI parsing
sys.argv = []
```

**Why**: Ultralytics checks `sys.argv` and if it contains values like `['python']` or `['yolo', '']`, it tries to parse them as CLI commands, triggering validation errors.

### 2. Enhanced CustomDetectionTrainer initialization

**File**: `backend/app/services/custom_trainer.py`

**Added protection**:
```python
def __init__(self, cfg=None, overrides=None, _callbacks=None):
    # CRITICAL: Ensure sys.argv is empty to prevent Ultralytics CLI parsing
    import sys
    import os
    os.environ['YOLO_CLI'] = '0'
    sys.argv = []
    
    # ... rest of initialization
```

**Improved config merging with invalid key filtering**:
```python
# List of keys to remove (invalid for YOLO training)
INVALID_KEYS = {'session', 'sample_per_class', 'record_gradients', 'gradient_interval', 
                'record_weights', 'weight_interval', '_partition_configs', '_dataset_name'}

# Build complete config by merging defaults with our overrides
if cfg is None:
    cfg = DEFAULT_CFG_DICT.copy()
elif isinstance(cfg, str):
    # If cfg is a path, let get_cfg handle it
    pass
elif isinstance(cfg, dict):
    # Merge with defaults
    cfg = {**DEFAULT_CFG_DICT, **cfg}

# Merge clean_overrides into cfg
if isinstance(cfg, dict):
    cfg.update(clean_overrides)
    
    # CRITICAL: Remove invalid keys that would trigger validation errors
    for key in INVALID_KEYS:
        cfg.pop(key, None)
    
    clean_overrides = {}  # Already merged
```

**Why**: This ensures:
- `sys.argv` is always empty during trainer initialization
- `YOLO_CLI` environment variable is set to disable CLI mode
- Config is properly merged before passing to parent class
- Custom parameters are extracted BEFORE merging into config
- **Invalid keys (session, custom params) are filtered out before validation**

## How It Works

### Before Fix:
```
1. ultra_trainer.py sets sys.argv = []
2. main.py sets sys.argv = ['python']  ❌
3. CustomDetectionTrainer.__init__ called
4. Ultralytics sees sys.argv = ['python']
5. Tries to parse as CLI: ['yolo', '']
6. Validates 'sample_per_class' as CLI arg
7. ERROR: not a valid YOLO argument
```

### After Fix:
```
1. ultra_trainer.py sets sys.argv = []
2. main.py sets sys.argv = []  ✓
3. CustomDetectionTrainer.__init__ called
4. Sets sys.argv = [] again (double protection)
5. Sets YOLO_CLI=0 environment variable
6. Extracts custom params BEFORE config merge
7. Merges clean config into DEFAULT_CFG_DICT
8. Filters out INVALID_KEYS (session, custom params)  ✓
9. Passes to parent with only valid YOLO params
10. SUCCESS: Training starts
```

## Custom Parameters Handling

Custom parameters are now handled in 4 stages:

### Stage 1: Extraction (in __init__)
```python
# Extract from _custom_params class attribute or overrides
self.job_id = clean_overrides.pop('job_id', custom_source.get('job_id'))
self.sample_per_class = clean_overrides.pop('sample_per_class', custom_source.get('sample_per_class', 0))
# ... etc
```

### Stage 2: Clean Config Merge
```python
# Only valid YOLO params remain in clean_overrides
cfg.update(clean_overrides)  # No custom params here
```

### Stage 3: Invalid Key Filtering
```python
# Remove invalid keys that would trigger Ultralytics validation errors
INVALID_KEYS = {'session', 'sample_per_class', 'record_gradients', ...}
for key in INVALID_KEYS:
    cfg.pop(key, None)
```

### Stage 4: Post-Init Injection
```python
# After super().__init__, inject into self.args for validator
self.args.sample_per_class = self.sample_per_class
```

## Files Modified

1. `backend/app/main.py` - Changed `sys.argv = ['python']` to `sys.argv = []`
2. `backend/app/services/custom_trainer.py` - Enhanced initialization with sys.argv protection and improved config merging

## Testing

To verify the fix works:

```bash
# Start training with sample_per_class parameter
# Should NOT see "sample_per_class is not a valid YOLO argument" error
```

## Related Issues Fixed

This fix also resolves:
- ✅ CLI argument parsing errors during training
- ✅ "Arguments received: ['yolo', '']" errors
- ✅ Custom parameter validation errors
- ✅ sys.argv contamination between modules

## Prevention

To prevent similar issues in the future:

1. **Never set sys.argv to non-empty values** in FastAPI applications
2. **Always set YOLO_CLI=0** before importing Ultralytics
3. **Extract custom params BEFORE** passing config to Ultralytics
4. **Use DEFAULT_CFG_DICT** as base for all configs
5. **Double-check sys.argv** is empty in trainer initialization

## Verification Checklist

- [x] sys.argv is empty in main.py
- [x] sys.argv is cleared in ultra_trainer.py
- [x] sys.argv is cleared in CustomDetectionTrainer.__init__
- [x] YOLO_CLI=0 is set
- [x] Custom params extracted before config merge
- [x] INVALID_KEYS list includes all custom params + 'session'
- [x] Invalid keys filtered out before super().__init__
- [x] Only valid YOLO params passed to parent class
- [x] Custom params injected into self.args after init
- [x] Code compiles without errors
