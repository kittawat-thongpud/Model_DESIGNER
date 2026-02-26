# Migration Plan: Old System → Ultralytics-Native Architecture

## Current vs Target Architecture

### OLD (PyTorch-centric)
```
Node Graph (Conv2d, ReLU, Linear, etc.)
  → codegen.py → nn.Module Python class
  → trainer.py → custom training loop (manual loss, optimizer, dataloader)
  → exporter.py → ONNX/script export
```

### NEW (Ultralytics-native)
```
Module Designer → custom nn.Module blocks → register into Ultralytics namespace
Model Designer  → YAML [from, repeats, module, args] → backbone + head sections
Train Designer  → model.train(**args) → Ultralytics handles everything
                  ↳ SSE monitoring via callbacks (on_train_epoch_end, etc.)
```

---

## File-by-File Audit: REMOVE / KEEP / REBUILD

### Backend: `plugins/` — REMOVE ENTIRELY
| File | Verdict | Reason |
|------|---------|--------|
| `plugins/base.py` | **REMOVE** | Old NodePlugin ABC, codegen_init, shape rules — not Ultralytics |
| `plugins/loader.py` | **REMOVE** | Old plugin discovery for PyTorch nodes |
| `plugins/__init__.py` | **REMOVE** | Old plugin exports |
| `plugins/nodes/*.py` (all 11 files) | **REMOVE** | Old PyTorch layer nodes (Conv2d, ReLU, Linear, etc.) + old YOLO wrappers |
| `plugins/losses/*.py` | **REMOVE** | Old custom loss plugins — Ultralytics has built-in losses |
| `plugins/tasks/*.py` | **REMOVE** | Old classification/detection task plugins — Ultralytics handles tasks |
| `plugins/datasets/*.py` | **KEEP** | Dataset registry is still useful (COCO, MNIST, etc.) — but may simplify |
| `plugins/training/*.py` | **EVALUATE** | Analysis plugins (metrics, visualization) — some monitoring logic reusable |
| `plugins/weight_sources/*.py` | **KEEP** | Ultralytics pretrained weight download — still needed |

### Backend: `registry/` — REMOVE & REBUILD
| File | Verdict | Reason |
|------|---------|--------|
| `registry/node_registry.py` | **REMOVE** | Old NodeCategory, NodeDef, param system — replaced by Ultralytics module catalog |
| `registry/training_registry.py` | **REMOVE** | Old training block registry — Ultralytics train() replaces this |

### Backend: `schemas/` — REMOVE & REBUILD
| File | Verdict | Reason |
|------|---------|--------|
| `schemas/core.py` | **REBUILD** | Old NodeSchema/EdgeSchema/ModelGraph → new YAML-layer schema |
| `schemas/build_schema.py` | **REMOVE** | Old build system (compiled PyTorch code) — not needed |
| `schemas/model_schema.py` | **REBUILD** | Model metadata — keep concept, change structure |
| `schemas/pipeline.py` | **REBUILD** | ExportRequest still useful, simplify |
| `schemas/training.py` | **REBUILD** | Training config → Ultralytics train() args |
| `schemas/training_pipeline.py` | **REMOVE** | Old training pipeline graph — replaced by simple config |
| `schemas/packages.py` | **REMOVE** | Old package system — not needed |
| `schemas/dataset_schema.py` | **KEEP** | Dataset schemas still useful |
| `schemas/job_schema.py` | **KEEP** | Job tracking still needed |
| `schemas/weight_schema.py` | **KEEP** | Weight management still needed |
| `schemas/inference.py` | **KEEP** | Inference schemas still useful |

### Backend: `services/` — REMOVE & REBUILD
| File | Verdict | Reason |
|------|---------|--------|
| `services/codegen.py` | **REMOVE** | Old PyTorch nn.Module code generation — not needed |
| `services/yaml_codegen.py` | **REMOVE** | Old graph→YAML conversion — will rebuild for new format |
| `services/training_codegen.py` | **REMOVE** | Old training script generation — Ultralytics handles training |
| `services/model_builder.py` | **REMOVE** | Old PyTorch model builder |
| `services/build_storage.py` | **REMOVE** | Old build artifact storage |
| `services/trainer.py` | **REMOVE** | Old custom training loop (82KB!) — replaced by ultra_trainer |
| `services/trainer_data.py` | **REMOVE** | Old custom dataloader logic — Ultralytics handles data |
| `services/ultra_trainer.py` | **REBUILD** | Keep concept, simplify to pure Ultralytics model.train() + callbacks |
| `services/pipeline_executor.py` | **REMOVE** | Old training pipeline executor |
| `services/model_presets.py` | **KEEP** | Model presets still useful |
| `services/exporter.py` | **REBUILD** | Keep Ultralytics export, remove old PyTorch export |
| `services/package_service.py` | **REMOVE** | Old package system |
| `services/validator.py` | **REMOVE** | Old graph validator — will rebuild for YAML validation |
| `services/inference.py` | **REBUILD** | Simplify to Ultralytics model.predict() |
| `services/event_bus.py` | **KEEP** | SSE pub/sub — core infrastructure |
| `services/job_storage.py` | **KEEP** | Job persistence — core infrastructure |
| `services/base_storage.py` | **KEEP** | Base storage — core infrastructure |
| `services/weight_storage.py` | **KEEP** | Weight management — still needed |
| `services/weight_transfer.py` | **KEEP** | Weight transfer — still needed |
| `services/weight_import.py` | **KEEP** | Weight import — still needed |
| `services/weight_snapshots.py` | **KEEP** | Weight snapshots — still needed |
| `services/gradient_snapshots.py` | **REMOVE** | Old gradient tracking — Ultralytics has its own |
| `services/classification_samples.py` | **REMOVE** | Old sample visualization — rebuild via callbacks |
| `services/detection_utils.py` | **REMOVE** | Old detection utilities |
| `services/dataset_registry.py` | **KEEP** | Dataset registry — still useful |
| `services/dataset_yaml.py` | **KEEP** | Ultralytics needs data.yaml — essential |
| `services/analysis_runner.py` | **EVALUATE** | Analysis hooks — some reusable for monitoring |
| `services/analysis_storage.py` | **EVALUATE** | Analysis persistence — reusable |

### Backend: `controllers/` — REMOVE & REBUILD
| File | Verdict | Reason |
|------|---------|--------|
| `controllers/model_controller.py` | **REBUILD** | Model CRUD — keep concept, change to YAML-native |
| `controllers/build_controller.py` | **REMOVE** | Old build system |
| `controllers/registry_controller.py` | **REMOVE** | Old node registry API — replaced by module catalog |
| `controllers/package_controller.py` | **REMOVE** | Old package system |
| `controllers/train_controller.py` | **REBUILD** | Training API — simplify to Ultralytics train() |
| `controllers/training_pipeline_controller.py` | **REMOVE** | Old training pipeline |
| `controllers/export_controller.py` | **REBUILD** | Keep, simplify |
| `controllers/job_controller.py` | **KEEP** | Job listing — still needed |
| `controllers/stream_controller.py` | **KEEP** | SSE streaming — core infrastructure |
| `controllers/dataset_controller.py` | **KEEP** | Dataset management — still needed |
| `controllers/dataset_samples_controller.py` | **KEEP** | Dataset preview — still needed |
| `controllers/weight_controller.py` | **KEEP** | Weight management — still needed |
| `controllers/weight_snapshot_controller.py` | **KEEP** | Weight snapshots — still needed |
| `controllers/gradient_snapshot_controller.py` | **REMOVE** | Old gradient system |
| `controllers/classification_sample_controller.py` | **REMOVE** | Old sample system |
| `controllers/analysis_controller.py` | **EVALUATE** | Analysis API — some reusable |
| `controllers/log_controller.py` | **KEEP** | System logs — core infrastructure |
| `controllers/stats_controller.py` | **KEEP** | Dashboard stats — still needed |

### Backend: Other
| File | Verdict | Reason |
|------|---------|--------|
| `main.py` | **REBUILD** | Update router imports, remove old plugin discovery |
| `config.py` | **KEEP** | Core config — add MODULES_DIR |
| `constants.py` | **KEEP** | Constants |
| `storage.py` | **KEEP** | Base storage |
| `logging_service.py` | **KEEP** | Logging |
| `modules/` | **KEEP** | Check what's in here |
| `utils/` | **KEEP** | Utilities |

---

## New Architecture: Three Designers

### 1. Module Designer (Custom Blocks)
**Purpose**: Create custom `nn.Module` blocks that can be used in Model YAML

**Backend**:
- `services/module_storage.py` — CRUD for custom module definitions (Python code + metadata)
- `services/module_registry.py` — catalog of all available modules (built-in Ultralytics + custom)
- `controllers/module_controller.py` — API endpoints

**Schema** (per custom module):
```json
{
  "id": "CustomBlock",
  "name": "CustomBlock",
  "code": "class CustomBlock(nn.Module):\n  def __init__(self, c1, c2):\n    ...",
  "args": [{"name": "c1", "type": "int"}, {"name": "c2", "type": "int"}],
  "category": "custom",
  "description": "Custom Conv-BN-ReLU block"
}
```

**Frontend**: Code editor (Monaco) + arg definition form + preview

### 2. Model Designer (YAML Architecture)
**Purpose**: Visual builder for Ultralytics model YAML

**Backend**:
- `services/model_storage.py` — CRUD for model YAML definitions
- `services/yaml_builder.py` — graph ↔ YAML conversion
- `controllers/model_controller.py` — API endpoints

**Schema** (model = YAML):
```yaml
nc: 80
scales:
  n: [0.33, 0.25, 1024]
backbone:
  - [-1, 1, Conv, [64, 3, 2]]     # layer 0
  - [-1, 1, Conv, [128, 3, 2]]    # layer 1
  - [-1, 3, C2f, [128, True]]     # layer 2
head:
  - [-1, 1, nn.Upsample, [None, 2, nearest]]
  - [[-1, 2], 1, Concat, [1]]
  - [[4, 6], 1, Detect, [nc]]
```

**Visual representation**: Each row = one node on canvas
- Node shows: `[from] → Module(args) ×repeats`
- Edges derived from `from` field (auto-drawn)
- Two sections: backbone (blue) and head (green)
- Sidebar: module catalog (built-in + custom)

### 3. Train Designer (Training Config + Monitoring)
**Purpose**: Configure Ultralytics `model.train()` and monitor in real-time

**What Ultralytics handles automatically**:
- Loss computation (box, cls, dfl)
- Optimizer (SGD, Adam, AdamW)
- LR scheduling (cosine, linear)
- Data augmentation (mosaic, mixup, copy_paste, etc.)
- EMA, AMP, gradient clipping
- Checkpointing (best.pt, last.pt)
- Validation after each epoch

**What we configure** (= `model.train()` kwargs):
- `data`: dataset YAML path
- `epochs`, `batch`, `imgsz`
- `lr0`, `lrf`, `momentum`, `weight_decay`
- `warmup_epochs`, `warmup_momentum`
- `box`, `cls`, `dfl` (loss weights)
- `optimizer` (SGD/Adam/AdamW/auto)
- `freeze` (layer freezing)
- Augmentation params (hsv_h/s/v, degrees, translate, scale, mosaic, mixup, etc.)
- `device`, `amp`, `seed`
- `patience` (early stopping)

**What we monitor** (via Ultralytics callbacks):
- `on_train_epoch_end` → epoch metrics (loss, lr)
- `on_fit_epoch_end` → validation metrics (mAP, precision, recall, F1)
- `on_train_end` → final results, best weights
- Progress: epoch/total, ETA
- Live charts: loss curves, mAP curves, LR schedule

**Frontend**: Config form (grouped sections) + live monitoring dashboard

---

## Migration Phases

### Phase 1: Clean old backend
Delete all files marked REMOVE. Update main.py imports.

### Phase 2: New schemas
- `schemas/module.py` — CustomModule definition
- `schemas/model.py` — YAML model (layers, params, scales)
- `schemas/train.py` — Training config (all model.train() args)

### Phase 3: Module Designer backend
- Module CRUD + built-in Ultralytics module catalog

### Phase 4: Model Designer backend  
- Model CRUD + YAML ↔ visual graph conversion
- Validation (channel compatibility, from references)

### Phase 5: Train Designer backend
- Ultralytics model.train() wrapper with SSE callbacks
- Job management (start/stop/monitor)

### Phase 6: Clean old frontend

### Phase 7-9: Frontend for each designer
