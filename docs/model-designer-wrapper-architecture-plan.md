# Model Designer Wrapper Architecture Plan

## Summary

Model Designer should be an Ultralytics-first wrapper platform. Ultralytics remains the default model, training, inference, and metrics foundation where possible. Anything outside the built-in path should enter the app through explicit plugins and adapters: model architecture plugins, dataset conversion plugins, trainer wrappers, installer actions, diagnostics, logs, metrics, jobs, and weights.

This plan captures the first architecture milestone: build a Plugin + TrainingPlan foundation without breaking existing APIs, old jobs, saved weights, or current frontend flows.

## Goals

- Keep Ultralytics as the primary execution backend for YAML-compatible models.
- Treat the app as the control plane for jobs, datasets, weights, logs, metrics, diagnostics, and plugin state.
- Make plugins responsible for declaring capabilities, install status, config defaults, metric groups, and trainer requirements.
- Create a `TrainingPlan` layer so training execution receives a resolved plan instead of re-deciding model, dataset, YAML patch, plugin, and trainer behavior inside one large worker function.
- Preserve current API compatibility while enabling cleaner internal contracts.

## Key Changes

### Plugin Contract

Add capability metadata to `ModelArchPlugin` while keeping existing methods compatible:

- `trainer_backend`: `ultralytics`, `custom_subprocess`, `dino`, `rtdetrv2`
- `model_format`: `ultralytics_yaml`, `external_repo`, `torch_module`
- `head_type`: `yolo`, `rtdetr`, `custom`
- `requires_install`: boolean
- `metric_groups`: examples include `sgb`, `cs2ga`, `decoder`, `dam`, `grad`
- `config_options`: plugin defaults currently provided by `get_config_options()`
- `diagnostics`: install/preflight/runtime warnings and errors

Add common installer-facing plugin methods:

- `install_status()`
- `install_actions()`
- Existing installer implementations for Mamba-YOLO, DINO, and RT-DETRv2 should be adapted into this common shape over time.

Keep `/api/plugins/archs` response compatible, but add optional fields:

- `capabilities`
- `install`
- `diagnostics`
- `config_options`

### TrainingPlan Layer

Add a training plan builder service that resolves a train request before it reaches the worker runtime.

The plan should resolve:

- model source: official YOLO, arch plugin, or saved YAML
- dataset source: dataset plugin, generated `data.yaml`, partition TXT splits
- trainer backend: Ultralytics-native, DINO wrapper, RT-DETRv2 wrapper, or custom subprocess
- plugin defaults plus user config overrides
- YAML patch strategy
- warm-start / pretrained policy
- metric capabilities
- preflight diagnostics

`/api/train/start` should keep the same request and response shape, but internally call `build_training_plan()` before submitting a job.

`ultra_trainer.py` should gradually become a worker executor that consumes a resolved plan, rather than a place where all model/plugin/dataset decisions are made.

### YAML Patch And Trainer Adapter

Move YAML mutation logic out of `ultra_trainer.py` into a dedicated YAML patch service.

The patch service should own:

- scale injection
- HSG-DETR V2c/V3 decoder args
- SGTokenBlockV2 soft-hard args
- custom module validation via `prepare_model_yaml`
- job-local YAML copy generation

Add a trainer adapter resolver:

- Ultralytics YAML-compatible models use `CustomDetectionTrainer`
- DINO and RT-DETRv2 keep their current wrappers
- plugins declare which backend they need

No behavior change is required in the first pass. The first milestone is moving decision points into testable services.

### Diagnostics And Logging

Add a structured diagnostic contract:

- `severity`: `info`, `warning`, `error`
- `code`: stable machine code such as `PLUGIN_NOT_INSTALLED`, `DATASET_MISSING_YAML`, `CUDA_OP_MISSING`
- `message`
- `source`: `plugin`, `preflight`, `trainplan`, `runtime`
- `recoverable`: boolean

Preflight should return structured diagnostics. Controllers can still convert errors to HTTP 400 and warnings to job/system logs.

Runtime logs should keep the current job log format, but include diagnostic codes in payloads where practical.

### Frontend Contract Usage

`CreateTrainJobModal` should prefer backend plugin metadata instead of local fallback hardcode.

The UI should read:

- supported scales
- labels and descriptions
- config defaults
- install status
- preflight diagnostics
- metric groups

Keep frontend fallbacks temporarily for compatibility with older backend responses.

## Public Interfaces And Types

Add internal backend schemas/dataclasses:

- `PluginCapabilities`
- `PluginInstallStatus`
- `DiagnosticMessage`
- `TrainingPlan`
- `ResolvedModelSource`
- `ResolvedDatasetSource`

Extend arch plugin API responses with optional fields:

- `capabilities?: object`
- `install?: object`
- `diagnostics?: DiagnosticMessage[]`
- `config_options?: object`

Do not remove or change:

- `/api/train/start` request body
- existing job records
- existing weight records
- existing job history / metrics layout
- existing model and dataset endpoints

## Test Plan

### Unit And Static Checks

- `py_compile` changed backend files.
- `npm run build`.
- Plugin discovery still finds legacy HSG, HSG-DETR V2/V2c/V3, V3-CS2GA, Mamba-YOLO, DINO, RT-DETR, and RT-DETRv2.
- `build_training_plan()` test cases:
  - official YOLO model
  - `arch:hsg_detr_v3` with `lean_n`
  - `arch:hsg_detr_v3_cs2ga` with `n`
  - saved custom YAML model
  - unavailable plugin installer case
- YAML patch service snapshot tests for HSG-DETR V3 config options.

### API Compatibility

- `/api/plugins/archs` still renders existing frontend.
- `/api/train/start` accepts existing request bodies.
- old jobs still load.
- resume and append keep current behavior.

### Smoke Tests

- API train smoke for:
  - official YOLO
  - HSG-DETR V3 Lean
  - HSG-DETR V3-CS2GA
- Plugin preflight failure path:
  - missing Mamba/DINO/RT-DETR dependency returns a clear diagnostic, not a stack trace.
- Dataset partition training still generates correct `data.yaml` or TXT split files.
- Metrics, logs, and weights are still saved under existing job directories.

## Milestone Order

### Milestone 1 — Foundation

1. Add plugin capability and diagnostic types.
2. Add optional capability/install fields to plugin API response.
3. Add `TrainingPlan` builder without changing training behavior.
4. Move YAML patch decisions into a dedicated service.
5. Add tests for plan building and plugin discovery.

### Milestone 2 — Training Runtime Cleanup

1. Refactor `ultra_trainer.py` into smaller executor-oriented services.
2. Keep job lifecycle, queue, subprocess runtime, retry policy, and platform policy separated.
3. Keep old job records compatible.

### Milestone 3 — Dataset Plugin Cleanup

1. Move dataset-specific import/extract/convert logic out of `dataset_controller.py`.
2. Add dataset plugin capabilities for import layout, conversion target, split strategy, and diagnostics.
3. Keep current dataset APIs compatible.

### Milestone 4 — Frontend Contract Cleanup

1. Replace hardcoded train modal fallbacks with plugin metadata.
2. Use backend metric group declarations where possible.
3. Split large pages/components into focused hooks and panels.

## Assumptions

- Milestone 1 must not break public APIs.
- Ultralytics remains the default backend for YAML-compatible models.
- Non-Ultralytics trainers are wrapped as trainer adapters, not forced into Ultralytics internals.
- Plugin installer scope for the first pass is status + actions + diagnostics, not a full installer queue manager.
- Dataset controller refactor comes after Plugin + TrainingPlan foundation.
