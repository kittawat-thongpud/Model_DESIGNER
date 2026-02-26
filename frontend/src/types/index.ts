/* ═══════════════════════════════════════════════════════════════════════════
   TypeScript interfaces — Ultralytics-native Model Designer
   ═══════════════════════════════════════════════════════════════════════════ */

// ─── Navigation ──────────────────────────────────────────────────────────────

export type PageName =
  | 'dashboard'
  | 'model-designer'
  | 'module-designer'
  | 'train-designer'
  | 'jobs'
  | 'job-detail'
  | 'weights'
  | 'weight-detail'
  | 'weight-editor'
  | 'datasets'
  | 'dataset-detail';

// ─── Module Designer (custom nn.Module blocks) ──────────────────────────────

export interface ModuleArg {
  name: string;
  type: string;
  default?: unknown;
  description?: string;
}

export interface ModuleRecord {
  module_id: string;
  name: string;
  code: string;
  args: ModuleArg[];
  category: string;
  description: string;
  created_at: string;
  updated_at: string;
}

export interface ModuleSummary {
  module_id: string;
  name: string;
  category: string;
  description: string;
  arg_count: number;
  created_at: string;
}

export interface CatalogModule {
  name: string;
  category: string;
  description: string;
  args: ModuleArg[];
  source: string;
}

// ─── Model Designer (Ultralytics YAML) ──────────────────────────────────────

export interface LayerDef {
  from: number | number[];
  repeats: number;
  module: string;
  args: unknown[];
}

export interface ModelYAML {
  nc: number;
  scales?: Record<string, number[]>;
  kpt_shape?: number[];
  backbone: LayerDef[];
  head: LayerDef[];
}

export interface ModelRecord {
  model_id: string;
  name: string;
  description: string;
  task: string;
  yaml_def: ModelYAML;
  created_at: string;
  updated_at: string;
}

export interface ModelSummary {
  model_id: string;
  name: string;
  task: string;
  layer_count: number;
  input_shape?: number[];
  params?: number;
  gradients?: number;
  flops?: number;
  created_at: string;
  updated_at: string;
}

export interface ExportRequest {
  model_id: string;
  format: string;
  weight_id?: string | null;
  imgsz?: number;
  scale?: string | null;
}

// ─── Training / Jobs (Ultralytics model.train()) ────────────────────────────

export interface TrainConfig {
  data: string;
  imgsz: number;
  batch: number;
  workers: number;
  epochs: number;
  patience: number;
  device: string;
  seed: number;
  deterministic: boolean;
  amp: boolean;
  close_mosaic: number;
  optimizer: string;
  lr0: number;
  lrf: number;
  momentum: number;
  weight_decay: number;
  warmup_epochs: number;
  warmup_momentum: number;
  warmup_bias_lr: number;
  cos_lr: boolean;
  pretrained: string;
  yolo_model: string;
  use_yolo_pretrained: boolean;
  freeze: number | number[];
  resume: boolean;
  box: number;
  cls: number;
  dfl: number;
  pose: number;
  nbs: number;
  conf: number | null;
  iou: number;
  max_det: number;
  agnostic_nms: boolean;
  rect: boolean;
  single_cls: boolean;
  hsv_h: number;
  hsv_s: number;
  hsv_v: number;
  degrees: number;
  translate: number;
  scale: number;
  shear: number;
  perspective: number;
  flipud: number;
  fliplr: number;
  bgr: number;
  mosaic: number;
  mixup: number;
  copy_paste: number;
  erasing: number;
  crop_fraction: number;
  auto_augment: string;
  save_period: number;
  val: boolean;
  plots: boolean;
  overlap_mask: boolean;
  mask_ratio: number;
  kobj: number;
  sample_per_class: number;
  [key: string]: unknown;
}

export interface EpochMetrics {
  epoch: number;
  box_loss: number;
  cls_loss: number;
  dfl_loss: number;
  precision?: number | null;
  recall?: number | null;
  mAP50?: number | null;
  mAP50_95?: number | null;
  fitness?: number | null;
  lr: number;
  epoch_time: number;
  gpu_memory_mb?: number | null;
  [key: string]: unknown;
}

export interface JobRecord {
  job_id: string;
  model_id: string;
  model_name: string;
  model_scale?: string;
  task?: string;
  status: string;
  epoch: number;
  total_epochs: number;
  message: string;
  config: Record<string, unknown>;
  partitions?: Array<{partition_id: string; train: boolean; val: boolean; test: boolean}>;
  history: EpochMetrics[];
  weight_id: string | null;
  // Ultralytics-native metrics
  best_fitness?: number | null;
  best_mAP50?: number | null;
  best_mAP50_95?: number | null;
  // Legacy / generic metrics (kept for backward compat with existing jobs)
  train_loss?: number;
  train_accuracy?: number;
  val_loss?: number | null;
  val_accuracy?: number | null;
  best_val_loss?: number | null;
  best_val_accuracy?: number | null;
  confusion_matrix?: number[][] | null;
  class_names?: string[];
  per_class_metrics?: Record<string, unknown>[] | null;
  device?: string | null;
  total_time?: number | null;
  model_params?: number | null;
  model_flops?: number | null;
  trainable_params?: number | null;
  inference_time_ms?: number | null;
  created_at: string;
  updated_at?: string;
  started_at?: string | null;
  completed_at?: string | null;
}

export interface JobSummary {
  job_id: string;
  model_id: string;
  model_name: string;
  task: string;
  status: string;
  epoch: number;
  total_epochs: number;
  best_fitness: number | null;
  created_at: string;
  completed_at: string | null;
}

export interface PartitionSplitConfig {
  partition_id: string;
  train: boolean;
  val: boolean;
  test: boolean;
}

export interface TrainRequest {
  model_id: string;
  model_scale?: string;  // Model scale (n, s, m, l, x)
  config: Partial<TrainConfig>;
  partitions?: PartitionSplitConfig[];  // Partition split configuration
}

// ─── Datasets ────────────────────────────────────────────────────────────────

export interface DatasetInfo {
  name: string;
  display_name: string;
  input_shape: number[];
  num_classes: number;
  train_size: number;
  test_size: number;
  val_size?: number;
  classes?: string[];
  class_names?: string[];
  task_type: string;
  available?: boolean;
  available_partitions?: string[];  // Partitions that exist after split: ['train', 'val', 'test']
}

export interface DatasetSplitMeta {
  total: number;
  labeled: number;
  effective?: number;
}

export interface DatasetMeta {
  name: string;
  available: boolean;
  disk_size_bytes: number;
  disk_size_human: string;
  splits: Record<string, DatasetSplitMeta>;
  split_config?: SplitTransferConfig;
  scanned_at: string;
}

export interface SplitTransferConfig {
  seed: number;
  train_to_val: number;
  train_to_test: number;
  test_to_train: number;
  test_to_val: number;
  val_to_train: number;
  val_to_test: number;
}

export interface SplitResponse extends SplitTransferConfig {
  orig_train: number;
  orig_test: number;
  orig_val: number;
  train_count: number;
  val_count: number;
  test_count: number;
}

// ─── Dataset Partitions ──────────────────────────────────────────────────────

export interface PartitionEntry {
  id: string;
  name: string;
  percent: number;
  train_count: number;
  val_count: number;
  test_count: number;
  class_counts?: {
    train: Record<string, number>;
    val: Record<string, number>;
    test: Record<string, number>;
  };
}

export interface PartitionSummary {
  seed: number;
  method: string;
  available_methods: string[];
  master: PartitionEntry;
  partitions: PartitionEntry[];
  total_train: number;
  total_val: number;
  total_test: number;
}

// ─── Dataset Samples ────────────────────────────────────────────────────────

export interface Annotation {
  bbox: number[];       // [x, y, w, h] in pixels (COCO format)
  category_id: number;
  category_name: string;
  area: number;
}

export interface SampleData {
  index: number;
  label: number;
  class_name: string;
  image_base64: string;
  mime: string;
  orig_w: number;
  orig_h: number;
  thumb_bytes: number;
  annotations?: Annotation[];
}

export interface DatasetSamplesResponse {
  dataset: string;
  split: string;
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
  class_idx: number | null;
  samples: SampleData[];
  avg_thumb_bytes: number;
  task_type: string;
}

// ─── Weights (weight_schema.py) ─────────────────────────────────────────────

export interface TrainingRun {
  run: number;
  job_id: string | null;
  weight_id: string;
  dataset: string;
  epochs: number;
  accuracy: number | null;
  loss: number | null;
  total_time: number | null;
  device: string | null;
  created_at: string;
}

export interface WeightRecord {
  weight_id: string;
  model_id: string;
  model_name: string;
  job_id: string | null;
  dataset: string;
  epochs_trained: number;
  total_epochs?: number;
  final_accuracy: number | null;
  final_loss: number | null;
  file_size_bytes: number;
  parent_weight_id: string | null;
  training_runs?: TrainingRun[];
  total_time?: number | null;
  device?: string | null;
  created_at: string;
}

// ─── Weight Transfer ────────────────────────────────────────────────────────

export interface WeightSourceInfo {
  name: string;
  display_name: string;
  file_extensions: string[];
  has_pretrained?: boolean;
}

export interface PretrainedModelInfo {
  model_key: string;
  display_name: string;
  description: string;
  param_count: number | null;
  task: string;
  source_plugin: string;
  downloaded: boolean;
}

export interface WeightGroupKey {
  key: string;
  shape: number[];
  dtype: string;
}

export interface WeightGroup {
  prefix: string;
  module_type: string;
  param_count: number;
  keys: WeightGroupKey[];
}

export interface MappingKey {
  src_key: string | null;
  tgt_key: string | null;
  src_shape: number[] | null;
  tgt_shape: number[] | null;
  matched: boolean;
}

export interface MappingEntry {
  src_prefix: string | null;
  tgt_prefix: string | null;
  status: 'matched' | 'shape_mismatch' | 'unmatched';
  keys: MappingKey[];
}

export interface MappingPreview {
  mapping: MappingEntry[];
  matched_groups: number;
  total_target_groups: number;
  match_ratio: number;
}

export interface ImportWeightResult {
  weight_id: string;
  source_plugin: string;
  source_display_name: string;
  key_count: number;
  groups: WeightGroup[];
  original_filename: string;
}

export interface ApplyMapResult {
  applied: number;
  skipped: number;
  total_target: number;
  applied_keys: string[];
  applied_node_ids: string[];
  freeze_node_ids: string[];
}

// ─── Compatibility Check ────────────────────────────────────────────────────

export interface CompatKeyDetail {
  suffix: string;
  src_key: string | null;
  tgt_key: string | null;
  src_shape: number[] | null;
  tgt_shape: number[] | null;
  src_dtype: string | null;
  tgt_dtype: string | null;
  src_numel?: number;
  tgt_numel?: number;
  shape_match: boolean;
  dtype_match: boolean;
  status: 'ok' | 'error' | 'warning' | 'extra_source' | 'missing_source';
  message?: string;
}

export interface CompatIssue {
  suffix: string;
  severity: 'error' | 'warning' | 'info';
  message: string;
}

export interface CompatCheckResult {
  overall: 'compatible' | 'partial' | 'incompatible';
  summary: string;
  ok_count: number;
  error_count: number;
  warning_count: number;
  info_count: number;
  total_keys: number;
  issues: CompatIssue[];
  keys: CompatKeyDetail[];
}

// ─── Layer Detail ───────────────────────────────────────────────────────────

export interface TensorMapSlice {
  label: string;
  values: number[][];
  rows: number;
  cols: number;
}

export interface TensorMap {
  ndim: number;
  shape: number[];
  description: string;
  slices: TensorMapSlice[];
}

export interface LayerDetailResult {
  key: string;
  weight_id: string;
  stats: {
    shape: number[];
    dtype: string;
    numel: number;
    min: number;
    max: number;
    mean: number;
    std: number;
    median: number;
    zeros_pct: number;
    near_zero_pct: number;
  };
  histogram: {
    counts: number[];
    bin_edges: number[];
    bin_min: number;
    bin_max: number;
  };
  tensor_map: TensorMap;
}

// ─── Logs ───────────────────────────────────────────────────────────────────

export interface LogEntry {
  timestamp: string;
  level: string;
  category: string;
  message: string;
  data?: Record<string, unknown>;
}

// ─── Stats ──────────────────────────────────────────────────────────────────

export interface DashboardStats {
  total_models: number;
  total_modules: number;
  total_jobs: number;
  active_jobs: number;
  total_weights: number;
}
