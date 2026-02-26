/**
 * TypeScript type definitions for Model DESIGNER.
 * Mirrors the backend Pydantic schemas.
 */

// ─── Global Config Variables ────────────────────────────────────────────────

export type GlobalVarType = 'bool' | 'float' | 'int' | 'str' | 'selector';

export interface GlobalVariable {
  id: string;
  name: string;             // e.g. "drop_rate"
  type: GlobalVarType;
  value: boolean | number | string;
  options?: string[];       // choices for 'selector' type
  description?: string;
}

// ─── Graph types ────────────────────────────────────────────────────────────

export interface Position {
  x: number;
  y: number;
}

export interface NodeParams {
  [key: string]: number | number[] | string | boolean | undefined;
}

export interface ModelNode {
  id: string;
  type: string;
  position: Position;
  params: NodeParams;
  enabledByGlobal?: string;  // global bool var name — skip layer when false
  packageId?: string;        // ID of the package if type="Package"
}

export interface ModelEdge {
  source: string;
  target: string;
  source_handle?: string;
  target_handle?: string;
}

export interface ModelMeta {
  name: string;
  version: string;
  created_at: string;
  updated_at: string;
  description: string;
}

export interface ModelGraph {
  id?: string;
  meta: ModelMeta;
  nodes: ModelNode[];
  edges: ModelEdge[];
  globals: GlobalVariable[];
}

export interface ModelSummary {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  node_count: number;
  edge_count: number;
}

export interface BuildResponse {
  model_id: string;
  code: string;
  class_name: string;
  exists?: boolean;
  build_id?: string;
  existing_build_id?: string;
  model_name?: string;
}

export interface TrainRequest {
  model_id: string;

  // Basic Training
  dataset: string;
  epochs: number;
  batch_size: number;
  imgsz?: number;
  device?: string;
  workers?: number;
  patience?: number;
  val?: boolean;
  seed?: number;
  deterministic?: boolean;

  // Optimizer
  optimizer?: 'Adam' | 'AdamW' | 'SGD';
  lr0?: number;
  lrf?: number;
  momentum?: number;
  weight_decay?: number;
  warmup_epochs?: number;
  warmup_momentum?: number;
  warmup_bias_lr?: number;
  cos_lr?: boolean;

  // Model Structure
  pretrained?: string;
  freeze?: number;
  amp?: boolean;

  // Augmentation (image only)
  hsv_h?: number;
  hsv_s?: number;
  hsv_v?: number;
  degrees?: number;
  translate?: number;
  scale?: number;
  shear?: number;
  flipud?: number;
  fliplr?: number;
  erasing?: number;
  auto_augment?: string;
  crop_fraction?: number;

  // Loss
  cls_weight?: number;

  // System
  save_period?: number;
  nbs?: number;

  // Weight Recording
  weight_record_enabled?: boolean;
  weight_record_layers?: string[];
  weight_record_frequency?: number;
}

export interface TrainStatus {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed" | "stopped";
  epoch: number;
  total_epochs: number;
  train_loss: number;
  train_accuracy: number;
  val_loss: number | null;
  val_accuracy: number | null;
  message: string;
}

export interface DatasetInfo {
  name: string;
  display_name: string;
  input_shape: number[];
  num_classes: number;
  train_size: number;
  test_size: number;
  classes: string[];
  task_type?: "classification" | "detection";
}

export interface PackageParameter {
  name: string;
  type: GlobalVarType;
  default: number | number[] | string | boolean | undefined;
  description: string;
  options?: string[];
}

export interface ModelPackage {
  id: string;
  name: string;
  description: string;
  nodes: ModelNode[];
  edges: ModelEdge[];
  globals: GlobalVariable[];
  exposed_params: PackageParameter[];
  created_at: string;
}

export interface CreatePackageRequest {
  graph: ModelGraph;
  name: string;
  description: string;
  exposed_globals: string[];
}


export interface LogEntry {
  timestamp: string;
  category: "model" | "dataset" | "training" | "system";
  level: "DEBUG" | "INFO" | "WARNING" | "ERROR";
  message: string;
  data: Record<string, unknown>;
}

// ─── Job types ──────────────────────────────────────────────────────────────

export interface EpochMetrics {
  epoch: number;

  // Losses
  train_loss: number;
  train_cls_loss: number;
  val_loss: number | null;
  val_cls_loss: number | null;

  // Accuracy
  train_accuracy: number;
  val_accuracy: number | null;

  // Classification metrics
  precision: number | null;
  recall: number | null;
  f1: number | null;

  // System
  lr: number;
  epoch_time: number;
  gpu_memory_mb: number | null;
}

export interface JobConfig {
  dataset: string;
  epochs: number;
  batch_size: number;
  imgsz: number;
  device: string;
  workers: number;
  patience: number;
  val: boolean;
  seed: number;
  deterministic: boolean;
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
  freeze: number;
  amp: boolean;
  hsv_h: number;
  hsv_s: number;
  hsv_v: number;
  degrees: number;
  translate: number;
  scale: number;
  shear: number;
  flipud: number;
  fliplr: number;
  erasing: number;
  auto_augment: string;
  crop_fraction: number;
  cls_weight: number;
  save_period: number;
  nbs: number;
  global_overrides: Record<string, unknown>;
  // Weight recording
  weight_record_enabled?: boolean;
  weight_record_layers?: string[];
  weight_record_frequency?: number;
  // Legacy alias
  learning_rate?: number;
}

export interface PerClassMetric {
  class: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface JobRecord {
  job_id: string;
  model_id: string;
  model_name: string;
  config: JobConfig;
  status: "pending" | "running" | "completed" | "failed" | "stopped";
  epoch: number;
  total_epochs: number;
  train_loss: number;
  train_accuracy: number;
  val_loss: number | null;
  val_accuracy: number | null;
  best_val_loss: number | null;
  best_val_accuracy: number | null;
  message: string;
  history: EpochMetrics[];
  weight_id: string | null;

  // Post-training analysis
  confusion_matrix: number[][] | null;
  class_names: string[];
  per_class_metrics: PerClassMetric[] | null;

  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface JobLogEntry {
  timestamp: string;
  level: string;
  message: string;
  data: Record<string, unknown>;
}

// ─── Weight types ───────────────────────────────────────────────────────────

export interface WeightRecord {
  weight_id: string;
  model_id: string;
  model_name: string;
  job_id: string | null;
  dataset: string;
  epochs_trained: number;
  final_accuracy: number | null;
  final_loss: number | null;
  file_size_bytes: number;
  created_at: string;
}

// ─── Dashboard stats ────────────────────────────────────────────────────────

export interface DashboardStats {
  total_models: number;
  total_jobs: number;
  active_jobs: number;
  total_weights: number;
}

// ─── Page navigation ────────────────────────────────────────────────────────

export type PageName = "dashboard" | "designer" | "models" | "jobs" | "weights" | "datasets" | "job-detail";

// ─── Build types ────────────────────────────────────────────────────────────

export interface LayerInfo {
  index: number;
  layer_type: string;
  params: Record<string, unknown>;
}

export interface BuildRecord {
  build_id: string;
  model_id: string;
  model_name: string;
  class_name: string;
  code: string;
  layers: LayerInfo[];
  node_count: number;
  edge_count: number;
  created_at: string;
}

export interface BuildSummary {
  build_id: string;
  model_id: string;
  model_name: string;
  class_name: string;
  layer_count: number;
  layer_types: string[];
  created_at: string;
}

// ─── Inference types ────────────────────────────────────────────────────────

export interface DetectionBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  label_id: number;
  label_name?: string;
}

export interface PredictRequest {
  model_id: string;
  weight_id?: string;
  image_base64: string;
}

export interface PredictResponse {
  model_id: string;
  task_type: "classification" | "detection";
  class_id?: number;
  class_name?: string;
  confidence?: number;
  boxes?: DetectionBox[];
}

// ─── Export types ────────────────────────────────────────────────────────────

export interface ExportRequest {
  model_id: string;
  format: 'module' | 'script' | 'onnx';
  weight_id?: string;
  dataset?: string;
  include_training?: boolean;
  batch_size?: number;
}

export interface ExportResponse {
  model_id: string;
  format: string;
  filename: string;
  code?: string;
  file_path?: string;
  message: string;
}

export interface ExportFormat {
  id: string;
  name: string;
  extension: string;
  description: string;
}

// ─── Registry types (from /api/registry) ────────────────────────────────────

export interface PinDef {
  name: string;
  pin_type: string;   // "Tensor2D" | "Tensor1D" | "Image" | "BBoxList" | "Scalar" | "Any" | "None"
  label: string;
}

export interface AutoParamDef {
  name: string;        // param name, e.g. "in_channels"
  from_field: string;  // upstream shape field: "channels" | "features" | "height" | "width"
}

export interface ShapeRule {
  rule_type: string;   // "none_to_2d" | "conv2d" | "pool2d" | "flatten" | "linear" | "upsample" | "passthrough" | "terminal"
  out_channels_param?: string;
  out_features_param?: string;
  kernel_param?: string;
  stride_param?: string;
  padding_param?: string;
  scale_param?: string;
}

export interface NodeRegistryEntry {
  type: string;
  label: string;
  category: string;
  color: string;
  icon: string;
  params: ParamDefinition[];
  inputs: PinDef[];
  outputs: PinDef[];
  hasInput: boolean;
  hasOutput: boolean;
  allowMultipleInputs: boolean;
  autoParams: AutoParamDef[];
  shapeRule?: ShapeRule;
}

export interface PinTypeEntry {
  value: string;
  name: string;
}

// ─── Layer definitions ──────────────────────────────────────────────────────
// LayerType is kept as a convenience alias (string) for backward compatibility.
// All node definitions are now fetched from the backend via nodeCatalogStore.

export type LayerType = string;

export interface ParamDefinition {
  name: string;
  type: "number" | "array" | "string" | "boolean";
  default: number | number[] | string | boolean;
  label: string;
  min?: number;
  max?: number;
}
