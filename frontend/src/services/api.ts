/**
 * API client for Ultralytics-native Model DESIGNER backend.
 * All requests use relative URLs — proxied by Vite to localhost:8000.
 */
import type {
  ModelRecord, ModelSummary, ExportRequest,
  ModuleRecord, ModuleSummary, CatalogModule, ModuleArg,
  TrainRequest, JobRecord,
  DatasetInfo, DatasetMeta, SplitResponse, SplitTransferConfig, PartitionSummary,
  DatasetSamplesResponse, SampleData,
  WeightRecord, WeightSourceInfo, WeightGroup, ImportWeightResult, PretrainedModelInfo,
  MappingPreview, MappingKey, ApplyMapResult,
  CompatCheckResult, LayerDetailResult,
  LogEntry, DashboardStats,
  InferenceResult, InferenceHistoryEntry, InferResult,
  BenchmarkResult, JobCheckpoint,
} from '../types';

const API = '';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

function get<T>(path: string): Promise<T> {
  return request<T>(path);
}

function post<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, { method: 'POST', body: body ? JSON.stringify(body) : undefined });
}

function patch<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, { method: 'PATCH', body: body ? JSON.stringify(body) : undefined });
}

function put<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, { method: 'PUT', body: body ? JSON.stringify(body) : undefined });
}

function del<T>(path: string): Promise<T> {
  return request<T>(path, { method: 'DELETE' });
}

export const api = {
  // ── Models (Ultralytics YAML) ──────────────────────────────────────────
  listModels: () => get<ModelSummary[]>('/api/models/'),
  loadModel: (id: string) => get<ModelRecord>(`/api/models/${id}`),
  importYAML: (yamlContent: string, name: string, task: string) => 
    post<{ model_id: string; name: string; task: string; message: string; graph: any }>('/api/models/import/yaml', { 
      yaml_content: yamlContent, 
      name, 
      task 
    }),
  saveModel: (body: { name: string; description?: string; task?: string; yaml_def: Record<string, unknown> }, replace = false) =>
    post<{ model_id: string; message: string }>(`/api/models/?replace=${replace}`, body),
  deleteModel: (id: string) => del<{ message: string }>(`/api/models/${id}`),
  getModelYaml: (id: string) => get<{ model_id: string; yaml: string }>(`/api/models/${id}/yaml`),
  validateModel: (id: string, scale?: string) => post<{ valid: boolean; model_id: string; params: number; gradients: number; flops?: number; layers: number; message: string }>(`/api/models/${id}/validate${scale ? `?scale=${scale}` : ''}`, {}),
  exportModel: (req: ExportRequest) => post<Record<string, unknown>>('/api/models/export', req),

  // ── Modules (custom nn.Module blocks) ──────────────────────────────────
  listModules: () => get<ModuleSummary[]>('/api/modules/'),
  loadModule: (id: string) => get<ModuleRecord>(`/api/modules/${id}`),
  getModuleCatalog: () => get<CatalogModule[]>('/api/modules/catalog'),
  getModuleCategories: () => get<string[]>('/api/modules/catalog/categories'),
  getModuleInfo: (name: string) => get<CatalogModule>(`/api/modules/catalog/${name}`),
  saveModule: (body: { name: string; code: string; args?: ModuleArg[]; category?: string; description?: string }) =>
    post<{ module_id: string; message: string }>('/api/modules/', body),
  updateModule: (id: string, body: { name: string; code: string; args?: ModuleArg[]; category?: string; description?: string }) =>
    put<{ module_id: string; message: string }>(`/api/modules/${id}`, body),
  deleteModule: (id: string) => del<{ message: string }>(`/api/modules/${id}`),
  validateModule: (id: string) => post<{ valid: boolean; message: string }>(`/api/modules/${id}/validate`),

  // ── Training ────────────────────────────────────────────────────────────
  startTraining: (req: TrainRequest) =>
    post<{ job_id: string; message: string }>('/api/train/start', req),
  stopTraining: (jobId: string) =>
    post<{ message: string }>(`/api/train/${jobId}/stop`),
  resumeTraining: (jobId: string) =>
    post<{ job_id: string; message: string }>(`/api/train/${jobId}/resume`),
  appendTraining: (jobId: string, additionalEpochs: number) =>
    post<{ job_id: string; message: string }>(`/api/train/${jobId}/append`, { additional_epochs: additionalEpochs }),
  getTrainJob: (jobId: string) => get<JobRecord>(`/api/train/${jobId}`),
  getTrainJobLogs: (jobId: string, limit = 200) =>
    get<unknown[]>(`/api/train/${jobId}/logs?limit=${limit}`),
  getTrainPlots: (jobId: string) =>
    get<{ plots: Array<{ name: string; path: string; size: number }> }>(`/api/train/${jobId}/plots`),
  getTrainPlotImage: (jobId: string, plotName: string) =>
    `/api/train/${jobId}/plots/${plotName}`,
  getGradientStats: (jobId: string) =>
    get<{ gradients: Array<{ epoch: number; file: string; size: number }> }>(`/api/train/${jobId}/gradients`),
  getGradientStatsEpoch: (jobId: string, epoch: number) =>
    get<Record<string, { mean: number; std: number; min: number; max: number; norm: number }>>(`/api/train/${jobId}/gradients/${epoch}`),
  getWeightStats: (jobId: string) =>
    get<{ weights: Array<{ epoch: number; file: string; size: number }> }>(`/api/train/${jobId}/weights_stats`),
  getWeightStatsEpoch: (jobId: string, epoch: number) =>
    get<Record<string, { mean: number; std: number; min: number; max: number; norm: number }>>(`/api/train/${jobId}/weights_stats/${epoch}`),
  getClassSamples: (jobId: string) =>
    get<{ classes: Array<{ name: string; count: number; images: string[] }> }>(`/api/train/${jobId}/samples`),
  getClassSampleImage: (jobId: string, className: string, filename: string) =>
    `/api/train/${jobId}/samples/${encodeURIComponent(className)}/${filename}`,

  // ── Jobs ────────────────────────────────────────────────────────────────
  listJobs: (params?: { status?: string; model_id?: string }) => {
    const sp = new URLSearchParams();
    if (params?.status) sp.set('status', params.status);
    if (params?.model_id) sp.set('model_id', params.model_id);
    const qs = sp.toString();
    return get<JobRecord[]>(`/api/jobs/${qs ? '?' + qs : ''}`);
  },
  loadJob: (id: string) => get<JobRecord>(`/api/jobs/${id}`),
  getJobLogs: (id: string, limit = 200) =>
    get<unknown[]>(`/api/jobs/${id}/logs?limit=${limit}`),
  deleteJob: (id: string) => del<{ message: string }>(`/api/jobs/${id}`),

  // ── Datasets ────────────────────────────────────────────────────────────
  listDatasets: () => get<DatasetInfo[]>('/api/datasets/'),
  getDatasetInfo: (name: string) => get<DatasetInfo>(`/api/datasets/${name}/info`),
  previewDataset: (name: string, count = 8) =>
    get<{ dataset: string; count: number; samples: unknown[] }>(`/api/datasets/${name}/preview?count=${count}`),
  getDatasetStatus: (name: string) => get<{
    name: string; available: boolean; manual_download?: boolean; instructions?: string;
    meta?: DatasetMeta;
  }>(`/api/datasets/${name}/status`),
  getDatasetMeta: (name: string) => get<DatasetMeta>(`/api/datasets/${name}/meta`),
  scanDatasetMeta: (name: string) => post<DatasetMeta>(`/api/datasets/${name}/scan`, {}),
  deleteDatasetData: (name: string) => del<{ name: string; deleted_dirs: string[]; available: boolean }>(`/api/datasets/${name}/data`),
  startDatasetDownload: (name: string) => post<{ status: string; progress: number; message: string }>(`/api/datasets/${name}/download`, {}),
  getDatasetDownloadStatus: (name: string) => get<{ status: string; progress: number; message: string; current_file?: string; bytes_downloaded?: number; bytes_total?: number }>(`/api/datasets/${name}/download-status`),
  workspaceScan: (name: string) => get<{ found: boolean; path: string; file_count: number; dir_count?: number; size_bytes?: number; pending_archive: { path: string; name: string; size_bytes: number } | null }>(`/api/datasets/${name}/workspace-scan`),
  resumeExtract: (name: string) => post<{ status: string; message: string }>(`/api/datasets/${name}/resume-extract`, {}),
  importLocal: (name: string) => post<{ status: string; message: string }>(`/api/datasets/${name}/import-local`, {}),
  downloadFromUrl: (name: string, url: string) => post<{ status: string; message: string }>(`/api/datasets/${name}/download-url`, { url }),
  uploadDataset: (name: string, file: File, onProgress?: (pct: number, msg: string) => void) => {
    const form = new FormData();
    form.append('file', file);
    return new Promise<{ status: string; message: string }>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `/api/datasets/${name}/upload`);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) {
          onProgress(Math.round((e.loaded / e.total) * 50), `Uploading ${(e.loaded / (1024 * 1024)).toFixed(1)} / ${(e.total / (1024 * 1024)).toFixed(1)} MB`);
        }
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText));
        else reject(new Error(JSON.parse(xhr.responseText)?.detail || 'Upload failed'));
      };
      xhr.onerror = () => reject(new Error('Network error'));
      xhr.send(form);
    });
  },
  getDatasetUploadStatus: (name: string) => get<{ status: string; progress: number; message: string; bytes_received?: number }>(`/api/datasets/${name}/upload-status`),
  getDatasetSplits: (name: string) => get<SplitResponse>(`/api/datasets/${name}/splits`),
  saveDatasetSplits: (name: string, body: SplitTransferConfig) =>
    post<SplitResponse>(`/api/datasets/${name}/splits`, body),
  getDatasetPartitions: (name: string) => get<PartitionSummary>(`/api/datasets/${name}/partitions`),
  createPartition: (name: string, body: { name: string; percent: number }) =>
    post<PartitionSummary>(`/api/datasets/${name}/partitions`, body),
  splitPartition: (name: string, partitionId: string, children: { name: string; percent: number }[]) =>
    post<PartitionSummary>(`/api/datasets/${name}/partitions/${partitionId}/split`, { children }),
  deletePartition: (name: string, partitionId: string) =>
    del<PartitionSummary>(`/api/datasets/${name}/partitions/${partitionId}`),
  updatePartitionMethod: (name: string, method: string) =>
    put<PartitionSummary>(`/api/datasets/${name}/partitions/method`, { method }),
  convertCocoToYolo: (name: string, params?: { use_segments?: boolean; use_keypoints?: boolean; cls91to80?: boolean }) =>
    post<{ status: string; message: string; labels_dir?: string; file_count?: number; error?: string }>(
      `/api/datasets/${name}/convert-coco`,
      params || {}
    ),
  getDatasetSamples: (name: string, params?: {
    page?: number; page_size?: number; class_idx?: number | null;
    class_indices?: string; split?: string; thumb_size?: number;
    include_annotations?: boolean; partition_id?: string;
  }) => {
    const sp = new URLSearchParams();
    if (params?.page != null) sp.set('page', String(params.page));
    if (params?.page_size != null) sp.set('page_size', String(params.page_size));
    if (params?.class_indices) sp.set('class_indices', params.class_indices);
    else if (params?.class_idx != null) sp.set('class_idx', String(params.class_idx));
    if (params?.split) sp.set('split', params.split);
    if (params?.thumb_size != null) sp.set('thumb_size', String(params.thumb_size));
    if (params?.include_annotations) sp.set('include_annotations', 'true');
    if (params?.partition_id) sp.set('partition_id', params.partition_id);
    const qs = sp.toString();
    return get<DatasetSamplesResponse>(`/api/datasets/${name}/samples${qs ? '?' + qs : ''}`);
  },
  getDatasetSampleDetail: (name: string, index: number, params?: {
    split?: string; include_annotations?: boolean;
  }) => {
    const sp = new URLSearchParams();
    if (params?.split) sp.set('split', params.split);
    if (params?.include_annotations != null) sp.set('include_annotations', String(params.include_annotations));
    const qs = sp.toString();
    return get<SampleData>(`/api/datasets/${name}/samples/${index}${qs ? '?' + qs : ''}`);
  },

  // ── Weights ─────────────────────────────────────────────────────────────
  listWeights: (modelId?: string) => {
    const qs = modelId ? `?model_id=${modelId}` : '';
    return get<WeightRecord[]>(`/api/weights/${qs}`);
  },
  getWeight: (id: string) => get<WeightRecord>(`/api/weights/${id}`),
  getWeightLineage: (id: string) => get<WeightRecord[]>(`/api/weights/${id}/lineage`),
  deleteWeight: (id: string) => del<{ message: string }>(`/api/weights/${id}`),
  renameWeight: (id: string, name: string) => patch<WeightRecord>(`/api/weights/${id}/rename`, { name }),
  downloadWeight: (id: string, filename?: string) => {
    const a = document.createElement('a');
    a.href = `/api/weights/${id}/download`;
    if (filename) a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  },
  createEmptyWeight: (modelId: string, name?: string, scale?: string, yoloModel?: string, usePretrained?: boolean) =>
    post<{ weight_id: string; model_id: string; model_name: string; key_count: number; file_size_bytes: number }>(
      '/api/weights/create-empty', {
        model_id: modelId,
        name: name || '',
        model_scale: scale || null,
        yolo_model: yoloModel || null,
        use_pretrained: usePretrained ?? true,
      },
    ),
  inspectWeightKeys: (id: string) => get<{ key: string; node_id: string; shape: number[]; dtype: string; numel: number }[]>(`/api/weights/${id}/keys`),
  getWeightInfo: (id: string) => get<{ params: number; gflops: number | null }>(`/api/weights/${id}/info`),
  extractPartialWeight: (id: string, nodeIds: string[]) =>
    post<{ weight_id: string; keys_extracted: number }>(`/api/weights/${id}/extract`, { node_ids: nodeIds }),
  transferWeights: (targetId: string, sourceId: string, nodeIdMap?: Record<string, string>) =>
    post<{ matched_keys: number; total_target_keys: number; match_ratio: number; keys: string[] }>(
      `/api/weights/${targetId}/transfer`,
      { source_weight_id: sourceId, node_id_map: nodeIdMap || null },
    ),

  // ── Weight Transfer (import, map, apply) ──────────────────────────────
  listWeightSources: () => get<WeightSourceInfo[]>('/api/weights/sources'),
  listPretrained: () => get<PretrainedModelInfo[]>('/api/weights/pretrained'),
  downloadPretrained: (modelKey: string) =>
    post<{ weight_id: string; model_key: string; source_plugin: string; key_count: number }>(
      '/api/weights/pretrained/download', { model_key: modelKey },
    ),
  importWeight: async (file: File, name: string): Promise<ImportWeightResult> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', name);
    const res = await fetch('/api/weights/import', { method: 'POST', body: formData });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${res.status}`);
    }
    return res.json();
  },
  getWeightGroups: (id: string) => get<WeightGroup[]>(`/api/weights/${id}/groups`),
  getWeightGroupsAnnotated: (id: string, modelId?: string) =>
    get<(WeightGroup & { node_label?: string })[]>(`/api/weights/${id}/groups-annotated${modelId ? `?model_id=${modelId}` : ''}`),
  autoMapWeights: (targetId: string, sourceId: string) =>
    post<MappingPreview>(`/api/weights/${targetId}/auto-map`, { source_weight_id: sourceId }),
  applyWeightMap: (targetId: string, sourceId: string, mapping: MappingKey[], freezeNodeIds: string[]) =>
    post<ApplyMapResult>(`/api/weights/${targetId}/apply-map`, {
      source_weight_id: sourceId,
      mapping,
      freeze_node_ids: freezeNodeIds,
    }),
  compatCheck: (targetId: string, sourceId: string, srcPrefix: string, tgtPrefix: string) =>
    post<CompatCheckResult>(`/api/weights/${targetId}/compat-check`, {
      source_weight_id: sourceId,
      src_prefix: srcPrefix,
      tgt_prefix: tgtPrefix,
    }),
  layerDetail: (weightId: string, key: string, bins = 50) =>
    get<LayerDetailResult>(`/api/weights/${weightId}/layer-detail?key=${encodeURIComponent(key)}&bins=${bins}`),

  // ── Logs ────────────────────────────────────────────────────────────────
  getLogs: (params?: { category?: string; level?: string; limit?: number; days?: number }) => {
    const sp = new URLSearchParams();
    if (params?.category) sp.set('category', params.category);
    if (params?.level) sp.set('level', params.level);
    if (params?.limit) sp.set('limit', String(params.limit));
    if (params?.days) sp.set('days', String(params.days));
    const qs = sp.toString();
    return get<LogEntry[]>(`/api/logs/${qs ? '?' + qs : ''}`);
  },

  // ── Stats ───────────────────────────────────────────────────────────────
  getStats: () => get<DashboardStats>('/api/stats'),

  // ── Job Checkpoints ─────────────────────────────────────────────────────────
  listJobCheckpoints: (jobId: string) =>
    get<{ checkpoints: JobCheckpoint[] }>(`/api/train/${jobId}/checkpoints`),
  createWeightFromCheckpoint: (jobId: string, checkpointName: string) =>
    post<{ weight_id: string; model_name: string; checkpoint: string; message: string }>(
      `/api/train/${jobId}/checkpoints/${encodeURIComponent(checkpointName)}/create-weight-profile`
    ),

  // ── Weight Export ───────────────────────────────────────────────────────────
  exportWeight: (weightId: string, body: { format: string; imgsz?: number; device?: string; half?: boolean; simplify?: boolean }) =>
    post<{ weight_id: string; format: string; exported_path: string; message: string }>(
      `/api/weights/${weightId}/export`, body
    ),
  downloadExportedWeight: (weightId: string, fmt: string) => {
    const a = document.createElement('a');
    a.href = `/api/weights/${weightId}/export/download?fmt=${fmt}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  },

  // ── Benchmark ───────────────────────────────────────────────────────────────
  listBenchmarkDatasets: (weightId?: string) => {
    const qs = weightId ? `?weight_id=${weightId}` : '';
    return get<{ label: string; value: string; yaml_path: string; nc: number | null; source: string }[]>(`/api/benchmark/datasets${qs}`);
  },
  runBenchmark: (body: { weight_id: string; dataset: string; split?: string; conf?: number; iou?: number; imgsz?: number; batch?: number; device?: string }) =>
    post<BenchmarkResult>('/api/benchmark/run', body),
  listBenchmarks: (weightId?: string, limit = 20) => {
    const qs = weightId ? `?weight_id=${weightId}&limit=${limit}` : `?limit=${limit}`;
    return get<BenchmarkResult[]>(`/api/benchmark/history${qs}`);
  },
  getBenchmark: (id: string) => get<BenchmarkResult>(`/api/benchmark/${id}`),
  deleteBenchmark: (id: string) => del<{ message: string }>(`/api/benchmark/${id}`),

  // ── Inference ───────────────────────────────────────────────────────────────
  predictImages: async (weightId: string, files: File[], conf = 0.25, iou = 0.45, imgsz = 640): Promise<InferenceResult> => {
    const formData = new FormData();
    formData.append('weight_id', weightId);
    formData.append('conf', String(conf));
    formData.append('iou', String(iou));
    formData.append('imgsz', String(imgsz));
    files.forEach(f => formData.append('files', f));
    const res = await fetch('/api/inference/predict', { method: 'POST', body: formData });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${res.status}`);
    }
    return res.json();
  },
  infer: async (
    weightId: string, file: File, conf = 0.25, iou = 0.45, imgsz = 640, visualizeSgbg = false
  ): Promise<InferResult> => {
    const fd = new FormData();
    fd.append('weight_id', weightId);
    fd.append('conf', String(conf));
    fd.append('iou', String(iou));
    fd.append('imgsz', String(imgsz));
    fd.append('visualize_sgbg', String(visualizeSgbg));
    fd.append('file', file);
    const res = await fetch('/api/inference/infer', { method: 'POST', body: fd });
    if (!res.ok) { const b = await res.json().catch(() => ({})); throw new Error(b.detail || `HTTP ${res.status}`); }
    return res.json();
  },
  inferAttention: async (
    weightId: string, file: File, imgsz: number,
    bbox: [number, number, number, number],  // [x1,y1,x2,y2]
    detLabel: string,
  ): Promise<{ weight_id: string; det_label: string; bbox_centroid: number[]; scales: Record<string, { scale: string; feature_hw: number[]; query_pixel: number[]; attention: string }> }> => {
    const fd = new FormData();
    fd.append('weight_id', weightId);
    fd.append('imgsz', String(imgsz));
    fd.append('bbox_x1', String(bbox[0]));
    fd.append('bbox_y1', String(bbox[1]));
    fd.append('bbox_x2', String(bbox[2]));
    fd.append('bbox_y2', String(bbox[3]));
    fd.append('det_label', detLabel);
    fd.append('file', file);
    const res = await fetch('/api/inference/infer/attention', { method: 'POST', body: fd });
    if (!res.ok) { const b = await res.json().catch(() => ({})); throw new Error(b.detail || `HTTP ${res.status}`); }
    return res.json();
  },
  getInferenceHistory: (limit = 50) => get<InferenceHistoryEntry[]>(`/api/inference/history?limit=${limit}`),
  clearInferenceHistory: () => del<{ message: string }>('/api/inference/history'),
  deleteInferenceEntry: (id: string) => del<{ message: string }>(`/api/inference/history/${id}`),

  // ── Packages (.mdpkg) ───────────────────────────────────────────────────
  exportWeightPackage: (weightId: string, includeJobs = false) => {
    const a = document.createElement('a');
    a.href = `/api/packages/weights/${weightId}/export?include_jobs=${includeJobs}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  },
  exportJobPackage: (jobId: string, includeJobs = false) => {
    const a = document.createElement('a');
    a.href = `/api/packages/jobs/${jobId}/export?include_jobs=${includeJobs}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  },
  peekPackage: async (
    file: File,
  ): Promise<{ version: string; weights: { id: string; model_name: string; dataset: string; epochs_trained: number }[]; jobs: string[] }> => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch('/api/packages/peek', { method: 'POST', body: formData });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${res.status}`);
    }
    return res.json();
  },
  importPackage: async (
    file: File,
    renameMap: Record<string, string> = {},
    includeJobs = false,
  ): Promise<{ weights_imported: { old_id: string; new_id: string; name: string }[]; jobs_imported: { old_id: string; new_id: string }[]; errors: string[] }> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('rename_map', JSON.stringify(renameMap));
    formData.append('include_jobs', String(includeJobs));
    const res = await fetch('/api/packages/import', { method: 'POST', body: formData });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${res.status}`);
    }
    return res.json();
  },
};
