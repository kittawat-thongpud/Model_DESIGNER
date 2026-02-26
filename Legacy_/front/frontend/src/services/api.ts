/**
 * API client for Model DESIGNER backend.
 */
const API_BASE = "";

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ─── Types ───────────────────────────────────────────────────────────────────

import type {
  ModelGraph,
  ModelSummary,
  BuildResponse,
  TrainRequest,
  TrainStatus,
  DatasetInfo,
  LogEntry,
  JobRecord,
  JobLogEntry,
  WeightRecord,
  DashboardStats,
  BuildRecord,
  BuildSummary,
  ModelPackage,
  CreatePackageRequest,
  NodeRegistryEntry,
  PinTypeEntry,
  ExportRequest,
  ExportResponse,
  ExportFormat,
} from "../types";

// ─── API ─────────────────────────────────────────────────────────────────────

export const api = {
  // Models
  saveModel: (graph: ModelGraph, replace = false) =>
    request<{ model_id: string; message: string; exists?: boolean; name?: string }>(
      `/api/models/?replace=${replace}`, 
      {
        method: "POST",
        body: JSON.stringify(graph),
      }
    ),

  listModels: () => request<ModelSummary[]>("/api/models/"),

  loadModel: (id: string) => request<ModelGraph>(`/api/models/${id}`),

  deleteModel: (id: string) =>
    request<{ message: string }>(`/api/models/${id}`, { method: "DELETE" }),

  buildModel: (id: string, replace = false) =>
    request<BuildResponse>(`/api/models/${id}/build?replace=${replace}`, { method: "POST" }),

  // Builds
  listBuilds: () => request<BuildSummary[]>("/api/builds/"),

  getBuild: (buildId: string) => request<BuildRecord>(`/api/builds/${buildId}`),

  deleteBuild: (buildId: string) =>
    request<{ message: string }>(`/api/builds/${buildId}`, { method: "DELETE" }),

  checkBuildName: (name: string) =>
    request<{ exists: boolean; build_id?: string; model_name?: string }>(
      `/api/builds/check?name=${encodeURIComponent(name)}`
    ),

  // Datasets
  listDatasets: () => request<DatasetInfo[]>("/api/datasets/"),

  getDatasetInfo: (name: string) =>
    request<DatasetInfo>(`/api/datasets/${name}/info`),

  // Training
  startTraining: (req: TrainRequest) =>
    request<{ job_id: string; status: string }>("/api/train", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  getTrainStatus: (jobId: string) =>
    request<TrainStatus>(`/api/train/${jobId}/status`),

  stopTraining: (jobId: string) =>
    request<{ message: string }>(`/api/train/${jobId}/stop`, {
      method: "POST",
    }),

  // Jobs
  listJobs: (params?: { status?: string; model_id?: string }) => {
    const qs = new URLSearchParams();
    if (params?.status) qs.set("status", params.status);
    if (params?.model_id) qs.set("model_id", params.model_id);
    const q = qs.toString();
    return request<JobRecord[]>(`/api/jobs/${q ? "?" + q : ""}`);
  },

  getJob: (jobId: string) => request<JobRecord>(`/api/jobs/${jobId}`),

  getJobLogs: (jobId: string, limit = 200) =>
    request<JobLogEntry[]>(`/api/jobs/${jobId}/logs?limit=${limit}`),

  deleteJob: (jobId: string) =>
    request<{ message: string }>(`/api/jobs/${jobId}`, { method: "DELETE" }),

  // Weight Snapshots
  getJobSnapshots: (jobId: string) =>
    request<Record<string, unknown>[]>(`/api/jobs/${jobId}/snapshots`),

  getSnapshotStats: (jobId: string) =>
    request<{ layers: string[]; epochs: number[]; timeline: Record<string, Record<string, unknown>[]> }>(
      `/api/jobs/${jobId}/snapshots/stats`
    ),

  getSnapshotLayers: (jobId: string) =>
    request<string[]>(`/api/jobs/${jobId}/snapshots/layers`),

  getSnapshotEpochs: (jobId: string) =>
    request<number[]>(`/api/jobs/${jobId}/snapshots/epochs`),

  getSnapshotData: (jobId: string, epoch: number, layer: string) =>
    request<{
      epoch: number;
      layer: string;
      shape: number[];
      rows: number;
      cols: number;
      values: number[][];
      min: number;
      max: number;
      mean: number;
      std: number;
    }>(`/api/jobs/${jobId}/snapshots/${epoch}/${layer}`),

  // Weights
  listWeights: (modelId?: string) => {
    const qs = modelId ? `?model_id=${modelId}` : "";
    return request<WeightRecord[]>(`/api/weights/${qs}`);
  },

  getWeight: (weightId: string) =>
    request<WeightRecord>(`/api/weights/${weightId}`),

  deleteWeight: (weightId: string) =>
    request<{ message: string }>(`/api/weights/${weightId}`, {
      method: "DELETE",
    }),

  // Logs
  getLogs: (params?: {
    category?: string;
    level?: string;
    limit?: number;
    offset?: number;
  }) => {
    const qs = new URLSearchParams();
    if (params?.category) qs.set("category", params.category);
    if (params?.level) qs.set("level", params.level);
    if (params?.limit) qs.set("limit", String(params.limit));
    if (params?.offset) qs.set("offset", String(params.offset));
    return request<LogEntry[]>(`/api/logs?${qs.toString()}`);
  },

  clearLogs: () =>
    request<{ message: string }>("/api/logs", { method: "DELETE" }),

  // Dashboard
  getStats: () => request<DashboardStats>("/api/stats"),

  // Packages
  listPackages: () => request<ModelPackage[]>("/api/packages/"),

  getPackage: (id: string) => request<ModelPackage>(`/api/packages/${id}`),

  createPackage: (req: CreatePackageRequest) =>
    request<ModelPackage>("/api/packages/", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  deletePackage: (id: string) =>
    request<{ message: string }>(`/api/packages/${id}`, { method: "DELETE" }),

  // Registry
  getNodeRegistry: () => request<NodeRegistryEntry[]>("/api/registry/nodes"),

  getPinTypes: () => request<PinTypeEntry[]>("/api/registry/pin-types"),

  // Export
  exportModel: (req: ExportRequest) =>
    request<ExportResponse>("/api/export/", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  getExportFormats: () => request<ExportFormat[]>("/api/export/formats"),
};
