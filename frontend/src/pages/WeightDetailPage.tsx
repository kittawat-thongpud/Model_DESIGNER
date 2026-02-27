import { useState, useEffect, useRef, useCallback } from 'react';
import { api } from '../services/api';
import type { WeightRecord, JobRecord } from '../types';
import {
  ArrowLeft, Box, Layers, Activity, Download,
  Cpu, Database, Clock, Zap, CheckCircle2, Copy, BarChart2,
  GitBranch, RefreshCw, Upload, X, Pencil, Check, Loader2,
  Image, Target, Package, ChevronDown,
} from 'lucide-react';
import ImportPackageModal from '../components/ImportPackageModal';
import WeightTransferCard from '../components/WeightTransferCard';
import { fmtSize, fmtTime, timeAgo, fmtDataset } from '../utils/format';
import ExportWeightPanel from '../components/ExportWeightPanel';
import BenchmarkPanel from '../components/BenchmarkPanel';
import JobCharts from '../components/JobCharts';
import PlotsGallery from '../components/PlotsGallery';
import JobConfiguration from '../components/JobConfiguration';
import ClassSamplesGallery from '../components/ClassSamplesGallery';

interface Props {
  weightId: string;
  onBack: () => void;
  onOpenJob?: (jobId: string) => void;
  onOpenWeight?: (weightId: string) => void;
  onEditWeight?: (weightId: string) => void;
}

/* ────────────────────────────── sub-components ────────────────────── */

function StatCard({ label, value, sub, trend, trendUp, color = 'text-white' }: {
  label: string; value: string; sub?: string;
  trend?: string; trendUp?: boolean; color?: string;
}) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
      <div className="text-slate-500 text-xs font-semibold uppercase tracking-wider mb-2">{label}</div>
      <div className="flex items-end gap-2">
        <div className={`text-2xl font-mono font-bold ${color}`}>{value}</div>
        {trend && (
          <div className={`text-xs font-medium mb-1 ${trendUp ? 'text-emerald-400' : 'text-amber-400'}`}>
            {trend}
          </div>
        )}
      </div>
      {sub && <div className="text-xs text-slate-600 mt-1">{sub}</div>}
    </div>
  );
}

function ConfigRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between items-center py-2 border-b border-slate-800/50 last:border-0">
      <span className="text-slate-500 text-sm">{label}</span>
      <span className="text-slate-200 text-sm font-mono">{value}</span>
    </div>
  );
}

function TabItem({ label, icon, active, onClick }: {
  label: string; icon: React.ReactNode; active: boolean; onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 pb-3 cursor-pointer transition-all border-b-2 text-sm font-medium ${
        active
          ? 'text-indigo-400 border-indigo-500'
          : 'text-slate-500 border-transparent hover:text-slate-300 hover:border-slate-700'
      }`}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}

/* ────────────────────────────── main component ─────────────────── */

export default function WeightDetailPage({ weightId, onBack, onOpenJob, onOpenWeight, onEditWeight }: Props) {
  const [weight, setWeight] = useState<WeightRecord | null>(null);
  const [job, setJob] = useState<JobRecord | null>(null);
  const [graph, setGraph] = useState<Record<string, unknown> | null>(null);
  const [analysis, setAnalysis] = useState<Record<string, unknown>[]>([]);
  const [lineage, setLineage] = useState<WeightRecord[]>([]);
  const [weightKeys, setWeightKeys] = useState<{key: string; node_id: string; shape: number[]; dtype: string; numel: number}[]>([]);
  const [weightGroups, setWeightGroups] = useState<{prefix: string; module_type: string; param_count: number; keys: {key: string; shape: number[]; dtype: string}[]}[]>([]);
  const [weightInfo, setWeightInfo] = useState<{params: number; gflops: number | null} | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'metrics' | 'layers' | 'code' | 'benchmark'>('overview');
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);
  const [plotsKey, setPlotsKey] = useState(0);

  // Continue Training modal state
  const [showTrainModal, setShowTrainModal] = useState(false);
  const [showExportPanel, setShowExportPanel] = useState(false);
  const [showBenchmarkPanel, setShowBenchmarkPanel] = useState(false);
  const [showExportMenu, setShowExportMenu] = useState(false);
  const [pkgIncludeJobs, setPkgIncludeJobs] = useState(false);
  const exportMenuRef = useRef<HTMLDivElement>(null);

  // Rename state
  const [renaming, setRenaming] = useState(false);
  const [renameName, setRenameName] = useState('');
  const [renameSaving, setRenameSaving] = useState(false);

  // Close export dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (exportMenuRef.current && !exportMenuRef.current.contains(e.target as Node)) setShowExportMenu(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const handleRename = async () => {
    if (!weight || !renameName.trim()) return;
    setRenameSaving(true);
    try {
      const updated = await api.renameWeight(weight.weight_id, renameName.trim());
      setWeight(updated);
    } catch { /* ignore */ }
    setRenameSaving(false);
    setRenaming(false);
  };

  useEffect(() => {
    setLoading(true);
    api.getWeight(weightId).then(async (w) => {
      setWeight(w);
      // Load related job data
      if (w.job_id) {
        api.loadJob(w.job_id).then(setJob).catch(() => {});
      }
      // Load model graph for architecture
      if (w.model_id) {
        api.loadModel(w.model_id).then((m) => setGraph(m as unknown as Record<string, unknown>)).catch(() => {});
      }
      // Load lineage chain
      api.getWeightLineage(w.weight_id).then(setLineage).catch(() => {});
      // Load weight keys for architecture tab
      api.inspectWeightKeys(w.weight_id).then(setWeightKeys).catch(() => {});
      // Load weight groups (for module_type tags)
      api.getWeightGroups(w.weight_id).then(setWeightGroups).catch(() => {});
      // Load weight info (params + GFLOPs) — lazy, only used in Architecture tab
      api.getWeightInfo(w.weight_id).then(setWeightInfo).catch(() => {});
    }).catch(() => {}).finally(() => setLoading(false));
  }, [weightId]);

  const handleExport = () => {
    if (!weight) return;
    const filename = `${weight.model_name}_${weight.weight_id.slice(0, 8)}.pt`.replace(/\s+/g, '_');
    api.downloadWeight(weight.weight_id, filename);
  };

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleOpenTrainModal = () => {
    setShowTrainModal(true);
  };

  if (loading && !weight) {
    return <div className="flex-1 flex items-center justify-center text-slate-500">Loading...</div>;
  }
  if (!weight) {
    return <div className="flex-1 flex items-center justify-center text-slate-500">Weight not found</div>;
  }

  const config = job?.config ?? {};
  const history = job?.history ?? [];

  // Architecture: extract layer info from graph
  const layers = (graph as any)?.nodes?.map((n: any) => {
    const p = n.params || {};
    return {
      name: (p.label as string) || n.id,
      type: n.type,
      params: p,
    };
  }) ?? [];

  const totalParams = job?.model_params ?? job?.trainable_params ?? null;
  const isDetection = (job?.task ?? weight.dataset_name ?? '').toLowerCase().includes('detect');
  const hasHistory = history.length > 0;

  // Usage code snippet
  const usageCode = `import torch
from pathlib import Path

# ─── 1. Rebuild model architecture ───
# (Use the same graph that produced this weight)
# Option A: If you exported via Model DESIGNER:
#   from my_model import GeneratedModel
#   model = GeneratedModel()

# Option B: Manual rebuild
#   import torch.nn as nn
#   model = nn.Sequential(...)  # match your architecture

# ─── 2. Load trained weights ───
weight_path = Path("weights/${weight.weight_id}.pt")
state_dict = torch.load(str(weight_path), map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# ─── 3. Inference ───
# For classification (e.g. MNIST 28×28 grayscale):
input_tensor = torch.randn(1, 1, 28, 28)  # adjust to your input shape
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1).max().item()

print(f"Predicted: {predicted_class}, Confidence: {confidence:.2%}")`;

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-7xl mx-auto p-8 space-y-6">

        {/* ── Breadcrumb ── */}
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <button onClick={onBack} className="hover:text-indigo-400 cursor-pointer transition-colors">Weights</button>
          <span>/</span>
          <span className="text-slate-300">{weight.model_name}</span>
        </div>

        {/* ── Top Banner: Status & Actions ── */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-slate-800 pb-6">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 shadow-[0_0_15px_rgba(99,102,241,0.15)]">
              <Box size={24} />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                {weight.weight_id.slice(0, 8)}
                {renaming ? (
                  <span className="flex items-center gap-2">
                    <input
                      autoFocus
                      value={renameName}
                      onChange={e => setRenameName(e.target.value)}
                      onKeyDown={e => { if (e.key === 'Enter') handleRename(); if (e.key === 'Escape') setRenaming(false); }}
                      className="bg-slate-800 border border-amber-500 rounded px-2 py-0.5 text-base font-normal text-white outline-none w-56"
                    />
                    <button onClick={handleRename} disabled={renameSaving} className="text-amber-400 hover:text-amber-300 cursor-pointer">
                      {renameSaving ? <Loader2 size={16} className="animate-spin" /> : <Check size={16} />}
                    </button>
                    <button onClick={() => setRenaming(false)} className="text-slate-500 hover:text-white cursor-pointer"><X size={16} /></button>
                  </span>
                ) : (
                  <span className="flex items-center gap-2 group">
                    <span className="text-slate-500 font-normal text-lg">/ {weight.model_name}({fmtDataset(weight.dataset, weight.dataset_name)})</span>
                    <button
                      onClick={() => { setRenameName(weight.model_name); setRenaming(true); }}
                      className="opacity-0 group-hover:opacity-100 text-slate-600 hover:text-amber-400 cursor-pointer transition-all"
                      title="Rename weight"
                    >
                      <Pencil size={14} />
                    </button>
                  </span>
                )}
              </h1>
              <div className="flex items-center gap-3 text-xs mt-1">
                <span className="flex items-center gap-1 text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded border border-emerald-400/20">
                  <CheckCircle2 size={12} /> Ready
                </span>
                <span className="text-slate-500 flex items-center gap-1">
                  <Clock size={12} /> {timeAgo(weight.created_at)}
                </span>
                <span className="text-slate-500 flex items-center gap-1">
                  <Database size={12} /> {fmtDataset(weight.dataset, weight.dataset_name)}
                </span>
                {weight.epochs_trained > 0 && (
                  <span className="text-slate-500 flex items-center gap-1">
                    <Activity size={12} /> {weight.total_epochs && weight.total_epochs > weight.epochs_trained
                      ? `${weight.epochs_trained} epochs (${weight.total_epochs} total)`
                      : `${weight.epochs_trained} epochs`}
                  </span>
                )}
              </div>
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              onClick={handleOpenTrainModal}
              className="bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 cursor-pointer shadow-lg shadow-indigo-500/20"
            >
              <RefreshCw size={16} /> Continue Training
            </button>

            {/* Benchmark */}
            <button
              onClick={() => setShowBenchmarkPanel(true)}
              className="bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 cursor-pointer border border-slate-600"
            >
              <BarChart2 size={16} /> Benchmark
            </button>

            {/* Export dropdown */}
            <div className="relative" ref={exportMenuRef}>
              <button
                onClick={() => setShowExportMenu(v => !v)}
                className="bg-emerald-700 hover:bg-emerald-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 cursor-pointer shadow-lg shadow-emerald-500/10"
              >
                <Download size={16} /> Export <ChevronDown size={14} className={`transition-transform ${showExportMenu ? 'rotate-180' : ''}`} />
              </button>
              {showExportMenu && (
                <div className="absolute right-0 top-full mt-1 w-52 bg-slate-800 border border-slate-700 rounded-xl shadow-xl z-30 overflow-hidden">
                  <button
                    onClick={() => { setShowExportPanel(true); setShowExportMenu(false); }}
                    className="w-full flex items-center gap-3 px-4 py-3 text-sm text-slate-200 hover:bg-slate-700 transition-colors cursor-pointer"
                  >
                    <Download size={15} className="text-emerald-400" />
                    <div className="text-left">
                      <div className="font-medium">Export .pt</div>
                      <div className="text-xs text-slate-500">Weight file only</div>
                    </div>
                  </button>
                  <div className="border-t border-slate-700/50" />
                  <button
                    onClick={() => { api.exportWeightPackage(weight.weight_id, pkgIncludeJobs); setShowExportMenu(false); }}
                    className="w-full flex items-center gap-3 px-4 py-3 text-sm text-slate-200 hover:bg-slate-700 transition-colors cursor-pointer"
                  >
                    <Package size={15} className="text-violet-400" />
                    <div className="text-left flex-1">
                      <div className="font-medium">Export Package</div>
                      <div className="text-xs text-slate-500">Lineage (.mdpkg)</div>
                    </div>
                  </button>
                  <div className="border-t border-slate-700/50" />
                  <div className="px-4 py-2.5 flex items-center justify-between">
                    <span className="text-xs text-slate-400">Include Jobs</span>
                    <button
                      onClick={e => { e.stopPropagation(); setPkgIncludeJobs(v => !v); }}
                      className={`w-8 h-4.5 rounded-full relative transition-colors cursor-pointer ${pkgIncludeJobs ? 'bg-violet-500' : 'bg-slate-600'}`}
                    >
                      <div className={`absolute top-0.5 w-3.5 h-3.5 bg-white rounded-full shadow transition-transform ${pkgIncludeJobs ? 'translate-x-4' : 'translate-x-0.5'}`} />
                    </button>
                  </div>
                </div>
              )}
            </div>


            {onEditWeight && (
              <button
                onClick={() => onEditWeight(weightId)}
                className="bg-amber-600 hover:bg-amber-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 cursor-pointer shadow-lg shadow-amber-500/20"
              >
                <RefreshCw size={16} /> Edit Weight
              </button>
            )}
            {job && onOpenJob && (
              <button
                onClick={() => onOpenJob(job.job_id)}
                className="bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors border border-slate-700 flex items-center gap-2 cursor-pointer"
              >
                <Activity size={16} /> View Job
              </button>
            )}
            <button
              onClick={onBack}
              className="bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors border border-slate-700 flex items-center gap-2 cursor-pointer"
            >
              <ArrowLeft size={16} /> Back
            </button>
          </div>
        </div>

        {/* ── KPI Cards ── */}
        <div className="flex flex-wrap gap-4">
          <StatCard
            label="Model Size"
            value={fmtSize(weight.file_size_bytes)}
            sub={totalParams != null ? `${totalParams.toLocaleString()} params` : undefined}
          />
          {job?.best_mAP50 != null && (
            <StatCard label="Best mAP50" value={`${(job.best_mAP50 * 100).toFixed(1)}%`} color="text-emerald-400" />
          )}
          {job?.best_mAP50_95 != null && (
            <StatCard label="Best mAP50-95" value={`${(job.best_mAP50_95 * 100).toFixed(1)}%`} color="text-indigo-400" />
          )}
          {job?.total_time != null && (
            <StatCard label="Train Time" value={fmtTime(job.total_time)} sub={job.device ?? undefined} />
          )}
        </div>

        {/* ── Tabs ── */}
        <div className="flex items-center gap-6 border-b border-slate-800 mt-4 overflow-x-auto">
          <TabItem label="Training & Metrics" icon={<Activity size={16} />} active={activeTab === 'overview'} onClick={() => setActiveTab('overview')} />
          <TabItem label="Architecture" icon={<Layers size={16} />} active={activeTab === 'layers'} onClick={() => setActiveTab('layers')} />
          <TabItem label="Benchmark" icon={<BarChart2 size={16} />} active={activeTab === 'benchmark'} onClick={() => setActiveTab('benchmark')} />
          <TabItem label="Lineage" icon={<GitBranch size={16} />} active={activeTab === 'code'} onClick={() => setActiveTab('code')} />
        </div>

        {/* ── Tab Content ── */}
        <div className="min-h-[400px]">

          {/* ═══ TAB 1: TRAINING & METRICS (merged) ═══ */}
          {activeTab === 'overview' && (
            <div className="space-y-6">
              {job ? (
                <>
                  {/* Training Charts */}
                  <JobCharts history={history} isDetection={isDetection} />

                  {/* Training Config */}
                  <JobConfiguration
                    config={job.config as any}
                    datasetName={weight.dataset_name || job.dataset_name}
                    partitions={job.partitions}
                    modelScale={job.model_scale}
                  />

                  {/* Class Samples */}
                  <div>
                    <h3 className="text-white font-semibold flex items-center gap-2 mb-4">
                      <Target size={16} className="text-indigo-400" /> Class Samples
                    </h3>
                    <ClassSamplesGallery jobId={job.job_id} />
                  </div>

                  {/* Visualization & Evaluation Plots */}
                  <div>
                    <h3 className="text-white font-semibold flex items-center gap-2 mb-4">
                      <Image size={16} className="text-purple-400" /> Plots
                    </h3>
                    <PlotsGallery jobId={job.job_id} />
                  </div>
                </>
              ) : (
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-12 text-center text-slate-500">
                  No training job linked to this weight.
                </div>
              )}
            </div>
          )}

          {/* ═══ TAB 3: ARCHITECTURE ═══ */}
          {activeTab === 'layers' && (() => {
            // Group by 2-level prefix (e.g. model.0, model.1) for YOLO weights
            // Fall back to 1-level if all keys share the same top-level prefix
            const getPrefix = (key: string) => {
              const parts = key.split('.');
              if (parts.length >= 2) return `${parts[0]}.${parts[1]}`;
              return parts[0];
            };
            const nodeMap = new Map<string, {keys: typeof weightKeys; params: number}>();
            for (const k of weightKeys) {
              const prefix = getPrefix(k.key);
              const entry = nodeMap.get(prefix) ?? { keys: [], params: 0 };
              entry.keys.push(k);
              entry.params += k.numel;
              nodeMap.set(prefix, entry);
            }
            const nodes = Array.from(nodeMap.entries());
            const totalP = weightKeys.reduce((s, k) => s + k.numel, 0);
            // Build module_type lookup from groups
            const moduleTypeMap = new Map<string, string>();
            for (const g of weightGroups) moduleTypeMap.set(g.prefix, g.module_type);

            return (
              <div className="space-y-4">
                {/* Summary bar */}
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 flex flex-wrap gap-4 items-center">
                  <div className="flex items-center gap-2">
                    <Layers size={16} className="text-indigo-400" />
                    <span className="text-white font-semibold">{weight.model_name}</span>
                  </div>
                  <div className="flex gap-4 text-xs text-slate-400 ml-auto flex-wrap">
                    <span><span className="text-slate-500">Layers:</span> <span className="text-white font-mono">{nodes.length}</span></span>
                    <span><span className="text-slate-500">Total Params:</span> <span className="text-indigo-300 font-mono">{(totalP || totalParams || 0).toLocaleString()}</span></span>
                    {(weightInfo?.gflops ?? job?.model_flops) != null && (
                      <span><span className="text-slate-500">GFLOPs:</span> <span className="text-purple-300 font-mono">{(weightInfo?.gflops ?? job?.model_flops)!.toFixed(1)}</span></span>
                    )}
                    <span><span className="text-slate-500">File Size:</span> <span className="text-white font-mono">{fmtSize(weight.file_size_bytes)}</span></span>
                    {job?.device && <span><span className="text-slate-500">Device:</span> <span className="text-white font-mono">{job.device}</span></span>}
                  </div>
                </div>

                {nodes.length > 0 ? (
                  <div className="bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden">
                    <table className="w-full text-sm text-left text-slate-400">
                      <thead className="text-xs text-slate-500 uppercase bg-slate-950/50 border-b border-slate-800">
                        <tr>
                          <th className="px-5 py-3">Node / Prefix</th>
                          <th className="px-5 py-3">Tensors</th>
                          <th className="px-5 py-3">Parameters</th>
                          <th className="px-5 py-3">Shapes (sample)</th>
                          <th className="px-5 py-3">dtype</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/50 font-mono text-xs">
                        {nodes.map(([nodeId, info]) => {
                          const sampleShapes = info.keys.slice(0, 3).map(k =>
                            `[${k.shape.join('×')}]`
                          ).join(' ');
                          const dtypes = [...new Set(info.keys.map(k => k.dtype.replace('torch.', '')))].join(', ');
                          const pct = totalP > 0 ? (info.params / totalP) * 100 : 0;
                          const moduleType = moduleTypeMap.get(nodeId.split('.')[0]) ?? '';
                          return (
                            <tr key={nodeId} className="hover:bg-slate-800/30 transition-colors group">
                              <td className="px-5 py-2.5">
                                <div className="flex items-center gap-2">
                                  <span className="text-white font-semibold font-mono">{nodeId}</span>
                                  {moduleType && <span className="text-[9px] px-1.5 py-0.5 rounded bg-slate-700/60 text-slate-400">{moduleType}</span>}
                                </div>
                              </td>
                              <td className="px-5 py-2.5 text-slate-400">{info.keys.length}</td>
                              <td className="px-5 py-2.5">
                                <div className="flex items-center gap-2">
                                  <span className="text-emerald-400">{info.params.toLocaleString()}</span>
                                  <div className="flex-1 max-w-[80px] h-1 bg-slate-800 rounded-full overflow-hidden">
                                    <div className="h-full bg-indigo-500/70 rounded-full" style={{ width: `${pct}%` }} />
                                  </div>
                                  <span className="text-slate-600 text-[10px]">{pct.toFixed(1)}%</span>
                                </div>
                              </td>
                              <td className="px-5 py-2.5 text-slate-500 truncate max-w-xs">{sampleShapes || '—'}</td>
                              <td className="px-5 py-2.5 text-amber-400/80">{dtypes}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-12 text-center text-slate-500">
                    Loading architecture…
                  </div>
                )}
              </div>
            );
          })()}

          {/* ═══ TAB 4: BENCHMARK ═══ */}
          {activeTab === 'benchmark' && (
            <BenchmarkPanel
              weightId={weight.weight_id}
              onClose={() => setActiveTab('overview')}
              inline
            />
          )}

          {/* ═══ TAB 5: LINEAGE ═══ */}
          {activeTab === 'code' && (
            <div className="space-y-6">
              {/* Lineage chain from API */}
              <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
                <h3 className="text-white font-semibold mb-5 flex items-center gap-2">
                  <GitBranch size={18} className="text-cyan-400" /> Full Lineage Chain
                </h3>
                {lineage.length === 0 ? (
                  <p className="text-slate-500 text-sm">No lineage data available.</p>
                ) : (
                  <div className="space-y-0">
                    {lineage.map((w, i) => {
                      const isCurrent = w.weight_id === weight.weight_id;
                      return (
                        <div key={w.weight_id} className={`flex gap-4 ${i < lineage.length - 1 ? 'pb-4' : ''}`}>
                          {/* Connector */}
                          <div className="flex flex-col items-center">
                            <div className={`w-3 h-3 rounded-full shrink-0 mt-1 ${
                              isCurrent ? 'bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.6)]' : 'bg-indigo-500'
                            }`} />
                            {i < lineage.length - 1 && (
                              <div className="w-px flex-1 bg-slate-700 mt-1" />
                            )}
                          </div>

                          {/* Card */}
                          <div className={`flex-1 rounded-xl border p-4 mb-1 ${
                            isCurrent
                              ? 'border-emerald-500/30 bg-emerald-500/5'
                              : 'border-slate-700/50 bg-slate-800/30'
                          }`}>
                            <div className="flex items-start justify-between gap-3">
                              <div>
                                <div className="flex items-center gap-2 text-sm">
                                  <span className={`font-semibold ${isCurrent ? 'text-emerald-400' : 'text-white'}`}>
                                    {w.model_name}
                                    {isCurrent && (
                                      <span className="ml-2 text-[10px] bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 rounded px-1.5 py-0.5">Current</span>
                                    )}
                                  </span>
                                  <span className="text-slate-500 font-mono text-xs">{w.weight_id.slice(0, 10)}</span>
                                </div>
                                <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-500 mt-1.5">
                                  <span className="flex items-center gap-1"><Database size={11} /> {fmtDataset(w.dataset, (w as any).dataset_name)}</span>
                                  <span className="flex items-center gap-1"><Activity size={11} /> {w.epochs_trained} epochs{w.total_epochs && w.total_epochs > w.epochs_trained ? ` (${w.total_epochs} cum.)` : ''}</span>
                                  {w.final_accuracy != null && (
                                    <span className="text-emerald-400 font-mono">{w.final_accuracy.toFixed(2)}% acc</span>
                                  )}
                                  {w.final_loss != null && (
                                    <span className="text-amber-400/70 font-mono">loss {w.final_loss.toFixed(4)}</span>
                                  )}
                                  <span className="flex items-center gap-1"><Clock size={11} /> {new Date(w.created_at).toLocaleDateString()}</span>
                                </div>
                              </div>
                              <div className="flex gap-2 shrink-0">
                                {!isCurrent && onOpenWeight && (
                                  <button
                                    onClick={() => onOpenWeight(w.weight_id)}
                                    className="text-xs px-2 py-1 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded border border-slate-600 transition-colors cursor-pointer"
                                  >
                                    View
                                  </button>
                                )}
                                <button
                                  onClick={() => {
                                    const fn = `${w.model_name}_${w.weight_id.slice(0, 8)}.pt`.replace(/\s+/g, '_');
                                    api.downloadWeight(w.weight_id, fn);
                                  }}
                                  className="text-xs px-2 py-1 bg-emerald-700/40 hover:bg-emerald-700/60 text-emerald-400 rounded border border-emerald-600/30 transition-colors cursor-pointer flex items-center gap-1"
                                  title="Download this weight"
                                >
                                  <Download size={11} /> .pt
                                </button>
                              </div>
                            </div>

                            {/* Training runs within this weight */}
                            {w.training_runs && w.training_runs.length > 0 && (
                              <div className="mt-3 pt-3 border-t border-slate-700/50 grid grid-cols-3 gap-2">
                                {w.training_runs.map((r) => (
                                  <div key={r.run} className="bg-slate-900/60 rounded-lg p-2 text-center">
                                    <div className="text-[10px] text-slate-500">Run {r.run}</div>
                                    <div className="text-xs font-mono text-white">{r.epochs}ep</div>
                                    {r.accuracy != null && (
                                      <div className="text-[10px] text-emerald-400 font-mono">{r.accuracy.toFixed(1)}%</div>
                                    )}
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Usage code */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 font-mono text-sm relative group">
                <div className="flex items-center justify-between mb-4">
                  <div className="text-slate-500"># Python — Load and use this weight</div>
                  <button
                    onClick={() => handleCopy(usageCode)}
                    className="bg-slate-800 hover:bg-slate-700 text-white px-3 py-1.5 rounded flex items-center gap-2 text-xs border border-slate-700 cursor-pointer"
                  >
                    <Copy size={14} /> {copied ? 'Copied!' : 'Copy'}
                  </button>
                </div>
                <pre className="text-indigo-300 whitespace-pre-wrap leading-relaxed overflow-x-auto">
                  {usageCode}
                </pre>
              </div>
            </div>
          )}

        </div>
      </div>

      {/* ── Continue Training placeholder ── */}
      {showTrainModal && weight && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => setShowTrainModal(false)}>
          <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-full max-w-md mx-4 p-6" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-lg font-semibold text-white mb-3">Continue Training</h3>
            <p className="text-sm text-slate-400 mb-4">Use the Train Designer to configure and launch training with this weight.</p>
            <button onClick={() => setShowTrainModal(false)} className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg text-sm transition-colors cursor-pointer">Close</button>
          </div>
        </div>
      )}

      {showExportPanel && (
        <ExportWeightPanel
          weightId={weight.weight_id}
          onClose={() => setShowExportPanel(false)}
        />
      )}

      {showBenchmarkPanel && (
        <BenchmarkPanel
          weightId={weight.weight_id}
          onClose={() => setShowBenchmarkPanel(false)}
        />
      )}

    </div>
  );
}
