import { useState, useEffect, useRef } from 'react';
import { api } from '../services/api';
import type { WeightRecord, JobRecord } from '../types';
import {
  ArrowLeft, Box, Layers, Activity, Terminal, Download,
  Cpu, Database, Clock, Zap, CheckCircle2, Copy, BarChart2,
  GitBranch, RefreshCw, Play, Upload, X,
} from 'lucide-react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, LineChart, Line, Legend,
} from 'recharts';
import WeightTransferCard from '../components/WeightTransferCard';
import ConfusionMatrixView from '../components/ConfusionMatrixView';
import { fmtSize, fmtTime, timeAgo } from '../utils/format';

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
  const [activeTab, setActiveTab] = useState<'overview' | 'metrics' | 'layers' | 'code'>('overview');
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);

  // Continue Training modal state
  const [showTrainModal, setShowTrainModal] = useState(false);

  // Import weight state
  const [importing, setImporting] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);
  const importFileRef = useRef<HTMLInputElement>(null);

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
    }).catch(() => {}).finally(() => setLoading(false));
  }, [weightId]);

  const handleExport = () => {
    if (!weight) return;
    const filename = `${weight.model_name}_${weight.weight_id.slice(0, 8)}.pt`.replace(/\s+/g, '_');
    api.downloadWeight(weight.weight_id, filename);
  };

  const handleImportFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImporting(true);
    setImportError(null);
    try {
      const result = await api.importWeight(file, file.name.replace(/\.pt[h]?$/, ''));
      // Navigate to the newly imported weight
      if (onOpenWeight) onOpenWeight(result.weight_id);
    } catch (err) {
      setImportError(err instanceof Error ? err.message : 'Import failed');
    } finally {
      setImporting(false);
      if (importFileRef.current) importFileRef.current.value = '';
    }
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
  const confMatrix = job?.confusion_matrix ?? null;
  const classNames = job?.class_names ?? [];
  const perClassMetrics = job?.per_class_metrics ?? null;

  // Build chart data from history
  const chartData = history.map((h) => ({
    epoch: h.epoch,
    train_loss: h.train_loss,
    val_loss: h.val_loss,
    train_acc: h.train_accuracy,
    val_acc: h.val_accuracy,
    lr: h.lr,
  }));

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
                <span className="text-slate-500 font-normal text-lg">/ {weight.model_name}({weight.dataset})</span>
              </h1>
              <div className="flex items-center gap-3 text-xs mt-1">
                <span className="flex items-center gap-1 text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded border border-emerald-400/20">
                  <CheckCircle2 size={12} /> Ready
                </span>
                <span className="text-slate-500 flex items-center gap-1">
                  <Clock size={12} /> {timeAgo(weight.created_at)}
                </span>
                <span className="text-slate-500 flex items-center gap-1">
                  <Database size={12} /> {weight.dataset}
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
              <Play size={16} /> Continue Training
            </button>

            {/* Export (Download) */}
            <button
              onClick={handleExport}
              className="bg-emerald-700 hover:bg-emerald-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 cursor-pointer shadow-lg shadow-emerald-500/10"
              title="Download .pt file"
            >
              <Download size={16} /> Export .pt
            </button>

            {/* Import (Upload) */}
            <label
              className={`bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 cursor-pointer border border-slate-600 ${importing ? 'opacity-60 cursor-not-allowed' : ''}`}
              title="Upload a .pt file to import as new weight"
            >
              {importing ? <RefreshCw size={16} className="animate-spin" /> : <Upload size={16} />}
              {importing ? 'Importing…' : 'Import .pt'}
              <input
                ref={importFileRef}
                type="file"
                accept=".pt,.pth"
                className="hidden"
                disabled={importing}
                onChange={handleImportFile}
              />
            </label>
            {importError && (
              <div className="w-full text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded px-3 py-1.5 flex items-center gap-2">
                <X size={12} /> {importError}
              </div>
            )}

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
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard
            label="Val Accuracy"
            value={weight.final_accuracy != null ? `${weight.final_accuracy.toFixed(1)}%` : '—'}
            color="text-emerald-400"
          />
          <StatCard
            label="Val Loss"
            value={weight.final_loss != null ? weight.final_loss.toFixed(4) : '—'}
            color="text-amber-400"
          />
          <StatCard
            label="Model Size"
            value={fmtSize(weight.file_size_bytes)}
            sub={totalParams != null ? `${totalParams.toLocaleString()} params` : undefined}
          />
          <StatCard
            label="Inference Time"
            value={job?.inference_time_ms != null ? `${job.inference_time_ms.toFixed(1)}ms` : '—'}
            sub={job?.device ?? undefined}
          />
        </div>

        {/* ── Tabs ── */}
        <div className="flex items-center gap-6 border-b border-slate-800 mt-4">
          <TabItem label="Overview" icon={<Activity size={16} />} active={activeTab === 'overview'} onClick={() => setActiveTab('overview')} />
          <TabItem label="Metrics & Evaluation" icon={<BarChart2 size={16} />} active={activeTab === 'metrics'} onClick={() => setActiveTab('metrics')} />
          <TabItem label="Architecture" icon={<Layers size={16} />} active={activeTab === 'layers'} onClick={() => setActiveTab('layers')} />
          <TabItem label="Lineage" icon={<GitBranch size={16} />} active={activeTab === 'code'} onClick={() => setActiveTab('code')} />
        </div>

        {/* ── Tab Content ── */}
        <div className="min-h-[400px]">

          {/* ═══ TAB 1: OVERVIEW ═══ */}
          {activeTab === 'overview' && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Training History Graph */}
              <div className="lg:col-span-2 space-y-6">
                {/* Accuracy Chart */}
                {chartData.length > 0 ? (
                  <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
                    <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
                      <Activity size={18} className="text-indigo-400" /> Training History
                    </h3>
                    <div className="h-[280px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData}>
                          <defs>
                            <linearGradient id="colorAcc" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                              <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                            </linearGradient>
                            <linearGradient id="colorValAcc" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#34d399" stopOpacity={0.3} />
                              <stop offset="95%" stopColor="#34d399" stopOpacity={0} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                          <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 11 }} />
                          <YAxis stroke="#475569" tick={{ fontSize: 11 }} />
                          <Tooltip
                            contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc', fontSize: 12 }}
                          />
                          <Legend wrapperStyle={{ fontSize: 12 }} />
                          <Area type="monotone" dataKey="train_acc" name="Train Acc" stroke="#6366f1" strokeWidth={2} fillOpacity={1} fill="url(#colorAcc)" />
                          <Area type="monotone" dataKey="val_acc" name="Val Acc" stroke="#34d399" strokeWidth={2} fillOpacity={1} fill="url(#colorValAcc)" />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                ) : (
                  <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-12 text-center text-slate-500">
                    No training history available
                  </div>
                )}

                {/* Loss Chart */}
                {chartData.length > 0 && (
                  <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
                    <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
                      <Zap size={18} className="text-amber-400" /> Loss Curve
                    </h3>
                    <div className="h-[220px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                          <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 11 }} />
                          <YAxis stroke="#475569" tick={{ fontSize: 11 }} />
                          <Tooltip
                            contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc', fontSize: 12 }}
                          />
                          <Legend wrapperStyle={{ fontSize: 12 }} />
                          <Line type="monotone" dataKey="train_loss" name="Train Loss" stroke="#f59e0b" strokeWidth={2} dot={false} />
                          <Line type="monotone" dataKey="val_loss" name="Val Loss" stroke="#ef4444" strokeWidth={2} dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}
              </div>

              {/* Right Column: Config + Lineage */}
              <div className="space-y-6">
                {/* Training Config */}
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
                  <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
                    <Cpu size={18} className="text-pink-400" /> Training Config
                  </h3>
                  <div className="space-y-0">
                    <ConfigRow label="Model" value={weight.model_name} />
                    <ConfigRow label="Dataset" value={weight.dataset} />
                    <ConfigRow label="Epochs" value={String(config.epochs ?? weight.epochs_trained)} />
                    <ConfigRow label="Optimizer" value={String(config.optimizer ?? 'Adam')} />
                    <ConfigRow label="Learning Rate" value={String(config.lr0 ?? config.learning_rate ?? '—')} />
                    <ConfigRow label="Batch Size" value={String(config.batch_size ?? '—')} />
                    <ConfigRow label="Loss" value={String(config.loss ?? 'CrossEntropyLoss')} />
                    {config.scheduler ? <ConfigRow label="Scheduler" value={String(config.scheduler)} /> : null}
                    {config.amp ? <ConfigRow label="AMP" value="Enabled" /> : null}
                    {job?.device && <ConfigRow label="Device" value={job.device} />}
                  </div>
                </div>

                {/* ── Training Lineage ── */}
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 space-y-4">
                  <h3 className="text-white font-semibold flex items-center gap-2">
                    <GitBranch size={18} className="text-cyan-400" /> Training Lineage
                  </h3>

                  {(() => {
                    const runs = weight.training_runs ?? [];
                    if (runs.length === 0 && !job) {
                      return <p className="text-xs text-slate-600">No training history recorded.</p>;
                    }

                    // Fall back to single run from current weight if no training_runs yet (legacy)
                    const displayRuns = runs.length > 0 ? runs : [{
                      run: 1,
                      job_id: weight.job_id,
                      weight_id: weight.weight_id,
                      dataset: weight.dataset,
                      epochs: weight.epochs_trained,
                      accuracy: weight.final_accuracy,
                      loss: weight.final_loss,
                      total_time: weight.total_time ?? (job?.total_time ?? null),
                      device: weight.device ?? (job?.device ?? null),
                      created_at: weight.created_at,
                    }];

                    const cumEpochs = displayRuns.reduce((s, r) => s + (r.epochs ?? 0), 0);
                    const cumTime = displayRuns.reduce((s, r) => s + (r.total_time ?? 0), 0);
                    const bestAcc = Math.max(...displayRuns.map((r) => r.accuracy ?? 0));

                    return (
                      <>
                        {/* Cumulative stats */}
                        <div className="grid grid-cols-3 gap-2 text-center">
                          <div className="bg-slate-800/50 rounded-lg p-2">
                            <div className="text-[10px] text-slate-500 uppercase">Total Runs</div>
                            <div className="text-sm font-bold text-white">{displayRuns.length}</div>
                          </div>
                          <div className="bg-slate-800/50 rounded-lg p-2">
                            <div className="text-[10px] text-slate-500 uppercase">Cum. Epochs</div>
                            <div className="text-sm font-bold text-white">{cumEpochs}</div>
                          </div>
                          <div className="bg-slate-800/50 rounded-lg p-2">
                            <div className="text-[10px] text-slate-500 uppercase">Best Acc</div>
                            <div className="text-sm font-bold text-emerald-400">{bestAcc > 0 ? `${bestAcc.toFixed(1)}%` : '—'}</div>
                          </div>
                        </div>

                        {/* Accuracy progression mini chart */}
                        {displayRuns.length > 1 && displayRuns.some((r) => r.accuracy != null) && (
                          <div className="h-24">
                            <ResponsiveContainer width="100%" height="100%">
                              <AreaChart data={displayRuns.map((r) => ({ name: `Run ${r.run}`, acc: r.accuracy ?? 0, loss: r.loss ?? 0 }))}>
                                <defs>
                                  <linearGradient id="lineageAccGrad" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#34d399" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#34d399" stopOpacity={0} />
                                  </linearGradient>
                                </defs>
                                <XAxis dataKey="name" tick={{ fontSize: 9, fill: '#64748b' }} axisLine={false} tickLine={false} />
                                <YAxis hide domain={[0, 100]} />
                                <Tooltip
                                  contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 11 }}
                                  formatter={(v: number) => [`${v.toFixed(1)}%`, 'Accuracy']}
                                />
                                <Area type="monotone" dataKey="acc" stroke="#34d399" fill="url(#lineageAccGrad)" strokeWidth={2} dot={{ r: 3 }} />
                              </AreaChart>
                            </ResponsiveContainer>
                          </div>
                        )}

                        {/* Timeline table */}
                        <div className="space-y-0">
                          {displayRuns.map((r, i) => {
                            const isCurrent = r.weight_id === weight.weight_id;
                            return (
                              <div
                                key={`${r.job_id ?? i}`}
                                className={`flex items-start gap-3 py-2 ${i < displayRuns.length - 1 ? 'border-b border-slate-800/50' : ''}`}
                              >
                                {/* Timeline dot + line */}
                                <div className="flex flex-col items-center shrink-0 pt-1">
                                  <div className={`w-2.5 h-2.5 rounded-full ${
                                    isCurrent
                                      ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.5)]'
                                      : 'bg-indigo-500'
                                  }`} />
                                  {i < displayRuns.length - 1 && <div className="w-px flex-1 bg-slate-700 mt-1 min-h-[16px]" />}
                                </div>

                                {/* Run details */}
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-center gap-2 text-xs">
                                    <span className={`font-medium ${isCurrent ? 'text-emerald-400' : 'text-white'}`}>
                                      Run {r.run}
                                    </span>
                                    {r.accuracy != null && (
                                      <span className="text-emerald-400 font-mono">{r.accuracy.toFixed(1)}%</span>
                                    )}
                                    {r.loss != null && (
                                      <span className="text-amber-400/70 font-mono text-[10px]">loss: {r.loss.toFixed(4)}</span>
                                    )}
                                  </div>
                                  <div className="flex items-center gap-2 text-[10px] text-slate-500 mt-0.5">
                                    <span>{r.epochs}ep</span>
                                    <span>·</span>
                                    <span>{r.dataset}</span>
                                    {r.total_time != null && r.total_time > 0 && (
                                      <><span>·</span><span>{fmtTime(r.total_time)}</span></>
                                    )}
                                    {r.device && (
                                      <><span>·</span><span>{r.device}</span></>
                                    )}
                                  </div>
                                  <div className="flex items-center gap-2 text-[10px] mt-0.5">
                                    {r.job_id && onOpenJob && (
                                      <button
                                        onClick={() => onOpenJob(r.job_id!)}
                                        className="text-indigo-400 hover:underline cursor-pointer"
                                      >
                                        Job {r.job_id.slice(0, 8)}
                                      </button>
                                    )}
                                    {!isCurrent && r.weight_id && onOpenWeight && (
                                      <>
                                        <span className="text-slate-700">·</span>
                                        <button
                                          onClick={() => onOpenWeight(r.weight_id)}
                                          className="text-cyan-400 hover:underline cursor-pointer"
                                        >
                                          Weight {r.weight_id.slice(0, 8)}
                                        </button>
                                      </>
                                    )}
                                    {r.created_at && (
                                      <span className="text-slate-600 ml-auto">{new Date(r.created_at).toLocaleDateString()}</span>
                                    )}
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                        </div>

                        {/* Cumulative time */}
                        {cumTime > 0 && (
                          <div className="text-[10px] text-slate-600 text-right border-t border-slate-800/50 pt-2">
                            Total training time: {fmtTime(cumTime)}
                          </div>
                        )}
                      </>
                    );
                  })()}
                </div>

                {/* ── Weight Transfer ── */}
                <WeightTransferCard weightId={weight.weight_id} />
              </div>
            </div>
          )}

          {/* ═══ TAB 2: METRICS ═══ */}
          {activeTab === 'metrics' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Confusion Matrix */}
              <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
                <h3 className="text-white font-semibold mb-6">Confusion Matrix</h3>
                {confMatrix && confMatrix.length > 0 ? (
                  <ConfusionMatrixView matrix={confMatrix} classNames={classNames} />
                ) : (
                  <div className="text-slate-500 text-sm text-center py-12">
                    No confusion matrix available for this weight.
                    {!job && <p className="mt-2 text-slate-600">Training job data not found.</p>}
                  </div>
                )}
              </div>

              {/* Per-class Metrics */}
              <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
                <h3 className="text-white font-semibold mb-4">Class Performance</h3>
                {perClassMetrics && perClassMetrics.length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left text-slate-400">
                      <thead className="text-xs uppercase bg-slate-950/50 text-slate-500">
                        <tr>
                          <th className="px-4 py-3 rounded-l-lg">Class</th>
                          <th className="px-4 py-3">Precision</th>
                          <th className="px-4 py-3">Recall</th>
                          <th className="px-4 py-3 rounded-r-lg">F1-Score</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/50">
                        {perClassMetrics.map((m, i) => {
                          const name = (m.class_name as string) ?? classNames[i] ?? `Class ${i}`;
                          const precision = m.precision as number | null;
                          const recall = m.recall as number | null;
                          const f1 = m.f1 as number | null;
                          return (
                            <tr key={i} className="hover:bg-slate-800/30">
                              <td className="px-4 py-3 font-medium text-white">{name}</td>
                              <td className="px-4 py-3 text-emerald-400 font-mono">
                                {precision != null ? precision.toFixed(3) : '—'}
                              </td>
                              <td className="px-4 py-3 font-mono">
                                {recall != null ? recall.toFixed(3) : '—'}
                              </td>
                              <td className="px-4 py-3 font-mono text-indigo-400">
                                {f1 != null ? f1.toFixed(3) : '—'}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="text-slate-500 text-sm text-center py-12">
                    No per-class metrics available.
                  </div>
                )}
              </div>

              {/* Analysis Results (from plugins) */}
              {analysis.length > 0 && (
                <div className="lg:col-span-2 bg-slate-900/50 border border-slate-800 rounded-xl p-6">
                  <h3 className="text-white font-semibold mb-4">Analysis Results</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {analysis
                      .filter((a) => a.renderer === 'metric_card' || a.renderer === 'scalar')
                      .map((a, i) => (
                        <div key={i} className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
                          <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">
                            {(a as any).display_name}
                          </div>
                          <div className="text-lg font-mono font-bold text-white">
                            {typeof a.data === 'object' && a.data !== null && 'value' in (a.data as Record<string, unknown>)
                              ? String((a.data as Record<string, unknown>).value)
                              : typeof a.data === 'number'
                                ? a.data.toFixed(4)
                                : String(a.data ?? '—')}
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ═══ TAB 3: ARCHITECTURE ═══ */}
          {activeTab === 'layers' && (
            <div className="bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden">
              <div className="p-4 border-b border-slate-800 bg-slate-950/30 flex justify-between items-center">
                <h3 className="text-white font-semibold">Model Architecture</h3>
                <span className="text-xs text-slate-500 bg-slate-900 px-2 py-1 rounded border border-slate-800">
                  {totalParams != null
                    ? `Total Params: ${totalParams.toLocaleString()} (${fmtSize(weight.file_size_bytes)})`
                    : `Size: ${fmtSize(weight.file_size_bytes)}`}
                </span>
              </div>
              {layers.length > 0 ? (
                <table className="w-full text-sm text-left text-slate-400">
                  <thead className="text-xs text-slate-500 uppercase bg-slate-950/50 border-b border-slate-800">
                    <tr>
                      <th className="px-6 py-3">Layer Name</th>
                      <th className="px-6 py-3">Type</th>
                      <th className="px-6 py-3">Key Parameters</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/50 font-mono text-xs">
                    {layers.map((layer: any, idx: number) => {
                      // Build a compact param summary
                      const paramEntries = Object.entries(layer.params)
                        .filter(([k]) => !['label'].includes(k))
                        .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
                        .slice(0, 5);
                      return (
                        <tr key={idx} className="hover:bg-slate-800/30 transition-colors">
                          <td className="px-6 py-3 text-white">{layer.name}</td>
                          <td className="px-6 py-3 text-indigo-400">{layer.type}</td>
                          <td className="px-6 py-3 text-slate-500 truncate max-w-xs">{paramEntries.join(', ') || '—'}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              ) : (
                <div className="text-slate-500 text-sm text-center py-12">
                  Model architecture not available.
                </div>
              )}
            </div>
          )}

          {/* ═══ TAB 4: LINEAGE ═══ */}
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
                                  <span className="flex items-center gap-1"><Database size={11} /> {w.dataset}</span>
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
    </div>
  );
}
