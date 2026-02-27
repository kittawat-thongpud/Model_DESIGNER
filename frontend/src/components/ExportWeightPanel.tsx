import { useState, useEffect } from 'react';
import { api } from '../services/api';
import type { JobCheckpoint } from '../types';
import {
  Download, Package, Cpu, FileCode2, X, RefreshCw,
  ChevronDown, CheckCircle2, AlertTriangle, Loader2,
} from 'lucide-react';

interface Props {
  weightId?: string;
  jobId?: string;
  onClose: () => void;
  onWeightCreated?: (weightId: string) => void;
}

const EXPORT_FORMATS = [
  { value: 'onnx', label: 'ONNX', desc: 'Cross-platform inference (recommended)' },
  { value: 'torchscript', label: 'TorchScript', desc: 'PyTorch optimized inference' },
  { value: 'engine', label: 'TensorRT', desc: 'NVIDIA GPU inference (requires TensorRT)' },
  { value: 'tflite', label: 'TFLite', desc: 'Mobile / Edge TPU' },
  { value: 'coreml', label: 'CoreML', desc: 'Apple devices' },
];

type Tab = 'download' | 'profile' | 'export';

export default function ExportWeightPanel({ weightId, jobId, onClose, onWeightCreated }: Props) {
  const [tab, setTab] = useState<Tab>('download');
  const [checkpoints, setCheckpoints] = useState<JobCheckpoint[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>('best.pt');
  const [exportFormat, setExportFormat] = useState('onnx');
  const [imgsz, setImgsz] = useState(640);
  const [half, setHalf] = useState(false);
  const [simplify, setSimplify] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ ok: boolean; message: string } | null>(null);

  useEffect(() => {
    if (jobId) {
      api.listJobCheckpoints(jobId).then(r => {
        setCheckpoints(r.checkpoints);
        if (r.checkpoints.length > 0) setSelectedCheckpoint(r.checkpoints[0].name);
      }).catch(() => {});
    }
  }, [jobId]);

  const resolvedWeightId = weightId ?? null;

  const handleDownloadPt = () => {
    if (!resolvedWeightId) return;
    api.downloadWeight(resolvedWeightId);
  };

  const handleCreateProfile = async () => {
    if (!jobId || !selectedCheckpoint) return;
    setLoading(true);
    setResult(null);
    try {
      const r = await api.createWeightFromCheckpoint(jobId, selectedCheckpoint);
      setResult({ ok: true, message: `Weight profile "${r.model_name}" created (ID: ${r.weight_id.slice(0, 8)})` });
      onWeightCreated?.(r.weight_id);
    } catch (e: unknown) {
      setResult({ ok: false, message: e instanceof Error ? e.message : 'Failed' });
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async () => {
    if (!resolvedWeightId) return;
    setLoading(true);
    setResult(null);
    try {
      const r = await api.exportWeight(resolvedWeightId, {
        format: exportFormat,
        imgsz,
        half,
        simplify,
      });
      setResult({ ok: true, message: `${r.format.toUpperCase()} exported. Downloading…` });
      setTimeout(() => api.downloadExportedWeight(resolvedWeightId, exportFormat), 800);
    } catch (e: unknown) {
      setResult({ ok: false, message: e instanceof Error ? e.message : 'Export failed' });
    } finally {
      setLoading(false);
    }
  };

  const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: 'download', label: 'Download .pt', icon: <Download size={14} /> },
    ...(jobId ? [{ id: 'profile' as Tab, label: 'Save to Library', icon: <Package size={14} /> }] : []),
    ...(resolvedWeightId ? [{ id: 'export' as Tab, label: 'Export Format', icon: <FileCode2 size={14} /> }] : []),
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div
        className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-full max-w-md mx-4"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800">
          <h3 className="text-white font-semibold flex items-center gap-2">
            <Cpu size={16} className="text-indigo-400" /> Export Weight
          </h3>
          <button onClick={onClose} className="text-slate-500 hover:text-white transition-colors cursor-pointer">
            <X size={16} />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-slate-800 px-5">
          {tabs.map(t => (
            <button
              key={t.id}
              onClick={() => { setTab(t.id); setResult(null); }}
              className={`flex items-center gap-1.5 px-3 py-2.5 text-xs font-medium border-b-2 transition-colors cursor-pointer -mb-px ${
                tab === t.id
                  ? 'text-indigo-400 border-indigo-500'
                  : 'text-slate-500 border-transparent hover:text-slate-300'
              }`}
            >
              {t.icon} {t.label}
            </button>
          ))}
        </div>

        <div className="p-5 space-y-4">

          {/* Download .pt */}
          {tab === 'download' && (
            <div className="space-y-3">
              {jobId && checkpoints.length > 0 && (
                <div>
                  <label className="block text-xs text-slate-400 mb-1.5">Checkpoint</label>
                  <div className="relative">
                    <select
                      value={selectedCheckpoint}
                      onChange={e => setSelectedCheckpoint(e.target.value)}
                      className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm appearance-none cursor-pointer focus:outline-none focus:border-indigo-500"
                    >
                      {checkpoints.map(c => (
                        <option key={c.name} value={c.name}>
                          {c.name} ({(c.size_bytes / 1024 / 1024).toFixed(1)} MB)
                        </option>
                      ))}
                    </select>
                    <ChevronDown size={14} className="absolute right-3 top-2.5 text-slate-400 pointer-events-none" />
                  </div>
                </div>
              )}
              <p className="text-xs text-slate-500">Download the raw PyTorch weight file (.pt) for use in other tools.</p>
              <button
                onClick={handleDownloadPt}
                disabled={!resolvedWeightId}
                className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors cursor-pointer"
              >
                <Download size={14} /> Download .pt
              </button>
            </div>
          )}

          {/* Save to Library (job checkpoints) */}
          {tab === 'profile' && jobId && (
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-slate-400 mb-1.5">Checkpoint to save</label>
                <div className="relative">
                  <select
                    value={selectedCheckpoint}
                    onChange={e => setSelectedCheckpoint(e.target.value)}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm appearance-none cursor-pointer focus:outline-none focus:border-indigo-500"
                  >
                    {checkpoints.map(c => (
                      <option key={c.name} value={c.name}>
                        {c.name} ({(c.size_bytes / 1024 / 1024).toFixed(1)} MB)
                      </option>
                    ))}
                  </select>
                  <ChevronDown size={14} className="absolute right-3 top-2.5 text-slate-400 pointer-events-none" />
                </div>
              </div>
              <p className="text-xs text-slate-500">
                Copies this checkpoint into the Weight Library so you can benchmark, transfer, or export it independently.
              </p>
              <button
                onClick={handleCreateProfile}
                disabled={loading || checkpoints.length === 0}
                className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors cursor-pointer"
              >
                {loading ? <RefreshCw size={14} className="animate-spin" /> : <Package size={14} />}
                {loading ? 'Saving…' : 'Save to Weight Library'}
              </button>
            </div>
          )}

          {/* Export Format */}
          {tab === 'export' && resolvedWeightId && (
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-slate-400 mb-1.5">Format</label>
                <div className="grid grid-cols-1 gap-1.5">
                  {EXPORT_FORMATS.map(f => (
                    <label
                      key={f.value}
                      className={`flex items-center gap-3 px-3 py-2.5 rounded-lg border cursor-pointer transition-colors ${
                        exportFormat === f.value
                          ? 'border-indigo-500 bg-indigo-500/10 text-white'
                          : 'border-slate-700 hover:border-slate-600 text-slate-400 hover:text-slate-300'
                      }`}
                    >
                      <input
                        type="radio"
                        name="export_format"
                        value={f.value}
                        checked={exportFormat === f.value}
                        onChange={() => setExportFormat(f.value)}
                        className="sr-only"
                      />
                      <span className="text-xs font-mono font-bold w-16">{f.label}</span>
                      <span className="text-xs">{f.desc}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Image Size</label>
                  <input
                    type="number"
                    value={imgsz}
                    min={32}
                    max={1920}
                    step={32}
                    onChange={e => setImgsz(parseInt(e.target.value) || 640)}
                    className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-1.5 text-white text-sm focus:outline-none focus:border-indigo-500"
                  />
                </div>
                <div className="flex flex-col gap-2 justify-end pb-0.5">
                  <label className="flex items-center gap-2 text-xs text-slate-400 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={half}
                      onChange={e => setHalf(e.target.checked)}
                      className="rounded"
                    />
                    FP16 Half
                  </label>
                  {exportFormat === 'onnx' && (
                    <label className="flex items-center gap-2 text-xs text-slate-400 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={simplify}
                        onChange={e => setSimplify(e.target.checked)}
                        className="rounded"
                      />
                      Simplify
                    </label>
                  )}
                </div>
              </div>

              <button
                onClick={handleExport}
                disabled={loading}
                className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors cursor-pointer"
              >
                {loading ? <Loader2 size={14} className="animate-spin" /> : <FileCode2 size={14} />}
                {loading ? 'Exporting…' : `Export as ${exportFormat.toUpperCase()}`}
              </button>
            </div>
          )}

          {/* Result feedback */}
          {result && (
            <div className={`flex items-start gap-2 px-3 py-2.5 rounded-lg text-xs ${
              result.ok
                ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-400'
                : 'bg-red-500/10 border border-red-500/20 text-red-400'
            }`}>
              {result.ok
                ? <CheckCircle2 size={14} className="mt-0.5 shrink-0" />
                : <AlertTriangle size={14} className="mt-0.5 shrink-0" />}
              {result.message}
            </div>
          )}

        </div>
      </div>
    </div>
  );
}
