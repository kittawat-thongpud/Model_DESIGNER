import React, { useState, useEffect, useRef, useCallback } from 'react';
import { api } from '../services/api';
import type {
  WeightRecord, InferenceHistoryEntry, InferenceClassSummary,
  InferResult, InferDetection, SgbgScaleVis,
} from '../types';
import {
  Upload, X, Play, RefreshCw, History, ChevronDown,
  Target, Layers, Trash2, ImageIcon, AlertTriangle,
  SlidersHorizontal, Eye, Brain, Cpu, FlipHorizontal2, Crosshair, Loader2,
} from 'lucide-react';
import { fmtDataset } from '../utils/format';

// ── Helpers ────────────────────────────────────────────────────────────────────

const confColor = (c: number) =>
  c >= 0.8 ? 'text-emerald-400' : c >= 0.5 ? 'text-amber-400' : 'text-red-400';

const barPct = (c: number) => `${Math.round(c * 100)}%`;

// ── Sub-components ─────────────────────────────────────────────────────────────

function SpeedBadge({ label, ms, color }: { label: string; ms: number; color: string }) {
  return (
    <div className="flex flex-col items-center bg-slate-800 rounded-lg px-3 py-2 min-w-[72px]">
      <span className="text-[9px] text-slate-500 uppercase tracking-wide">{label}</span>
      <span className={`text-sm font-bold font-mono ${color}`}>{ms.toFixed(1)}</span>
      <span className="text-[9px] text-slate-600">ms</span>
    </div>
  );
}

function DetectionRow({ det, idx, onQuery, querying, isActive }: {
  det: InferDetection;
  idx: number;
  onQuery: (det: InferDetection) => void;
  querying: boolean;
  isActive: boolean;
}) {
  const colors = [
    'border-indigo-500', 'border-emerald-500', 'border-amber-500',
    'border-pink-500', 'border-cyan-500', 'border-purple-500',
  ];
  const dotColor = colors[idx % colors.length];
  return (
    <tr
      className={`border-b border-slate-800/40 transition-colors cursor-pointer ${
        isActive ? 'bg-violet-900/30 hover:bg-violet-900/40' : 'hover:bg-slate-800/30'
      }`}
      onClick={() => onQuery(det)}
    >
      <td className="py-1.5 pl-2 pr-2">
        <div className="flex items-center gap-1.5">
          <span className={`w-2 h-2 rounded-full border-2 shrink-0 ${dotColor}`} />
          <span className="text-slate-200 text-xs">{det.class_name}</span>
        </div>
      </td>
      <td className={`py-1.5 pr-2 text-right font-mono text-xs ${confColor(det.confidence)}`}>
        {(det.confidence * 100).toFixed(1)}%
      </td>
      <td className="py-1.5 pr-2 text-right font-mono text-[10px] text-slate-500">
        [{det.bbox.map(v => Math.round(v)).join(', ')}]
      </td>
      <td className="py-1.5 pr-2 text-right">
        {isActive && querying
          ? <Loader2 size={11} className="text-violet-400 animate-spin inline" />
          : <Crosshair size={11} className={isActive ? 'text-violet-400' : 'text-slate-700 group-hover:text-slate-500'} />}
      </td>
    </tr>
  );
}

// ── Attention Query Result Panel ───────────────────────────────────────────────

type AttentionScaleResult = {
  scale: string;
  feature_hw: number[];
  query_pixel: number[];
  attention: string;
};

function AttentionQueryPanel({
  detLabel, scales, onClose,
}: {
  detLabel: string;
  scales: Record<string, AttentionScaleResult>;
  onClose: () => void;
}) {
  const entries = Object.entries(scales);
  return (
    <div className="bg-slate-900 border border-violet-700/50 rounded-xl overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b border-slate-800 bg-violet-900/20">
        <div className="flex items-center gap-2">
          <Crosshair size={13} className="text-violet-400" />
          <span className="text-sm font-semibold text-white">Attention Query</span>
          <span className="text-xs text-violet-300 font-mono bg-violet-900/40 px-2 py-0.5 rounded">
            {detLabel}
          </span>
        </div>
        <button onClick={onClose} className="text-slate-500 hover:text-white transition-colors cursor-pointer p-1">
          <X size={13} />
        </button>
      </div>
      <div className="p-3 space-y-3">
        <p className="text-[10px] text-slate-500">
          ★ = token nearest to detection centroid | JET: red = strong attention from that token
        </p>
        {entries.map(([slug, s]) => (
          <div key={slug}>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-semibold text-indigo-400 font-mono">{slug.toUpperCase()}</span>
              <span className="text-[10px] text-slate-500">{s.scale}</span>
              <span className="text-[10px] text-slate-600 font-mono">
                hw={s.feature_hw[0]}×{s.feature_hw[1]}
              </span>
            </div>
            <img
              src={s.attention}
              alt={`attn-${slug}`}
              className="w-full rounded-lg object-contain bg-slate-950"
            />
          </div>
        ))}
      </div>
    </div>
  );
}

function SgbgScalePanel({ slug, vis }: { slug: string; vis: SgbgScaleVis }) {
  const [tab, setTab] = useState<'selection' | 'attention' | 'delta'>('selection');
  const tabs: { key: typeof tab; label: string; icon: React.ReactNode }[] = [
    { key: 'selection', label: 'Selection', icon: <Target size={11} /> },
    { key: 'attention', label: 'Attention', icon: <Brain size={11} /> },
    { key: 'delta', label: 'Δ Refinement', icon: <FlipHorizontal2 size={11} /> },
  ];
  const imgMap: Record<'selection' | 'attention' | 'delta', string> = {
    selection: vis.selection,
    attention: vis.attention,
    delta: vis.delta,
  };
  const imgSrc = imgMap[tab];
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
      {/* Scale header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-indigo-400 font-mono">{slug.toUpperCase()}</span>
          <span className="text-[10px] text-slate-500">{vis.scale}</span>
        </div>
        <div className="flex items-center gap-3 text-[10px] font-mono text-slate-500">
          <span>hw={vis.feature_hw[0]}×{vis.feature_hw[1]}</span>
          <span>k={vis.k}</span>
          <span className={vis.gate > 0.01 ? 'text-emerald-400' : 'text-slate-600'}>
            α={vis.gate.toFixed(4)}
          </span>
        </div>
      </div>

      {/* Tab switcher */}
      <div className="flex border-b border-slate-800">
        {tabs.map(t => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 text-[11px] transition-colors cursor-pointer ${
              tab === t.key
                ? 'bg-indigo-600/20 text-indigo-300 border-b-2 border-indigo-500'
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            {t.icon}{t.label}
          </button>
        ))}
      </div>

      {/* Visualization image */}
      <div className="p-2">
        <img
          src={imgSrc}
          alt={`${slug}-${tab}`}
          className="w-full rounded-lg object-contain bg-slate-950"
        />
        <div className="text-[9px] text-slate-600 mt-1 text-center">
          {tab === 'selection' && 'HOT: high-energy tokens selected for attention'}
          {tab === 'attention' && 'JET: attention influence from query token ★ | k×k matrix inset'}
          {tab === 'delta' && 'INFERNO: Δ magnitude | TWILIGHT: amplify(bright) / suppress(dark)'}
        </div>
      </div>
    </div>
  );
}

function SgbgVisPanel({ sgbg_vis }: { sgbg_vis: InferResult['sgbg_vis'] }) {
  if (!sgbg_vis) return null;
  const errObj = sgbg_vis as { error?: string };
  if (errObj.error) {
    return (
      <div className="text-xs text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-xl p-3 flex gap-2">
        <AlertTriangle size={14} className="shrink-0 mt-0.5" />
        SGBG visualization error: {errObj.error}
      </div>
    );
  }
  const entries = Object.entries(sgbg_vis as Record<string, SgbgScaleVis>);
  if (entries.length === 0) return null;
  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-white flex items-center gap-2">
        <Brain size={14} className="text-violet-400" />
        SGBG Feature Analysis
        <span className="text-[10px] text-slate-500 font-normal">SparseGlobalBlockGated internals</span>
      </h3>
      <div className="flex flex-col gap-4">
        {entries.map(([slug, vis]) => (
          <SgbgScalePanel key={slug} slug={slug} vis={vis} />
        ))}
      </div>
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────────

export default function InferencePage() {
  const [weights, setWeights] = useState<WeightRecord[]>([]);
  const [selectedWeight, setSelectedWeight] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [conf, setConf] = useState(0.25);
  const [iou, setIou] = useState(0.45);
  const [imgsz, setImgsz] = useState(640);
  const [visualizeSgbg, setVisualizeSgbg] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<InferResult | null>(null);
  const [history, setHistory] = useState<InferenceHistoryEntry[]>([]);
  const [activeTab, setActiveTab] = useState<'infer' | 'history'>('infer');

  // Attention query state
  const [attnQuerying, setAttnQuerying] = useState(false);
  const [activeDet, setActiveDet] = useState<InferDetection | null>(null);
  const [attnResult, setAttnResult] = useState<Record<string, AttentionScaleResult> | null>(null);
  const [attnError, setAttnError] = useState<string | null>(null);

  const dropRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.listWeights().then(setWeights).catch(() => {});
    api.getInferenceHistory().then(setHistory).catch(() => {});
  }, []);

  const setImageFile = useCallback((f: File) => {
    setFile(f);
    const url = URL.createObjectURL(f);
    setPreview(prev => { if (prev) URL.revokeObjectURL(prev); return url; });
    setResult(null);
    setError(null);
    setActiveDet(null);
    setAttnResult(null);
    setAttnError(null);
  }, []);

  const handleQueryAttention = async (det: InferDetection) => {
    if (!file || !selectedWeight) return;
    // Toggle off if same detection clicked again
    if (activeDet && activeDet.bbox.join() === det.bbox.join()) {
      setActiveDet(null);
      setAttnResult(null);
      setAttnError(null);
      return;
    }
    setActiveDet(det);
    setAttnQuerying(true);
    setAttnResult(null);
    setAttnError(null);
    try {
      const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
      const r = await api.inferAttention(
        selectedWeight, file, imgsz,
        det.bbox as [number, number, number, number],
        label,
      );
      setAttnResult(r.scales);
    } catch (e: unknown) {
      setAttnError(e instanceof Error ? e.message : 'Attention query failed');
    } finally {
      setAttnQuerying(false);
    }
  };

  const clearFile = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  // Drag-and-drop
  useEffect(() => {
    const el = dropRef.current;
    if (!el) return;
    const onDrop = (e: DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer?.files?.[0];
      if (f && f.type.startsWith('image/')) setImageFile(f);
    };
    const onDragOver = (e: DragEvent) => e.preventDefault();
    el.addEventListener('drop', onDrop);
    el.addEventListener('dragover', onDragOver);
    return () => { el.removeEventListener('drop', onDrop); el.removeEventListener('dragover', onDragOver); };
  }, [setImageFile]);

  const handleRun = async () => {
    if (!selectedWeight || !file) return;
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const r = await api.infer(selectedWeight, file, conf, iou, imgsz, visualizeSgbg);
      setResult(r);
      api.getInferenceHistory().then(setHistory).catch(() => {});
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Inference failed');
    } finally {
      setRunning(false);
    }
  };

  const handleClearHistory = async () => {
    await api.clearInferenceHistory();
    setHistory([]);
  };

  const handleDeleteEntry = async (id: string) => {
    await api.deleteInferenceEntry(id);
    setHistory(prev => prev.filter(h => h.id !== id));
  };

  const hasSgbg = result?.sgbg_vis != null && !('error' in (result.sgbg_vis ?? {}));
  const isHsgDet = result?.is_hsg_det ?? false;

  return (
    <div className="flex-1 overflow-y-auto bg-[#0f1117]">
      <div className="max-w-7xl mx-auto p-6 space-y-6">

        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-5">
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">Inference</h1>
            <p className="text-slate-500 text-sm mt-0.5">Run detection on a single image with full feature visualization</p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setActiveTab('infer')}
              className={`px-3 py-1.5 text-sm rounded-lg border transition-colors cursor-pointer ${activeTab === 'infer' ? 'bg-indigo-600 border-indigo-500 text-white' : 'border-slate-700 text-slate-400 hover:text-white hover:border-slate-600'}`}
            >
              <Play size={13} className="inline mr-1.5" />Infer
            </button>
            <button
              onClick={() => setActiveTab('history')}
              className={`px-3 py-1.5 text-sm rounded-lg border transition-colors cursor-pointer ${activeTab === 'history' ? 'bg-indigo-600 border-indigo-500 text-white' : 'border-slate-700 text-slate-400 hover:text-white hover:border-slate-600'}`}
            >
              <History size={13} className="inline mr-1.5" />History ({history.length})
            </button>
          </div>
        </div>

        {/* ── Infer tab ── */}
        {activeTab === 'infer' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

            {/* ── Left: controls ── */}
            <div className="space-y-4">

              {/* Model select */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3">
                <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                  <Layers size={15} className="text-indigo-400" /> Model
                </h3>
                <div className="relative">
                  <select
                    value={selectedWeight}
                    onChange={e => setSelectedWeight(e.target.value)}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm appearance-none cursor-pointer focus:outline-none focus:border-indigo-500"
                  >
                    <option value="">— select weight —</option>
                    {weights.map(w => (
                      <option key={w.weight_id} value={w.weight_id}>
                        {w.model_name} ({w.weight_id.slice(0, 8)})
                      </option>
                    ))}
                  </select>
                  <ChevronDown size={14} className="absolute right-3 top-2.5 text-slate-400 pointer-events-none" />
                </div>
                {selectedWeight && (() => {
                  const w = weights.find(x => x.weight_id === selectedWeight);
                  return w ? (
                    <div className="text-[10px] text-slate-500 space-y-0.5 border-t border-slate-800 pt-2">
                      <div>Dataset: <span className="text-slate-400">{fmtDataset(w.dataset, w.dataset_name)}</span></div>
                      <div>Epochs: <span className="text-slate-400">{w.total_epochs ?? w.epochs_trained ?? '—'}</span></div>
                      <div>Accuracy: <span className="text-emerald-400">{w.final_accuracy != null ? `${(w.final_accuracy * 100).toFixed(1)}%` : '—'}</span></div>
                    </div>
                  ) : null;
                })()}
              </div>

              {/* Settings */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
                <button
                  onClick={() => setShowSettings(v => !v)}
                  className="w-full flex items-center justify-between px-4 py-3 text-sm font-semibold text-white hover:bg-slate-800/40 transition-colors cursor-pointer"
                >
                  <span className="flex items-center gap-2"><SlidersHorizontal size={14} className="text-slate-400" /> Settings</span>
                  <ChevronDown size={14} className={`text-slate-400 transition-transform ${showSettings ? 'rotate-180' : ''}`} />
                </button>
                {showSettings && (
                  <div className="px-4 pb-4 space-y-3 border-t border-slate-800 pt-3">
                    {[
                      { label: 'Confidence', id: 'conf', val: conf, set: setConf, min: 0.01, max: 1, step: 0.01 },
                      { label: 'IoU Threshold', id: 'iou', val: iou, set: setIou, min: 0.1, max: 1, step: 0.05 },
                      { label: 'Image Size', id: 'imgsz', val: imgsz, set: setImgsz, min: 32, max: 1920, step: 32 },
                    ].map(({ label, id, val, set, min, max, step }) => (
                      <div key={id}>
                        <div className="flex justify-between text-xs text-slate-400 mb-1">
                          <span>{label}</span>
                          <span className="font-mono text-slate-300">{val}</span>
                        </div>
                        <input type="range" min={min} max={max} step={step} value={val}
                          onChange={e => set(parseFloat(e.target.value))}
                          className="w-full accent-indigo-500" />
                      </div>
                    ))}

                    {/* SGBG toggle — shown always so user can enable before first run */}
                    <div className="flex items-center justify-between pt-1 border-t border-slate-800">
                      <div>
                        <div className="text-xs text-slate-300 flex items-center gap-1.5">
                          <Brain size={12} className="text-violet-400" /> SGBG Visualization
                        </div>
                        <div className="text-[10px] text-slate-600">
                          {result && !isHsgDet ? 'Not available — model has no SGBG modules' : 'HSG-DET models only — slower'}
                        </div>
                      </div>
                      <button
                        onClick={() => setVisualizeSgbg(v => !v)}
                        disabled={result != null && !isHsgDet}
                        className={`relative w-9 h-5 rounded-full transition-colors cursor-pointer ${visualizeSgbg ? 'bg-violet-600' : 'bg-slate-700'} disabled:opacity-30 disabled:cursor-not-allowed`}
                      >
                        <span className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition-transform ${visualizeSgbg ? 'translate-x-4' : ''}`} />
                      </button>
                    </div>
                  </div>
                )}
              </div>

              {/* Run button */}
              <button
                onClick={handleRun}
                disabled={running || !selectedWeight || !file}
                className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white font-medium rounded-xl transition-colors cursor-pointer"
              >
                {running
                  ? <><RefreshCw size={15} className="animate-spin" /> Running…</>
                  : <><Play size={15} /> Run Inference</>}
              </button>

              {error && (
                <div className="flex gap-2 text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-xl p-3">
                  <AlertTriangle size={15} className="mt-0.5 shrink-0" />{error}
                </div>
              )}

              {/* Result summary */}
              {result && (
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3">
                  <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                    <Target size={14} className="text-emerald-400" /> Result
                  </h3>
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      { label: 'Detections', value: result.total_detections, color: 'text-emerald-400' },
                      { label: 'Total', value: `${result.elapsed_ms.toFixed(0)} ms`, color: 'text-white' },
                    ].map(({ label, value, color }) => (
                      <div key={label} className="bg-slate-800 rounded-lg p-2.5 text-center">
                        <div className="text-[10px] text-slate-500 uppercase">{label}</div>
                        <div className={`text-base font-bold font-mono ${color}`}>{value}</div>
                      </div>
                    ))}
                  </div>
                  {/* Speed breakdown */}
                  <div className="flex gap-1.5 justify-center flex-wrap border-t border-slate-800 pt-3">
                    <SpeedBadge label="Pre" ms={result.speed.preprocess_ms} color="text-slate-300" />
                    <SpeedBadge label="Infer" ms={result.speed.inference_ms} color="text-amber-400" />
                    <SpeedBadge label="Post" ms={result.speed.postprocess_ms} color="text-slate-300" />
                  </div>
                  {hasSgbg && (
                    <div className="text-[10px] text-violet-400 flex items-center gap-1 justify-center">
                      <Brain size={10} /> SGBG visualization ready — see panels below
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* ── Right: image + results ── */}
            <div className="lg:col-span-2 space-y-4">

              {/* Drop zone / preview */}
              {!file ? (
                <div
                  ref={dropRef}
                  onClick={() => fileRef.current?.click()}
                  className="border-2 border-dashed border-slate-700 hover:border-indigo-500 rounded-xl p-10 text-center cursor-pointer transition-colors bg-slate-900/30 hover:bg-slate-900/60"
                >
                  <input ref={fileRef} type="file" accept="image/*" className="hidden"
                    onChange={e => { const f = e.target.files?.[0]; if (f) setImageFile(f); }} />
                  <Upload size={32} className="mx-auto text-slate-600 mb-3" />
                  <p className="text-slate-400 text-sm">Drop an image here or <span className="text-indigo-400 underline">browse</span></p>
                  <p className="text-slate-600 text-xs mt-1">Single image — JPEG, PNG, WebP</p>
                </div>
              ) : (
                <div className="relative bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
                  {/* filename bar */}
                  <div className="flex items-center justify-between px-3 py-2 border-b border-slate-800">
                    <div className="flex items-center gap-2 text-xs text-slate-400">
                      <ImageIcon size={12} className="text-slate-500" />
                      <span className="truncate max-w-xs">{file.name}</span>
                      <span className="text-slate-600">({(file.size / 1024).toFixed(0)} KB)</span>
                    </div>
                    <button onClick={clearFile} className="text-slate-600 hover:text-red-400 transition-colors cursor-pointer p-1">
                      <X size={13} />
                    </button>
                  </div>
                  {/* annotated result or original preview */}
                  <img
                    src={result?.image_b64 ?? preview ?? ''}
                    alt="inference"
                    className="w-full object-contain max-h-[480px] bg-slate-950"
                  />
                  {result && (
                    <div className="absolute top-10 right-2 bg-black/70 text-emerald-400 text-[10px] font-mono px-2 py-1 rounded">
                      {result.total_detections} obj
                    </div>
                  )}
                </div>
              )}

              {/* Detection table */}
              {result && (
                <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
                  <div className="px-3 py-2 border-b border-slate-800 flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                      <Eye size={14} className="text-indigo-400" /> Detections
                      <span className="text-slate-500 font-normal text-xs">{result.detections.length} objects</span>
                    </h3>
                    <div className="flex items-center gap-2 text-[10px] text-slate-500">
                      <Cpu size={10} className="text-amber-400" />
                      <span>{result.speed.inference_ms.toFixed(1)} ms inference</span>
                    </div>
                  </div>
                  {result.detections.length > 0 ? (
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="text-[10px] text-slate-500 uppercase border-b border-slate-800">
                            <th className="text-left px-3 py-2">Class</th>
                            <th className="text-right px-3 py-2">Confidence</th>
                            <th className="text-right px-3 py-2">BBox [x1,y1,x2,y2]</th>
                            {isHsgDet && <th className="text-right px-3 py-2 text-violet-600"><Crosshair size={10} className="inline" /></th>}
                          </tr>
                        </thead>
                        <tbody>
                          {result.detections.map((d: InferDetection, i: number) => (
                            <DetectionRow
                              key={i} det={d} idx={i}
                              onQuery={isHsgDet ? handleQueryAttention : () => {}}
                              querying={attnQuerying}
                              isActive={activeDet?.bbox.join() === d.bbox.join()}
                            />
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="text-center text-slate-600 text-sm py-6">No objects detected above threshold</div>
                  )}

                  {/* Confidence bars per class */}
                  {result.detections.length > 0 && (
                    <div className="px-3 pb-3 pt-2 border-t border-slate-800 space-y-1.5">
                      <div className="text-[10px] text-slate-600 uppercase tracking-wider">Confidence distribution</div>
                      {result.detections.map((d, i) => (
                        <div key={i} className="flex items-center gap-2">
                          <span className="text-[10px] text-slate-400 w-28 truncate shrink-0">{d.class_name}</span>
                          <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                            <div className="h-full bg-indigo-500 rounded-full" style={{ width: barPct(d.confidence) }} />
                          </div>
                          <span className={`text-[10px] font-mono w-10 text-right shrink-0 ${confColor(d.confidence)}`}>
                            {(d.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Attention query result */}
              {attnQuerying && !attnResult && (
                <div className="bg-slate-900 border border-violet-700/40 rounded-xl p-4 flex items-center gap-3">
                  <Loader2 size={16} className="text-violet-400 animate-spin shrink-0" />
                  <div>
                    <div className="text-sm text-white">Querying attention…</div>
                    <div className="text-[10px] text-slate-500">
                      {activeDet?.class_name} — running inference with SGBG hooks
                    </div>
                  </div>
                </div>
              )}
              {attnError && (
                <div className="flex gap-2 text-xs text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-xl p-3">
                  <AlertTriangle size={13} className="shrink-0 mt-0.5" />{attnError}
                </div>
              )}
              {attnResult && activeDet && (
                <AttentionQueryPanel
                  detLabel={`${activeDet.class_name} ${(activeDet.confidence * 100).toFixed(0)}%`}
                  scales={attnResult}
                  onClose={() => { setActiveDet(null); setAttnResult(null); }}
                />
              )}

              {/* SGBG panels */}
              {result?.sgbg_vis && <SgbgVisPanel sgbg_vis={result.sgbg_vis} />}
            </div>
          </div>
        )}

        {/* ── History tab ── */}
        {activeTab === 'history' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-white font-semibold">{history.length} run{history.length !== 1 ? 's' : ''}</h2>
              {history.length > 0 && (
                <button onClick={handleClearHistory}
                  className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-red-400 transition-colors cursor-pointer">
                  <Trash2 size={12} /> Clear All
                </button>
              )}
            </div>

            {history.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-20 text-slate-600 gap-3">
                <History size={40} className="text-slate-800" />
                <span>No inference history yet</span>
              </div>
            ) : (
              <div className="space-y-2">
                {history.map(entry => {
                  const w = weights.find(x => x.weight_id === entry.weight_id);
                  return (
                    <div key={entry.id} className="bg-slate-900 border border-slate-800 rounded-xl p-4">
                      <div className="flex items-start justify-between">
                        <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            <span className="text-white text-sm font-medium">{w?.model_name || entry.weight_id.slice(0, 12)}</span>
                            <span className="text-[10px] text-slate-500 font-mono">{entry.weight_id.slice(0, 8)}</span>
                          </div>
                          <div className="text-xs text-slate-500">
                            {new Date(entry.timestamp).toLocaleString()} · <span className="text-slate-400">{entry.source_name}</span>
                          </div>
                          <div className="flex items-center gap-4 text-xs pt-1">
                            <span className="text-emerald-400 font-mono">{entry.total_detections} det</span>
                            <span className="text-blue-400 font-mono">{entry.avg_latency_ms.toFixed(0)} ms</span>
                            <span className="text-amber-400 font-mono">{entry.fps.toFixed(1)} FPS</span>
                            <span className="text-slate-500">conf={entry.conf} iou={entry.iou}</span>
                          </div>
                        </div>
                        <button onClick={() => handleDeleteEntry(entry.id)}
                          className="text-slate-600 hover:text-red-400 transition-colors cursor-pointer p-1">
                          <Trash2 size={14} />
                        </button>
                      </div>
                      {entry.class_summary.length > 0 && (
                        <div className="flex flex-wrap gap-1.5 mt-3 pt-3 border-t border-slate-800">
                          {entry.class_summary.map((cls: InferenceClassSummary) => (
                            <span key={cls.class_id} className="px-2 py-0.5 bg-slate-800 text-slate-300 text-[10px] rounded-full font-mono">
                              {cls.class_name} <span className="text-indigo-400">×{cls.count}</span>
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
