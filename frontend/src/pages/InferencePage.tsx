import { useState, useEffect, useRef, useCallback } from 'react';
import { api } from '../services/api';
import type {
  WeightRecord, InferenceResult, InferenceImageResult,
  InferenceHistoryEntry, InferenceClassSummary, InferenceDetection,
} from '../types';
import {
  Upload, X, Play, RefreshCw, History, ChevronDown, ChevronRight,
  Zap, Timer, Target, Layers, Trash2, ImageIcon, AlertTriangle,
  SlidersHorizontal, CheckCircle2, Eye,
} from 'lucide-react';
import { fmtDataset } from '../utils/format';

// ── Helpers ───────────────────────────────────────────────────────────────────

const confColor = (c: number) =>
  c >= 0.8 ? 'text-emerald-400' : c >= 0.5 ? 'text-amber-400' : 'text-red-400';

const barWidth = (c: number) => `${Math.round(c * 100)}%`;

function ClassBar({ cls }: { cls: InferenceClassSummary }) {
  return (
    <div className="space-y-0.5">
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-300 truncate max-w-[120px]">{cls.class_name}</span>
        <div className="flex items-center gap-2 shrink-0">
          <span className="text-slate-500 text-[10px]">×{cls.count}</span>
          <span className={`font-mono text-[10px] ${confColor(cls.avg_conf)}`}>{(cls.avg_conf * 100).toFixed(0)}%</span>
        </div>
      </div>
      <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
        <div className="h-full bg-indigo-500 rounded-full" style={{ width: barWidth(cls.avg_conf) }} />
      </div>
    </div>
  );
}

function DetectionBox({ det, scale }: {
  det: { class_name: string; confidence: number; bbox: number[] };
  scale: { x: number; y: number };
}) {
  const [x1, y1, x2, y2] = det.bbox;
  return (
    <div
      className="absolute border-2 border-indigo-500 rounded pointer-events-none"
      style={{
        left: x1 * scale.x,
        top: y1 * scale.y,
        width: (x2 - x1) * scale.x,
        height: (y2 - y1) * scale.y,
      }}
    >
      <span
        className="absolute -top-5 left-0 bg-indigo-600 text-white text-[9px] px-1 py-0.5 rounded whitespace-nowrap"
        style={{ fontSize: 9 }}
      >
        {det.class_name} {(det.confidence * 100).toFixed(0)}%
      </span>
    </div>
  );
}

function DetectionRow({ det, idx }: {
  det: InferenceDetection;
  idx: number;
}) {
  const [open, setOpen] = useState(false);
  const hasTopClasses = det.top_classes && det.top_classes.length > 1;

  return (
    <>
      <tr
        className={`border-b border-slate-800/50 transition-colors ${hasTopClasses ? 'cursor-pointer hover:bg-slate-800/40' : 'hover:bg-slate-800/20'}`}
        onClick={() => hasTopClasses && setOpen(v => !v)}
      >
        <td className="py-1.5 pr-3">
          <div className="flex items-center gap-1.5">
            {hasTopClasses ? (
              open
                ? <ChevronDown size={11} className="text-indigo-400 shrink-0" />
                : <ChevronRight size={11} className="text-slate-500 shrink-0" />
            ) : (
              <span className="w-[11px] shrink-0" />
            )}
            <span className="text-slate-200">{det.class_name}</span>
          </div>
        </td>
        <td className={`py-1.5 pr-3 text-right font-mono ${confColor(det.confidence)}`}>
          {(det.confidence * 100).toFixed(1)}%
        </td>
        <td className="py-1.5 text-right font-mono text-slate-500 text-[10px]">
          [{det.bbox.map((v: number) => v.toFixed(0)).join(', ')}]
        </td>
      </tr>

      {/* Top-classes dropdown */}
      {open && hasTopClasses && (
        <tr className="bg-slate-950/60">
          <td colSpan={3} className="px-2 pb-2 pt-1">
            <div className="ml-4 space-y-1">
              <div className="text-[10px] text-slate-600 uppercase tracking-wider mb-1.5">
                Class distribution (top {det.top_classes.length})
              </div>
              {det.top_classes.map((tc: { class_id: number; class_name: string; score: number }, ti: number) => {
                const isTop = tc.class_id === det.class_id;
                const pct = Math.round(tc.score * 100);
                return (
                  <div key={ti} className="flex items-center gap-2">
                    <span
                      className={`text-[10px] w-28 truncate shrink-0 ${isTop ? 'text-white font-semibold' : 'text-slate-400'}`}
                    >
                      {isTop && <span className="text-indigo-400 mr-1">▶</span>}
                      {tc.class_name}
                    </span>
                    <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all ${isTop ? 'bg-indigo-500' : 'bg-slate-600'}`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <span className={`text-[10px] font-mono w-10 text-right shrink-0 ${confColor(tc.score)}`}>
                      {pct}%
                    </span>
                  </div>
                );
              })}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

function ImageResultCard({ imgResult, index, showAnnotated }: {
  imgResult: InferenceImageResult;
  index: number;
  showAnnotated: boolean;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
      <div
        className="flex items-center justify-between px-3 py-2 cursor-pointer hover:bg-slate-800/40 transition-colors"
        onClick={() => setExpanded(v => !v)}
      >
        <div className="flex items-center gap-2 text-sm">
          {expanded ? <ChevronDown size={14} className="text-slate-400" /> : <ChevronRight size={14} className="text-slate-400" />}
          <ImageIcon size={14} className="text-slate-500" />
          <span className="text-slate-300 font-medium">Image {index + 1}</span>
          <span className="text-slate-500 text-xs">{imgResult.detections.length} detections</span>
        </div>
        <div className="flex items-center gap-3 text-[10px] font-mono text-slate-500">
          <span><Timer size={9} className="inline mr-0.5 text-blue-400" />{imgResult.total_ms.toFixed(0)} ms</span>
          <span><Zap size={9} className="inline mr-0.5 text-amber-400" />{imgResult.inference_ms.toFixed(0)} ms</span>
        </div>
      </div>

      {expanded && (
        <div className="border-t border-slate-800 p-3 space-y-3">
          {/* Annotated image */}
          {imgResult.image_b64 && (
            <div className="relative">
              <img
                src={imgResult.image_b64}
                alt={`result-${index}`}
                className="w-full rounded-lg object-contain max-h-96 bg-slate-950"
              />
            </div>
          )}

          {/* Detections table */}
          {imgResult.detections.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-[10px] text-slate-500 uppercase border-b border-slate-800">
                    <th className="text-left py-1.5 pr-3">Class</th>
                    <th className="text-right py-1.5 pr-3">Confidence</th>
                    <th className="text-right py-1.5">BBox (x1,y1,x2,y2)</th>
                  </tr>
                </thead>
                <tbody>
                  {imgResult.detections.map((d, i) => (
                    <DetectionRow key={i} det={d} idx={i} />
                  ))}
                </tbody>
              </table>
              <p className="text-[10px] text-slate-600 mt-1.5 pl-1">
                ▶ Click a row to expand class distribution
              </p>
            </div>
          ) : (
            <p className="text-xs text-slate-600 text-center py-2">No detections</p>
          )}
        </div>
      )}
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────

export default function InferencePage() {
  const [weights, setWeights] = useState<WeightRecord[]>([]);
  const [selectedWeight, setSelectedWeight] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [conf, setConf] = useState(0.25);
  const [iou, setIou] = useState(0.45);
  const [imgsz, setImgsz] = useState(640);
  const [showSettings, setShowSettings] = useState(false);

  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [history, setHistory] = useState<InferenceHistoryEntry[]>([]);
  const [activeTab, setActiveTab] = useState<'predict' | 'history'>('predict');
  const [showAnnotated] = useState(true);

  const dropRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.listWeights().then(setWeights).catch(() => {});
    api.getInferenceHistory().then(setHistory).catch(() => {});
  }, []);

  const addFiles = useCallback((newFiles: FileList | File[]) => {
    const arr = Array.from(newFiles).filter(f => f.type.startsWith('image/'));
    setFiles(prev => {
      const combined = [...prev, ...arr].slice(0, 32);
      // Create previews
      combined.forEach((f, i) => {
        if (i >= prev.length) {
          const url = URL.createObjectURL(f);
          setPreviews(p => {
            const next = [...p];
            next[i] = url;
            return next;
          });
        }
      });
      return combined;
    });
  }, []);

  const removeFile = (i: number) => {
    URL.revokeObjectURL(previews[i] || '');
    setFiles(prev => prev.filter((_, idx) => idx !== i));
    setPreviews(prev => prev.filter((_, idx) => idx !== i));
  };

  // Drag and drop
  useEffect(() => {
    const el = dropRef.current;
    if (!el) return;
    const onDrop = (e: DragEvent) => {
      e.preventDefault();
      if (e.dataTransfer?.files) addFiles(e.dataTransfer.files);
    };
    const onDragOver = (e: DragEvent) => e.preventDefault();
    el.addEventListener('drop', onDrop);
    el.addEventListener('dragover', onDragOver);
    return () => { el.removeEventListener('drop', onDrop); el.removeEventListener('dragover', onDragOver); };
  }, [addFiles]);

  const handleRun = async () => {
    if (!selectedWeight || files.length === 0) return;
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const r = await api.predictImages(selectedWeight, files, conf, iou, imgsz);
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

  return (
    <div className="flex-1 overflow-y-auto bg-[#0f1117]">
      <div className="max-w-7xl mx-auto p-6 space-y-6">

        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-5">
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">Inference</h1>
            <p className="text-slate-500 text-sm mt-0.5">Test your trained models on images</p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setActiveTab('predict')}
              className={`px-3 py-1.5 text-sm rounded-lg border transition-colors cursor-pointer ${activeTab === 'predict' ? 'bg-indigo-600 border-indigo-500 text-white' : 'border-slate-700 text-slate-400 hover:text-white hover:border-slate-600'}`}
            >
              <Play size={13} className="inline mr-1.5" />Predict
            </button>
            <button
              onClick={() => setActiveTab('history')}
              className={`px-3 py-1.5 text-sm rounded-lg border transition-colors cursor-pointer ${activeTab === 'history' ? 'bg-indigo-600 border-indigo-500 text-white' : 'border-slate-700 text-slate-400 hover:text-white hover:border-slate-600'}`}
            >
              <History size={13} className="inline mr-1.5" />History ({history.length})
            </button>
          </div>
        </div>

        {activeTab === 'predict' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

            {/* Left column: config */}
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
                      <div>Epochs: <span className="text-slate-400">{w.total_epochs || w.epochs_trained || '—'}</span></div>
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
                  <div className="px-4 pb-4 space-y-3 border-t border-slate-800">
                    {[
                      { label: 'Confidence', key: 'conf', val: conf, set: setConf, min: 0.01, max: 1, step: 0.01 },
                      { label: 'IoU Threshold', key: 'iou', val: iou, set: setIou, min: 0.1, max: 1, step: 0.05 },
                      { label: 'Image Size', key: 'imgsz', val: imgsz, set: setImgsz, min: 32, max: 1920, step: 32 },
                    ].map(({ label, key, val, set, min, max, step }) => (
                      <div key={key}>
                        <div className="flex justify-between text-xs text-slate-400 mb-1">
                          <span>{label}</span>
                          <span className="font-mono text-slate-300">{val}</span>
                        </div>
                        <input
                          type="range"
                          min={min}
                          max={max}
                          step={step}
                          value={val}
                          onChange={e => set(parseFloat(e.target.value) as any)}
                          className="w-full accent-indigo-500"
                        />
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Run button */}
              <button
                onClick={handleRun}
                disabled={running || !selectedWeight || files.length === 0}
                className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white font-medium rounded-xl transition-colors cursor-pointer"
              >
                {running
                  ? <><RefreshCw size={15} className="animate-spin" /> Running inference…</>
                  : <><Play size={15} /> Run Inference ({files.length} image{files.length !== 1 ? 's' : ''})</>}
              </button>

              {error && (
                <div className="flex gap-2 text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-xl p-3">
                  <AlertTriangle size={15} className="mt-0.5 shrink-0" /> {error}
                </div>
              )}

              {/* Summary stats when result available */}
              {result && (
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3">
                  <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                    <Target size={14} className="text-emerald-400" /> Summary
                  </h3>
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      { label: 'Images', value: result.image_count, color: 'text-white' },
                      { label: 'Detections', value: result.total_detections, color: 'text-emerald-400' },
                      { label: 'Avg Latency', value: `${result.avg_latency_ms.toFixed(0)} ms`, color: 'text-blue-400' },
                      { label: 'FPS', value: result.fps.toFixed(1), color: 'text-amber-400' },
                    ].map(({ label, value, color }) => (
                      <div key={label} className="bg-slate-800 rounded-lg p-2.5 text-center">
                        <div className="text-[10px] text-slate-500 uppercase">{label}</div>
                        <div className={`text-base font-bold font-mono ${color}`}>{value}</div>
                      </div>
                    ))}
                  </div>

                  {/* Class summary */}
                  {result.class_summary.length > 0 && (
                    <div className="space-y-2 border-t border-slate-800 pt-3">
                      <div className="text-[10px] text-slate-500 uppercase tracking-wider">Detected Classes</div>
                      {result.class_summary.map(cls => <ClassBar key={cls.class_id} cls={cls} />)}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Right column: upload + results */}
            <div className="lg:col-span-2 space-y-4">

              {/* Drop zone */}
              <div
                ref={dropRef}
                onClick={() => fileRef.current?.click()}
                className="border-2 border-dashed border-slate-700 hover:border-indigo-500 rounded-xl p-6 text-center cursor-pointer transition-colors bg-slate-900/30 hover:bg-slate-900/60"
              >
                <input
                  ref={fileRef}
                  type="file"
                  accept="image/*"
                  multiple
                  className="hidden"
                  onChange={e => e.target.files && addFiles(e.target.files)}
                />
                <Upload size={28} className="mx-auto text-slate-600 mb-2" />
                <p className="text-slate-400 text-sm">Drop images here or <span className="text-indigo-400 underline">browse</span></p>
                <p className="text-slate-600 text-xs mt-1">Up to 32 images — JPEG, PNG, WebP</p>
              </div>

              {/* Thumbnail grid */}
              {files.length > 0 && (
                <div className="grid grid-cols-4 sm:grid-cols-6 gap-2">
                  {files.map((f, i) => (
                    <div key={i} className="relative group aspect-square">
                      <img
                        src={previews[i] || ''}
                        alt={f.name}
                        className="w-full h-full object-cover rounded-lg bg-slate-800"
                      />
                      <button
                        onClick={() => removeFile(i)}
                        className="absolute top-0.5 right-0.5 bg-black/70 text-white rounded p-0.5 opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
                      >
                        <X size={10} />
                      </button>
                    </div>
                  ))}
                  <button
                    onClick={() => fileRef.current?.click()}
                    className="aspect-square rounded-lg border-2 border-dashed border-slate-700 hover:border-indigo-500 flex items-center justify-center text-slate-600 hover:text-indigo-400 transition-colors cursor-pointer"
                  >
                    <Upload size={18} />
                  </button>
                </div>
              )}

              {/* Per-image results */}
              {result && result.images.length > 0 && (
                <div className="space-y-2">
                  <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                    <Eye size={14} className="text-indigo-400" /> Results — {result.images.length} image{result.images.length !== 1 ? 's' : ''}
                  </h3>
                  {result.images.map((img, i) => (
                    <ImageResultCard key={i} imgResult={img} index={i} showAnnotated={showAnnotated} />
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* History tab */}
        {activeTab === 'history' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-white font-semibold">{history.length} inference run{history.length !== 1 ? 's' : ''}</h2>
              {history.length > 0 && (
                <button
                  onClick={handleClearHistory}
                  className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-red-400 transition-colors cursor-pointer"
                >
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
                            {new Date(entry.timestamp).toLocaleString()} · {entry.image_count} image{entry.image_count !== 1 ? 's' : ''} · <span className="text-slate-400">{entry.source_name}</span>
                          </div>
                          <div className="flex items-center gap-4 text-xs pt-1">
                            <span className="text-emerald-400 font-mono">{entry.total_detections} detections</span>
                            <span className="text-blue-400 font-mono">{entry.avg_latency_ms.toFixed(0)} ms/img</span>
                            <span className="text-amber-400 font-mono">{entry.fps.toFixed(1)} FPS</span>
                            <span className="text-slate-500">conf={entry.conf} iou={entry.iou} imgsz={entry.imgsz}</span>
                          </div>
                        </div>
                        <button
                          onClick={() => handleDeleteEntry(entry.id)}
                          className="text-slate-600 hover:text-red-400 transition-colors cursor-pointer p-1"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>

                      {/* Class summary chips */}
                      {entry.class_summary.length > 0 && (
                        <div className="flex flex-wrap gap-1.5 mt-3 pt-3 border-t border-slate-800">
                          {entry.class_summary.map(cls => (
                            <span
                              key={cls.class_id}
                              className="px-2 py-0.5 bg-slate-800 text-slate-300 text-[10px] rounded-full font-mono"
                            >
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
