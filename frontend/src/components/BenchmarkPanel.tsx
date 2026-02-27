import { useState, useEffect } from 'react';
import { api } from '../services/api';
import type { BenchmarkResult, WeightRecord } from '../types';
import {
  BarChart2, X, RefreshCw, ChevronDown, AlertTriangle,
  Target, Activity, Zap, Cpu, Trash2, Database,
} from 'lucide-react';
import { fmtDataset } from '../utils/format';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts';

interface Props {
  weightId: string;
  onClose: () => void;
  inline?: boolean;
}

const pct = (v: number | null | undefined) =>
  v != null ? `${(v * 100).toFixed(1)}%` : '—';
const fmt4 = (v: number | null | undefined) =>
  v != null ? v.toFixed(4) : '—';

function ConfusionMatrix({ data }: { data: { matrix: number[][]; names: string[] } }) {
  const { matrix, names } = data;
  if (!matrix || matrix.length === 0) return null;
  const maxVal = Math.max(...matrix.flat());

  return (
    <div className="overflow-auto">
      <div className="text-[10px] text-slate-500 mb-2">Confusion Matrix (row=actual, col=predicted)</div>
      <table className="text-[10px] border-collapse">
        <thead>
          <tr>
            <th className="w-16 pr-1 text-right text-slate-600"></th>
            {names.map((n, i) => (
              <th key={i} className="text-center text-slate-400 px-1 pb-1 max-w-[48px] truncate" style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', height: 60 }}>
                {n.length > 8 ? n.slice(0, 8) + '…' : n}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, ri) => (
            <tr key={ri}>
              <td className="text-right text-slate-400 pr-2 text-[10px] max-w-[60px] truncate">{(names[ri] || String(ri)).slice(0, 8)}</td>
              {row.map((val, ci) => {
                const intensity = maxVal > 0 ? val / maxVal : 0;
                const bg = ri === ci
                  ? `rgba(99,102,241,${Math.max(0.1, intensity)})`
                  : `rgba(239,68,68,${Math.max(0, intensity * 0.7)})`;
                return (
                  <td
                    key={ci}
                    className="text-center text-[9px] font-mono w-8 h-7 border border-slate-800/50"
                    style={{ background: bg, color: intensity > 0.5 ? 'white' : '#94a3b8' }}
                    title={`${names[ri] || ri} → ${names[ci] || ci}: ${val}`}
                  >
                    {val > 0 ? val : ''}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MetricBadge({ label, value, color = 'text-white' }: { label: string; value: string; color?: string }) {
  return (
    <div className="bg-slate-800 rounded-lg px-3 py-2 text-center">
      <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-0.5">{label}</div>
      <div className={`text-base font-bold font-mono ${color}`}>{value}</div>
    </div>
  );
}

interface DatasetOption {
  label: string;
  value: string;
  nc: number | null;
  source: string;
}

export default function BenchmarkPanel({ weightId, onClose, inline = false }: Props) {
  const [weightMeta, setWeightMeta] = useState<WeightRecord | null>(null);
  const [datasets, setDatasets] = useState<DatasetOption[]>([]);
  const [dataset, setDataset] = useState('');
  const [split, setSplit] = useState('val');
  const [conf, setConf] = useState(0.001);
  const [iou, setIou] = useState(0.6);
  const [imgsz, setImgsz] = useState(640);
  const [batch, setBatch] = useState(16);

  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BenchmarkResult | null>(null);
  const [history, setHistory] = useState<BenchmarkResult[]>([]);
  const [activeResult, setActiveResult] = useState<BenchmarkResult | null>(null);
  const [sortKey, setSortKey] = useState<'class_name' | 'ap50' | 'ap50_95' | 'precision' | 'recall' | 'f1'>('ap50');
  const [sortAsc, setSortAsc] = useState(false);

  const handleSort = (key: typeof sortKey) => {
    if (sortKey === key) setSortAsc(a => !a);
    else { setSortKey(key); setSortAsc(false); }
  };

  useEffect(() => {
    api.getWeight(weightId).then(setWeightMeta).catch(() => {});
    api.listBenchmarkDatasets(weightId).then(setDatasets).catch(() => {});
    api.listBenchmarks(weightId).then(h => {
      setHistory(h);
      if (h.length > 0) setActiveResult(h[0]);
    }).catch(() => {});
  }, [weightId]);

  const handleRun = async () => {
    if (!dataset) return;
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const r = await api.runBenchmark({ weight_id: weightId, dataset, split, conf, iou, imgsz, batch });
      setResult(r);
      setActiveResult(r);
      setHistory(prev => [r, ...prev]);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Benchmark failed');
    } finally {
      setRunning(false);
    }
  };

  const handleDelete = async (id: string) => {
    await api.deleteBenchmark(id).catch(() => {});
    setHistory(prev => prev.filter(h => h.benchmark_id !== id));
    if (activeResult?.benchmark_id === id) setActiveResult(history[1] || null);
  };

  const displayResult = activeResult;

  const panelBody = (
    <>
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800 shrink-0">
          <div className="flex items-center gap-3">
            <BarChart2 size={16} className="text-indigo-400 shrink-0" />
            <div>
              <h3 className="text-white font-semibold leading-tight">
                {weightMeta ? weightMeta.model_name : 'Benchmark'}
              </h3>
              {weightMeta && (
                <div className="flex items-center gap-2 text-[11px] text-slate-500 mt-0.5">
                  <Database size={10} />
                  <span>{fmtDataset(weightMeta.dataset, weightMeta.dataset_name)}</span>
                  {weightMeta.epochs_trained > 0 && <><span>·</span><span>{weightMeta.epochs_trained} ep</span></>}
                </div>
              )}
            </div>
          </div>
          {!inline && (
            <button onClick={onClose} className="text-slate-500 hover:text-white cursor-pointer"><X size={16} /></button>
          )}
        </div>

        <div className="flex flex-1 min-h-0 overflow-hidden">

          {/* Left: Config + Run */}
          <div className="w-64 shrink-0 border-r border-slate-800 p-4 space-y-3 overflow-y-auto">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Dataset</label>
              <div className="relative">
                <select
                  value={dataset}
                  onChange={e => setDataset(e.target.value)}
                  className="w-full bg-slate-800 border border-slate-700 rounded px-2.5 py-1.5 text-white text-xs appearance-none cursor-pointer focus:outline-none focus:border-indigo-500"
                >
                  <option value="">— select dataset —</option>
                  {datasets.map(d => (
                    <option key={d.value} value={d.value}>
                      {d.source === 'job' ? `📋 ${d.label}` : `📁 ${d.label}`}{d.nc != null ? ` (${d.nc} cls)` : ''}
                    </option>
                  ))}
                </select>
                <ChevronDown size={12} className="absolute right-2 top-2 text-slate-400 pointer-events-none" />
              </div>
            </div>

            <div>
              <label className="block text-xs text-slate-400 mb-1">Split</label>
              <div className="relative">
                <select
                  value={split}
                  onChange={e => setSplit(e.target.value)}
                  className="w-full bg-slate-800 border border-slate-700 rounded px-2.5 py-1.5 text-white text-xs appearance-none cursor-pointer focus:outline-none focus:border-indigo-500"
                >
                  <option value="val">val</option>
                  <option value="test">test</option>
                  <option value="train">train</option>
                </select>
                <ChevronDown size={12} className="absolute right-2 top-2 text-slate-400 pointer-events-none" />
              </div>
            </div>

            {[
              { label: 'Conf Thresh', val: conf, set: setConf, min: 0.001, max: 1, step: 0.001 },
              { label: 'IoU Thresh', val: iou, set: setIou, min: 0.1, max: 1, step: 0.05 },
              { label: 'Image Size', val: imgsz, set: setImgsz, min: 32, max: 1920, step: 32 },
              { label: 'Batch Size', val: batch, set: setBatch, min: 1, max: 64, step: 1 },
            ].map(({ label, val, set, min, max, step }) => (
              <div key={label}>
                <label className="block text-xs text-slate-400 mb-1">{label}</label>
                <input
                  type="number"
                  value={val}
                  min={min}
                  max={max}
                  step={step}
                  onChange={e => set(parseFloat(e.target.value) as any)}
                  className="w-full bg-slate-800 border border-slate-700 rounded px-2.5 py-1.5 text-white text-xs focus:outline-none focus:border-indigo-500"
                />
              </div>
            ))}

            {error && (
              <div className="flex gap-2 text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded p-2">
                <AlertTriangle size={12} className="mt-0.5 shrink-0" /> {error}
              </div>
            )}

            <button
              onClick={handleRun}
              disabled={running || !dataset}
              className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs font-medium rounded-lg transition-colors cursor-pointer"
            >
              {running
                ? <><RefreshCw size={12} className="animate-spin" /> Running…</>
                : <><BarChart2 size={12} /> Run Benchmark</>}
            </button>

            {/* History list */}
            {history.length > 0 && (
              <div className="pt-2 space-y-1">
                <div className="text-[10px] text-slate-500 uppercase tracking-wider">Past Runs</div>
                {history.map(h => (
                  <div
                    key={h.benchmark_id}
                    onClick={() => setActiveResult(h)}
                    className={`flex items-center justify-between px-2 py-1.5 rounded cursor-pointer text-xs transition-colors ${
                      activeResult?.benchmark_id === h.benchmark_id
                        ? 'bg-indigo-500/20 text-indigo-300'
                        : 'hover:bg-slate-800 text-slate-400'
                    }`}
                  >
                    <div>
                      <div className="font-medium">{h.dataset}/{h.split}</div>
                      <div className="text-[10px] text-slate-500">{new Date(h.timestamp).toLocaleDateString()}</div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-emerald-400">{pct(h.mAP50)}</span>
                      <button
                        onClick={ev => { ev.stopPropagation(); handleDelete(h.benchmark_id); }}
                        className="text-slate-600 hover:text-red-400 cursor-pointer"
                      >
                        <Trash2 size={11} />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Right: Results */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {running && (
              <div className="flex flex-col items-center justify-center h-48 gap-3 text-slate-400">
                <RefreshCw size={28} className="animate-spin text-indigo-400" />
                <span className="text-sm">Running validation… this may take a few minutes</span>
              </div>
            )}

            {!running && !displayResult && (
              <div className="flex flex-col items-center justify-center h-48 gap-2 text-slate-500">
                <BarChart2 size={32} className="text-slate-700" />
                <span className="text-sm">Select a dataset and run benchmark</span>
              </div>
            )}

            {!running && displayResult && (
              <>
                {/* Overall metrics */}
                <div>
                  <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">Overall — {displayResult.dataset}/{displayResult.split}</div>
                  <div className="grid grid-cols-4 gap-2">
                    <MetricBadge label="mAP@0.5" value={pct(displayResult.mAP50)} color="text-emerald-400" />
                    <MetricBadge label="mAP@0.5:0.95" value={pct(displayResult.mAP50_95)} color="text-sky-400" />
                    <MetricBadge label="Precision" value={pct(displayResult.precision)} color="text-violet-400" />
                    <MetricBadge label="Recall" value={pct(displayResult.recall)} color="text-rose-400" />
                  </div>
                </div>

                {/* Latency + Model info */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-slate-800/50 rounded-lg p-3 space-y-1.5">
                    <div className="text-[10px] text-slate-500 uppercase tracking-wider flex items-center gap-1.5"><Zap size={10} /> Latency</div>
                    {[
                      ['Preprocess', displayResult.preprocess_ms],
                      ['Inference', displayResult.inference_ms],
                      ['Postprocess', displayResult.postprocess_ms],
                    ].map(([k, v]) => (
                      <div key={k as string} className="flex justify-between text-xs">
                        <span className="text-slate-500">{k}</span>
                        <span className="font-mono text-slate-300">{(v as number).toFixed(1)} ms</span>
                      </div>
                    ))}
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-3 space-y-1.5">
                    <div className="text-[10px] text-slate-500 uppercase tracking-wider flex items-center gap-1.5"><Cpu size={10} /> Model Info</div>
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-500">Parameters</span>
                      <span className="font-mono text-slate-300">{displayResult.params ? (displayResult.params / 1e6).toFixed(2) + 'M' : '—'}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-500">FLOPs</span>
                      <span className="font-mono text-slate-300">{displayResult.flops_gflops ? displayResult.flops_gflops.toFixed(2) + ' GFLOPs' : '—'}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-500">Duration</span>
                      <span className="font-mono text-slate-300">{displayResult.elapsed_s.toFixed(0)}s</span>
                    </div>
                  </div>
                </div>

                {/* Per-class mAP bar chart */}
                {displayResult.per_class.length > 0 && (
                  <div>
                    <div className="text-xs text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5"><Target size={11} /> Per-Class mAP@0.5</div>
                    <div style={{ height: Math.max(200, displayResult.per_class.length * 24) }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={displayResult.per_class.map(c => ({
                            name: c.class_name.length > 14 ? c.class_name.slice(0, 14) + '…' : c.class_name,
                            ap50: c.ap50 != null ? parseFloat((c.ap50 * 100).toFixed(1)) : 0,
                            precision: c.precision != null ? parseFloat((c.precision * 100).toFixed(1)) : 0,
                            recall: c.recall != null ? parseFloat((c.recall * 100).toFixed(1)) : 0,
                          }))}
                          layout="vertical"
                          margin={{ top: 0, right: 30, left: 90, bottom: 0 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                          <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 10, fill: '#64748b' }} tickFormatter={v => `${v}%`} />
                          <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: '#94a3b8' }} width={90} />
                          <Tooltip
                            contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8, fontSize: 11 }}
                            formatter={(v: number, name: string) => [`${v}%`, name]}
                          />
                          <Bar dataKey="ap50" name="mAP@0.5" fill="#6366f1" radius={[0, 3, 3, 0]} />
                          <Bar dataKey="precision" name="Precision" fill="#8b5cf6" radius={[0, 3, 3, 0]} />
                          <Bar dataKey="recall" name="Recall" fill="#ec4899" radius={[0, 3, 3, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Per-class table */}
                    <div className="mt-3 overflow-x-auto">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="text-[10px] text-slate-500 uppercase">
                            {([
                              { key: 'class_name', label: 'Class', align: 'left' },
                              { key: 'ap50',       label: 'mAP@0.5',      align: 'right' },
                              { key: 'ap50_95',    label: 'mAP@0.5:0.95', align: 'right' },
                              { key: 'precision',  label: 'Precision',     align: 'right' },
                              { key: 'recall',     label: 'Recall',        align: 'right' },
                              { key: 'f1',         label: 'F1',            align: 'right' },
                            ] as const).map(col => (
                              <th
                                key={col.key}
                                className={`py-1.5 ${col.align === 'right' ? 'text-right' : 'text-left'} cursor-pointer select-none hover:text-slate-300 transition-colors`}
                                onClick={() => handleSort(col.key)}
                              >
                                {col.label}
                                {sortKey === col.key ? (sortAsc ? ' ↑' : ' ↓') : ' ·'}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {[...displayResult.per_class].sort((a, b) => {
                            const av = sortKey === 'class_name' ? a.class_name : (a[sortKey] ?? -1);
                            const bv = sortKey === 'class_name' ? b.class_name : (b[sortKey] ?? -1);
                            if (typeof av === 'string') return sortAsc ? av.localeCompare(bv as string) : (bv as string).localeCompare(av);
                            return sortAsc ? (av as number) - (bv as number) : (bv as number) - (av as number);
                          }).map(c => (
                            <tr key={c.class_id} className="border-t border-slate-800 hover:bg-slate-800/30">
                              <td className="py-1 text-slate-300">{c.class_name}</td>
                              <td className="py-1 text-right font-mono text-emerald-400">{pct(c.ap50)}</td>
                              <td className="py-1 text-right font-mono text-sky-400">{pct(c.ap50_95)}</td>
                              <td className="py-1 text-right font-mono text-violet-400">{pct(c.precision)}</td>
                              <td className="py-1 text-right font-mono text-rose-400">{pct(c.recall)}</td>
                              <td className="py-1 text-right font-mono text-amber-400">{pct(c.f1)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Confusion matrix */}
                {displayResult.confusion_matrix && (
                  <div>
                    <div className="text-xs text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5"><Activity size={11} /> Confusion Matrix</div>
                    <ConfusionMatrix data={displayResult.confusion_matrix} />
                  </div>
                )}
              </>
            )}
          </div>
        </div>
    </>
  );

  if (inline) {
    return (
      <div className="flex flex-col min-h-[500px]">
        {panelBody}
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm" onClick={onClose}>
      <div
        className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-full max-w-4xl mx-4 max-h-[90vh] flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        {panelBody}
      </div>
    </div>
  );
}
