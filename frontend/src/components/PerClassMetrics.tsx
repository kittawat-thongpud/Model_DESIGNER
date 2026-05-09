import React, { useState, useMemo } from 'react';
import { BarChart2, ChevronDown, ChevronUp, ArrowUpDown, Search, LineChart as LineChartIcon, ToggleLeft, ToggleRight, Eye, EyeOff } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface EpochData {
  ap_per_class?: number[];
  ap50_per_class?: number[];
  precision_per_class?: number[];
  recall_per_class?: number[];
  f1_per_class?: number[];
  epoch?: number;
  [key: string]: unknown;
}

interface Props {
  history: EpochData[];
  classNames?: string[];
}

type SortKey = 'index' | 'mAP50_95' | 'mAP50' | 'precision' | 'recall' | 'f1';
type SortDir = 'asc' | 'desc';

const COCO_NAMES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
  'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
  'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
  'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
  'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
  'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
  'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
  'toothbrush',
];

function MetricBar({ value, color, max = 1.0 }: { value: number; color: string; max?: number }) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  return (
    <div className="flex items-center gap-2 w-full">
      <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-[11px] font-mono text-slate-300 w-12 text-right">
        {(value * 100).toFixed(1)}
      </span>
    </div>
  );
}

const PerClassMetrics: React.FC<Props> = ({ history, classNames }) => {
  const [expanded, setExpanded] = useState(false);
  const [viewMode, setViewMode] = useState<'table' | 'graph'>('table');
  const [selectedEpoch, setSelectedEpoch] = useState<number>(-1); // -1 = latest
  const [sortKey, setSortKey] = useState<SortKey>('mAP50_95');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [searchTerm, setSearchTerm] = useState('');
  const [visibleClasses, setVisibleClasses] = useState<Set<number>>(new Set());
  const [metricType, setMetricType] = useState<'mAP50_95' | 'mAP50' | 'precision' | 'recall' | 'f1'>('mAP50_95');

  // Get selected epoch data
  const epochData = useMemo(() => {
    if (selectedEpoch === -1) {
      return history[history.length - 1] || {};
    }
    return history.find(e => e.epoch === selectedEpoch) || history[history.length - 1] || {};
  }, [history, selectedEpoch]);

  const ap = epochData.ap_per_class as number[] | undefined;
  const ap50 = epochData.ap50_per_class as number[] | undefined;
  const prec = epochData.precision_per_class as number[] | undefined;
  const rec = epochData.recall_per_class as number[] | undefined;
  const f1 = epochData.f1_per_class as number[] | undefined;

  const nc = ap?.length || ap50?.length || prec?.length || 0;
  if (nc === 0) return null;

  const names = classNames && classNames.length === nc
    ? classNames
    : nc === 80
      ? COCO_NAMES
      : Array.from({ length: nc }, (_, i) => `class_${i}`);

  // Initialize visible classes (show top 10 by mAP50_95)
  useMemo(() => {
    if (visibleClasses.size === 0 && ap) {
      const top10 = Array.from({ length: nc }, (_, i) => ({ index: i, value: ap[i] }))
        .sort((a, b) => b.value - a.value)
        .slice(0, 10)
        .map(x => x.index);
      setVisibleClasses(new Set(top10));
    }
  }, [ap, nc, visibleClasses.size]);

  const rows = useMemo(() => {
    const data = Array.from({ length: nc }, (_, i) => ({
      index: i,
      name: names[i],
      mAP50_95: ap?.[i] ?? 0,
      mAP50: ap50?.[i] ?? 0,
      precision: prec?.[i] ?? 0,
      recall: rec?.[i] ?? 0,
      f1: f1?.[i] ?? 0,
    }));

    // Filter by search
    const filtered = searchTerm
      ? data.filter(r => r.name.toLowerCase().includes(searchTerm.toLowerCase()))
      : data;

    // Sort
    filtered.sort((a, b) => {
      const va = a[sortKey] as number;
      const vb = b[sortKey] as number;
      return sortDir === 'desc' ? vb - va : va - vb;
    });

    return filtered;
  }, [nc, ap, ap50, prec, rec, f1, names, sortKey, sortDir, searchTerm]);

  // Summary stats
  const meanAP = ap ? ap.reduce((s, v) => s + v, 0) / ap.length : 0;
  const meanAP50 = ap50 ? ap50.reduce((s, v) => s + v, 0) / ap50.length : 0;
  const meanPrec = prec ? prec.reduce((s, v) => s + v, 0) / prec.length : 0;
  const meanRec = rec ? rec.reduce((s, v) => s + v, 0) / rec.length : 0;
  const meanF1 = f1 ? f1.reduce((s, v) => s + v, 0) / f1.length : 0;

  // Prepare graph data
  const graphData = useMemo(() => {
    return history.map(epoch => {
      const epochAp = epoch.ap_per_class as number[] | undefined;
      const epochAp50 = epoch.ap50_per_class as number[] | undefined;
      const epochPrec = epoch.precision_per_class as number[] | undefined;
      const epochRec = epoch.recall_per_class as number[] | undefined;
      const epochF1 = epoch.f1_per_class as number[] | undefined;

      const data: any = { epoch: epoch.epoch };

      // Add visible classes
      visibleClasses.forEach(classIndex => {
        data[`class_${classIndex}`] = (() => {
          switch (metricType) {
            case 'mAP50_95': return epochAp?.[classIndex] ?? 0;
            case 'mAP50': return epochAp50?.[classIndex] ?? 0;
            case 'precision': return epochPrec?.[classIndex] ?? 0;
            case 'recall': return epochRec?.[classIndex] ?? 0;
            case 'f1': return epochF1?.[classIndex] ?? 0;
            default: return 0;
          }
        })();
      });

      return data;
    });
  }, [history, visibleClasses, metricType]);

  // Calculate Y-axis domain with margin
  const yAxisDomain = useMemo(() => {
    if (graphData.length === 0 || visibleClasses.size === 0) return [0, 1];

    let min = Infinity;
    let max = -Infinity;

    graphData.forEach(epoch => {
      visibleClasses.forEach(classIndex => {
        const value = epoch[`class_${classIndex}`];
        if (typeof value === 'number' && !isNaN(value)) {
          min = Math.min(min, value);
          max = Math.max(max, value);
        }
      });
    });

    if (!isFinite(min) || !isFinite(max)) return [0, 1];

    // Add 10% margin on top and bottom
    const margin = (max - min) * 0.1;
    return [Math.max(0, min - margin), Math.min(1, max + margin)];
  }, [graphData, visibleClasses]);

  // Toggle class visibility
  const toggleClass = (classIndex: number) => {
    const newSet = new Set(visibleClasses);
    if (newSet.has(classIndex)) {
      newSet.delete(classIndex);
    } else {
      newSet.add(classIndex);
    }
    setVisibleClasses(newSet);
  };

  const toggleAllClasses = () => {
    if (visibleClasses.size === nc) {
      setVisibleClasses(new Set());
    } else {
      setVisibleClasses(new Set(Array.from({ length: nc }, (_, i) => i)));
    }
  };

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(d => d === 'desc' ? 'asc' : 'desc');
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
  };

  const SortHeader = ({ label, field }: { label: string; field: SortKey }) => (
    <th
      className="px-3 py-2 text-right cursor-pointer hover:text-white transition-colors select-none"
      onClick={() => toggleSort(field)}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        {sortKey === field ? (
          sortDir === 'desc' ? <ChevronDown size={10} /> : <ChevronUp size={10} />
        ) : (
          <ArrowUpDown size={9} className="opacity-30" />
        )}
      </span>
    </th>
  );

  const currentEpoch = epochData.epoch ?? history[history.length - 1]?.epoch ?? 0;

  return (
    <div className="bg-[#0f1117] border border-slate-800 rounded-xl overflow-hidden">
      {/* Header */}
      <div
        className="px-4 py-3 border-b border-slate-800/50 bg-slate-900/30 flex items-center gap-3 cursor-pointer hover:bg-slate-800/30 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <BarChart2 size={16} className="text-cyan-400" />
        <span className="text-sm font-semibold text-slate-200">Per-Class Metrics</span>
        <span className="text-[10px] text-slate-500 uppercase">{nc} classes &middot; {history.length} epochs</span>
        <div className="ml-auto flex items-center gap-4">
          {/* Summary chips */}
          <div className="hidden sm:flex gap-2 text-[10px]">
            <span className="px-2 py-0.5 bg-blue-500/10 text-blue-400 border border-blue-500/20 rounded">
              mAP50: {(meanAP50 * 100).toFixed(1)}%
            </span>
            <span className="px-2 py-0.5 bg-violet-500/10 text-violet-400 border border-violet-500/20 rounded">
              mAP50-95: {(meanAP * 100).toFixed(1)}%
            </span>
            <span className="px-2 py-0.5 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded">
              F1: {(meanF1 * 100).toFixed(1)}%
            </span>
          </div>
          {expanded ? <ChevronUp size={16} className="text-slate-400" /> : <ChevronDown size={16} className="text-slate-400" />}
        </div>
      </div>

      {expanded && (
        <div className="p-4 space-y-3">
          {/* Controls */}
          <div className="flex flex-wrap items-center gap-3">
            {/* View toggle */}
            <button
              onClick={() => setViewMode(viewMode === 'table' ? 'graph' : 'table')}
              className="flex items-center gap-2 px-3 py-1.5 bg-slate-900/60 border border-slate-800 rounded-lg text-xs text-white hover:bg-slate-800 transition-colors"
            >
              {viewMode === 'table' ? <BarChart2 size={14} className="text-cyan-400" /> : <LineChartIcon size={14} className="text-violet-400" />}
              {viewMode === 'table' ? 'Table' : 'Graph'}
            </button>

            {/* Epoch selector (table mode only) */}
            {viewMode === 'table' && (
              <select
                value={selectedEpoch}
                onChange={e => setSelectedEpoch(parseInt(e.target.value))}
                className="px-3 py-1.5 bg-slate-900/60 border border-slate-800 rounded-lg text-xs text-white"
              >
                <option value={-1}>Latest (Epoch {currentEpoch})</option>
                {history.slice().reverse().map(e => (
                  <option key={e.epoch} value={e.epoch}>Epoch {e.epoch}</option>
                ))}
              </select>
            )}

            {/* Metric type selector (graph mode only) */}
            {viewMode === 'graph' && (
              <select
                value={metricType}
                onChange={e => setMetricType(e.target.value as any)}
                className="px-3 py-1.5 bg-slate-900/60 border border-slate-800 rounded-lg text-xs text-white"
              >
                <option value="mAP50_95">mAP50-95</option>
                <option value="mAP50">mAP50</option>
                <option value="precision">Precision</option>
                <option value="recall">Recall</option>
                <option value="f1">F1</option>
              </select>
            )}

            {/* Toggle all classes (graph mode only) */}
            {viewMode === 'graph' && (
              <button
                onClick={toggleAllClasses}
                className="flex items-center gap-2 px-3 py-1.5 bg-slate-900/60 border border-slate-800 rounded-lg text-xs text-white hover:bg-slate-800 transition-colors"
              >
                {visibleClasses.size === nc ? <EyeOff size={14} className="text-slate-400" /> : <Eye size={14} className="text-emerald-400" />}
                {visibleClasses.size === nc ? 'Hide All' : 'Show All'} ({visibleClasses.size}/{nc})
              </button>
            )}
          </div>

          {/* Mean row */}
          <div className="grid grid-cols-5 gap-3">
            <div className="bg-slate-900/60 border border-slate-800 rounded-lg px-3 py-2 text-center">
              <div className="text-[9px] text-slate-500 uppercase">mAP50-95</div>
              <div className="text-lg font-mono font-bold text-violet-400">{(meanAP * 100).toFixed(1)}%</div>
            </div>
            <div className="bg-slate-900/60 border border-slate-800 rounded-lg px-3 py-2 text-center">
              <div className="text-[9px] text-slate-500 uppercase">mAP50</div>
              <div className="text-lg font-mono font-bold text-blue-400">{(meanAP50 * 100).toFixed(1)}%</div>
            </div>
            <div className="bg-slate-900/60 border border-slate-800 rounded-lg px-3 py-2 text-center">
              <div className="text-[9px] text-slate-500 uppercase">Precision</div>
              <div className="text-lg font-mono font-bold text-amber-400">{(meanPrec * 100).toFixed(1)}%</div>
            </div>
            <div className="bg-slate-900/60 border border-slate-800 rounded-lg px-3 py-2 text-center">
              <div className="text-[9px] text-slate-500 uppercase">Recall</div>
              <div className="text-lg font-mono font-bold text-rose-400">{(meanRec * 100).toFixed(1)}%</div>
            </div>
            <div className="bg-slate-900/60 border border-slate-800 rounded-lg px-3 py-2 text-center">
              <div className="text-[9px] text-slate-500 uppercase">F1</div>
              <div className="text-lg font-mono font-bold text-emerald-400">{(meanF1 * 100).toFixed(1)}%</div>
            </div>
          </div>

          {/* Table view */}
          {viewMode === 'table' && (
            <>
              {/* Search */}
              <div className="relative">
                <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                <input
                  type="text"
                  placeholder="Search class..."
                  value={searchTerm}
                  onChange={e => setSearchTerm(e.target.value)}
                  className="w-full pl-9 pr-3 py-2 bg-slate-900/60 border border-slate-800 rounded-lg text-xs text-white placeholder-slate-600 focus:outline-none focus:border-slate-600"
                />
              </div>

              {/* Table */}
              <div className="overflow-x-auto max-h-[500px] overflow-y-auto rounded-lg border border-slate-800">
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-slate-900/95 backdrop-blur-sm z-10">
                    <tr className="text-slate-500 uppercase tracking-wider border-b border-slate-800">
                      <SortHeader label="#" field="index" />
                      <th className="px-3 py-2 text-left">Class</th>
                      <SortHeader label="mAP50-95" field="mAP50_95" />
                      <SortHeader label="mAP50" field="mAP50" />
                      <SortHeader label="Precision" field="precision" />
                      <SortHeader label="Recall" field="recall" />
                      <SortHeader label="F1" field="f1" />
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map(row => (
                      <tr key={row.index} className="border-b border-slate-800/30 hover:bg-slate-800/20">
                        <td className="px-3 py-1.5 text-right text-slate-600 font-mono">{row.index}</td>
                        <td className="px-3 py-1.5 text-white font-medium whitespace-nowrap">{row.name}</td>
                        <td className="px-3 py-1.5 w-32"><MetricBar value={row.mAP50_95} color="#a78bfa" /></td>
                        <td className="px-3 py-1.5 w-32"><MetricBar value={row.mAP50} color="#38bdf8" /></td>
                        <td className="px-3 py-1.5 w-32"><MetricBar value={row.precision} color="#f59e0b" /></td>
                        <td className="px-3 py-1.5 w-32"><MetricBar value={row.recall} color="#f43f5e" /></td>
                        <td className="px-3 py-1.5 w-32"><MetricBar value={row.f1} color="#10b981" /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="text-[10px] text-slate-600 text-right">
                Showing {rows.length} of {nc} classes
              </div>
            </>
          )}

          {/* Graph view */}
          {viewMode === 'graph' && (
            <>
              {/* Class toggles */}
              <div className="max-h-[200px] overflow-y-auto rounded-lg border border-slate-800 p-2">
                <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-1">
                  {Array.from({ length: nc }, (_, i) => (
                    <button
                      key={i}
                      onClick={() => toggleClass(i)}
                      className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] transition-colors ${
                        visibleClasses.has(i)
                          ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                          : 'bg-slate-800 text-slate-500 border border-slate-700'
                      }`}
                      title={names[i]}
                    >
                      {visibleClasses.has(i) ? <Eye size={10} /> : <EyeOff size={10} />}
                      <span className="truncate">{names[i]}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Line chart */}
              <div className="h-[400px] rounded-lg border border-slate-800 p-2">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={graphData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
                    <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
                    <YAxis stroke="#475569" tick={{ fontSize: 9 }} domain={yAxisDomain} />
                    <Tooltip
                      content={({ active, payload, label }) => {
                        if (!active || !payload?.length) return null;
                        return (
                          <div className="bg-slate-950/90 border border-slate-800 p-2 rounded-lg shadow-xl backdrop-blur-md text-xs z-50">
                            <p className="text-slate-400 font-bold mb-1.5 border-b border-slate-800 pb-1">Epoch {label}</p>
                            {payload.map((entry: any, i: number) => (
                              <div key={i} className="flex items-center justify-between gap-4 mb-0.5">
                                <span className="flex items-center gap-1.5" style={{ color: entry.color }}>
                                  <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: entry.color }} />
                                  {names[parseInt(entry.dataKey.replace('class_', ''))]}
                                </span>
                                <span className="font-mono text-white">
                                  {typeof entry.value === 'number' ? (entry.value * 100).toFixed(1) + '%' : '-'}
                                </span>
                              </div>
                            ))}
                          </div>
                        );
                      }}
                    />
                    <Legend
                      iconType="circle"
                      wrapperStyle={{ fontSize: '10px' }}
                      onClick={(e: any) => {
                        const classIndex = parseInt(e.dataKey.replace('class_', ''));
                        if (!isNaN(classIndex)) toggleClass(classIndex);
                      }}
                    />
                    {Array.from(visibleClasses).map((classIndex, i) => {
                      const colors = ['#a78bfa', '#38bdf8', '#f59e0b', '#f43f5e', '#10b981', '#64748b', '#fb923c', '#22c55e', '#e11d48', '#8b5cf6'];
                      return (
                        <Line
                          key={classIndex}
                          type="monotone"
                          dataKey={`class_${classIndex}`}
                          name={names[classIndex]}
                          stroke={colors[i % colors.length]}
                          strokeWidth={2}
                          dot={false}
                          connectNulls
                        />
                      );
                    })}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default PerClassMetrics;
