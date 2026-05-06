import React, { useState, useMemo } from 'react';
import { BarChart2, ChevronDown, ChevronUp, ArrowUpDown, Search } from 'lucide-react';

interface EpochData {
  ap_per_class?: number[];
  ap50_per_class?: number[];
  precision_per_class?: number[];
  recall_per_class?: number[];
  f1_per_class?: number[];
  [key: string]: unknown;
}

interface Props {
  epochData: EpochData;
  epoch: number;
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

const PerClassMetrics: React.FC<Props> = ({ epochData, epoch, classNames }) => {
  const [expanded, setExpanded] = useState(false);
  const [sortKey, setSortKey] = useState<SortKey>('mAP50_95');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [searchTerm, setSearchTerm] = useState('');

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

  return (
    <div className="bg-[#0f1117] border border-slate-800 rounded-xl overflow-hidden">
      {/* Header */}
      <div
        className="px-4 py-3 border-b border-slate-800/50 bg-slate-900/30 flex items-center gap-3 cursor-pointer hover:bg-slate-800/30 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <BarChart2 size={16} className="text-cyan-400" />
        <span className="text-sm font-semibold text-slate-200">Per-Class Metrics</span>
        <span className="text-[10px] text-slate-500 uppercase">Epoch {epoch} &middot; {nc} classes</span>
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
        </div>
      )}
    </div>
  );
};

export default PerClassMetrics;
