/**
 * WeightLayerDetailPanel — Shows histogram, tensor map, and stats for a
 * specific weight tensor. Used as a modal overlay inside
 * the WeightMappingPanel when a user clicks a layer.
 */
import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { X, Loader2, BarChart2, Grid3X3, Info } from 'lucide-react';
import { api } from '../services/api';
import type { LayerDetailResult, TensorMap, TensorMapSlice } from '../types';

/* ─── Colormaps ────────────────────────────────────────────────────────── */

type ColormapName = 'blue_red' | 'viridis' | 'inferno' | 'grayscale' | 'plasma' | 'coolwarm';

const COLORMAP_LABELS: Record<ColormapName, string> = {
  blue_red: 'Blue → Red',
  viridis: 'Viridis',
  inferno: 'Inferno',
  grayscale: 'Grayscale',
  plasma: 'Plasma',
  coolwarm: 'Cool → Warm',
};

function lerp(a: number, b: number, t: number) { return a + (b - a) * t; }

function colormapFn(name: ColormapName, t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t));
  switch (name) {
    case 'blue_red': {
      if (t < 0.5) { const s = t * 2; return [Math.round(lerp(30, 255, s)), Math.round(lerp(64, 255, s)), Math.round(lerp(175, 255, s))]; }
      else { const s = (t - 0.5) * 2; return [255, Math.round(lerp(255, 55, s)), Math.round(lerp(255, 35, s))]; }
    }
    case 'viridis': {
      // Simplified 5-stop viridis: dark purple → teal → green → yellow
      const stops: [number, number, number][] = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
      const idx = t * (stops.length - 1);
      const lo = Math.floor(idx), hi = Math.min(lo + 1, stops.length - 1);
      const f = idx - lo;
      return [Math.round(lerp(stops[lo][0], stops[hi][0], f)), Math.round(lerp(stops[lo][1], stops[hi][1], f)), Math.round(lerp(stops[lo][2], stops[hi][2], f))];
    }
    case 'inferno': {
      const stops: [number, number, number][] = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
      const idx = t * (stops.length - 1);
      const lo = Math.floor(idx), hi = Math.min(lo + 1, stops.length - 1);
      const f = idx - lo;
      return [Math.round(lerp(stops[lo][0], stops[hi][0], f)), Math.round(lerp(stops[lo][1], stops[hi][1], f)), Math.round(lerp(stops[lo][2], stops[hi][2], f))];
    }
    case 'grayscale':
      { const v = Math.round(t * 255); return [v, v, v]; }
    case 'plasma': {
      const stops: [number, number, number][] = [[13, 8, 135], [84, 2, 163], [139, 10, 165], [185, 50, 137], [219, 92, 104], [244, 136, 73], [254, 188, 43], [240, 249, 33]];
      const idx = t * (stops.length - 1);
      const lo = Math.floor(idx), hi = Math.min(lo + 1, stops.length - 1);
      const f = idx - lo;
      return [Math.round(lerp(stops[lo][0], stops[hi][0], f)), Math.round(lerp(stops[lo][1], stops[hi][1], f)), Math.round(lerp(stops[lo][2], stops[hi][2], f))];
    }
    case 'coolwarm': {
      const stops: [number, number, number][] = [[59, 76, 192], [116, 142, 227], [180, 201, 237], [220, 220, 220], [237, 179, 160], [220, 109, 87], [180, 4, 38]];
      const idx = t * (stops.length - 1);
      const lo = Math.floor(idx), hi = Math.min(lo + 1, stops.length - 1);
      const f = idx - lo;
      return [Math.round(lerp(stops[lo][0], stops[hi][0], f)), Math.round(lerp(stops[lo][1], stops[hi][1], f)), Math.round(lerp(stops[lo][2], stops[hi][2], f))];
    }
  }
}

function colormapCSS(name: ColormapName, steps = 8): string {
  const cols = Array.from({ length: steps }, (_, i) => {
    const [r, g, b] = colormapFn(name, i / (steps - 1));
    return `rgb(${r},${g},${b})`;
  });
  return `linear-gradient(to right, ${cols.join(', ')})`;
}

/* ─── Component ────────────────────────────────────────────────────────── */

interface Props {
  weightId: string;
  keyName: string;
  onClose: () => void;
}

export default function WeightLayerDetailPanel({ weightId, keyName, onClose }: Props) {
  const [data, setData] = useState<LayerDetailResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<'histogram' | 'tensor_map' | 'stats'>('tensor_map');

  useEffect(() => {
    setLoading(true);
    setError(null);
    api.layerDetail(weightId, keyName)
      .then(setData)
      .catch((e) => setError((e as Error).message))
      .finally(() => setLoading(false));
  }, [weightId, keyName]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 size={20} className="animate-spin text-indigo-400" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-4">
        <div className="text-red-400 text-xs">{error || 'No data'}</div>
        <button onClick={onClose} className="mt-2 text-xs text-slate-400 hover:text-white cursor-pointer">Close</button>
      </div>
    );
  }

  const { stats, histogram, tensor_map } = data;
  const hasTensorMap = tensor_map.slices.length > 0;

  return (
    <div className="flex flex-col h-full bg-slate-900">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 shrink-0">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold text-white truncate">{keyName}</h3>
          <div className="text-[10px] text-slate-500 mt-0.5">
            {stats.dtype} · [{stats.shape.join('×')}] · {stats.numel.toLocaleString()} params
          </div>
        </div>
        <button onClick={onClose} className="p-1.5 text-slate-500 hover:text-white rounded-lg hover:bg-slate-800 transition-colors cursor-pointer shrink-0">
          <X size={14} />
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-slate-800 shrink-0 overflow-x-auto">
        {([
          { id: 'tensor_map' as const, label: 'Tensor Map', icon: Grid3X3, disabled: !hasTensorMap },
          { id: 'histogram' as const, label: 'Histogram', icon: BarChart2, disabled: false },
          { id: 'stats' as const, label: 'Stats', icon: Info, disabled: false },
        ]).map((t) => (
          <button
            key={t.id}
            onClick={() => !t.disabled && setTab(t.id)}
            disabled={t.disabled}
            className={`flex items-center gap-1.5 px-3 py-2 text-[11px] font-medium border-b-2 transition-colors cursor-pointer whitespace-nowrap ${
              tab === t.id
                ? 'border-indigo-500 text-indigo-400'
                : t.disabled
                  ? 'border-transparent text-slate-700 cursor-not-allowed'
                  : 'border-transparent text-slate-500 hover:text-slate-300'
            }`}
          >
            <t.icon size={12} />
            {t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 min-h-0">
        {tab === 'tensor_map' && hasTensorMap && <TensorMapView tensorMap={tensor_map} />}
        {tab === 'histogram' && <HistogramView histogram={histogram} stats={stats} />}
        {tab === 'stats' && <StatsView stats={stats} />}
      </div>
    </div>
  );
}

/* ─── Tensor Map (unified shape-aware view) ───────────────────────────── */

function TensorMapView({ tensorMap }: { tensorMap: TensorMap }) {
  const [colormap, setColormap] = useState<ColormapName>('blue_red');

  // Compute global min/max across ALL slices for consistent coloring
  const { gMin, gMax } = useMemo(() => {
    let mn = Infinity, mx = -Infinity;
    for (const s of tensorMap.slices) {
      for (const row of s.values) {
        for (const v of row) {
          if (v < mn) mn = v;
          if (v > mx) mx = v;
        }
      }
    }
    return { gMin: mn, gMax: mx };
  }, [tensorMap]);

  const isGrid = tensorMap.slices.length > 1;
  const is1D = tensorMap.ndim === 1;

  return (
    <div>
      {/* Top bar: description + colormap selector */}
      <div className="flex items-center justify-between mb-3 gap-2">
        <div className="text-[10px] text-slate-500">{tensorMap.description}</div>
        <select
          value={colormap}
          onChange={(e) => setColormap(e.target.value as ColormapName)}
          className="bg-slate-800 border border-slate-700 rounded px-2 py-1 text-[10px] text-slate-300 outline-none focus:border-indigo-500 shrink-0 cursor-pointer"
        >
          {(Object.keys(COLORMAP_LABELS) as ColormapName[]).map((k) => (
            <option key={k} value={k}>{COLORMAP_LABELS[k]}</option>
          ))}
        </select>
      </div>

      {/* Render based on layout */}
      {is1D ? (
        // 1D strip — single wide horizontal bar
        <Strip1D slice={tensorMap.slices[0]} colormap={colormap} gMin={gMin} gMax={gMax} />
      ) : !isGrid ? (
        // Single 2D slice — large canvas with tooltip
        <SingleSlice slice={tensorMap.slices[0]} colormap={colormap} gMin={gMin} gMax={gMax} />
      ) : (
        // Multi-slice grid (3D channels / 4D conv kernels)
        <SliceGrid slices={tensorMap.slices} colormap={colormap} gMin={gMin} gMax={gMax} ndim={tensorMap.ndim} />
      )}

      {/* Color scale legend */}
      <div className="flex items-center gap-2 mt-3">
        <span className="text-[8px] text-slate-500 font-mono">{gMin.toFixed(3)}</span>
        <div className="flex-1 h-3 rounded" style={{ background: colormapCSS(colormap) }} />
        <span className="text-[8px] text-slate-500 font-mono">{gMax.toFixed(3)}</span>
      </div>
    </div>
  );
}

/* ─── 1D Strip ─────────────────────────────────────────────────────────── */

function Strip1D({ slice, colormap, gMin, gMax }: {
  slice: TensorMapSlice; colormap: ColormapName; gMin: number; gMax: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const vals = slice.values[0] || [];
  const n = vals.length;
  const cellW = Math.max(3, Math.min(12, Math.floor(500 / n)));
  const w = n * cellW;
  const h = Math.max(32, cellW * 2);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const range = gMax - gMin || 1;
    for (let i = 0; i < n; i++) {
      const t = (vals[i] - gMin) / range;
      const [r, g, b] = colormapFn(colormap, t);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(i * cellW, 0, cellW, h);
    }
  }, [vals, colormap, gMin, gMax, w, h, n, cellW]);

  const [tooltip, setTooltip] = useState<{ x: number; idx: number; val: number } | null>(null);
  const handleMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const idx = Math.floor((e.clientX - rect.left) / rect.width * n);
    if (idx >= 0 && idx < n) {
      setTooltip({ x: e.clientX - rect.left, idx, val: vals[idx] });
    } else setTooltip(null);
  }, [n, vals]);

  return (
    <div className="overflow-x-auto border border-slate-700/30 rounded-lg bg-slate-800/20 p-1 relative">
      <canvas
        ref={canvasRef}
        style={{ width: w, height: h, imageRendering: 'pixelated' }}
        className="block rounded-sm"
        onMouseMove={handleMove}
        onMouseLeave={() => setTooltip(null)}
      />
      {tooltip && (
        <div className="absolute pointer-events-none bg-slate-900/95 border border-slate-600 rounded px-2 py-1 text-[9px] text-white font-mono whitespace-nowrap z-10"
          style={{ left: tooltip.x + 12, top: 4 }}>
          [{tooltip.idx}] = {tooltip.val.toFixed(6)}
        </div>
      )}
    </div>
  );
}

/* ─── Single 2D Slice (full canvas with tooltip) ──────────────────────── */

function SingleSlice({ slice, colormap, gMin, gMax }: {
  slice: TensorMapSlice; colormap: ColormapName; gMin: number; gMax: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; row: number; col: number; val: number } | null>(null);

  const cellPx = Math.max(2, Math.min(10, Math.floor(450 / Math.max(slice.rows, slice.cols))));
  const canvasW = slice.cols * cellPx;
  const canvasH = slice.rows * cellPx;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = canvasW;
    canvas.height = canvasH;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const range = gMax - gMin || 1;
    for (let r = 0; r < slice.rows; r++) {
      for (let c = 0; c < slice.cols; c++) {
        const v = slice.values[r][c];
        const t = (v - gMin) / range;
        const [cr, cg, cb] = colormapFn(colormap, t);
        ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
        ctx.fillRect(c * cellPx, r * cellPx, cellPx, cellPx);
      }
    }
  }, [slice, colormap, gMin, gMax, canvasW, canvasH, cellPx]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const col = Math.floor((e.clientX - rect.left) * scaleX / cellPx);
    const row = Math.floor((e.clientY - rect.top) * scaleY / cellPx);
    if (row >= 0 && row < slice.rows && col >= 0 && col < slice.cols) {
      setTooltip({ x: e.clientX - rect.left, y: e.clientY - rect.top, row, col, val: slice.values[row][col] });
    } else setTooltip(null);
  }, [slice, cellPx]);

  return (
    <div className="overflow-auto border border-slate-700/30 rounded-lg bg-slate-800/20 p-1 relative"
      style={{ maxHeight: 420 }}>
      <canvas
        ref={canvasRef}
        style={{ width: canvasW, height: canvasH, imageRendering: 'pixelated' }}
        className="block"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setTooltip(null)}
      />
      {tooltip && (
        <div className="absolute pointer-events-none bg-slate-900/95 border border-slate-600 rounded px-2 py-1 text-[9px] text-white font-mono whitespace-nowrap z-10"
          style={{ left: tooltip.x + 12, top: tooltip.y - 8 }}>
          [{tooltip.row},{tooltip.col}] = {tooltip.val.toFixed(6)}
        </div>
      )}
    </div>
  );
}

/* ─── Multi-slice Grid (3D/4D) ─────────────────────────────────────────── */

function SliceGrid({ slices, colormap, gMin, gMax, ndim }: {
  slices: TensorMapSlice[]; colormap: ColormapName; gMin: number; gMax: number; ndim: number;
}) {
  // Determine cell size based on slice dimensions
  const firstSlice = slices[0];
  const kH = firstSlice.rows, kW = firstSlice.cols;
  const cellPx = kH <= 3 ? 14 : kH <= 5 ? 10 : kH <= 7 ? 7 : kH <= 16 ? 4 : 2;

  return (
    <div>
      <div className="text-[9px] text-slate-600 mb-2">
        {ndim === 4
          ? 'Each tile = mean kernel across input channels for one output channel.'
          : `${slices.length} slices of [${kH}×${kW}]`}
      </div>
      <div className="flex flex-wrap gap-1.5 overflow-auto" style={{ maxHeight: 380 }}>
        {slices.map((s, idx) => (
          <SliceTile key={idx} slice={s} cellPx={cellPx} colormap={colormap} gMin={gMin} gMax={gMax} />
        ))}
      </div>
    </div>
  );
}

function SliceTile({ slice, cellPx, colormap, gMin, gMax }: {
  slice: TensorMapSlice; cellPx: number; colormap: ColormapName; gMin: number; gMax: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const w = slice.cols * cellPx;
  const h = slice.rows * cellPx;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const range = gMax - gMin || 1;
    for (let r = 0; r < slice.rows; r++) {
      for (let c = 0; c < slice.cols; c++) {
        const v = slice.values[r][c];
        const t = (v - gMin) / range;
        const [cr, cg, cb] = colormapFn(colormap, t);
        ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
        ctx.fillRect(c * cellPx, r * cellPx, cellPx, cellPx);
      }
    }
  }, [slice, cellPx, colormap, gMin, gMax, w, h]);

  return (
    <div className="flex flex-col items-center">
      <canvas
        ref={canvasRef}
        style={{ width: w, height: h, imageRendering: 'pixelated' }}
        className="block rounded-sm border border-slate-700/30"
        title={slice.label}
      />
      <span className="text-[7px] text-slate-600 mt-0.5">{slice.label}</span>
    </div>
  );
}

/* ─── Histogram ────────────────────────────────────────────────────────── */

function HistogramView({ histogram, stats }: {
  histogram: LayerDetailResult['histogram'];
  stats: LayerDetailResult['stats'];
}) {
  const maxCount = Math.max(...histogram.counts, 1);
  const barWidth = 100 / histogram.counts.length;

  return (
    <div>
      {/* Quick stats row */}
      <div className="grid grid-cols-4 gap-2 mb-4">
        {[
          { label: 'Mean', value: stats.mean.toFixed(4) },
          { label: 'Std', value: stats.std.toFixed(4) },
          { label: 'Min', value: stats.min.toFixed(4) },
          { label: 'Max', value: stats.max.toFixed(4) },
        ].map((s) => (
          <div key={s.label} className="bg-slate-800/50 rounded-lg px-2 py-1.5 text-center">
            <div className="text-[9px] text-slate-500 uppercase">{s.label}</div>
            <div className="text-[11px] text-white font-mono">{s.value}</div>
          </div>
        ))}
      </div>

      {/* Histogram bars */}
      <div className="bg-slate-800/30 rounded-lg p-3 border border-slate-700/30">
        <svg viewBox="0 0 100 40" className="w-full h-40" preserveAspectRatio="none">
          {histogram.counts.map((count, i) => {
            const height = (count / maxCount) * 38;
            const x = i * barWidth;
            return (
              <rect
                key={i}
                x={x + barWidth * 0.05}
                y={40 - height}
                width={barWidth * 0.9}
                height={height}
                fill="#6366f1"
                opacity={0.8}
                rx={0.3}
              >
                <title>{`[${histogram.bin_edges[i]?.toFixed(4)}, ${histogram.bin_edges[i + 1]?.toFixed(4)}): ${count}`}</title>
              </rect>
            );
          })}
          {/* Zero line if in range */}
          {histogram.bin_min <= 0 && histogram.bin_max >= 0 && (() => {
            const zeroX = ((0 - histogram.bin_min) / (histogram.bin_max - histogram.bin_min || 1)) * 100;
            return <line x1={zeroX} y1={0} x2={zeroX} y2={40} stroke="#ef4444" strokeWidth={0.3} opacity={0.6} />;
          })()}
        </svg>
        <div className="flex justify-between text-[8px] text-slate-600 mt-1 font-mono">
          <span>{histogram.bin_min.toFixed(3)}</span>
          <span>0</span>
          <span>{histogram.bin_max.toFixed(3)}</span>
        </div>
      </div>

      {/* Distribution badges */}
      <div className="flex gap-2 mt-3">
        <span className="text-[9px] px-2 py-1 rounded bg-slate-800 text-slate-400">
          Zeros: {stats.zeros_pct.toFixed(1)}%
        </span>
        <span className="text-[9px] px-2 py-1 rounded bg-slate-800 text-slate-400">
          Near-zero (&lt;1e-6): {stats.near_zero_pct.toFixed(1)}%
        </span>
      </div>
    </div>
  );
}

/* ─── Stats ────────────────────────────────────────────────────────────── */

function StatsView({ stats }: { stats: LayerDetailResult['stats'] }) {
  const rows = [
    { label: 'Shape', value: `[${stats.shape.join(' × ')}]` },
    { label: 'Data Type', value: stats.dtype },
    { label: 'Parameters', value: stats.numel.toLocaleString() },
    { label: 'Min', value: stats.min.toFixed(6) },
    { label: 'Max', value: stats.max.toFixed(6) },
    { label: 'Mean', value: stats.mean.toFixed(6) },
    { label: 'Std Dev', value: stats.std.toFixed(6) },
    { label: 'Median', value: stats.median.toFixed(6) },
    { label: 'Zeros', value: `${stats.zeros_pct.toFixed(2)}%` },
    { label: 'Near-zero (<1e-6)', value: `${stats.near_zero_pct.toFixed(2)}%` },
  ];

  return (
    <div className="space-y-1">
      {rows.map((r) => (
        <div key={r.label} className="flex items-center justify-between px-3 py-1.5 rounded bg-slate-800/30 border border-slate-700/20">
          <span className="text-[10px] text-slate-400">{r.label}</span>
          <span className="text-[11px] text-white font-mono">{r.value}</span>
        </div>
      ))}
    </div>
  );
}
