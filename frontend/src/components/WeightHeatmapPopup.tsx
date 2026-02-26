/**
 * WeightHeatmapPopup — full-screen overlay showing a weight matrix as a
 * zoomable/pannable heatmap with cell values visible when zoomed in.
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import { api } from '../services/api';
import { X, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';
import { colormapRGB, colormapCSS, type ColormapName } from '../utils/colormaps';

interface Props {
  jobId: string;
  epoch: number;
  layerName: string;
  colormap: ColormapName;
  onClose: () => void;
}

interface SnapshotData {
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
}

export default function WeightHeatmapPopup({ jobId, epoch, layerName, colormap, onClose }: Props) {
  const [data, setData] = useState<SnapshotData | null>(null);
  const [loading, setLoading] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
  const dragRef = useRef<{ active: boolean; sx: number; sy: number; tx: number; ty: number }>({
    active: false, sx: 0, sy: 0, tx: 0, ty: 0,
  });

  useEffect(() => {
    (api as any).getSnapshotData(jobId, epoch, layerName).then((d: any) => {
      setData(d as SnapshotData);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, [jobId, epoch, layerName]);

  const fitToView = useCallback(() => {
    if (!data || !containerRef.current) return;
    const cw = containerRef.current.clientWidth;
    const ch = containerRef.current.clientHeight;
    const sx = (cw - 40) / data.cols;
    const sy = (ch - 40) / data.rows;
    const scale = Math.min(sx, sy, 40);
    setTransform({
      x: (cw - data.cols * scale) / 2,
      y: (ch - data.rows * scale) / 2,
      scale,
    });
  }, [data]);

  useEffect(() => { fitToView(); }, [fitToView]);

  // ── Canvas rendering ──
  const render = useCallback(() => {
    if (!data || !canvasRef.current || !containerRef.current) return;
    const canvas = canvasRef.current;
    const container = containerRef.current;
    const dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    const h = container.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const { x, y, scale } = transform;
    const range = data.max - data.min || 1;

    // Draw cells
    for (let r = 0; r < data.rows; r++) {
      for (let c = 0; c < data.cols; c++) {
        const cx = x + c * scale;
        const cy = y + r * scale;
        if (cx + scale < 0 || cx > w || cy + scale < 0 || cy > h) continue;

        const val = data.values[r][c];
        const norm = (val - data.min) / range;
        const [cr, cg, cb] = colormapRGB(norm, colormap);

        ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
        ctx.fillRect(cx, cy, scale + 0.5, scale + 0.5);

        if (scale >= 30) {
          const brightness = (cr * 299 + cg * 587 + cb * 114) / 1000;
          ctx.fillStyle = brightness > 128 ? '#000' : '#fff';
          ctx.font = `${Math.min(scale * 0.28, 11)}px monospace`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(val.toFixed(3), cx + scale / 2, cy + scale / 2);
        }
      }
    }

    // Grid lines when zoomed
    if (scale >= 8) {
      ctx.strokeStyle = 'rgba(255,255,255,0.06)';
      ctx.lineWidth = 0.5;
      for (let r = 0; r <= data.rows; r++) {
        const cy = y + r * scale;
        ctx.beginPath(); ctx.moveTo(x, cy); ctx.lineTo(x + data.cols * scale, cy); ctx.stroke();
      }
      for (let c = 0; c <= data.cols; c++) {
        const cx = x + c * scale;
        ctx.beginPath(); ctx.moveTo(cx, y); ctx.lineTo(cx, y + data.rows * scale); ctx.stroke();
      }
    }
  }, [data, transform, colormap]);

  useEffect(() => { render(); }, [render]);

  // ── Zoom (scroll wheel) ──
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    setTransform((prev) => {
      const ns = Math.max(0.5, Math.min(200, prev.scale * factor));
      const r = ns / prev.scale;
      return { scale: ns, x: mx - r * (mx - prev.x), y: my - r * (my - prev.y) };
    });
  }, []);

  // ── Drag (pan) ──
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    dragRef.current = { active: true, sx: e.clientX, sy: e.clientY, tx: transform.x, ty: transform.y };
  }, [transform]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const d = dragRef.current;
    if (!d.active) return;
    setTransform((prev) => ({ ...prev, x: d.tx + (e.clientX - d.sx), y: d.ty + (e.clientY - d.sy) }));
  }, []);

  const handleMouseUp = useCallback(() => { dragRef.current.active = false; }, []);

  // ── Keyboard shortcuts ──
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-[60] bg-black/80 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3 bg-slate-900 border-b border-slate-800 shrink-0">
        <div className="flex items-center gap-4 min-w-0">
          <h3 className="text-sm font-semibold text-white truncate">{layerName}</h3>
          {data && (
            <span className="text-[11px] text-slate-400 shrink-0">
              [{data.shape.join('×')}] · {data.rows}×{data.cols} · min {data.min.toFixed(4)} · max {data.max.toFixed(4)} · μ {data.mean.toFixed(4)}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1 shrink-0">
          <button onClick={() => setTransform((p) => ({ ...p, scale: Math.min(200, p.scale * 1.3) }))}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded cursor-pointer" title="Zoom In">
            <ZoomIn size={14} />
          </button>
          <button onClick={() => setTransform((p) => ({ ...p, scale: Math.max(0.5, p.scale / 1.3) }))}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded cursor-pointer" title="Zoom Out">
            <ZoomOut size={14} />
          </button>
          <button onClick={fitToView}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded cursor-pointer" title="Fit to View">
            <Maximize2 size={14} />
          </button>
          <button onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded ml-2 cursor-pointer">
            <X size={16} />
          </button>
        </div>
      </div>

      {/* Canvas area */}
      <div
        ref={containerRef}
        className="flex-1 overflow-hidden cursor-grab active:cursor-grabbing bg-slate-950"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        {loading ? (
          <div className="flex items-center justify-center h-full text-slate-500 text-sm">Loading weight data…</div>
        ) : !data ? (
          <div className="flex items-center justify-center h-full text-slate-500 text-sm">No data available</div>
        ) : (
          <canvas ref={canvasRef} />
        )}
      </div>

      {/* Color legend bar */}
      {data && (
        <div className="px-5 py-2 bg-slate-900 border-t border-slate-800 flex items-center gap-3 shrink-0">
          <span className="text-[10px] text-slate-500 w-16 text-right shrink-0">{data.min.toFixed(4)}</span>
          <div className="flex-1 h-3 rounded-full overflow-hidden flex">
            {Array.from({ length: 80 }, (_, i) => (
              <div key={i} className="flex-1" style={{ backgroundColor: colormapCSS(i / 79, colormap) }} />
            ))}
          </div>
          <span className="text-[10px] text-slate-500 w-16 shrink-0">{data.max.toFixed(4)}</span>
          <span className="text-[10px] text-slate-400 ml-2">Zoom: {transform.scale.toFixed(1)}×</span>
        </div>
      )}
    </div>
  );
}
