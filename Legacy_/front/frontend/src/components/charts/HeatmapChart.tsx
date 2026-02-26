/**
 * HeatmapChart — Canvas-based heatmap for confusion matrices and weight visualization.
 */
import { useRef, useEffect, useState } from 'react';

type Colormap = 'viridis' | 'plasma' | 'inferno' | 'magma' | 'coolwarm';

// Simplified colormaps (10 stops each)
const COLORMAPS: Record<Colormap, [number, number, number][]> = {
  viridis: [
    [68, 1, 84], [72, 35, 116], [64, 67, 135], [52, 94, 141],
    [33, 145, 140], [44, 160, 44], [94, 201, 98], [165, 219, 85],
    [215, 226, 82], [253, 231, 37],
  ],
  plasma: [
    [13, 8, 135], [75, 3, 161], [126, 3, 168], [168, 34, 150],
    [204, 71, 120], [230, 109, 87], [248, 149, 64], [253, 191, 73],
    [252, 230, 97], [240, 249, 33],
  ],
  inferno: [
    [0, 0, 4], [22, 11, 57], [66, 10, 104], [106, 23, 110],
    [147, 38, 103], [188, 55, 84], [223, 81, 58], [246, 120, 32],
    [251, 175, 35], [252, 255, 164],
  ],
  magma: [
    [0, 0, 4], [18, 13, 54], [51, 16, 104], [89, 22, 126],
    [128, 37, 130], [170, 51, 119], [211, 77, 105], [240, 119, 107],
    [253, 176, 122], [252, 253, 191],
  ],
  coolwarm: [
    [59, 76, 192], [98, 130, 234], [141, 176, 254], [184, 208, 249],
    [221, 221, 221], [245, 196, 173], [244, 154, 123], [222, 96, 77],
    [180, 4, 38], [180, 4, 38],
  ],
};

function getColor(value: number, min: number, max: number, colormap: Colormap): string {
  const t = max !== min ? (value - min) / (max - min) : 0.5;
  const stops = COLORMAPS[colormap];
  const idx = Math.min(Math.floor(t * (stops.length - 1)), stops.length - 2);
  const frac = t * (stops.length - 1) - idx;
  const a = stops[idx];
  const b = stops[idx + 1];
  const r = Math.round(a[0] + (b[0] - a[0]) * frac);
  const g = Math.round(a[1] + (b[1] - a[1]) * frac);
  const bv = Math.round(a[2] + (b[2] - a[2]) * frac);
  return `rgb(${r},${g},${bv})`;
}

interface Props {
  values: number[][];
  rowLabels?: string[];
  colLabels?: string[];
  colormap?: Colormap;
  height?: number;
  showValues?: boolean;
  title?: string;
  onColormapChange?: (cm: Colormap) => void;
}

export default function HeatmapChart({
  values,
  rowLabels,
  colLabels,
  colormap = 'viridis',
  height = 400,
  showValues = true,
  title,
  onColormapChange,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    row: string;
    col: string;
    val: number;
  } | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || values.length === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    const h = height;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const rows = values.length;
    const cols = values[0]?.length || 0;
    if (rows === 0 || cols === 0) return;

    // Padding for labels
    const labelPad = rowLabels ? 60 : 20;
    const topPad = colLabels ? 50 : 20;
    const rightPad = 20;
    const bottomPad = 10;

    const plotW = w - labelPad - rightPad;
    const plotH = h - topPad - bottomPad;
    const cellW = plotW / cols;
    const cellH = plotH / rows;

    // Compute range
    // Compute range safely
    let vMin = Infinity;
    let vMax = -Infinity;
    
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = values[r][c];
        if (v < vMin) vMin = v;
        if (v > vMax) vMax = v;
      }
    }

    // Clear
    ctx.clearRect(0, 0, w, h);

    // Draw cells
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = values[r][c];
        ctx.fillStyle = getColor(v, vMin, vMax, colormap);
        ctx.fillRect(labelPad + c * cellW, topPad + r * cellH, cellW - 1, cellH - 1);

        if (showValues && cellW > 20 && cellH > 15) {
          ctx.fillStyle = v > (vMin + vMax) / 2 ? 'rgba(0,0,0,0.7)' : 'rgba(255,255,255,0.8)';
          ctx.font = `${Math.min(10, cellW / 3)}px monospace`;
          ctx.textAlign = 'center';
          ctx.fillText(
            v >= 1000 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(3),
            labelPad + c * cellW + cellW / 2,
            topPad + r * cellH + cellH / 2 + 4
          );
        }
      }
    }

    // Row labels
    if (rowLabels) {
      ctx.fillStyle = 'rgba(255,255,255,0.6)';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'right';
      for (let r = 0; r < rows; r++) {
        const label = rowLabels[r] ?? String(r);
        ctx.fillText(label, labelPad - 4, topPad + r * cellH + cellH / 2 + 3);
      }
    }

    // Col labels
    if (colLabels) {
      ctx.fillStyle = 'rgba(255,255,255,0.6)';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      for (let c = 0; c < cols; c++) {
        const label = colLabels[c] ?? String(c);
        ctx.save();
        ctx.translate(labelPad + c * cellW + cellW / 2, topPad - 6);
        ctx.rotate(-Math.PI / 4);
        ctx.fillText(label, 0, 0);
        ctx.restore();
      }
    }

    // Mouse → tooltip
    const handleMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const c = Math.floor((mx - labelPad) / cellW);
      const r = Math.floor((my - topPad) / cellH);
      if (r >= 0 && r < rows && c >= 0 && c < cols) {
        setTooltip({
          x: mx,
          y: my,
          row: rowLabels?.[r] ?? String(r),
          col: colLabels?.[c] ?? String(c),
          val: values[r][c],
        });
      } else {
        setTooltip(null);
      }
    };

    canvas.addEventListener('mousemove', handleMove);
    canvas.addEventListener('mouseleave', () => setTooltip(null));
    return () => {
      canvas.removeEventListener('mousemove', handleMove);
      canvas.removeEventListener('mouseleave', () => setTooltip(null));
    };
  }, [values, rowLabels, colLabels, colormap, height, showValues]);

  const colormaps: Colormap[] = ['viridis', 'plasma', 'inferno', 'magma', 'coolwarm'];

  return (
    <div ref={containerRef} className="chart-container" style={{ position: 'relative' }}>
      {title && <div className="chart-title">{title}</div>}
      <canvas ref={canvasRef} />
      {tooltip && (
        <div className="chart-tooltip" style={{ left: tooltip.x, top: tooltip.y - 40 }}>
          [{tooltip.row}] → [{tooltip.col}]: <strong>{tooltip.val}</strong>
        </div>
      )}
      {onColormapChange && (
        <div className="colormap-picker">
          {colormaps.map((cm) => (
            <button
              key={cm}
              className={`colormap-btn ${cm === colormap ? 'active' : ''}`}
              onClick={() => onColormapChange(cm)}
              title={cm}
            >
              {cm}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export type { Colormap };
