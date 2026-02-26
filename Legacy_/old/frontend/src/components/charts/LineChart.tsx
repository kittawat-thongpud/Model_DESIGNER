/**
 * LineChart — Canvas-based interactive line chart.
 * Usage: <LineChart data={series} xLabel="Epoch" yLabel="Loss" />
 */
import { useRef, useEffect, useState } from 'react';

export interface LineSeries {
  label: string;
  color: string;
  data: { x: number; y: number }[];
  dashed?: boolean;
}

interface Props {
  series: LineSeries[];
  xLabel?: string;
  yLabel?: string;
  height?: number;
  showLegend?: boolean;
}

export default function LineChart({
  series,
  xLabel = '',
  yLabel = '',
  height = 240,
  showLegend = true,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    label: string;
    value: string;
    epoch: number;
  } | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

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

    // Padding
    const pad = { top: 20, right: 20, bottom: 35, left: 55 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    // Compute ranges
    const allPts = series.flatMap((s) => s.data);
    if (allPts.length === 0) return;

    const xMin = Math.min(...allPts.map((p) => p.x));
    const xMax = Math.max(...allPts.map((p) => p.x));
    const yVals = allPts.map((p) => p.y).filter((v) => v != null && !isNaN(v));
    if (yVals.length === 0) return;
    const yMinRaw = Math.min(...yVals);
    const yMaxRaw = Math.max(...yVals);
    const yPad = (yMaxRaw - yMinRaw) * 0.1 || 0.1;
    const yMin = yMinRaw - yPad;
    const yMax = yMaxRaw + yPad;

    const toX = (v: number) => pad.left + ((v - xMin) / (xMax - xMin || 1)) * plotW;
    const toY = (v: number) => pad.top + (1 - (v - yMin) / (yMax - yMin || 1)) * plotH;

    // Clear
    ctx.clearRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const y = pad.top + (plotH / yTicks) * i;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(w - pad.right, y);
      ctx.stroke();
    }

    // Y labels
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    for (let i = 0; i <= yTicks; i++) {
      const val = yMax - ((yMax - yMin) / yTicks) * i;
      const y = pad.top + (plotH / yTicks) * i;
      ctx.fillText(val.toFixed(4), pad.left - 6, y + 3);
    }

    // X labels
    ctx.textAlign = 'center';
    const xStep = Math.max(1, Math.floor((xMax - xMin) / 10));
    for (let x = xMin; x <= xMax; x += xStep) {
      ctx.fillText(String(Math.round(x)), toX(x), h - pad.bottom + 18);
    }

    // Axis labels
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '11px sans-serif';
    if (xLabel) {
      ctx.textAlign = 'center';
      ctx.fillText(xLabel, pad.left + plotW / 2, h - 3);
    }
    if (yLabel) {
      ctx.save();
      ctx.translate(12, pad.top + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.fillText(yLabel, 0, 0);
      ctx.restore();
    }

    // Draw lines
    series.forEach((s) => {
      if (s.data.length === 0) return;
      const pts = s.data.filter((p) => p.y != null && !isNaN(p.y));
      if (pts.length === 0) return;

      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      if (s.dashed) ctx.setLineDash([5, 4]);
      else ctx.setLineDash([]);

      pts.forEach((p, i) => {
        const px = toX(p.x);
        const py = toY(p.y);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
      ctx.setLineDash([]);

      // Dots
      pts.forEach((p) => {
        ctx.beginPath();
        ctx.arc(toX(p.x), toY(p.y), 3, 0, Math.PI * 2);
        ctx.fillStyle = s.color;
        ctx.fill();
      });
    });

    // Mouse handler for tooltip
    const handleMouse = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      let closest: { dist: number; label: string; value: string; epoch: number; px: number; py: number } | null = null;
      series.forEach((s) => {
        s.data.forEach((p) => {
          if (p.y == null || isNaN(p.y)) return;
          const px = toX(p.x);
          const py = toY(p.y);
          const dist = Math.sqrt((mx - px) ** 2 + (my - py) ** 2);
          if (dist < 20 && (!closest || dist < closest.dist)) {
            closest = { dist, label: s.label, value: p.y.toFixed(4), epoch: p.x, px, py };
          }
        });
      });

      if (closest) {
        setTooltip({ x: closest.px, y: closest.py, label: closest.label, value: closest.value, epoch: closest.epoch });
      } else {
        setTooltip(null);
      }
    };

    canvas.addEventListener('mousemove', handleMouse);
    canvas.addEventListener('mouseleave', () => setTooltip(null));

    return () => {
      canvas.removeEventListener('mousemove', handleMouse);
      canvas.removeEventListener('mouseleave', () => setTooltip(null));
    };
  }, [series, height, xLabel, yLabel]);

  return (
    <div ref={containerRef} className="chart-container" style={{ position: 'relative' }}>
      <canvas ref={canvasRef} />
      {tooltip && (
        <div
          className="chart-tooltip"
          style={{
            left: tooltip.x,
            top: tooltip.y - 40,
          }}
        >
          <strong>{tooltip.label}</strong>
          <br />
          E{tooltip.epoch}: {tooltip.value}
        </div>
      )}
      {showLegend && series.length > 1 && (
        <div className="chart-legend-row">
          {series.map((s) => (
            <span key={s.label} className="legend-chip">
              <span className="legend-swatch" style={{ background: s.color }} />
              {s.label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
