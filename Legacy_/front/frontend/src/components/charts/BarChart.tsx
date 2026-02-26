/**
 * BarChart — Canvas-based bar chart for per-class metrics, epoch timing, etc.
 */
import { useRef, useEffect, useState } from 'react';

export interface BarData {
  label: string;
  value: number;
  color?: string;
}

interface Props {
  data: BarData[];
  yLabel?: string;
  height?: number;
  color?: string;
  horizontal?: boolean;
}

export default function BarChart({
  data,
  yLabel = '',
  height = 200,
  color = '#6366f1',
  horizontal = false,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    label: string;
    value: string;
  } | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || data.length === 0) return;

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

    const pad = { top: 15, right: 15, bottom: 40, left: 50 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    const maxVal = Math.max(...data.map((d) => d.value), 0.001);

    // Clear
    ctx.clearRect(0, 0, w, h);

    if (!horizontal) {
      const barW = Math.min(plotW / data.length - 4, 40);
      const gap = (plotW - barW * data.length) / (data.length + 1);

      // Grid
      ctx.strokeStyle = 'rgba(255,255,255,0.06)';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const y = pad.top + (plotH / 4) * i;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(w - pad.right, y);
        ctx.stroke();
      }

      // Y labels
      ctx.fillStyle = 'rgba(255,255,255,0.4)';
      ctx.font = '10px monospace';
      ctx.textAlign = 'right';
      for (let i = 0; i <= 4; i++) {
        const val = maxVal - (maxVal / 4) * i;
        const y = pad.top + (plotH / 4) * i;
        ctx.fillText(val.toFixed(2), pad.left - 5, y + 3);
      }

      // Bars
      data.forEach((d, i) => {
        const x = pad.left + gap + i * (barW + gap);
        const barH = (d.value / maxVal) * plotH;
        const y = pad.top + plotH - barH;

        // Bar with gradient
        const gradient = ctx.createLinearGradient(x, y, x, pad.top + plotH);
        const c = d.color || color;
        gradient.addColorStop(0, c);
        gradient.addColorStop(1, c + '40');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.roundRect(x, y, barW, barH, [3, 3, 0, 0]);
        ctx.fill();

        // Label
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'center';
        const label = d.label.length > 8 ? d.label.slice(0, 7) + '…' : d.label;
        ctx.fillText(label, x + barW / 2, h - pad.bottom + 15);
      });
    }

    // Y-axis label
    if (yLabel) {
      ctx.fillStyle = 'rgba(255,255,255,0.5)';
      ctx.font = '11px sans-serif';
      ctx.save();
      ctx.translate(12, pad.top + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.fillText(yLabel, 0, 0);
      ctx.restore();
    }

    // Tooltip
    const handleMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;

      const barW = Math.min(plotW / data.length - 4, 40);
      const gap = (plotW - barW * data.length) / (data.length + 1);

      for (let i = 0; i < data.length; i++) {
        const x = pad.left + gap + i * (barW + gap);
        if (mx >= x && mx <= x + barW) {
          setTooltip({
            x: mx,
            y: pad.top + plotH - (data[i].value / maxVal) * plotH - 10,
            label: data[i].label,
            value: data[i].value.toFixed(4),
          });
          return;
        }
      }
      setTooltip(null);
    };

    canvas.addEventListener('mousemove', handleMove);
    canvas.addEventListener('mouseleave', () => setTooltip(null));
    return () => {
      canvas.removeEventListener('mousemove', handleMove);
      canvas.removeEventListener('mouseleave', () => setTooltip(null));
    };
  }, [data, height, color, yLabel, horizontal]);

  return (
    <div ref={containerRef} className="chart-container" style={{ position: 'relative' }}>
      <canvas ref={canvasRef} />
      {tooltip && (
        <div className="chart-tooltip" style={{ left: tooltip.x, top: tooltip.y }}>
          <strong>{tooltip.label}</strong>: {tooltip.value}
        </div>
      )}
    </div>
  );
}
