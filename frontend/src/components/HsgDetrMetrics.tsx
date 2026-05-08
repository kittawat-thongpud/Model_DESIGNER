import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import { Brain, Activity, AlertTriangle, Zap, Layers } from 'lucide-react';

// ─── Types ───────────────────────────────────────────────────────────────────

export interface HsgDetrMetricsEntry {
  epoch: number;
  // TGSR SGB metrics per scale (chan_w = multiplicative channel weights)
  'sgb/P3_ratio'?: number;
  'sgb/P3_chan_w_mean'?: number;         // TGSR: channel recalibration weight mean
  'sgb/P3_chan_w_deviation'?: number;   // TGSR: |chan_w - 1.0| (active contribution)
  'sgb/P3_saliency_mean'?: number;
  'sgb/P3_k_over_N'?: number;
  'sgb/P4_ratio'?: number;
  'sgb/P4_chan_w_mean'?: number;
  'sgb/P4_chan_w_deviation'?: number;
  'sgb/P4_saliency_mean'?: number;
  'sgb/P4_k_over_N'?: number;
  'sgb/P5_ratio'?: number;
  'sgb/P5_chan_w_mean'?: number;
  'sgb/P5_chan_w_deviation'?: number;
  'sgb/P5_saliency_mean'?: number;
  'sgb/P5_k_over_N'?: number;
  // Decoder
  'decoder/alpha'?: number;
  'decoder/num_queries'?: number;
  // Gradients (TGSR: recal = chan_fc, attn = q/k/v projections)
  'grad/backbone_norm'?: number;
  'grad/neck_norm'?: number;
  'grad/sgb_norm'?: number;
  'grad/sgb_recal_norm'?: number;   // TGSR: chan_fc + spatial_conv
  'grad/sgb_attn_norm'?: number;   // TGSR: q/k/v projections
  'grad/decoder_norm'?: number;
  'grad/has_nan'?: number;
  'grad/has_inf'?: number;
  [key: string]: number | undefined;
}

interface Props {
  history: HsgDetrMetricsEntry[];
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

const SCALES = ['P3', 'P4', 'P5'] as const;
const SCALE_COLORS: Record<string, string> = { P3: '#f59e0b', P4: '#38bdf8', P5: '#a78bfa' };

const fmt = (v: number | undefined, digits = 4) =>
  v != null ? (v < 0.001 && v > 0 ? v.toExponential(2) : v.toFixed(digits)) : '—';

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload?.length) {
    return (
      <div className="bg-slate-950/90 border border-slate-800 p-2.5 rounded-lg shadow-xl backdrop-blur-md text-xs z-50">
        <p className="text-slate-400 font-bold mb-1.5 border-b border-slate-800 pb-1">Epoch {label}</p>
        {payload.map((entry: any, i: number) => (
          <div key={i} className="flex items-center justify-between gap-4 mb-0.5">
            <span className="flex items-center gap-1.5" style={{ color: entry.color }}>
              <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: entry.color }} />
              {entry.name}
            </span>
            <span className="font-mono text-white">
              {typeof entry.value === 'number'
                ? entry.value < 0.001 && entry.value > 0
                  ? entry.value.toExponential(2)
                  : entry.value.toFixed(4)
                : '—'}
            </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

// ─── Sub-components ──────────────────────────────────────────────────────────

const StatusBadge = ({ ok, label }: { ok: boolean; label: string }) => (
  <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border ${
    ok
      ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
      : 'bg-red-500/10 text-red-400 border-red-500/20'
  }`}>
    {ok ? '✓' : '✗'} {label}
  </span>
);

const MetricCard = ({ label, value, color, sub }: { label: string; value: string; color?: string; sub?: string }) => (
  <div className="bg-slate-900/60 border border-slate-800 rounded-lg px-3 py-2 min-w-[90px]">
    <div className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</div>
    <div className={`text-sm font-mono font-bold ${color || 'text-white'}`}>{value}</div>
    {sub && <div className="text-[9px] text-slate-600">{sub}</div>}
  </div>
);

// ─── Main Component ──────────────────────────────────────────────────────────

const HsgDetrMetrics: React.FC<Props> = ({ history }) => {
  if (!history || history.length === 0) return null;

  const latest = history[history.length - 1];
  const hasNan = latest['grad/has_nan'] === 1;
  const hasInf = latest['grad/has_inf'] === 1;

  return (
    <div className="space-y-6">
      {/* Section Header */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-violet-500/10 border border-violet-500/20 flex items-center justify-center">
          <Brain size={16} className="text-violet-400" />
        </div>
        <div>
          <h3 className="text-white font-semibold text-sm">HSG-DETR Internals</h3>
          <p className="text-[10px] text-slate-500 uppercase tracking-wider">Sparse Global Token Block & Decoder Metrics</p>
        </div>
        {/* Health badges */}
        <div className="ml-auto flex gap-2">
          <StatusBadge ok={!hasNan} label={hasNan ? 'NaN!' : 'No NaN'} />
          <StatusBadge ok={!hasInf} label={hasInf ? 'Inf!' : 'No Inf'} />
        </div>
      </div>

      {/* NaN/Inf Warning */}
      {(hasNan || hasInf) && (
        <div className="bg-red-500/5 border border-red-500/20 rounded-lg p-3 flex items-center gap-3 text-red-400 text-xs">
          <AlertTriangle size={16} />
          <span>Gradient anomaly detected at epoch {latest.epoch}. Consider reducing learning rate or checking data pipeline.</span>
        </div>
      )}

      {/* KPI Row: Decoder Alpha + TGSR Chan W */}
      <div className="flex flex-wrap gap-3">
        <MetricCard
          label="Decoder α"
          value={fmt(latest['decoder/alpha'], 3)}
          color="text-emerald-400"
          sub={`target 0.30 | epoch ${latest.epoch}`}
        />
        {SCALES.map(s => {
          const dev = latest[`sgb/${s}_chan_w_deviation`];
          // TGSR: deviation > 0.05 = active learning
          const isActive = dev != null && dev > 0.05;
          return (
            <MetricCard
              key={s}
              label={`${s} chan_w`}
              value={fmt(latest[`sgb/${s}_chan_w_mean`])}
              color={isActive ? 'text-emerald-400' : 'text-amber-400'}
              sub={`dev=${fmt(dev, 3)} | ratio=${fmt(latest[`sgb/${s}_ratio`], 3)}`}
            />
          );
        })}
        <MetricCard
          label="Grad SGB Recal"
          value={fmt(latest['grad/sgb_recal_norm'], 4)}
          color={latest['grad/sgb_recal_norm'] != null && latest['grad/sgb_recal_norm']! > 0.01
            ? 'text-emerald-400' : 'text-amber-400'}
          sub="chan_fc (must > 0)"
        />
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

        {/* Chart 1: Decoder Alpha Schedule */}
        <div className="bg-[#0f1117] border border-slate-800 rounded-xl overflow-hidden">
          <div className="px-4 py-2.5 border-b border-slate-800/50 bg-slate-900/30 flex items-center gap-2">
            <Zap size={14} className="text-emerald-400" />
            <span className="text-xs font-semibold text-slate-200">Decoder Alpha Schedule</span>
          </div>
          <div className="p-3 h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
                <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
                <YAxis stroke="#475569" tick={{ fontSize: 9 }} domain={[0, 0.35]} />
                <Tooltip content={<CustomTooltip />} />
                <Line type="monotone" dataKey="decoder/alpha" name="Alpha" stroke="#10b981" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 2: TGSR Chan W Mean (per scale) */}
        <div className="bg-[#0f1117] border border-slate-800 rounded-xl overflow-hidden">
          <div className="px-4 py-2.5 border-b border-slate-800/50 bg-slate-900/30 flex items-center gap-2">
            <Layers size={14} className="text-violet-400" />
            <span className="text-xs font-semibold text-slate-200">TGSR Chan W Mean (multiplicative)</span>
          </div>
          <div className="p-3 h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
                <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
                <YAxis stroke="#475569" tick={{ fontSize: 9 }} domain={[0.8, 1.2]} />
                <Tooltip content={<CustomTooltip />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
                {SCALES.map(s => (
                  <Line key={s} type="monotone" dataKey={`sgb/${s}_chan_w_mean`} name={s} stroke={SCALE_COLORS[s]} strokeWidth={2} dot={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 3: TGSR Chan W Deviation (active contribution indicator) */}
        <div className="bg-[#0f1117] border border-slate-800 rounded-xl overflow-hidden">
          <div className="px-4 py-2.5 border-b border-slate-800/50 bg-slate-900/30 flex items-center gap-2">
            <Activity size={14} className="text-rose-400" />
            <span className="text-xs font-semibold text-slate-200">TGSR Chan W Deviation |1-w| (active?)</span>
          </div>
          <div className="p-3 h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
                <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
                <YAxis stroke="#475569" tick={{ fontSize: 9 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
                {SCALES.map(s => (
                  <Line key={s} type="monotone" dataKey={`sgb/${s}_chan_w_deviation`} name={s} stroke={SCALE_COLORS[s]} strokeWidth={2} dot={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 4: TGSR Gradient Norms */}
        <div className="bg-[#0f1117] border border-slate-800 rounded-xl overflow-hidden">
          <div className="px-4 py-2.5 border-b border-slate-800/50 bg-slate-900/30 flex items-center gap-2">
            <Activity size={14} className="text-blue-400" />
            <span className="text-xs font-semibold text-slate-200">TGSR Gradient Norms (max per group)</span>
          </div>
          <div className="p-3 h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
                <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
                <YAxis stroke="#475569" tick={{ fontSize: 9 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
                <Line type="monotone" dataKey="grad/backbone_norm" name="Backbone" stroke="#64748b" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="grad/neck_norm" name="Neck" stroke="#fb923c" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="grad/sgb_recal_norm" name="SGB Recal" stroke="#a78bfa" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="grad/sgb_attn_norm" name="SGB Attn" stroke="#f43f5e" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="grad/decoder_norm" name="Decoder" stroke="#10b981" strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>

      {/* TGSR SGB Detail Table */}
      <div className="bg-slate-900/40 border border-slate-800 rounded-xl overflow-hidden">
        <div className="px-4 py-2.5 border-b border-slate-800/50 bg-slate-900/30">
          <span className="text-xs font-semibold text-slate-200">TGSR Token Selection (Latest — Epoch {latest.epoch})</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-500 uppercase tracking-wider border-b border-slate-800">
                <th className="px-4 py-2 text-left">Scale</th>
                <th className="px-4 py-2 text-right">Ratio (ρ)</th>
                <th className="px-4 py-2 text-right">k / N</th>
                <th className="px-4 py-2 text-right">Chan W</th>
                <th className="px-4 py-2 text-right">Deviation</th>
                <th className="px-4 py-2 text-right">Saliency</th>
                <th className="px-4 py-2 text-center">Status</th>
              </tr>
            </thead>
            <tbody>
              {SCALES.map(s => {
                const chanW = latest[`sgb/${s}_chan_w_mean`];
                const dev = latest[`sgb/${s}_chan_w_deviation`];
                // TGSR: deviation > 0.05 = active contribution (target at epoch 5)
                const isActive = dev != null && dev > 0.05;
                const isLearning = dev != null && dev > 0.10; // Strong learning by epoch 30
                return (
                  <tr key={s} className="border-b border-slate-800/50 hover:bg-slate-800/20">
                    <td className="px-4 py-2 font-bold" style={{ color: SCALE_COLORS[s] }}>{s}</td>
                    <td className="px-4 py-2 text-right font-mono text-slate-300">{fmt(latest[`sgb/${s}_ratio`], 3)}</td>
                    <td className="px-4 py-2 text-right font-mono text-slate-300">{fmt(latest[`sgb/${s}_k_over_N`], 4)}</td>
                    <td className={`px-4 py-2 text-right font-mono ${chanW != null && chanW > 0.9 && chanW < 1.1 ? 'text-slate-300' : 'text-amber-400'}`}>
                      {fmt(chanW)}
                    </td>
                    <td className={`px-4 py-2 text-right font-mono ${isLearning ? 'text-emerald-400' : isActive ? 'text-blue-400' : 'text-slate-500'}`}>
                      {fmt(dev, 3)}
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-slate-300">{fmt(latest[`sgb/${s}_saliency_mean`])}</td>
                    <td className="px-4 py-2 text-center">
                      {isLearning
                        ? <span className="text-emerald-400 text-[10px]">● Strong</span>
                        : isActive
                          ? <span className="text-blue-400 text-[10px]">◐ Active</span>
                          : <span className="text-slate-500 text-[10px]">○ Identity</span>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default HsgDetrMetrics;
