import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import { Brain, Activity, AlertTriangle, Zap, Layers, ShieldCheck } from 'lucide-react';

export interface HsgDetrMetricsEntry {
  epoch: number;
  'decoder/alpha'?: number;
  'decoder/num_queries'?: number;
  'grad/backbone_norm'?: number;
  'grad/neck_norm'?: number;
  'grad/sgb_norm'?: number;
  'grad/sgb_sparse_norm'?: number;
  'grad/sgb_gamma_norm'?: number;
  'grad/sgb_norm_norm'?: number;
  'grad/decoder_norm'?: number;
  'grad/has_nan'?: number;
  'grad/has_inf'?: number;
  [key: string]: number | undefined;
}

interface Props {
  history: HsgDetrMetricsEntry[];
}

const SCALES = ['P3', 'P4', 'P5'] as const;
const SCALE_COLORS: Record<string, string> = { P3: '#f59e0b', P4: '#38bdf8', P5: '#a78bfa' };
const SCALE_HELP: Record<string, string> = {
  P3: 'high-res',
  P4: 'mid-res',
  P5: 'low-res',
};
const EPS = 1e-8;

const metricKey = (scale: string, name: string) => `sgb/${scale}_${name}`;

const fmt = (v: number | undefined, digits = 4) =>
  v != null ? (Math.abs(v) < 0.001 && v !== 0 ? v.toExponential(2) : v.toFixed(digits)) : '-';

const metricColor = (v: number | undefined, threshold = EPS) =>
  v != null && Math.abs(v) > threshold ? 'text-emerald-400' : 'text-amber-400';

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
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
              ? Math.abs(entry.value) < 0.001 && entry.value !== 0
                ? entry.value.toExponential(2)
                : entry.value.toFixed(4)
              : '-'}
          </span>
        </div>
      ))}
    </div>
  );
};

const StatusBadge = ({ ok, label }: { ok: boolean; label: string }) => (
  <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border ${
    ok
      ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
      : 'bg-red-500/10 text-red-400 border-red-500/20'
  }`}>
    {ok ? 'OK' : 'FAIL'} {label}
  </span>
);

const MetricCard = ({ label, value, color, sub }: { label: string; value: string; color?: string; sub?: string }) => (
  <div className="bg-slate-900/60 border border-slate-800 rounded-lg px-3 py-2 min-w-[118px]">
    <div className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</div>
    <div className={`text-sm font-mono font-bold ${color || 'text-white'}`}>{value}</div>
    {sub && <div className="text-[9px] text-slate-600">{sub}</div>}
  </div>
);

const ChartPanel = ({
  title,
  icon,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) => (
  <div className="bg-[#0f1117] border border-slate-800 rounded-xl overflow-hidden">
    <div className="px-4 py-2.5 border-b border-slate-800/50 bg-slate-900/30 flex items-center gap-2">
      {icon}
      <span className="text-xs font-semibold text-slate-200">{title}</span>
    </div>
    <div className="p-3 h-[210px]">{children}</div>
  </div>
);

const scaleLines = (name: string) => SCALES.map(scale => (
  <Line
    key={`${scale}-${name}`}
    type="monotone"
    dataKey={metricKey(scale, name)}
    name={scale}
    stroke={SCALE_COLORS[scale]}
    strokeWidth={2}
    dot={false}
    connectNulls
  />
));

const HsgDetrMetrics: React.FC<Props> = ({ history }) => {
  if (!history || history.length === 0) return null;

  const latest = history[history.length - 1];
  const hasNan = latest['grad/has_nan'] === 1;
  const hasInf = latest['grad/has_inf'] === 1;
  const anyFiniteGuard = SCALES.some(s => (latest[metricKey(s, 'finite_guard_count')] ?? 0) > 0);
  const sparseGrad = latest['grad/sgb_sparse_norm'];
  const gammaGrad = latest['grad/sgb_gamma_norm'];
  const alphaMax = Math.max(0.55, ...history.map(h => Number(h['decoder/alpha'] ?? 0) * 1.1));
  const hasGammaFloorMetrics = SCALES.some(scale => latest[metricKey(scale, 'gamma_floor')] != null);
  const hasScaledDeltaMetrics = SCALES.some(scale => latest[metricKey(scale, 'delta_scaled_norm_selected')] != null);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-violet-500/10 border border-violet-500/20 flex items-center justify-center">
          <Brain size={16} className="text-violet-400" />
        </div>
        <div>
          <h3 className="text-white font-semibold text-sm">HSG-DETR Sparse Debug</h3>
          <p className="text-[10px] text-slate-500 uppercase tracking-wider">Stride-labeled SGB, gamma floor, decoder alpha, gradient contract</p>
        </div>
        <div className="ml-auto flex gap-2">
          <StatusBadge ok={!hasNan} label={hasNan ? 'NaN' : 'No NaN'} />
          <StatusBadge ok={!hasInf} label={hasInf ? 'Inf' : 'No Inf'} />
          <StatusBadge ok={!anyFiniteGuard} label={anyFiniteGuard ? 'Guard hit' : 'Finite'} />
        </div>
      </div>

      {(hasNan || hasInf || anyFiniteGuard) && (
        <div className="bg-red-500/5 border border-red-500/20 rounded-lg p-3 flex items-center gap-3 text-red-400 text-xs">
          <AlertTriangle size={16} />
          <span>Numeric anomaly detected at epoch {latest.epoch}. Check finite guard count, AMP scale, gradients, and the latest training logs.</span>
        </div>
      )}

      <div className="flex flex-wrap gap-3">
        <MetricCard
          label="Decoder alpha"
          value={fmt(latest['decoder/alpha'], 3)}
          color="text-emerald-400"
          sub={`queries=${fmt(latest['decoder/num_queries'], 0)} | epoch ${latest.epoch}`}
        />
        <MetricCard
          label="SGB sparse grad"
          value={fmt(sparseGrad, 4)}
          color={metricColor(sparseGrad)}
          sub="q/k/v/out params"
        />
        <MetricCard
          label="Gamma grad"
          value={fmt(gammaGrad, 4)}
          color={metricColor(gammaGrad)}
          sub="LayerScale update"
        />
        {SCALES.map(scale => {
          const selected = latest[metricKey(scale, 'selected_ratio')] ?? latest[metricKey(scale, 'k_over_N')];
          const gamma = latest[metricKey(scale, 'gamma_abs_mean')];
          const gammaRaw = latest[metricKey(scale, 'gamma_raw_abs_mean')];
          const gammaFloor = latest[metricKey(scale, 'gamma_floor')];
          const delta = latest[metricKey(scale, 'delta_scaled_norm_selected')] ?? latest[metricKey(scale, 'delta_norm_selected')];
          return (
            <MetricCard
              key={scale}
              label={`${scale} selected`}
              value={fmt(selected, 4)}
              color={metricColor(delta)}
              sub={`${SCALE_HELP[scale]} | gamma=${fmt(gamma, 3)} raw=${fmt(gammaRaw, 3)} floor=${fmt(gammaFloor, 3)} | delta=${fmt(delta, 3)}`}
            />
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ChartPanel title="Decoder Alpha Schedule" icon={<Zap size={14} className="text-emerald-400" />}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
              <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
              <YAxis stroke="#475569" tick={{ fontSize: 9 }} domain={[0, alphaMax]} />
              <Tooltip content={<CustomTooltip />} />
              <Line type="monotone" dataKey="decoder/alpha" name="Alpha" stroke="#10b981" strokeWidth={2} dot={false} connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </ChartPanel>

        <ChartPanel title="Sparse LayerScale Gamma" icon={<Layers size={14} className="text-violet-400" />}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
              <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
              <YAxis stroke="#475569" tick={{ fontSize: 9 }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
              {scaleLines('gamma_abs_mean')}
              {hasGammaFloorMetrics && scaleLines('gamma_floor').map(line => React.cloneElement(line, {
                strokeDasharray: '3 3',
                strokeWidth: 1,
                name: `${line.props.name} floor`,
              }))}
            </LineChart>
          </ResponsiveContainer>
        </ChartPanel>

        <ChartPanel title={hasScaledDeltaMetrics ? 'Selected Delta After Gamma' : 'Selected Delta Norm'} icon={<Activity size={14} className="text-rose-400" />}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
              <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
              <YAxis stroke="#475569" tick={{ fontSize: 9 }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
              {scaleLines(hasScaledDeltaMetrics ? 'delta_scaled_norm_selected' : 'delta_norm_selected')}
            </LineChart>
          </ResponsiveContainer>
        </ChartPanel>

        <ChartPanel title="Selected-token Sparse Grad" icon={<Activity size={14} className="text-blue-400" />}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
              <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
              <YAxis stroke="#475569" tick={{ fontSize: 9 }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
              {scaleLines('selected_grad_norm')}
            </LineChart>
          </ResponsiveContainer>
        </ChartPanel>

        <ChartPanel title="Selection Ratio" icon={<Layers size={14} className="text-amber-400" />}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
              <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
              <YAxis stroke="#475569" tick={{ fontSize: 9 }} domain={[0, 1]} />
              <Tooltip content={<CustomTooltip />} />
              <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
              {scaleLines('selected_ratio')}
            </LineChart>
          </ResponsiveContainer>
        </ChartPanel>

        <ChartPanel title="Optimizer Grad Groups" icon={<ShieldCheck size={14} className="text-cyan-400" />}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
              <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
              <YAxis stroke="#475569" tick={{ fontSize: 9 }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
              <Line type="monotone" dataKey="grad/backbone_norm" name="Backbone" stroke="#64748b" strokeWidth={1.5} dot={false} connectNulls />
              <Line type="monotone" dataKey="grad/neck_norm" name="Neck" stroke="#fb923c" strokeWidth={1.5} dot={false} connectNulls />
              <Line type="monotone" dataKey="grad/sgb_sparse_norm" name="SGB Sparse" stroke="#f43f5e" strokeWidth={2} dot={false} connectNulls />
              <Line type="monotone" dataKey="grad/sgb_gamma_norm" name="SGB Gamma" stroke="#a78bfa" strokeWidth={2} dot={false} connectNulls />
              <Line type="monotone" dataKey="grad/decoder_norm" name="Decoder" stroke="#10b981" strokeWidth={1.5} dot={false} connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </ChartPanel>
      </div>

      <div className="bg-slate-900/40 border border-slate-800 rounded-xl overflow-hidden">
        <div className="px-4 py-2.5 border-b border-slate-800/50 bg-slate-900/30">
          <span className="text-xs font-semibold text-slate-200">Sparse Contract Latest - Epoch {latest.epoch}</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-500 uppercase tracking-wider border-b border-slate-800">
                <th className="px-4 py-2 text-left">Block</th>
                <th className="px-4 py-2 text-right">Ratio</th>
                <th className="px-4 py-2 text-right">k / N</th>
                <th className="px-4 py-2 text-right">Gamma raw</th>
                <th className="px-4 py-2 text-right">Gamma eff</th>
                <th className="px-4 py-2 text-right">Floor</th>
                <th className="px-4 py-2 text-right">Delta selected</th>
                <th className="px-4 py-2 text-right">Delta scaled</th>
                <th className="px-4 py-2 text-right">Delta non-selected</th>
                <th className="px-4 py-2 text-right">Selected grad</th>
                <th className="px-4 py-2 text-right">Non-selected sparse grad</th>
                <th className="px-4 py-2 text-right">Guard hits</th>
                <th className="px-4 py-2 text-center">Contract</th>
              </tr>
            </thead>
            <tbody>
              {SCALES.map(scale => {
                const deltaNon = latest[metricKey(scale, 'delta_norm_nonselected')];
                const deltaSelected = latest[metricKey(scale, 'delta_norm_selected')];
                const deltaScaled = latest[metricKey(scale, 'delta_scaled_norm_selected')];
                const sparseNon = latest[metricKey(scale, 'nonselected_sparse_grad')];
                const selectedGrad = latest[metricKey(scale, 'selected_grad_norm')];
                const guardHits = latest[metricKey(scale, 'finite_guard_count')] ?? 0;
                const contractOk = (deltaNon ?? 0) <= EPS && (sparseNon ?? 0) <= EPS && guardHits === 0;
                return (
                  <tr key={scale} className="border-b border-slate-800/50 hover:bg-slate-800/20">
                    <td className="px-4 py-2 font-bold" style={{ color: SCALE_COLORS[scale] }}>
                      {scale}
                      <span className="block text-[9px] font-normal text-slate-500">{SCALE_HELP[scale]}</span>
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-slate-300">{fmt(latest[metricKey(scale, 'ratio')], 3)}</td>
                    <td className="px-4 py-2 text-right font-mono text-slate-300">
                      {fmt(latest[metricKey(scale, 'k_over_N')], 4)}
                      <span className="text-slate-600 ml-1">({fmt(latest[metricKey(scale, 'k')], 0)}/{fmt(latest[metricKey(scale, 'N')], 0)})</span>
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-slate-300">{fmt(latest[metricKey(scale, 'gamma_raw_abs_mean')], 4)}</td>
                    <td className="px-4 py-2 text-right font-mono text-emerald-400">{fmt(latest[metricKey(scale, 'gamma_abs_mean')], 4)}</td>
                    <td className="px-4 py-2 text-right font-mono text-cyan-400">{fmt(latest[metricKey(scale, 'gamma_floor')], 4)}</td>
                    <td className={`px-4 py-2 text-right font-mono ${metricColor(deltaSelected)}`}>
                      {fmt(deltaSelected, 4)}
                    </td>
                    <td className={`px-4 py-2 text-right font-mono ${metricColor(deltaScaled ?? deltaSelected)}`}>
                      {fmt(deltaScaled, 4)}
                    </td>
                    <td className={`px-4 py-2 text-right font-mono ${(deltaNon ?? 0) <= EPS ? 'text-emerald-400' : 'text-red-400'}`}>
                      {fmt(deltaNon, 4)}
                    </td>
                    <td className={`px-4 py-2 text-right font-mono ${metricColor(selectedGrad)}`}>{fmt(selectedGrad, 4)}</td>
                    <td className={`px-4 py-2 text-right font-mono ${(sparseNon ?? 0) <= EPS ? 'text-emerald-400' : 'text-red-400'}`}>
                      {fmt(sparseNon, 4)}
                    </td>
                    <td className={`px-4 py-2 text-right font-mono ${guardHits === 0 ? 'text-slate-400' : 'text-red-400'}`}>{fmt(guardHits, 0)}</td>
                    <td className="px-4 py-2 text-center">
                      {contractOk
                        ? <span className="text-emerald-400 text-[10px]">PASS</span>
                        : <span className="text-red-400 text-[10px]">CHECK</span>}
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
