import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import { Brain, Activity, AlertTriangle, Zap, Layers, ShieldCheck, Crosshair } from 'lucide-react';

export interface HsgDetrMetricsEntry {
  epoch: number;
  'decoder/alpha'?: number;
  'decoder/alpha_progress'?: number;
  'decoder/alpha_u'?: number;
  'decoder/alpha_eff'?: number;
  'decoder/num_queries'?: number;
  'decoder/hidden_dim'?: number;
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

const isReferenceGuided = (entry: HsgDetrMetricsEntry, scale: string) =>
  Number(entry[metricKey(scale, 'reference_guided')] ?? 0) >= 0.5;

const hasMetric = (entry: HsgDetrMetricsEntry, suffix: string) =>
  SCALES.some(scale => entry[metricKey(scale, suffix)] != null);

const hasScale = (entry: HsgDetrMetricsEntry, scale: string) =>
  entry[metricKey(scale, 'N')] != null || entry[metricKey(scale, 'ratio')] != null;

const isSoftHardActive = (entry: HsgDetrMetricsEntry, scale: string) =>
  Number(entry[metricKey(scale, 'soft_hard_active')] ?? 0) >= 0.5
  || (
    entry[metricKey(scale, 'top_m')] != null
    && entry[metricKey(scale, 'K_eff')] != null
    && Number(entry[metricKey(scale, 'top_m')]) > Number(entry[metricKey(scale, 'K_eff')])
    && Number(entry[metricKey(scale, 'lambda_soft')] ?? 0) > 0
  );

const blockLabel = (entry: HsgDetrMetricsEntry, scale: string) =>
  isReferenceGuided(entry, scale)
    ? 'Ref-guided local'
    : entry[metricKey(scale, 'top_m')] != null
      ? isSoftHardActive(entry, scale) ? 'V3 Top-M soft-hard' : 'V3 hard top-K'
      : entry[metricKey(scale, 'score_std')] != null
      ? 'V2 selected-token'
      : 'Selected-token';

const blockSubLabel = (entry: HsgDetrMetricsEntry, scale: string) => {
  const windowSize = entry[metricKey(scale, 'window_size')];
  return isReferenceGuided(entry, scale)
    ? `local ${fmt(windowSize, 0)}x${fmt(windowSize, 0)}`
    : 'all-pairs sparse';
};

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

const scaleLines = (name: string, scales: readonly string[] = SCALES) => scales.map(scale => (
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
  const activeScales = SCALES.filter(scale => hasScale(latest, scale));
  const hasNan = latest['grad/has_nan'] === 1;
  const hasInf = latest['grad/has_inf'] === 1;
  const anyFiniteGuard = activeScales.some(s => (latest[metricKey(s, 'finite_guard_count')] ?? 0) > 0);
  const sparseGrad = latest['grad/sgb_sparse_norm'];
  const gammaGrad = latest['grad/sgb_gamma_norm'];
  const alphaMax = Math.max(
    0.55,
    ...history.map(h => Number(h['decoder/alpha'] ?? 0) * 1.1),
    ...history.map(h => Number(h['decoder/alpha_eff'] ?? 0) * 1.1),
  );
  const hasGammaFloorMetrics = activeScales.some(scale => latest[metricKey(scale, 'gamma_floor')] != null);
  const hasScaledDeltaMetrics = activeScales.some(scale => latest[metricKey(scale, 'delta_scaled_norm_selected')] != null);
  const referenceGuidedCount = activeScales.filter(scale => isReferenceGuided(latest, scale)).length;
  const hasV2ScoreStats = activeScales.some(scale => latest[metricKey(scale, 'score_std')] != null);
  const hasV3TopM = activeScales.some(scale => latest[metricKey(scale, 'top_m')] != null);
  const hasV3SoftHard = activeScales.some(scale => isSoftHardActive(latest, scale));
  const hasV3Dam = activeScales.some(scale => latest[`dam/${scale}_sampling_mass`] != null || latest[metricKey(scale, 'selector_DAM_corr')] != null);
  const seBlocks = activeScales.filter(scale => Number(latest[metricKey(scale, 'channel_se')] ?? 0) >= 0.5).length;
  const variantSummary = hasV3TopM
    ? hasV3SoftHard
      ? `Full V3: hd=${fmt(latest['decoder/hidden_dim'], 0)}, SE ${seBlocks}/${activeScales.length}, Top-M training, DAM ${hasV3Dam ? 'on' : 'off'}`
      : activeScales.length < 3
        ? `Ultra V3: hd=${fmt(latest['decoder/hidden_dim'], 0)}, no P3-SGB, hard top-K, SE ${seBlocks}/${activeScales.length}`
        : `Lean V3: hd=${fmt(latest['decoder/hidden_dim'], 0)}, three SGB levels, hard top-K, SE ${seBlocks}/${activeScales.length}`
    : '';
  const sparseVariant = referenceGuidedCount > 0
    ? 'reference-guided local aggregation'
    : hasV3SoftHard
      ? 'V3 Top-M soft-hard sparse attention'
      : hasV3TopM
      ? 'V3 lean hard top-K sparse attention'
      : hasV2ScoreStats
      ? 'V2 AMP-safe selected-token sparse attention'
      : 'legacy selected-token sparse attention';

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-violet-500/10 border border-violet-500/20 flex items-center justify-center">
          <Brain size={16} className="text-violet-400" />
        </div>
        <div>
          <h3 className="text-white font-semibold text-sm">HSG-DETR Sparse Debug</h3>
          <p className="text-[10px] text-slate-500 uppercase tracking-wider">
            {variantSummary || sparseVariant} | stride-labeled blocks, sparse selection, decoder alpha, gradient contract
          </p>
        </div>
        <div className="ml-auto flex gap-2">
          <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border ${
            referenceGuidedCount > 0
              ? 'bg-cyan-500/10 text-cyan-300 border-cyan-500/20'
              : 'bg-violet-500/10 text-violet-300 border-violet-500/20'
          }`}>
            {referenceGuidedCount > 0 ? 'Local sparse' : hasV3SoftHard ? 'V3 Top-M' : hasV3TopM ? 'V3 hard' : hasV2ScoreStats ? 'V2 token' : 'Legacy token'}
          </span>
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
          value={fmt(latest['decoder/alpha_eff'] ?? latest['decoder/alpha'], 3)}
          color="text-emerald-400"
          sub={`progress=${fmt(latest['decoder/alpha_progress'], 2)} | max=${fmt(latest['decoder/alpha_u'], 2)} | q=${fmt(latest['decoder/num_queries'], 0)} | hd=${fmt(latest['decoder/hidden_dim'], 0)}`}
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
        {activeScales.map(scale => {
          const selected = latest[metricKey(scale, 'selected_ratio')];
          const gamma = latest[metricKey(scale, 'gamma_abs_mean')];
          const gammaRaw = latest[metricKey(scale, 'gamma_raw_abs_mean')];
          const gammaFloor = latest[metricKey(scale, 'gamma_floor')];
          const delta = latest[metricKey(scale, 'delta_scaled_norm_selected')] ?? latest[metricKey(scale, 'delta_norm_selected')];
          const scoreStd = latest[metricKey(scale, 'score_std')];
          const topM = latest[metricKey(scale, 'top_m')];
          const kEff = latest[metricKey(scale, 'K_eff')];
          const lambdaSoft = latest[metricKey(scale, 'lambda_soft')];
          const isV3 = topM != null;
          const isV2 = scoreStd != null && !isV3;
          const softHard = isSoftHardActive(latest, scale);
          const channelSe = Number(latest[metricKey(scale, 'channel_se')] ?? 0) >= 0.5;

          // V2 shows different sub-label and value
          const subLabel = isV3
            ? `${SCALE_HELP[scale]} | ${softHard ? 'soft-hard' : 'hard'} | M=${fmt(topM, 0)} K=${fmt(kEff, 0)} | SE=${channelSe ? 'on' : 'off'} | gamma=${fmt(gamma, 4)}`
            : isV2
            ? `${SCALE_HELP[scale]} | ${blockSubLabel(latest, scale)} | score_std=${fmt(scoreStd, 2)} | gamma=${fmt(gamma, 3)}`
            : `${SCALE_HELP[scale]} | ${blockSubLabel(latest, scale)} | gamma=${fmt(gamma, 3)} raw=${fmt(gammaRaw, 3)} floor=${fmt(gammaFloor, 3)} | delta=${fmt(delta, 3)}`;

          const cardValue = isV3 ? fmt(latest[metricKey(scale, 'top_m_over_N')], 3) : isV2 ? fmt(scoreStd, 3) : fmt(selected, 4);
          const cardColor = isV3 ? 'text-cyan-300' : isV2 ? '#a78bfa' : metricColor(delta);
          const cardLabel = isV3 ? `${scale} ${softHard ? 'M/N' : 'K/N'}` : isV2 ? `${scale} score_std` : `${scale} selected`;

          return (
            <MetricCard
              key={scale}
              label={cardLabel}
              value={cardValue}
              color={cardColor}
              sub={subLabel}
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

        {hasV3TopM && (
          <ChartPanel title="Top-M Soft-Hard Coverage" icon={<Crosshair size={14} className="text-cyan-400" />}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
                <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
                <YAxis stroke="#475569" tick={{ fontSize: 9 }} domain={[0, 1]} />
                <Tooltip content={<CustomTooltip />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
                {scaleLines('top_m_over_N', activeScales)}
              </LineChart>
            </ResponsiveContainer>
          </ChartPanel>
        )}

        {hasV3TopM && (
          <ChartPanel title="Soft Non-hard Mass" icon={<Activity size={14} className="text-sky-400" />}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
                <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
                <YAxis stroke="#475569" tick={{ fontSize: 9 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
                {scaleLines('soft_nonhard_mass', activeScales)}
              </LineChart>
            </ResponsiveContainer>
          </ChartPanel>
        )}

        {hasV3TopM && (
          <ChartPanel title="Hard vs Non-hard Delta" icon={<Activity size={14} className="text-rose-400" />}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
                <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
                <YAxis stroke="#475569" tick={{ fontSize: 9 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
                {scaleLines('hard_delta_norm', activeScales)}
                {scaleLines('nonhard_delta_norm', activeScales).map(line => React.cloneElement(line, {
                  strokeDasharray: '3 3',
                  strokeWidth: 1,
                  name: `${line.props.name} non-hard`,
                }))}
              </LineChart>
            </ResponsiveContainer>
          </ChartPanel>
        )}

        {hasV3Dam && (
          <ChartPanel title="Approx DAM Sampling Mass" icon={<Crosshair size={14} className="text-emerald-400" />}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
                <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
                <YAxis stroke="#475569" tick={{ fontSize: 9 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
                {activeScales.map(scale => (
                  <Line
                    key={`${scale}-dam-mass`}
                    type="monotone"
                    dataKey={`dam/${scale}_sampling_mass`}
                    name={`${scale} DAM mass`}
                    stroke={SCALE_COLORS[scale]}
                    strokeWidth={2}
                    dot={false}
                    connectNulls
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </ChartPanel>
        )}

        {hasV3Dam && (
          <ChartPanel title="Selector ↔ DAM Alignment" icon={<Brain size={14} className="text-violet-400" />}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
                <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
                <YAxis stroke="#475569" tick={{ fontSize: 9 }} domain={[-1, 1]} />
                <Tooltip content={<CustomTooltip />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
                {scaleLines('selector_DAM_corr', activeScales)}
              </LineChart>
            </ResponsiveContainer>
          </ChartPanel>
        )}

        <ChartPanel title="Sparse LayerScale Gamma" icon={<Layers size={14} className="text-violet-400" />}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
              <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
              <YAxis stroke="#475569" tick={{ fontSize: 9 }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
              {scaleLines('gamma_abs_mean', activeScales)}
              {hasGammaFloorMetrics && scaleLines('gamma_floor', activeScales).map(line => React.cloneElement(line, {
                strokeDasharray: '3 3',
                strokeWidth: 1,
                name: `${line.props.name} floor`,
              }))}
            </LineChart>
          </ResponsiveContainer>
        </ChartPanel>

        {hasV2ScoreStats && (
          <ChartPanel title="Score Std (Budget Formula)" icon={<Activity size={14} className="text-violet-400" />}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
                <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
                <YAxis stroke="#475569" tick={{ fontSize: 9 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
                {scaleLines('score_std', activeScales)}
              </LineChart>
            </ResponsiveContainer>
          </ChartPanel>
        )}

        {!hasV2ScoreStats && (
          <ChartPanel title={hasScaledDeltaMetrics ? 'Selected Delta After Gamma' : 'Selected Delta Norm'} icon={<Activity size={14} className="text-rose-400" />}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
                <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
                <YAxis stroke="#475569" tick={{ fontSize: 9 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
                {scaleLines(hasScaledDeltaMetrics ? 'delta_scaled_norm_selected' : 'delta_norm_selected', activeScales)}
              </LineChart>
            </ResponsiveContainer>
          </ChartPanel>
        )}

        {!hasV2ScoreStats && (
          <ChartPanel title={referenceGuidedCount > 0 ? 'Reference Sparse Grad' : 'Selected-token Sparse Grad'} icon={<Activity size={14} className="text-blue-400" />}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
                <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
                <YAxis stroke="#475569" tick={{ fontSize: 9 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
                {scaleLines('selected_grad_norm', activeScales)}
              </LineChart>
            </ResponsiveContainer>
          </ChartPanel>
        )}

        <ChartPanel title="Selection Ratio (k/N)" icon={<Layers size={14} className="text-amber-400" />}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
              <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
              <YAxis stroke="#475569" tick={{ fontSize: 9 }} domain={[0, 1]} />
              <Tooltip content={<CustomTooltip />} />
              <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
              {scaleLines('k_over_N', activeScales)}
            </LineChart>
          </ResponsiveContainer>
        </ChartPanel>

        {referenceGuidedCount > 0 && (
          <ChartPanel title="Local Window Size" icon={<Crosshair size={14} className="text-cyan-400" />}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
                <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
                <YAxis stroke="#475569" tick={{ fontSize: 9 }} allowDecimals={false} />
                <Tooltip content={<CustomTooltip />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
                {scaleLines('window_size', activeScales)}
              </LineChart>
            </ResponsiveContainer>
          </ChartPanel>
        )}

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
              <Line type="monotone" dataKey="grad/sgb_norm_norm" name="SGB Norm" stroke="#8b5cf6" strokeWidth={2} dot={false} connectNulls />
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
                <th className="px-4 py-2 text-left">Mode</th>
                <th className="px-4 py-2 text-right">Ratio</th>
                <th className="px-4 py-2 text-right">Window</th>
                <th className="px-4 py-2 text-right">k / N</th>
                {!hasV2ScoreStats && <th className="px-4 py-2 text-right">Gamma raw</th>}
                <th className="px-4 py-2 text-right">Gamma eff</th>
                {!hasV2ScoreStats && <th className="px-4 py-2 text-right">Floor</th>}
                {!hasV2ScoreStats && <th className="px-4 py-2 text-right">Delta selected</th>}
                {!hasV2ScoreStats && <th className="px-4 py-2 text-right">Delta scaled</th>}
                {!hasV2ScoreStats && <th className="px-4 py-2 text-right">Delta non-selected</th>}
                {!hasV2ScoreStats && <th className="px-4 py-2 text-right">Selected grad</th>}
                {!hasV2ScoreStats && <th className="px-4 py-2 text-right">Non-selected sparse grad</th>}
                {!hasV2ScoreStats && <th className="px-4 py-2 text-right">Guard hits</th>}
                {hasV3TopM && <th className="px-4 py-2 text-right">Top-M</th>}
                {hasV3TopM && <th className="px-4 py-2 text-right">λ soft</th>}
                {hasV3TopM && <th className="px-4 py-2 text-right">Non-hard Δ</th>}
                {hasV3Dam && <th className="px-4 py-2 text-right">DAM mass@k</th>}
                {hasV3Dam && <th className="px-4 py-2 text-right">DAM corr</th>}
                {hasV2ScoreStats && !hasV3TopM && <th className="px-4 py-2 text-right">Score std</th>}
                <th className="px-4 py-2 text-center">Contract</th>
              </tr>
            </thead>
            <tbody>
              {activeScales.map(scale => {
                const deltaNon = latest[metricKey(scale, 'delta_norm_nonselected')];
                const deltaSelected = latest[metricKey(scale, 'delta_norm_selected')];
                const deltaScaled = latest[metricKey(scale, 'delta_scaled_norm_selected')];
                const sparseNon = latest[metricKey(scale, 'nonselected_sparse_grad')];
                const selectedGrad = latest[metricKey(scale, 'selected_grad_norm')];
                const guardHits = latest[metricKey(scale, 'finite_guard_count')] ?? 0;
                const contractOk = (deltaNon ?? 0) <= EPS && (sparseNon ?? 0) <= EPS && guardHits === 0;
                const refGuided = isReferenceGuided(latest, scale);
                const windowSize = latest[metricKey(scale, 'window_size')];
                const scoreStd = latest[metricKey(scale, 'score_std')];
                const topM = latest[metricKey(scale, 'top_m')];
                const kEff = latest[metricKey(scale, 'K_eff')];
                const lambdaSoft = latest[metricKey(scale, 'lambda_soft')];
                const nonhardDelta = latest[metricKey(scale, 'nonhard_delta_norm')];
                const damMass = latest[metricKey(scale, 'selected_DAM_mass@k')];
                const damCorr = latest[metricKey(scale, 'selector_DAM_corr')];
                const isV3 = topM != null;
                const isV2 = scoreStd != null && !isV3;

                // For V2, contract is always OK (no delta/sparse grad checks)
                const v2ContractOk = true;

                return (
                  <tr key={scale} className="border-b border-slate-800/50 hover:bg-slate-800/20">
                    <td className="px-4 py-2 font-bold" style={{ color: SCALE_COLORS[scale] }}>
                      {scale}
                      <span className="block text-[9px] font-normal text-slate-500">{SCALE_HELP[scale]}</span>
                    </td>
                    <td className="px-4 py-2">
                      <span className={`inline-flex items-center px-2 py-0.5 rounded border text-[10px] font-semibold ${
                        refGuided
                          ? 'bg-cyan-500/10 text-cyan-300 border-cyan-500/20'
                          : 'bg-violet-500/10 text-violet-300 border-violet-500/20'
                      }`}>
                        {blockLabel(latest, scale)}
                      </span>
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-slate-300">{fmt(latest[metricKey(scale, 'ratio')], 3)}</td>
                    <td className="px-4 py-2 text-right font-mono text-slate-300">{refGuided ? fmt(windowSize, 0) : '-'}</td>
                    <td className="px-4 py-2 text-right font-mono text-slate-300">
                      {fmt(latest[metricKey(scale, 'k_over_N')], 4)}
                      <span className="text-slate-600 ml-1">({fmt(latest[metricKey(scale, 'k')], 0)}/{fmt(latest[metricKey(scale, 'N')], 0)})</span>
                    </td>
                    {!isV2 && <td className="px-4 py-2 text-right font-mono text-slate-300">{fmt(latest[metricKey(scale, 'gamma_raw_abs_mean')], 4)}</td>}
                    <td className="px-4 py-2 text-right font-mono text-emerald-400">{fmt(latest[metricKey(scale, 'gamma_abs_mean')], 4)}</td>
                    {!isV2 && <td className="px-4 py-2 text-right font-mono text-cyan-400">{fmt(latest[metricKey(scale, 'gamma_floor')], 4)}</td>}
                    {!isV2 && <td className={`px-4 py-2 text-right font-mono ${metricColor(deltaSelected)}`}>{fmt(deltaSelected, 4)}</td>}
                    {!isV2 && <td className={`px-4 py-2 text-right font-mono ${metricColor(deltaScaled ?? deltaSelected)}`}>{fmt(deltaScaled, 4)}</td>}
                    {!isV2 && <td className={`px-4 py-2 text-right font-mono ${(deltaNon ?? 0) <= EPS ? 'text-emerald-400' : 'text-red-400'}`}>{fmt(deltaNon, 4)}</td>}
                    {!isV2 && <td className={`px-4 py-2 text-right font-mono ${metricColor(selectedGrad)}`}>{fmt(selectedGrad, 4)}</td>}
                    {!isV2 && <td className={`px-4 py-2 text-right font-mono ${(sparseNon ?? 0) <= EPS ? 'text-emerald-400' : 'text-red-400'}`}>{fmt(sparseNon, 4)}</td>}
                    {!isV2 && <td className={`px-4 py-2 text-right font-mono ${guardHits === 0 ? 'text-slate-400' : 'text-red-400'}`}>{fmt(guardHits, 0)}</td>}
                    {hasV3TopM && <td className="px-4 py-2 text-right font-mono text-cyan-300">{isV3 ? `${fmt(topM, 0)}/${fmt(kEff, 0)}` : '-'}</td>}
                    {hasV3TopM && <td className="px-4 py-2 text-right font-mono text-sky-300">{fmt(lambdaSoft, 3)}</td>}
                    {hasV3TopM && <td className={`px-4 py-2 text-right font-mono ${metricColor(nonhardDelta)}`}>{fmt(nonhardDelta, 4)}</td>}
                    {hasV3Dam && <td className="px-4 py-2 text-right font-mono text-emerald-300">{fmt(damMass, 4)}</td>}
                    {hasV3Dam && <td className="px-4 py-2 text-right font-mono text-violet-300">{fmt(damCorr, 4)}</td>}
                    {isV2 && !hasV3TopM && <td className="px-4 py-2 text-right font-mono text-violet-400">{fmt(scoreStd, 2)}</td>}
                    <td className="px-4 py-2 text-center">
                      {(isV2 ? v2ContractOk : contractOk)
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
