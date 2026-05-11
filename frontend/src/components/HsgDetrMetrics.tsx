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
  'decoder/loc_quality_mode'?: string;
  'grad/backbone_norm'?: number;
  'grad/neck_norm'?: number;
  'grad/sgb_norm'?: number;
  'grad/sgb_sparse_norm'?: number;
  'grad/sgb_gamma_norm'?: number;
  'grad/sgb_norm_norm'?: number;
  'grad/decoder_norm'?: number;
  'grad/has_nan'?: number;
  'grad/has_inf'?: number;
  [key: string]: any;
}

interface Props {
  history: HsgDetrMetricsEntry[];
}

const SCALES = ['P3', 'P4', 'P5'] as const;
const SCALE_COLORS: Record<string, string> = { P3: '#f59e0b', P4: '#38bdf8', P5: '#a78bfa' };
const CS2GA_COLORS: Record<string, string> = {
  p3: '#f59e0b',
  p4: '#38bdf8',
  p5: '#a78bfa',
  within: '#10b981',
  cross: '#f43f5e',
};
const SCALE_HELP: Record<string, string> = {
  P3: 'high-res',
  P4: 'mid-res',
  P5: 'low-res',
};
const EPS = 1e-8;

type HsgMetricVariant = 'cs2ga' | 'v3-topm' | 'v3-hard' | 'v2-token' | 'legacy-token';

const metricKey = (scale: string, name: string) => `sgb/${scale}_${name}`;
const cs2gaKey = (block: number, name: string) => `cs2ga/${block}/${name}`;
const fmt = (v: number | undefined, digits = 4) =>
  v != null ? (Math.abs(v) < 0.001 && v !== 0 ? v.toExponential(2) : v.toFixed(digits)) : '-';
const str = (v: unknown) => (typeof v === 'string' ? v : undefined);

const hasKeyPrefix = (entry: HsgDetrMetricsEntry, prefix: string) =>
  Object.keys(entry).some(key => key.startsWith(prefix) && entry[key] != null);

const isReferenceGuided = (entry: HsgDetrMetricsEntry, scale: string) =>
  Number(entry[metricKey(scale, 'reference_guided')] ?? 0) >= 0.5;

const hasMetric = (entry: HsgDetrMetricsEntry, suffix: string) =>
  SCALES.some(scale => entry[metricKey(scale, suffix)] != null);

const hasScale = (entry: HsgDetrMetricsEntry, scale: string) =>
  entry[metricKey(scale, 'N')] != null || entry[metricKey(scale, 'ratio')] != null;

const getActiveSgbScales = (entry: HsgDetrMetricsEntry) =>
  SCALES.filter(scale => hasScale(entry, scale));

const getCs2gaBlocks = (entry: HsgDetrMetricsEntry) => {
  const indices = new Set<number>();
  Object.keys(entry).forEach(key => {
    const match = key.match(/^cs2ga\/(\d+)\//);
    if (match && entry[key] != null) indices.add(Number(match[1]));
  });
  return Array.from(indices).sort((a, b) => a - b);
};

const hasDecoderMetrics = (entry: HsgDetrMetricsEntry) =>
  entry['decoder/alpha'] != null
  || entry['decoder/num_queries'] != null
  || entry['decoder/cls_conf_mean'] != null;

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

const detectHsgMetricVariant = (entry: HsgDetrMetricsEntry): HsgMetricVariant => {
  if (hasKeyPrefix(entry, 'cs2ga/')) return 'cs2ga';
  const activeScales = getActiveSgbScales(entry);
  const hasV3TopM = activeScales.some(scale => entry[metricKey(scale, 'top_m')] != null);
  if (hasV3TopM) {
    return activeScales.some(scale => isSoftHardActive(entry, scale)) ? 'v3-topm' : 'v3-hard';
  }
  if (activeScales.some(scale => entry[metricKey(scale, 'score_std')] != null)) return 'v2-token';
  return 'legacy-token';
};

const cs2gaLine = (block: number, name: string, label: string, color: string, dashed = false) => (
  <Line
    key={`cs2ga-${block}-${name}`}
    type="monotone"
    dataKey={cs2gaKey(block, name)}
    name={`B${block} ${label}`}
    stroke={color}
    strokeWidth={dashed ? 1.5 : 2}
    strokeDasharray={dashed ? '3 3' : undefined}
    dot={false}
    connectNulls
  />
);

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

const SimpleLineChart = ({
  history,
  children,
  domain,
  allowDecimals,
}: {
  history: HsgDetrMetricsEntry[];
  children: React.ReactNode;
  domain?: [number, number];
  allowDecimals?: boolean;
}) => (
  <ResponsiveContainer width="100%" height="100%">
    <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
      <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
      <YAxis stroke="#475569" tick={{ fontSize: 9 }} domain={domain} allowDecimals={allowDecimals} />
      <Tooltip content={<CustomTooltip />} />
      <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
      {children}
    </LineChart>
  </ResponsiveContainer>
);

const Cs2gaSummaryCards = ({ latest, blocks }: { latest: HsgDetrMetricsEntry; blocks: number[] }) => (
  <>
    {blocks.map(block => (
      <React.Fragment key={`cs2ga-cards-${block}`}>
        <MetricCard
          label={`CS2GA B${block} gates`}
          value={`${fmt(latest[cs2gaKey(block, 'gate_p3')], 2)}/${fmt(latest[cs2gaKey(block, 'gate_p4')], 2)}/${fmt(latest[cs2gaKey(block, 'gate_p5')], 2)}`}
          color="text-violet-300"
          sub="P3 / P4 / P5 residual gates"
        />
        <MetricCard
          label={`CS2GA B${block} k budget`}
          value={`${fmt(latest[cs2gaKey(block, 'k3')], 0)}/${fmt(latest[cs2gaKey(block, 'k4')], 0)}/${fmt(latest[cs2gaKey(block, 'k5')], 0)}`}
          color="text-cyan-300"
          sub="P3 / P4 / P5 selected tokens"
        />
        <MetricCard
          label={`CS2GA B${block} cross mass`}
          value={fmt(latest[cs2gaKey(block, 'attn_cross_frac')], 4)}
          color="text-rose-300"
          sub={`within=${fmt(latest[cs2gaKey(block, 'attn_within_frac')], 4)}`}
        />
        <MetricCard
          label={`CS2GA B${block} delta abs`}
          value={`${fmt(latest[cs2gaKey(block, 'delta_abs_p3')], 2)}/${fmt(latest[cs2gaKey(block, 'delta_abs_p4')], 2)}/${fmt(latest[cs2gaKey(block, 'delta_abs_p5')], 2)}`}
          color="text-emerald-300"
          sub="P3 / P4 / P5 before gate"
        />
      </React.Fragment>
    ))}
  </>
);

const Cs2gaCharts = ({ history, blocks }: { history: HsgDetrMetricsEntry[]; blocks: number[] }) => (
  <>
    <ChartPanel title="CS2GA Gate Evolution" icon={<Layers size={14} className="text-violet-400" />}>
      <SimpleLineChart history={history} domain={[0, 1]}>
        {blocks.flatMap(block => [
          cs2gaLine(block, 'gate_p3', 'gate P3', CS2GA_COLORS.p3),
          cs2gaLine(block, 'gate_p4', 'gate P4', CS2GA_COLORS.p4),
          cs2gaLine(block, 'gate_p5', 'gate P5', CS2GA_COLORS.p5),
        ])}
      </SimpleLineChart>
    </ChartPanel>

    <ChartPanel title="CS2GA Attention Split" icon={<Crosshair size={14} className="text-rose-400" />}>
      <SimpleLineChart history={history} domain={[0, 1]}>
        {blocks.flatMap(block => [
          cs2gaLine(block, 'attn_within_frac', 'within', CS2GA_COLORS.within),
          cs2gaLine(block, 'attn_cross_frac', 'cross', CS2GA_COLORS.cross, true),
        ])}
      </SimpleLineChart>
    </ChartPanel>

    <ChartPanel title="CS2GA Delta Magnitude" icon={<Activity size={14} className="text-emerald-400" />}>
      <SimpleLineChart history={history}>
        {blocks.flatMap(block => [
          cs2gaLine(block, 'delta_abs_p3', 'delta P3', CS2GA_COLORS.p3),
          cs2gaLine(block, 'delta_abs_p4', 'delta P4', CS2GA_COLORS.p4),
          cs2gaLine(block, 'delta_abs_p5', 'delta P5', CS2GA_COLORS.p5),
        ])}
      </SimpleLineChart>
    </ChartPanel>
  </>
);

const Cs2gaContractTable = ({ latest, blocks }: { latest: HsgDetrMetricsEntry; blocks: number[] }) => (
  <div className="bg-slate-900/40 border border-slate-800 rounded-xl overflow-hidden">
    <div className="px-4 py-2.5 border-b border-slate-800/50 bg-slate-900/30">
      <span className="text-xs font-semibold text-slate-200">CS2GA Contract Latest - Epoch {latest.epoch}</span>
    </div>
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-slate-500 uppercase tracking-wider border-b border-slate-800">
            <th className="px-4 py-2 text-left">Block</th>
            <th className="px-4 py-2 text-right">k P3/P4/P5</th>
            <th className="px-4 py-2 text-right">Gate P3</th>
            <th className="px-4 py-2 text-right">Gate P4</th>
            <th className="px-4 py-2 text-right">Gate P5</th>
            <th className="px-4 py-2 text-right">Within mass</th>
            <th className="px-4 py-2 text-right">Cross mass</th>
            <th className="px-4 py-2 text-right">Delta P3</th>
            <th className="px-4 py-2 text-right">Delta P4</th>
            <th className="px-4 py-2 text-right">Delta P5</th>
          </tr>
        </thead>
        <tbody>
          {blocks.map(block => (
            <tr key={`cs2ga-row-${block}`} className="border-b border-slate-800/50 hover:bg-slate-800/20">
              <td className="px-4 py-2 font-bold text-violet-300">CS2GA {block}</td>
              <td className="px-4 py-2 text-right font-mono text-cyan-300">
                {fmt(latest[cs2gaKey(block, 'k3')], 0)}/{fmt(latest[cs2gaKey(block, 'k4')], 0)}/{fmt(latest[cs2gaKey(block, 'k5')], 0)}
              </td>
              <td className="px-4 py-2 text-right font-mono" style={{ color: CS2GA_COLORS.p3 }}>{fmt(latest[cs2gaKey(block, 'gate_p3')], 4)}</td>
              <td className="px-4 py-2 text-right font-mono" style={{ color: CS2GA_COLORS.p4 }}>{fmt(latest[cs2gaKey(block, 'gate_p4')], 4)}</td>
              <td className="px-4 py-2 text-right font-mono" style={{ color: CS2GA_COLORS.p5 }}>{fmt(latest[cs2gaKey(block, 'gate_p5')], 4)}</td>
              <td className="px-4 py-2 text-right font-mono text-emerald-300">{fmt(latest[cs2gaKey(block, 'attn_within_frac')], 4)}</td>
              <td className="px-4 py-2 text-right font-mono text-rose-300">{fmt(latest[cs2gaKey(block, 'attn_cross_frac')], 4)}</td>
              <td className="px-4 py-2 text-right font-mono" style={{ color: CS2GA_COLORS.p3 }}>{fmt(latest[cs2gaKey(block, 'delta_abs_p3')], 4)}</td>
              <td className="px-4 py-2 text-right font-mono" style={{ color: CS2GA_COLORS.p4 }}>{fmt(latest[cs2gaKey(block, 'delta_abs_p4')], 4)}</td>
              <td className="px-4 py-2 text-right font-mono" style={{ color: CS2GA_COLORS.p5 }}>{fmt(latest[cs2gaKey(block, 'delta_abs_p5')], 4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </div>
);

const DecoderMetricsTable = ({ latest }: { latest: HsgDetrMetricsEntry }) => {
  const rows: Array<[string, string]> = [
    ['alpha', fmt(latest['decoder/alpha'], 4)],
    ['alpha_progress', fmt(latest['decoder/alpha_progress'], 4)],
    ['alpha_u', fmt(latest['decoder/alpha_u'], 4)],
    ['alpha_eff', fmt(latest['decoder/alpha_eff'], 4)],
    ['num_queries', fmt(latest['decoder/num_queries'], 0)],
    ['hidden_dim', fmt(latest['decoder/hidden_dim'], 0)],
    ['loc_quality_mode', str(latest['decoder/loc_quality_mode']) || '-'],
    ['cls_conf_mean/std', `${fmt(latest['decoder/cls_conf_mean'], 4)} / ${fmt(latest['decoder/cls_conf_std'], 4)}`],
    ['loc_conf_mean/std', `${fmt(latest['decoder/loc_conf_mean'], 4)} / ${fmt(latest['decoder/loc_conf_std'], 4)}`],
    ['uncertainty_mean/std', `${fmt(latest['decoder/uncertainty_mean'], 4)} / ${fmt(latest['decoder/uncertainty_std'], 4)}`],
    ['selected_cls_conf_mean', fmt(latest['decoder/selected_cls_conf_mean'], 4)],
    ['selected_loc_conf_mean', fmt(latest['decoder/selected_loc_conf_mean'], 4)],
    ['selected_uncertainty_mean', fmt(latest['decoder/selected_uncertainty_mean'], 4)],
    ['score_mean/std', `${fmt(latest['decoder/score_mean'], 4)} / ${fmt(latest['decoder/score_std'], 4)}`],
    ['selected_score_mean/std', `${fmt(latest['decoder/selected_score_mean'], 4)} / ${fmt(latest['decoder/selected_score_std'], 4)}`],
    ['score_entropy_mean', fmt(latest['decoder/score_entropy_mean'], 4)],
    ['selected_box_area_mean', fmt(latest['decoder/selected_box_area_mean'], 4)],
    ['beta_s', fmt(latest['decoder/beta_s'], 4)],
  ];

  return (
    <div className="bg-slate-900/40 border border-slate-800 rounded-xl overflow-hidden">
      <div className="px-4 py-2.5 border-b border-slate-800/50 bg-slate-900/30">
        <span className="text-xs font-semibold text-slate-200">Decoder Metrics Latest - Epoch {latest.epoch}</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-px bg-slate-800">
        {rows.filter(([, value]) => value !== '-').map(([label, value]) => (
          <div key={label} className="bg-slate-950/40 px-4 py-2">
            <div className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</div>
            <div className="font-mono text-xs text-slate-200">{value}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

const GradientMetricsTable = ({ latest }: { latest: HsgDetrMetricsEntry }) => {
  const rows: Array<[string, number | undefined]> = [
    ['backbone_norm', latest['grad/backbone_norm']],
    ['neck_norm', latest['grad/neck_norm']],
    ['decoder_norm', latest['grad/decoder_norm']],
    ['sgb_norm', latest['grad/sgb_norm']],
    ['sgb_gamma_norm', latest['grad/sgb_gamma_norm']],
    ['sgb_sparse_norm', latest['grad/sgb_sparse_norm']],
    ['sgb_norm_norm', latest['grad/sgb_norm_norm']],
    ['has_nan', latest['grad/has_nan']],
    ['has_inf', latest['grad/has_inf']],
  ];
  const present = rows.filter(([, value]) => value != null);
  if (!present.length) return null;

  return (
    <div className="bg-slate-900/40 border border-slate-800 rounded-xl overflow-hidden">
      <div className="px-4 py-2.5 border-b border-slate-800/50 bg-slate-900/30">
        <span className="text-xs font-semibold text-slate-200">Gradient Norms Latest - Epoch {latest.epoch}</span>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-px bg-slate-800">
        {present.map(([label, value]) => (
          <div key={label} className="bg-slate-950/40 px-4 py-2">
            <div className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</div>
            <div className={`font-mono text-xs ${label.startsWith('has_') && value ? 'text-red-400' : 'text-slate-200'}`}>
              {fmt(value, label.startsWith('has_') ? 0 : 4)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const SgbChecklistTable = ({
  latest,
  activeScales,
  hasV3Dam,
}: {
  latest: HsgDetrMetricsEntry;
  activeScales: readonly string[];
  hasV3Dam: boolean;
}) => (
  <div className="bg-slate-900/40 border border-slate-800 rounded-xl overflow-hidden">
    <div className="px-4 py-2.5 border-b border-slate-800/50 bg-slate-900/30">
      <span className="text-xs font-semibold text-slate-200">SGB Block Metrics Latest - Epoch {latest.epoch}</span>
    </div>
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-slate-500 uppercase tracking-wider border-b border-slate-800">
            <th className="px-4 py-2 text-left">Block</th>
            <th className="px-4 py-2 text-right">ratio</th>
            <th className="px-4 py-2 text-right">N</th>
            <th className="px-4 py-2 text-right">k / N</th>
            <th className="px-4 py-2 text-right">gate</th>
            <th className="px-4 py-2 text-right">score std</th>
            <th className="px-4 py-2 text-right">K_eff</th>
            <th className="px-4 py-2 text-right">Top-M</th>
            <th className="px-4 py-2 text-right">tau</th>
            <th className="px-4 py-2 text-right">lambda</th>
            <th className="px-4 py-2 text-right">soft mass</th>
            <th className="px-4 py-2 text-right">extra mass</th>
            <th className="px-4 py-2 text-right">hard delta</th>
            <th className="px-4 py-2 text-right">non-hard delta</th>
            <th className="px-4 py-2 text-right">SE</th>
            <th className="px-4 py-2 text-right">soft cfg/active</th>
            <th className="px-4 py-2 text-right">saliency</th>
            {hasV3Dam && <th className="px-4 py-2 text-right">DAM mass@k</th>}
            {hasV3Dam && <th className="px-4 py-2 text-right">DAM corr</th>}
            <th className="px-4 py-2 text-right">ref</th>
          </tr>
        </thead>
        <tbody>
          {activeScales.map(scale => (
            <tr key={scale} className="border-b border-slate-800/50 hover:bg-slate-800/20">
              <td className="px-4 py-2 font-bold" style={{ color: SCALE_COLORS[scale] }}>{scale}</td>
              <td className="px-4 py-2 text-right font-mono text-slate-300">{fmt(latest[metricKey(scale, 'ratio')], 3)}</td>
              <td className="px-4 py-2 text-right font-mono text-slate-300">{fmt(latest[metricKey(scale, 'N')], 0)}</td>
              <td className="px-4 py-2 text-right font-mono text-slate-300">
                {fmt(latest[metricKey(scale, 'k_over_N')], 4)}
                <span className="text-slate-600 ml-1">({fmt(latest[metricKey(scale, 'k')], 0)})</span>
              </td>
              <td className="px-4 py-2 text-right font-mono text-emerald-400">{fmt(latest[metricKey(scale, 'gamma_abs_mean')], 4)}</td>
              <td className="px-4 py-2 text-right font-mono text-violet-300">{fmt(latest[metricKey(scale, 'score_std')], 3)}</td>
              <td className="px-4 py-2 text-right font-mono text-cyan-300">{fmt(latest[metricKey(scale, 'K_eff')], 0)}</td>
              <td className="px-4 py-2 text-right font-mono text-cyan-300">
                {fmt(latest[metricKey(scale, 'top_m')], 0)}
                <span className="text-slate-600 ml-1">({fmt(latest[metricKey(scale, 'top_m_over_N')], 3)})</span>
              </td>
              <td className="px-4 py-2 text-right font-mono text-slate-300">{fmt(latest[metricKey(scale, 'tau')], 3)}</td>
              <td className="px-4 py-2 text-right font-mono text-sky-300">{fmt(latest[metricKey(scale, 'lambda_soft')], 3)}</td>
              <td className="px-4 py-2 text-right font-mono text-slate-300">{fmt(latest[metricKey(scale, 'soft_nonhard_mass')], 4)}</td>
              <td className="px-4 py-2 text-right font-mono text-slate-300">{fmt(latest[metricKey(scale, 'topm_extra_mass')], 4)}</td>
              <td className="px-4 py-2 text-right font-mono text-rose-300">{fmt(latest[metricKey(scale, 'hard_delta_norm')], 4)}</td>
              <td className="px-4 py-2 text-right font-mono text-rose-300">{fmt(latest[metricKey(scale, 'nonhard_delta_norm')], 4)}</td>
              <td className="px-4 py-2 text-right font-mono text-slate-300">{fmt(latest[metricKey(scale, 'channel_se')], 0)}</td>
              <td className="px-4 py-2 text-right font-mono text-slate-300">
                {fmt(latest[metricKey(scale, 'soft_hard_config')], 0)}/{fmt(latest[metricKey(scale, 'soft_hard_active')], 0)}
              </td>
              <td className="px-4 py-2 text-right font-mono text-slate-300">{fmt(latest[metricKey(scale, 'saliency_mean')], 4)}</td>
              {hasV3Dam && <td className="px-4 py-2 text-right font-mono text-emerald-300">{fmt(latest[metricKey(scale, 'selected_DAM_mass@k')], 4)}</td>}
              {hasV3Dam && <td className="px-4 py-2 text-right font-mono text-violet-300">{fmt(latest[metricKey(scale, 'selector_DAM_corr')], 4)}</td>}
              <td className="px-4 py-2 text-right font-mono text-slate-300">{fmt(latest[metricKey(scale, 'reference_guided')], 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </div>
);

const HsgDetrMetrics: React.FC<Props> = ({ history }) => {
  if (!history || history.length === 0) return null;

  const latest = history[history.length - 1];
  const variant = detectHsgMetricVariant(latest);
  const isCs2ga = variant === 'cs2ga';
  const cs2gaBlocks = getCs2gaBlocks(latest);
  const activeScales = getActiveSgbScales(latest);
  const hasNan = latest['grad/has_nan'] === 1;
  const hasInf = latest['grad/has_inf'] === 1;
  const anyFiniteGuard = activeScales.some(s => (latest[metricKey(s, 'finite_guard_count')] ?? 0) > 0);
  const sparseGrad = latest['grad/sgb_sparse_norm'];
  const gammaGrad = latest['grad/sgb_gamma_norm'];
  const hasDecoderAlpha = latest['decoder/alpha'] != null || latest['decoder/alpha_eff'] != null;
  const alphaMax = Math.max(
    0.55,
    ...history.map(h => Number(h['decoder/alpha'] ?? 0) * 1.1),
    ...history.map(h => Number(h['decoder/alpha_eff'] ?? 0) * 1.1),
  );
  const referenceGuidedCount = activeScales.filter(scale => isReferenceGuided(latest, scale)).length;
  const hasV2ScoreStats = activeScales.some(scale => latest[metricKey(scale, 'score_std')] != null);
  const hasV3TopM = activeScales.some(scale => latest[metricKey(scale, 'top_m')] != null);
  const hasV3SoftHard = activeScales.some(scale => isSoftHardActive(latest, scale));
  const hasV3Dam = activeScales.some(scale => latest[`dam/${scale}_sampling_mass`] != null || latest[metricKey(scale, 'selector_DAM_corr')] != null);
  const seBlocks = activeScales.filter(scale => Number(latest[metricKey(scale, 'channel_se')] ?? 0) >= 0.5).length;
  const variantSummary = isCs2ga
    ? `HSG-DETR V3-CS2GA: ${cs2gaBlocks.length} cross-scale block${cs2gaBlocks.length === 1 ? '' : 's'}, joint P3/P4/P5 sparse attention, YOLO Detect head`
    : hasV3TopM
    ? hasV3SoftHard
      ? `Full V3: hd=${fmt(latest['decoder/hidden_dim'], 0)}, SE ${seBlocks}/${activeScales.length}, Top-M training, DAM ${hasV3Dam ? 'on' : 'off'}`
      : activeScales.length < 3
        ? `Ultra V3: hd=${fmt(latest['decoder/hidden_dim'], 0)}, no P3-SGB, hard top-K, SE ${seBlocks}/${activeScales.length}`
        : `Lean V3: hd=${fmt(latest['decoder/hidden_dim'], 0)}, three SGB levels, hard top-K, SE ${seBlocks}/${activeScales.length}`
    : '';
  const sparseVariant = referenceGuidedCount > 0
    ? 'reference-guided local aggregation'
    : isCs2ga
      ? 'CS2GA cross-scale sparse attention'
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
            {variantSummary || sparseVariant} | {isCs2ga ? 'cross-scale gates, attention split, delta magnitude' : 'stride-labeled blocks, sparse selection, decoder alpha, gradient contract'}
          </p>
        </div>
        <div className="ml-auto flex gap-2">
          <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border ${
            referenceGuidedCount > 0
              ? 'bg-cyan-500/10 text-cyan-300 border-cyan-500/20'
              : 'bg-violet-500/10 text-violet-300 border-violet-500/20'
          }`}>
            {isCs2ga ? 'CS2GA' : referenceGuidedCount > 0 ? 'Local sparse' : hasV3SoftHard ? 'V3 Top-M' : hasV3TopM ? 'V3 hard' : hasV2ScoreStats ? 'V2 token' : 'Legacy token'}
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
        {hasDecoderAlpha && (
          <MetricCard
            label="Decoder alpha"
            value={fmt(latest['decoder/alpha_eff'] ?? latest['decoder/alpha'], 3)}
            color="text-emerald-400"
            sub={`progress=${fmt(latest['decoder/alpha_progress'], 2)} | max=${fmt(latest['decoder/alpha_u'], 2)} | q=${fmt(latest['decoder/num_queries'], 0)} | hd=${fmt(latest['decoder/hidden_dim'], 0)}`}
          />
        )}
        {sparseGrad != null && (
          <MetricCard
            label={isCs2ga ? 'Sparse grad' : 'SGB sparse grad'}
            value={fmt(sparseGrad, 4)}
            color={metricColor(sparseGrad)}
            sub={isCs2ga ? 'cross-scale sparse params' : 'q/k/v/out params'}
          />
        )}
        {gammaGrad != null && (
          <MetricCard
            label="Gamma grad"
            value={fmt(gammaGrad, 4)}
            color={metricColor(gammaGrad)}
            sub="LayerScale update"
          />
        )}
        {isCs2ga && <Cs2gaSummaryCards latest={latest} blocks={cs2gaBlocks} />}
        {!isCs2ga && hasV3TopM && activeScales.map(scale => {
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
        {isCs2ga ? (
          <>
            <Cs2gaCharts history={history} blocks={cs2gaBlocks} />
            <ChartPanel title="Optimizer Grad Groups" icon={<ShieldCheck size={14} className="text-cyan-400" />}>
              <SimpleLineChart history={history}>
                <Line type="monotone" dataKey="grad/backbone_norm" name="Backbone" stroke="#64748b" strokeWidth={1.5} dot={false} connectNulls />
                <Line type="monotone" dataKey="grad/neck_norm" name="Neck" stroke="#fb923c" strokeWidth={1.5} dot={false} connectNulls />
                <Line type="monotone" dataKey="grad/sgb_sparse_norm" name="Sparse" stroke="#f43f5e" strokeWidth={2} dot={false} connectNulls />
                <Line type="monotone" dataKey="grad/decoder_norm" name="Head" stroke="#10b981" strokeWidth={1.5} dot={false} connectNulls />
              </SimpleLineChart>
            </ChartPanel>
          </>
        ) : (
          <>
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

        {hasV3TopM && (
        <ChartPanel title="Sparse LayerScale Gamma" icon={<Layers size={14} className="text-violet-400" />}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
              <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
              <YAxis stroke="#475569" tick={{ fontSize: 9 }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
              {scaleLines('gamma_abs_mean', activeScales)}
            </LineChart>
          </ResponsiveContainer>
        </ChartPanel>
        )}

        {hasV3TopM && hasV2ScoreStats && (
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

        {hasV3TopM && (
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
          </>
        )}
      </div>

      {isCs2ga && <Cs2gaContractTable latest={latest} blocks={cs2gaBlocks} />}
      {!isCs2ga && hasV3TopM && (
        <SgbChecklistTable latest={latest} activeScales={activeScales} hasV3Dam={hasV3Dam} />
      )}
      {hasDecoderMetrics(latest) && <DecoderMetricsTable latest={latest} />}
      <GradientMetricsTable latest={latest} />
    </div>
  );
};

export default HsgDetrMetrics;
