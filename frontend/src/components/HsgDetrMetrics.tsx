import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import { Activity, AlertTriangle, Brain, Crosshair, Layers, ShieldCheck, Zap } from 'lucide-react';

export interface HsgDetrMetricsEntry {
  epoch: number;
  [key: string]: any;
}

interface Props {
  history: HsgDetrMetricsEntry[];
}

type MetricValue = number | string | boolean | null | undefined;
type LineSpec = { key: string; label: string; color: string; dashed?: boolean };
type CardSpec = { label: string; value: string; sub?: string; color?: string };
type TableColumn = { key: string; label: string; digits?: number; color?: string };
type TableSpec = { title: string; rowLabel: string; rows: Array<{ id: string; label: string; values: Record<string, MetricValue> }>; columns: TableColumn[] };
type ChartSpec = { title: string; icon: React.ReactNode; lines: LineSpec[]; domain?: [number, number] };
type MetricPanel = {
  id: string;
  title: string;
  summary: string;
  badge: string;
  priority: number;
  cards: CardSpec[];
  charts: ChartSpec[];
  tables: TableSpec[];
};

const SCALES = ['P3', 'P4', 'P5'] as const;
const SCALE_COLORS: Record<string, string> = { P3: '#f59e0b', P4: '#38bdf8', P5: '#a78bfa' };
const CS2GA_COLORS: Record<string, string> = {
  p3: '#f59e0b',
  p4: '#38bdf8',
  p5: '#a78bfa',
  within: '#10b981',
  cross: '#f43f5e',
};
const GRAD_COLORS: Record<string, string> = {
  'grad/backbone_norm': '#64748b',
  'grad/neck_norm': '#fb923c',
  'grad/decoder_norm': '#10b981',
  'grad/sgb_norm': '#38bdf8',
  'grad/sgb_sparse_norm': '#f43f5e',
  'grad/sgb_gamma_norm': '#a78bfa',
  'grad/sgb_norm_norm': '#8b5cf6',
};
const EPS = 1e-8;

const metricKey = (scale: string, name: string) => `sgb/${scale}_${name}`;
const cs2gaKey = (block: number, name: string) => `cs2ga/${block}/${name}`;
const present = (v: unknown) => v !== undefined && v !== null;
const isFiniteNumber = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const asNumber = (v: unknown) => (isFiniteNumber(v) ? v : undefined);
const fmt = (v: MetricValue, digits = 4) => {
  if (typeof v === 'string') return v;
  if (typeof v === 'boolean') return v ? '1' : '0';
  if (!isFiniteNumber(v)) return '-';
  return Math.abs(v) < 0.001 && v !== 0 ? v.toExponential(2) : v.toFixed(digits);
};
const metricColor = (v: unknown, threshold = EPS) =>
  isFiniteNumber(v) && Math.abs(v) > threshold ? 'text-emerald-400' : 'text-amber-400';

const normalizeEntry = (entry: HsgDetrMetricsEntry): HsgDetrMetricsEntry => {
  const nested = entry.plot ?? entry.plots ?? entry.hsg_detr;
  if (nested && typeof nested === 'object' && !Array.isArray(nested)) {
    return { ...(nested as Record<string, unknown>), ...entry, epoch: entry.epoch };
  }
  return entry;
};

const hasPrefix = (entry: HsgDetrMetricsEntry, prefix: string) =>
  Object.keys(entry).some(key => key.startsWith(prefix) && present(entry[key]));

const getActiveSgbScales = (entry: HsgDetrMetricsEntry) =>
  SCALES.filter(scale => present(entry[metricKey(scale, 'N')]) || present(entry[metricKey(scale, 'ratio')]) || present(entry[metricKey(scale, 'top_m')]));

const getCs2gaBlocks = (entry: HsgDetrMetricsEntry) => {
  const indices = new Set<number>();
  Object.keys(entry).forEach(key => {
    const match = key.match(/^cs2ga\/(\d+)\//);
    if (match && present(entry[key])) indices.add(Number(match[1]));
  });
  return Array.from(indices).sort((a, b) => a - b);
};

const dataHasAny = (history: HsgDetrMetricsEntry[], keys: string[]) =>
  keys.some(key => history.some(row => present(row[key])));

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
          <span className="font-mono text-white">{fmt(entry.value)}</span>
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

const MetricCard = ({ label, value, color, sub }: CardSpec) => (
  <div className="bg-slate-900/60 border border-slate-800 rounded-lg px-3 py-2 min-w-[118px]">
    <div className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</div>
    <div className={`text-sm font-mono font-bold ${color || 'text-white'}`}>{value}</div>
    {sub && <div className="text-[9px] text-slate-600">{sub}</div>}
  </div>
);

const ChartPanel = ({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) => (
  <div className="bg-[#0f1117] border border-slate-800 rounded-xl overflow-hidden">
    <div className="px-4 py-2.5 border-b border-slate-800/50 bg-slate-900/30 flex items-center gap-2">
      {icon}
      <span className="text-xs font-semibold text-slate-200">{title}</span>
    </div>
    <div className="p-3 h-[210px]">{children}</div>
  </div>
);

const SimpleLineChart = ({ history, lines, domain }: { history: HsgDetrMetricsEntry[]; lines: LineSpec[]; domain?: [number, number] }) => (
  <ResponsiveContainer width="100%" height="100%">
    <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
      <XAxis dataKey="epoch" stroke="#475569" tick={{ fontSize: 9 }} />
      <YAxis stroke="#475569" tick={{ fontSize: 9 }} domain={domain} />
      <Tooltip content={<CustomTooltip />} />
      <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
      {lines.map(line => (
        <Line
          key={line.key}
          type="monotone"
          dataKey={line.key}
          name={line.label}
          stroke={line.color}
          strokeWidth={line.dashed ? 1.5 : 2}
          strokeDasharray={line.dashed ? '3 3' : undefined}
          dot={false}
          connectNulls
        />
      ))}
    </LineChart>
  </ResponsiveContainer>
);

const MetricsTable = ({ table }: { table: TableSpec }) => (
  <div className="bg-slate-900/40 border border-slate-800 rounded-xl overflow-hidden">
    <div className="px-4 py-2.5 border-b border-slate-800/50 bg-slate-900/30">
      <span className="text-xs font-semibold text-slate-200">{table.title}</span>
    </div>
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-slate-500 uppercase tracking-wider border-b border-slate-800">
            <th className="px-4 py-2 text-left">{table.rowLabel}</th>
            {table.columns.map(col => <th key={col.key} className="px-4 py-2 text-right">{col.label}</th>)}
          </tr>
        </thead>
        <tbody>
          {table.rows.map(row => (
            <tr key={row.id} className="border-b border-slate-800/50 hover:bg-slate-800/20">
              <td className="px-4 py-2 font-bold text-slate-200">{row.label}</td>
              {table.columns.map(col => (
                <td key={`${row.id}-${col.key}`} className={`px-4 py-2 text-right font-mono ${col.color || 'text-slate-300'}`}>
                  {fmt(row.values[col.key], col.digits ?? 4)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </div>
);

const compactRowsTable = (title: string, rows: Array<[string, MetricValue]>) => {
  const filtered = rows.filter(([, value]) => present(value));
  if (!filtered.length) return null;
  return (
    <div className="bg-slate-900/40 border border-slate-800 rounded-xl overflow-hidden">
      <div className="px-4 py-2.5 border-b border-slate-800/50 bg-slate-900/30">
        <span className="text-xs font-semibold text-slate-200">{title}</span>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-px bg-slate-800">
        {filtered.map(([label, value]) => (
          <div key={label} className="bg-slate-950/40 px-4 py-2">
            <div className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</div>
            <div className={`font-mono text-xs ${label.startsWith('has_') && Number(value) ? 'text-red-400' : 'text-slate-200'}`}>
              {fmt(value, label.startsWith('has_') ? 0 : 4)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const buildCs2gaPanel = (latest: HsgDetrMetricsEntry): MetricPanel | null => {
  const blocks = getCs2gaBlocks(latest);
  if (!blocks.length) return null;
  const cards = blocks.flatMap<CardSpec>(block => [
    {
      label: `CS2GA B${block} gates`,
      value: `${fmt(latest[cs2gaKey(block, 'gate_p3')], 2)}/${fmt(latest[cs2gaKey(block, 'gate_p4')], 2)}/${fmt(latest[cs2gaKey(block, 'gate_p5')], 2)}`,
      color: 'text-violet-300',
      sub: 'P3 / P4 / P5 residual gates',
    },
    {
      label: `CS2GA B${block} k budget`,
      value: `${fmt(latest[cs2gaKey(block, 'k3')], 0)}/${fmt(latest[cs2gaKey(block, 'k4')], 0)}/${fmt(latest[cs2gaKey(block, 'k5')], 0)}`,
      color: 'text-cyan-300',
      sub: 'P3 / P4 / P5 selected tokens',
    },
    {
      label: `CS2GA B${block} cross mass`,
      value: fmt(latest[cs2gaKey(block, 'attn_cross_frac')], 4),
      color: 'text-rose-300',
      sub: `within=${fmt(latest[cs2gaKey(block, 'attn_within_frac')], 4)}`,
    },
    {
      label: `CS2GA B${block} delta abs`,
      value: `${fmt(latest[cs2gaKey(block, 'delta_abs_p3')], 2)}/${fmt(latest[cs2gaKey(block, 'delta_abs_p4')], 2)}/${fmt(latest[cs2gaKey(block, 'delta_abs_p5')], 2)}`,
      color: 'text-emerald-300',
      sub: 'P3 / P4 / P5 before gate',
    },
  ]);

  const blockLines = (name: string, label: string, color: string, dashed = false) =>
    blocks.map(block => ({ key: cs2gaKey(block, name), label: `B${block} ${label}`, color, dashed }));

  return {
    id: 'cs2ga',
    title: 'HSG-DETR V3-CS2GA',
    summary: `${blocks.length} cross-scale sparse block${blocks.length === 1 ? '' : 's'} | joint P3/P4/P5 attention | YOLO Detect head`,
    badge: 'CS2GA',
    priority: 10,
    cards,
    charts: [
      {
        title: 'CS2GA Gate Evolution',
        icon: <Layers size={14} className="text-violet-400" />,
        domain: [0, 1],
        lines: [
          ...blockLines('gate_p3', 'gate P3', CS2GA_COLORS.p3),
          ...blockLines('gate_p4', 'gate P4', CS2GA_COLORS.p4),
          ...blockLines('gate_p5', 'gate P5', CS2GA_COLORS.p5),
        ],
      },
      {
        title: 'CS2GA Attention Split',
        icon: <Crosshair size={14} className="text-rose-400" />,
        domain: [0, 1],
        lines: [
          ...blockLines('attn_within_frac', 'within', CS2GA_COLORS.within),
          ...blockLines('attn_cross_frac', 'cross', CS2GA_COLORS.cross, true),
        ],
      },
      {
        title: 'CS2GA Delta Magnitude',
        icon: <Activity size={14} className="text-emerald-400" />,
        lines: [
          ...blockLines('delta_abs_p3', 'delta P3', CS2GA_COLORS.p3),
          ...blockLines('delta_abs_p4', 'delta P4', CS2GA_COLORS.p4),
          ...blockLines('delta_abs_p5', 'delta P5', CS2GA_COLORS.p5),
        ],
      },
    ],
    tables: [{
      title: `CS2GA Contract Latest - Epoch ${latest.epoch}`,
      rowLabel: 'Block',
      rows: blocks.map(block => ({
        id: `cs2ga-${block}`,
        label: `CS2GA ${block}`,
        values: {
          k: `${fmt(latest[cs2gaKey(block, 'k3')], 0)}/${fmt(latest[cs2gaKey(block, 'k4')], 0)}/${fmt(latest[cs2gaKey(block, 'k5')], 0)}`,
          gate_p3: latest[cs2gaKey(block, 'gate_p3')],
          gate_p4: latest[cs2gaKey(block, 'gate_p4')],
          gate_p5: latest[cs2gaKey(block, 'gate_p5')],
          within: latest[cs2gaKey(block, 'attn_within_frac')],
          cross: latest[cs2gaKey(block, 'attn_cross_frac')],
          delta_p3: latest[cs2gaKey(block, 'delta_abs_p3')],
          delta_p4: latest[cs2gaKey(block, 'delta_abs_p4')],
          delta_p5: latest[cs2gaKey(block, 'delta_abs_p5')],
        },
      })),
      columns: [
        { key: 'k', label: 'k P3/P4/P5' },
        { key: 'gate_p3', label: 'Gate P3', color: 'text-amber-300' },
        { key: 'gate_p4', label: 'Gate P4', color: 'text-sky-300' },
        { key: 'gate_p5', label: 'Gate P5', color: 'text-violet-300' },
        { key: 'within', label: 'Within mass', color: 'text-emerald-300' },
        { key: 'cross', label: 'Cross mass', color: 'text-rose-300' },
        { key: 'delta_p3', label: 'Delta P3' },
        { key: 'delta_p4', label: 'Delta P4' },
        { key: 'delta_p5', label: 'Delta P5' },
      ],
    }],
  };
};

const buildSgbPanel = (latest: HsgDetrMetricsEntry): MetricPanel | null => {
  const scales = getActiveSgbScales(latest);
  if (!scales.length) return null;
  const softHard = scales.some(scale => Number(latest[metricKey(scale, 'soft_hard_active')] ?? 0) >= 0.5);
  const hasTopM = scales.some(scale => present(latest[metricKey(scale, 'top_m')]));
  const seCount = scales.filter(scale => Number(latest[metricKey(scale, 'channel_se')] ?? 0) >= 0.5).length;
  const cards = scales.map<CardSpec>(scale => {
    const topM = latest[metricKey(scale, 'top_m')];
    const kEff = latest[metricKey(scale, 'K_eff')];
    const channelSe = Number(latest[metricKey(scale, 'channel_se')] ?? 0) >= 0.5;
    const active = Number(latest[metricKey(scale, 'soft_hard_active')] ?? 0) >= 0.5;
    return {
      label: `${scale} ${active ? 'M/N' : 'K/N'}`,
      value: fmt(latest[metricKey(scale, active ? 'top_m_over_N' : 'k_over_N')], 3),
      color: 'text-cyan-300',
      sub: `${active ? 'soft-hard' : 'hard'} | M=${fmt(topM, 0)} K=${fmt(kEff ?? latest[metricKey(scale, 'k')], 0)} | SE=${channelSe ? 'on' : 'off'} | gate=${fmt(latest[metricKey(scale, 'gamma_abs_mean')], 3)}`,
    };
  });

  const scaleLines = (suffix: string, dashed = false) => scales.map(scale => ({
    key: metricKey(scale, suffix),
    label: `${scale}${dashed ? ' non-hard' : ''}`,
    color: SCALE_COLORS[scale],
    dashed,
  }));
  const chartIf = (title: string, suffixes: string[], icon: React.ReactNode, lines: LineSpec[], domain?: [number, number]) => ({ title, icon, lines, domain, suffixes });
  const rawCharts = [
    chartIf('Top-M / K Coverage', ['top_m_over_N', 'k_over_N'], <Crosshair size={14} className="text-cyan-400" />, [
      ...scaleLines(hasTopM ? 'top_m_over_N' : 'k_over_N'),
    ], [0, 1]),
    chartIf('Soft Non-hard Mass', ['soft_nonhard_mass'], <Activity size={14} className="text-sky-400" />, scaleLines('soft_nonhard_mass')),
    chartIf('Hard vs Non-hard Delta', ['hard_delta_norm', 'nonhard_delta_norm'], <Activity size={14} className="text-rose-400" />, [
      ...scaleLines('hard_delta_norm'),
      ...scaleLines('nonhard_delta_norm', true),
    ]),
    chartIf('Sparse LayerScale Gate', ['gamma_abs_mean'], <Layers size={14} className="text-violet-400" />, scaleLines('gamma_abs_mean')),
    chartIf('Score Std', ['score_std'], <Activity size={14} className="text-violet-400" />, scaleLines('score_std')),
    chartIf('Selection Ratio (k/N)', ['k_over_N'], <Layers size={14} className="text-amber-400" />, scaleLines('k_over_N'), [0, 1]),
  ];

  return {
    id: 'sgb',
    title: softHard ? 'HSG-DETR V3 SGB Top-M' : hasTopM ? 'HSG-DETR V3 SGB Hard Top-K' : 'HSG-DETR SGB',
    summary: `${scales.join('/')} sparse token blocks | SE ${seCount}/${scales.length} | ${softHard ? 'Top-M soft-hard active' : 'hard sparse selection'}`,
    badge: softHard ? 'V3 Top-M' : hasTopM ? 'V3 hard' : 'SGB',
    priority: 20,
    cards,
    charts: rawCharts
      .filter(chart => scales.some(scale => chart.suffixes.some(suffix => present(latest[metricKey(scale, suffix)]))))
      .map(({ suffixes, ...chart }) => chart),
    tables: [{
      title: `SGB Contract Latest - Epoch ${latest.epoch}`,
      rowLabel: 'Scale',
      rows: scales.map(scale => ({
        id: scale,
        label: scale,
        values: {
          ratio: latest[metricKey(scale, 'ratio')],
          N: latest[metricKey(scale, 'N')],
          k: latest[metricKey(scale, 'k')],
          k_over_N: latest[metricKey(scale, 'k_over_N')],
          gate: latest[metricKey(scale, 'gamma_abs_mean')],
          score_std: latest[metricKey(scale, 'score_std')],
          K_eff: latest[metricKey(scale, 'K_eff')],
          top_m: latest[metricKey(scale, 'top_m')],
          tau: latest[metricKey(scale, 'tau')],
          lambda_soft: latest[metricKey(scale, 'lambda_soft')],
          soft_mass: latest[metricKey(scale, 'soft_nonhard_mass')],
          extra_mass: latest[metricKey(scale, 'topm_extra_mass')],
          hard_delta: latest[metricKey(scale, 'hard_delta_norm')],
          nonhard_delta: latest[metricKey(scale, 'nonhard_delta_norm')],
          se: latest[metricKey(scale, 'channel_se')],
          soft_active: present(latest[metricKey(scale, 'soft_hard_config')]) || present(latest[metricKey(scale, 'soft_hard_active')])
            ? `${fmt(latest[metricKey(scale, 'soft_hard_config')], 0)}/${fmt(latest[metricKey(scale, 'soft_hard_active')], 0)}`
            : undefined,
          saliency: latest[metricKey(scale, 'saliency_mean')],
          dam_mass: latest[metricKey(scale, 'selected_DAM_mass@k')],
          dam_corr: latest[metricKey(scale, 'selector_DAM_corr')],
          ref: latest[metricKey(scale, 'reference_guided')],
        },
      })),
      columns: [
        { key: 'ratio', label: 'ratio', digits: 3 },
        { key: 'N', label: 'N', digits: 0 },
        { key: 'k', label: 'k', digits: 0 },
        { key: 'k_over_N', label: 'k/N' },
        { key: 'gate', label: 'gate', color: 'text-emerald-300' },
        { key: 'score_std', label: 'score std' },
        { key: 'K_eff', label: 'K_eff', digits: 0 },
        { key: 'top_m', label: 'Top-M', digits: 0 },
        { key: 'tau', label: 'tau', digits: 3 },
        { key: 'lambda_soft', label: 'lambda', digits: 3 },
        { key: 'soft_mass', label: 'soft mass' },
        { key: 'extra_mass', label: 'extra mass' },
        { key: 'hard_delta', label: 'hard delta' },
        { key: 'nonhard_delta', label: 'non-hard delta' },
        { key: 'se', label: 'SE', digits: 0 },
        { key: 'soft_active', label: 'soft cfg/active' },
        { key: 'saliency', label: 'saliency' },
        { key: 'dam_mass', label: 'DAM mass@k' },
        { key: 'dam_corr', label: 'DAM corr' },
        { key: 'ref', label: 'ref', digits: 0 },
      ].filter(col => {
        const suffixByColumn: Record<string, string[]> = {
          ratio: ['ratio'],
          N: ['N'],
          k: ['k'],
          k_over_N: ['k_over_N'],
          gate: ['gamma_abs_mean'],
          score_std: ['score_std'],
          K_eff: ['K_eff'],
          top_m: ['top_m'],
          tau: ['tau'],
          lambda_soft: ['lambda_soft'],
          soft_mass: ['soft_nonhard_mass'],
          extra_mass: ['topm_extra_mass'],
          hard_delta: ['hard_delta_norm'],
          nonhard_delta: ['nonhard_delta_norm'],
          se: ['channel_se'],
          soft_active: ['soft_hard_config', 'soft_hard_active'],
          saliency: ['saliency_mean'],
          dam_mass: ['selected_DAM_mass@k'],
          dam_corr: ['selector_DAM_corr'],
          ref: ['reference_guided'],
        };
        return scales.some(scale => (suffixByColumn[col.key] || [col.key]).some(suffix => present(latest[metricKey(scale, suffix)])));
      }),
    }],
  };
};

const buildDecoderPanel = (latest: HsgDetrMetricsEntry): MetricPanel | null => {
  if (!hasPrefix(latest, 'decoder/')) return null;
  const rows: Array<[string, MetricValue]> = [
    ['alpha', latest['decoder/alpha']],
    ['alpha_progress', latest['decoder/alpha_progress']],
    ['alpha_u', latest['decoder/alpha_u']],
    ['alpha_eff', latest['decoder/alpha_eff']],
    ['num_queries', latest['decoder/num_queries']],
    ['hidden_dim', latest['decoder/hidden_dim']],
    ['loc_quality_mode', latest['decoder/loc_quality_mode']],
    ['cls_conf_mean', latest['decoder/cls_conf_mean']],
    ['cls_conf_std', latest['decoder/cls_conf_std']],
    ['loc_conf_mean', latest['decoder/loc_conf_mean']],
    ['loc_conf_std', latest['decoder/loc_conf_std']],
    ['uncertainty_mean', latest['decoder/uncertainty_mean']],
    ['uncertainty_std', latest['decoder/uncertainty_std']],
    ['selected_cls_conf_mean', latest['decoder/selected_cls_conf_mean']],
    ['selected_loc_conf_mean', latest['decoder/selected_loc_conf_mean']],
    ['selected_uncertainty_mean', latest['decoder/selected_uncertainty_mean']],
    ['score_mean', latest['decoder/score_mean']],
    ['score_std', latest['decoder/score_std']],
    ['selected_score_mean', latest['decoder/selected_score_mean']],
    ['selected_score_std', latest['decoder/selected_score_std']],
    ['score_entropy_mean', latest['decoder/score_entropy_mean']],
    ['selected_box_area_mean', latest['decoder/selected_box_area_mean']],
    ['beta_s', latest['decoder/beta_s']],
  ];
  return {
    id: 'decoder',
    title: 'RT-DETR Decoder',
    summary: `q=${fmt(latest['decoder/num_queries'], 0)} | hd=${fmt(latest['decoder/hidden_dim'], 0)} | loc=${fmt(latest['decoder/loc_quality_mode'])}`,
    badge: 'Decoder',
    priority: 30,
    cards: [
      {
        label: 'Decoder alpha',
        value: fmt(latest['decoder/alpha_eff'] ?? latest['decoder/alpha'], 3),
        color: 'text-emerald-400',
        sub: `progress=${fmt(latest['decoder/alpha_progress'], 2)} | max=${fmt(latest['decoder/alpha_u'], 2)}`,
      },
      {
        label: 'Queries / hidden',
        value: `${fmt(latest['decoder/num_queries'], 0)} / ${fmt(latest['decoder/hidden_dim'], 0)}`,
        color: 'text-cyan-300',
        sub: `loc=${fmt(latest['decoder/loc_quality_mode'])}`,
      },
    ].filter(card => !card.value.includes('-')),
    charts: [{
      title: 'Decoder Alpha Schedule',
      icon: <Zap size={14} className="text-emerald-400" />,
      lines: [
        { key: 'decoder/alpha', label: 'alpha raw', color: '#64748b', dashed: true },
        { key: 'decoder/alpha_eff', label: 'alpha eff', color: '#10b981' },
        { key: 'decoder/alpha_progress', label: 'progress', color: '#38bdf8' },
      ],
    }],
    tables: [],
  };
};

const buildDamPanel = (latest: HsgDetrMetricsEntry): MetricPanel | null => {
  if (!hasPrefix(latest, 'dam/')) return null;
  const lines = SCALES
    .map(scale => ({ key: `dam/${scale}_sampling_mass`, label: `${scale} sampling`, color: SCALE_COLORS[scale] }))
    .filter(line => present(latest[line.key]));
  if (!lines.length) return null;
  return {
    id: 'dam',
    title: 'Approx DAM Sampling Mass',
    summary: 'MSDeformAttn sampling-mass diagnostics, not dense attention',
    badge: 'DAM',
    priority: 40,
    cards: lines.map(line => ({ label: line.label, value: fmt(latest[line.key], 4), color: 'text-emerald-300' })),
    charts: [{ title: 'Approx DAM Sampling Mass', icon: <Crosshair size={14} className="text-emerald-400" />, lines }],
    tables: [],
  };
};

const buildGradPanel = (latest: HsgDetrMetricsEntry): MetricPanel | null => {
  if (!hasPrefix(latest, 'grad/')) return null;
  const normKeys = Object.keys(GRAD_COLORS).filter(key => present(latest[key]));
  const hasNan = Number(latest['grad/has_nan'] ?? 0) === 1;
  const hasInf = Number(latest['grad/has_inf'] ?? 0) === 1;
  return {
    id: 'grad',
    title: 'Gradient Diagnostics',
    summary: `${hasNan ? 'NaN detected' : 'No NaN'} | ${hasInf ? 'Inf detected' : 'No Inf'}`,
    badge: 'Grad',
    priority: 90,
    cards: [
      { label: 'NaN / Inf', value: `${fmt(latest['grad/has_nan'], 0)} / ${fmt(latest['grad/has_inf'], 0)}`, color: hasNan || hasInf ? 'text-red-400' : 'text-emerald-400' },
      ...normKeys.slice(0, 3).map(key => ({ label: key.replace('grad/', ''), value: fmt(latest[key], 4), color: metricColor(latest[key]) })),
    ],
    charts: normKeys.length ? [{
      title: 'Optimizer Grad Groups',
      icon: <ShieldCheck size={14} className="text-cyan-400" />,
      lines: normKeys.map(key => ({ key, label: key.replace('grad/', '').replace('_norm', ''), color: GRAD_COLORS[key] })),
    }] : [],
    tables: [],
  };
};

const buildGenericPanel = (latest: HsgDetrMetricsEntry, knownPrefixes: string[]): MetricPanel | null => {
  const keys = Object.keys(latest).filter(key =>
    key !== 'epoch'
    && !knownPrefixes.some(prefix => key.startsWith(prefix))
    && present(latest[key])
    && (typeof latest[key] === 'number' || typeof latest[key] === 'string')
  );
  if (!keys.length) return null;
  return {
    id: 'generic',
    title: 'Additional Debug Metrics',
    summary: `${keys.length} unclassified metric${keys.length === 1 ? '' : 's'}`,
    badge: 'Extra',
    priority: 100,
    cards: [],
    charts: [],
    tables: [],
  };
};

const buildPanels = (history: HsgDetrMetricsEntry[]) => {
  const latest = history[history.length - 1];
  const builders = [buildCs2gaPanel, buildSgbPanel, buildDecoderPanel, buildDamPanel, buildGradPanel];
  const panels = builders.map(builder => builder(latest)).filter(Boolean) as MetricPanel[];
  const generic = buildGenericPanel(latest, ['cs2ga/', 'sgb/', 'decoder/', 'dam/', 'grad/']);
  if (generic) panels.push(generic);
  return panels.sort((a, b) => a.priority - b.priority);
};

const HsgDetrMetrics: React.FC<Props> = ({ history }) => {
  const normalizedHistory = (history || []).map(normalizeEntry).filter(entry => typeof entry.epoch === 'number');
  if (!normalizedHistory.length) return null;

  const latest = normalizedHistory[normalizedHistory.length - 1];
  const panels = buildPanels(normalizedHistory);
  if (!panels.length) return null;

  const hasNan = Number(latest['grad/has_nan'] ?? 0) === 1;
  const hasInf = Number(latest['grad/has_inf'] ?? 0) === 1;
  const primary = panels[0];
  const hasAnomaly = hasNan || hasInf;

  const visibleCharts = panels.flatMap(panel => panel.charts.map(chart => ({ panel, chart })))
    .filter(({ chart }) => dataHasAny(normalizedHistory, chart.lines.map(line => line.key)));
  const visibleTables = panels.flatMap(panel => panel.tables.map(table => ({ panel, table })));

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-violet-500/10 border border-violet-500/20 flex items-center justify-center">
          <Brain size={16} className="text-violet-400" />
        </div>
        <div>
          <h3 className="text-white font-semibold text-sm">Model Debug Metrics</h3>
          <p className="text-[10px] text-slate-500 uppercase tracking-wider">
            {primary.title}: {primary.summary}
          </p>
        </div>
        <div className="ml-auto flex flex-wrap justify-end gap-2">
          {panels.map(panel => (
            <span key={panel.id} className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border bg-violet-500/10 text-violet-300 border-violet-500/20">
              {panel.badge}
            </span>
          ))}
          <StatusBadge ok={!hasNan} label={hasNan ? 'NaN' : 'No NaN'} />
          <StatusBadge ok={!hasInf} label={hasInf ? 'Inf' : 'No Inf'} />
        </div>
      </div>

      {hasAnomaly && (
        <div className="bg-red-500/5 border border-red-500/20 rounded-lg p-3 flex items-center gap-3 text-red-400 text-xs">
          <AlertTriangle size={16} />
          <span>Numeric anomaly detected at epoch {latest.epoch}. Check AMP scale, gradients, and latest training logs.</span>
        </div>
      )}

      <div className="flex flex-wrap gap-3">
        {panels.flatMap(panel => panel.cards.map(card => <MetricCard key={`${panel.id}-${card.label}`} {...card} />))}
      </div>

      {visibleCharts.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {visibleCharts.map(({ panel, chart }) => (
            <ChartPanel key={`${panel.id}-${chart.title}`} title={chart.title} icon={chart.icon}>
              <SimpleLineChart history={normalizedHistory} lines={chart.lines} domain={chart.domain} />
            </ChartPanel>
          ))}
        </div>
      )}

      {visibleTables.map(({ panel, table }) => <MetricsTable key={`${panel.id}-${table.title}`} table={table} />)}

      {compactRowsTable(`Latest Metrics - Epoch ${latest.epoch}`, [
        ...Object.keys(latest)
          .filter(key => key.startsWith('decoder/') && present(latest[key]))
          .map(key => [key.replace('decoder/', ''), latest[key]] as [string, MetricValue]),
      ])}

      {compactRowsTable(`Gradient Metrics - Epoch ${latest.epoch}`, [
        ...Object.keys(latest)
          .filter(key => key.startsWith('grad/') && present(latest[key]))
          .map(key => [key.replace('grad/', ''), latest[key]] as [string, MetricValue]),
      ])}
    </div>
  );
};

export default HsgDetrMetrics;
