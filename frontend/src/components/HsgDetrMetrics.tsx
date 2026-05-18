import React from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Activity, AlertTriangle, Brain, Crosshair, Layers, ShieldCheck, Table2, Zap } from 'lucide-react';

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
type TableSpec = {
  title: string;
  rowLabel: string;
  rows: Array<{ id: string; label: string; values: Record<string, MetricValue> }>;
  columns: TableColumn[];
};
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

type MetricSchema = {
  namespace: string;
  title: string;
  badge: string;
  priority: number;
  icon: React.ReactNode;
};

const PALETTE = [
  '#10b981',
  '#38bdf8',
  '#a78bfa',
  '#f59e0b',
  '#f43f5e',
  '#8b5cf6',
  '#fb923c',
  '#22c55e',
  '#60a5fa',
  '#e879f9',
  '#f97316',
  '#14b8a6',
];

const SCHEMAS: MetricSchema[] = [
  { namespace: 'cs2ga', title: 'HSG-DETR V3-CS2GA', badge: 'CS2GA', priority: 10, icon: <Crosshair size={14} className="text-rose-400" /> },
  { namespace: 'sgb', title: 'Sparse Global Blocks', badge: 'SGB', priority: 20, icon: <Layers size={14} className="text-violet-400" /> },
  { namespace: 'decoder', title: 'RT-DETR Decoder', badge: 'Decoder', priority: 30, icon: <Zap size={14} className="text-emerald-400" /> },
  { namespace: 'dam', title: 'Approx DAM Sampling Mass', badge: 'DAM', priority: 40, icon: <Crosshair size={14} className="text-emerald-400" /> },
  { namespace: 'grad', title: 'Gradient Diagnostics', badge: 'Grad', priority: 90, icon: <ShieldCheck size={14} className="text-cyan-400" /> },
];

const CS2GA_GROUPS: Array<{ title: string; match: RegExp; domain?: [number, number] }> = [
  { title: 'CS2GA Gate / LayerScale', match: /(^|\/)(gate_|ls_)/ },
  { title: 'CS2GA Attention Split', match: /attn_(within|cross)_frac$/, domain: [0, 1] },
  { title: 'CS2GA Delta Magnitude', match: /delta_abs_/ },
  { title: 'CS2GA Score Diagnostics', match: /score_(max|min_selected|std)$/ },
  { title: 'CS2GA K Budget', match: /\/k[345]$/ },
  { title: 'CS2GA Attention Entropy', match: /attn_entropy$/ },
];

const SGB_GROUPS: Array<{ title: string; match: RegExp; domain?: [number, number] }> = [
  { title: 'SGB Selection Coverage', match: /(k_over_N|top_m_over_N)$/, domain: [0, 1] },
  { title: 'SGB Soft-Hard State', match: /(soft_nonhard_mass|topm_extra_mass|lambda_soft|tau)$/ },
  { title: 'SGB Delta Norm', match: /(hard_delta_norm|nonhard_delta_norm)$/ },
  { title: 'SGB Gate / Saliency', match: /(gamma_abs_mean|score_std|saliency_mean)$/ },
  { title: 'SGB DAM Alignment', match: /(selector_DAM_corr|selected_DAM_mass@k)$/ },
];

const DECODER_GROUPS: Array<{ title: string; match: RegExp; domain?: [number, number] }> = [
  { title: 'Decoder Alpha Schedule', match: /^decoder\/alpha(_progress|_u|_eff)?$/ },
  { title: 'Decoder Confidence', match: /(cls_conf|loc_conf|uncertainty|score).*_(mean|std)$/ },
  { title: 'Decoder Selected Queries', match: /^decoder\/selected_/ },
];

const GRAD_GROUPS: Array<{ title: string; match: RegExp; domain?: [number, number] }> = [
  { title: 'Gradient Norms', match: /_norm$/ },
  { title: 'Gradient Anomaly Flags', match: /(has_nan|has_inf)$/, domain: [0, 1] },
];

const present = (v: unknown) => v !== undefined && v !== null;
const isFiniteNumber = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const isMetricLeaf = (v: unknown) => typeof v === 'number' || typeof v === 'string' || typeof v === 'boolean';

const fmt = (v: MetricValue, digits = 4) => {
  if (typeof v === 'string') return v;
  if (typeof v === 'boolean') return v ? '1' : '0';
  if (!isFiniteNumber(v)) return '-';
  return Math.abs(v) < 0.001 && v !== 0 ? v.toExponential(2) : v.toFixed(digits);
};

const prettyKey = (key: string) =>
  key
    .replace(/^cs2ga\/(\d+)\//, 'B$1 ')
    .replace(/^sgb\//, '')
    .replace(/^decoder\//, '')
    .replace(/^dam\//, '')
    .replace(/^grad\//, '')
    .replace(/_/g, ' ');

const namespaceOf = (key: string) => {
  const slash = key.indexOf('/');
  return slash > 0 ? key.slice(0, slash) : 'misc';
};

const flattenMetrics = (source: unknown, prefix = '', out: Record<string, MetricValue> = {}) => {
  if (!source || typeof source !== 'object' || Array.isArray(source)) return out;
  Object.entries(source as Record<string, unknown>).forEach(([key, value]) => {
    if (key === 'epoch') return;
    if (isMetricLeaf(value)) {
      out[prefix ? `${prefix}/${key}` : key] = value as MetricValue;
      return;
    }
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      flattenMetrics(value, prefix ? `${prefix}/${key}` : key, out);
    }
  });
  return out;
};

const normalizeEntry = (entry: HsgDetrMetricsEntry): HsgDetrMetricsEntry => {
  const epoch = Number(entry.epoch);
  const flat: Record<string, MetricValue> = {};

  // New jobs may store debug metrics under hsg_detr, plot.hsg_detr, plots.hsg_detr,
  // plot, or directly on METRICS log entries. Flatten all known wrappers so new
  // namespaces render automatically while old job records still work.
  [
    entry.hsg_detr,
    entry.plot?.hsg_detr,
    entry.plots?.hsg_detr,
    entry.metrics?.hsg_detr,
    entry.payload?.hsg_detr,
    entry.plot,
    entry.plots,
    entry.metrics,
    entry.payload,
    entry,
  ].forEach(candidate => flattenMetrics(candidate, '', flat));

  delete flat.type;
  delete flat.timestamp;
  return { ...flat, epoch };
};

const keysForNamespace = (latest: HsgDetrMetricsEntry, namespace: string) =>
  Object.keys(latest)
    .filter(key => key.startsWith(`${namespace}/`) && present(latest[key]) && isMetricLeaf(latest[key]))
    .sort((a, b) => a.localeCompare(b));

const dataHasAny = (history: HsgDetrMetricsEntry[], keys: string[]) =>
  keys.some(key => history.some(row => present(row[key]) && isMetricLeaf(row[key])));

const numericKeys = (history: HsgDetrMetricsEntry[], keys: string[]) =>
  keys.filter(key => history.some(row => isFiniteNumber(row[key])));

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

const makeLines = (keys: string[]) =>
  keys.map((key, i) => ({ key, label: prettyKey(key), color: PALETTE[i % PALETTE.length], dashed: key.includes('cross') || key.includes('nonhard') }));

const makeGroupedCharts = (
  history: HsgDetrMetricsEntry[],
  namespaceKeys: string[],
  groups: Array<{ title: string; match: RegExp; domain?: [number, number] }>,
  icon: React.ReactNode,
) => {
  const charts: ChartSpec[] = [];
  const used = new Set<string>();
  groups.forEach(group => {
    const keys = numericKeys(history, namespaceKeys.filter(key => group.match.test(key))).slice(0, 12);
    if (!keys.length) return;
    keys.forEach(key => used.add(key));
    charts.push({ title: group.title, icon, domain: group.domain, lines: makeLines(keys) });
  });

  const remaining = numericKeys(history, namespaceKeys.filter(key => !used.has(key))).slice(0, 12);
  if (remaining.length) {
    charts.push({ title: 'Additional Metric Trends', icon: <Activity size={14} className="text-slate-400" />, lines: makeLines(remaining) });
  }
  return charts;
};

const makeCards = (latest: HsgDetrMetricsEntry, keys: string[]) => {
  const preferred = [
    /gate_/,
    /^cs2ga\/\d+\/ls_/,
    /attn_cross_frac/,
    /attn_within_frac/,
    /attn_entropy/,
    /delta_abs_/,
    /score_std/,
    /alpha_eff/,
    /num_queries/,
    /hidden_dim/,
    /has_nan/,
    /has_inf/,
    /_norm$/,
  ];
  const sorted = [...keys].sort((a, b) => {
    const ai = preferred.findIndex(rx => rx.test(a));
    const bi = preferred.findIndex(rx => rx.test(b));
    return (ai < 0 ? 99 : ai) - (bi < 0 ? 99 : bi) || a.localeCompare(b);
  });
  return sorted.slice(0, 8).map<CardSpec>(key => ({
    label: prettyKey(key),
    value: fmt(latest[key], key.includes('k3') || key.includes('k4') || key.includes('k5') || key.includes('queries') ? 0 : 4),
    color: key.includes('has_nan') || key.includes('has_inf')
      ? Number(latest[key]) ? 'text-red-400' : 'text-emerald-400'
      : 'text-slate-100',
  }));
};

const tableForNamespace = (latest: HsgDetrMetricsEntry, namespace: string, keys: string[]): TableSpec => {
  if (namespace === 'cs2ga') {
    const blocks = Array.from(new Set(keys.map(key => key.match(/^cs2ga\/(\d+)\//)?.[1]).filter(Boolean) as string[])).sort();
    const metrics = Array.from(new Set(keys.map(key => key.replace(/^cs2ga\/\d+\//, '')))).sort();
    return {
      title: `CS2GA Latest - Epoch ${latest.epoch}`,
      rowLabel: 'Block',
      rows: blocks.map(block => ({
        id: block,
        label: `B${block}`,
        values: Object.fromEntries(metrics.map(metric => [metric, latest[`cs2ga/${block}/${metric}`]])),
      })),
      columns: metrics.map(metric => ({ key: metric, label: metric.replace(/_/g, ' ') })),
    };
  }

  if (namespace === 'sgb') {
    const scales = Array.from(new Set(keys.map(key => key.match(/^sgb\/(P\d)_/)?.[1]).filter(Boolean) as string[])).sort();
    const metrics = Array.from(new Set(keys.map(key => key.replace(/^sgb\/P\d_/, '')))).sort();
    return {
      title: `SGB Latest - Epoch ${latest.epoch}`,
      rowLabel: 'Scale',
      rows: scales.map(scale => ({
        id: scale,
        label: scale,
        values: Object.fromEntries(metrics.map(metric => [metric, latest[`sgb/${scale}_${metric}`]])),
      })),
      columns: metrics.map(metric => ({ key: metric, label: metric.replace(/_/g, ' ') })),
    };
  }

  return {
    title: `${namespace.toUpperCase()} Latest - Epoch ${latest.epoch}`,
    rowLabel: 'Metric',
    rows: keys.map(key => ({ id: key, label: prettyKey(key), values: { value: latest[key] } })),
    columns: [{ key: 'value', label: 'Value' }],
  };
};

const buildPanel = (history: HsgDetrMetricsEntry[], namespace: string): MetricPanel | null => {
  const latest = history[history.length - 1];
  const keys = keysForNamespace(latest, namespace);
  if (!keys.length) return null;

  const schema = SCHEMAS.find(item => item.namespace === namespace) || {
    namespace,
    title: `${namespace.toUpperCase()} Metrics`,
    badge: namespace,
    priority: 80,
    icon: <Table2 size={14} className="text-slate-400" />,
  };
  const numeric = numericKeys(history, keys);
  const groups = namespace === 'cs2ga'
    ? CS2GA_GROUPS
    : namespace === 'sgb'
      ? SGB_GROUPS
      : namespace === 'decoder'
        ? DECODER_GROUPS
        : namespace === 'grad'
          ? GRAD_GROUPS
          : [];

  const charts = groups.length
    ? makeGroupedCharts(history, keys, groups, schema.icon)
    : numeric.length
      ? [{ title: `${schema.title} Trends`, icon: schema.icon, lines: makeLines(numeric.slice(0, 12)) }]
      : [];

  return {
    id: namespace,
    title: schema.title,
    summary: `${keys.length} metric keys | ${numeric.length} numeric trend${numeric.length === 1 ? '' : 's'}`,
    badge: schema.badge,
    priority: schema.priority,
    cards: makeCards(latest, keys),
    charts,
    tables: [tableForNamespace(latest, namespace, keys)],
  };
};

const buildPanels = (history: HsgDetrMetricsEntry[]) => {
  const latest = history[history.length - 1];
  const namespaces = Array.from(new Set(
    Object.keys(latest)
      .filter(key => key !== 'epoch' && key.includes('/') && present(latest[key]))
      .map(namespaceOf),
  ));
  return namespaces
    .map(namespace => buildPanel(history, namespace))
    .filter(Boolean)
    .sort((a, b) => (a as MetricPanel).priority - (b as MetricPanel).priority) as MetricPanel[];
};

const HsgDetrMetrics: React.FC<Props> = ({ history }) => {
  const normalizedHistory = (history || [])
    .map(normalizeEntry)
    .filter(entry => Number.isFinite(entry.epoch))
    .sort((a, b) => a.epoch - b.epoch);

  if (!normalizedHistory.length) return null;

  const latest = normalizedHistory[normalizedHistory.length - 1];
  const panels = buildPanels(normalizedHistory);
  if (!panels.length) return null;

  const hasNan = Number(latest['grad/has_nan'] ?? 0) === 1;
  const hasInf = Number(latest['grad/has_inf'] ?? 0) === 1;
  const primary = panels[0];
  const visibleCharts = panels
    .flatMap(panel => panel.charts.map(chart => ({ panel, chart })))
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
            Dynamic schema | {primary.title}: {primary.summary}
          </p>
        </div>
        <div className="ml-auto flex flex-wrap justify-end gap-2">
          {panels.map(panel => (
            <span key={panel.id} className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border bg-violet-500/10 text-violet-300 border-violet-500/20">
              {panel.badge}
            </span>
          ))}
          {(present(latest['grad/has_nan']) || present(latest['grad/has_inf'])) && (
            <>
              <StatusBadge ok={!hasNan} label={hasNan ? 'NaN' : 'No NaN'} />
              <StatusBadge ok={!hasInf} label={hasInf ? 'Inf' : 'No Inf'} />
            </>
          )}
        </div>
      </div>

      {(hasNan || hasInf) && (
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
    </div>
  );
};

export default HsgDetrMetrics;
