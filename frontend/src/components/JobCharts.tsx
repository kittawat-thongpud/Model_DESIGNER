import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
  Legend,
  BarChart,
  Bar,
  Brush,
  ReferenceLine,
} from 'recharts';
import {
  TrendingUp,
  Activity,
  Zap,
  Clock,
  Target,
  Crosshair,
  Cpu,
  Timer,
  Eye,
} from 'lucide-react';
import { EpochMetrics } from '../types';

interface JobChartsProps {
  history: EpochMetrics[];
  isDetection: boolean;
  isSelfSupervised?: boolean;
}

// ─── 1. Custom Tooltip ───────────────────────────────────────────────────────

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-slate-950/90 border border-slate-800 p-2.5 rounded-lg shadow-xl backdrop-blur-md text-xs z-50">
        <p className="text-slate-400 font-bold mb-1.5 border-b border-slate-800 pb-1">Epoch {label}</p>
        {payload.map((entry: any, index: number) => (
          <div key={index} className="flex items-center justify-between gap-4 mb-0.5">
            <span className="flex items-center gap-1.5" style={{ color: entry.color || entry.payload.fill }}>
              <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: entry.color || entry.payload.fill }} />
              {entry.name || entry.dataKey}
            </span>
            <span className="font-mono text-white">
              {typeof entry.value === 'number' 
                ? entry.value < 0.001 
                  ? entry.value.toExponential(2) 
                  : entry.value.toFixed(4) 
                : entry.value}
            </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

// ─── 2. Chart Card Container ─────────────────────────────────────────────────

interface ChartCardProps {
  title: string;
  subtitle?: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  height?: number;
  rightElement?: React.ReactNode;
  latestValue?: string | number;
  trend?: 'up' | 'down' | 'neutral';
}

const ChartCard: React.FC<ChartCardProps> = ({
  title,
  subtitle,
  icon,
  children,
  height = 300,
  rightElement,
  latestValue,
  trend,
}) => {
  return (
    <div className="bg-[#0f1117] border border-slate-800 rounded-xl overflow-hidden shadow-sm flex flex-col group hover:border-slate-700 transition-all duration-300">
      <div className="px-5 py-3 border-b border-slate-800/50 bg-slate-900/30 flex justify-between items-center backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <div className="text-slate-400 group-hover:text-white transition-colors duration-300">
            {icon}
          </div>
          <div>
            <h3 className="text-slate-200 font-semibold text-sm tracking-wide flex items-center gap-2">
              {title}
            </h3>
            {subtitle && (
              <p className="text-[10px] text-slate-500 font-medium uppercase tracking-wider">
                {subtitle}
              </p>
            )}
          </div>
        </div>
        <div className="flex items-center gap-4">
          {latestValue !== undefined && (
            <div className="flex items-center gap-2 bg-slate-800/40 px-3 py-1.5 rounded-md border border-slate-800 group-hover:border-slate-700/80 transition-colors">
              <span className="text-[10px] text-slate-500 font-bold uppercase">Latest</span>
              <div className="flex items-center gap-1.5">
                <span className={`font-mono font-bold text-sm ${
                  trend === 'up' ? 'text-emerald-400' : 
                  trend === 'down' ? 'text-rose-400' : 'text-slate-200'
                }`}>
                  {latestValue}
                </span>
                {trend === 'up' && <TrendingUp size={12} className="text-emerald-500" />}
                {trend === 'down' && <TrendingUp size={12} className="text-rose-500 transform rotate-180" />}
              </div>
            </div>
          )}
          {rightElement}
        </div>
      </div>
      <div className="p-4 w-full relative" style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          {children as React.ReactElement}
        </ResponsiveContainer>
      </div>
    </div>
  );
};

// ─── 3. Small Loss Chart ─────────────────────────────────────────────────────

interface SmallLossChartProps {
  title: string;
  data: EpochMetrics[];
  trainKey: string;
  valKey: string;
  color: string;
}

const SmallLossChart: React.FC<SmallLossChartProps> = ({ title, data, trainKey, valKey, color }) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 hover:border-slate-700 transition-colors">
      <div className="flex justify-between items-center mb-2">
        <span className="text-xs font-bold text-slate-400 uppercase" title="Train uses trainer.loss_items. For end2end heads this is one2many + one2one summed; validation uses the validator path, typically one2one for end2end.">{title}</span>
        <div className="flex gap-2 items-center">
           <div className="w-2 h-2 rounded-full" style={{backgroundColor: color}}></div>
           <span className="text-[10px] text-slate-500">Train</span>
           <div className="w-2 h-2 rounded-full border border-slate-500 ml-2"></div>
           <span className="text-[10px] text-slate-500">Val</span>
        </div>
      </div>
      <div className="h-[120px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
            <XAxis dataKey="epoch" hide />
            <YAxis stroke="#475569" tick={{fontSize: 9}} width={25} domain={['auto', 'auto']} />
            <Tooltip content={<CustomTooltip />} />
            <Line type="monotone" dataKey={trainKey} name="Train" stroke={color} strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey={valKey} name="Val" stroke={color} strokeWidth={2} strokeDasharray="4 4" dot={false} strokeOpacity={0.6} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

// ─── 4. Main Charts Component ────────────────────────────────────────────────

const JobCharts: React.FC<JobChartsProps> = ({ history, isDetection, isSelfSupervised }) => {
  if (!history || history.length === 0) return null;

  const latest = history[history.length - 1];
  const prev = history.length > 1 ? history[history.length - 2] : null;

  // Helper to get trend
  const getTrend = (key: string, higherIsBetter = true) => {
    const val = Number(latest[key] || 0);
    const pVal = Number(prev?.[key] || 0);
    if (val === pVal) return 'neutral';
    return (val > pVal) === higherIsBetter ? 'up' : 'down';
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      
      {/* SECTION 1: PERFORMANCE METRICS */}
      {isSelfSupervised && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* k-NN Accuracy Chart */}
          <ChartCard 
            title="k-NN Accuracy" 
            icon={<Target size={18} className="text-emerald-400" />}
            latestValue={latest.knn_accuracy ? `${(Number(latest.knn_accuracy) * 100).toFixed(2)}%` : '-'}
            trend={getTrend('knn_accuracy')}
          >
            <AreaChart data={history.filter(h => h.knn_accuracy)} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="colorkNN" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
              <XAxis dataKey="epoch" stroke="#475569" tick={{fontSize: 10}} minTickGap={20} />
              <YAxis stroke="#475569" tick={{fontSize: 10}} domain={[0, 1]} tickFormatter={(v) => Number(v).toFixed(4)} />
              <Tooltip content={<CustomTooltip />} />
              <Legend iconType="circle" wrapperStyle={{fontSize: '12px'}} />
              <Area type="monotone" dataKey="knn_accuracy" name="k-NN Accuracy" stroke="#10b981" strokeWidth={2} fillOpacity={1} fill="url(#colorkNN)" />
              <Brush 
                dataKey="epoch" 
                height={20} 
                stroke="#475569" 
                fill="#0f1117" 
                tickFormatter={() => ''}
                travellerWidth={10}
              />
            </AreaChart>
          </ChartCard>
        </div>
      )}
      
      {isDetection && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* mAP Chart — mAP50, mAP50-95, mAP75 (optional), fitness (optional) */}
          <ChartCard
            title="Mean Average Precision (mAP)"
            icon={<Target size={18} className="text-emerald-400" />}
            latestValue={latest.mAP50 ? `${(Number(latest.mAP50) * 100).toFixed(2)}%` : '-'}
            trend={getTrend('mAP50')}
          >
            <AreaChart data={history} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="colormAP" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
              <XAxis dataKey="epoch" stroke="#475569" tick={{fontSize: 10}} minTickGap={20} />
              <YAxis stroke="#475569" tick={{fontSize: 10}} domain={[0, 1]} tickFormatter={(v) => `${(Number(v)*100).toFixed(0)}%`} />
              <Tooltip content={<CustomTooltip />} />
              <Legend iconType="circle" wrapperStyle={{fontSize: '12px'}} />
              <Area type="monotone" dataKey="mAP50" name="mAP@50" stroke="#10b981" strokeWidth={2} fillOpacity={1} fill="url(#colormAP)" />
              <Area type="monotone" dataKey="mAP50_95" name="mAP@50-95" stroke="#34d399" strokeWidth={2} fillOpacity={0} strokeDasharray="4 4" />
              {history.some(h => h.mAP75 != null) && (
                <Line type="monotone" dataKey="mAP75" name="mAP@75" stroke="#6ee7b7" strokeWidth={1.5} strokeDasharray="3 3" dot={false} />
              )}
              {history.some(h => h.fitness != null) && (
                <Line type="monotone" dataKey="fitness" name="Fitness" stroke="#f59e0b" strokeWidth={1.5} strokeDasharray="6 2" dot={false} strokeOpacity={0.7} />
              )}
              <Brush
                dataKey="epoch"
                height={20}
                stroke="#475569"
                fill="#0f1117"
                tickFormatter={() => ''}
                travellerWidth={10}
              />
            </AreaChart>
          </ChartCard>

          {/* Precision & Recall */}
          <ChartCard
            title="Precision & Recall"
            icon={<Crosshair size={18} className="text-indigo-400" />}
            latestValue={latest.precision ? Number(latest.precision).toFixed(4) : '-'}
            trend={getTrend('precision')}
          >
            <LineChart data={history} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
              <XAxis dataKey="epoch" stroke="#475569" tick={{fontSize: 10}} minTickGap={20} />
              <YAxis stroke="#475569" tick={{fontSize: 10}} domain={[0, 1]} tickFormatter={(v) => `${(Number(v)*100).toFixed(0)}%`} />
              <Tooltip content={<CustomTooltip />} />
              <Legend iconType="plainline" wrapperStyle={{fontSize: '12px'}} />
              <Line type="monotone" dataKey="precision" name="Precision" stroke="#6366f1" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="recall" name="Recall" stroke="#a855f7" strokeWidth={2} dot={false} />
            </LineChart>
          </ChartCard>
        </div>
      )}

      {/* SECTION 2: LOSS ANALYSIS (GRID) */}
      <div>
        <h3 className="text-white font-semibold flex items-center gap-2 mb-4">
          <Activity size={18} className="text-rose-400" /> Loss Analysis
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <SmallLossChart 
            title="Box Loss" 
            data={history} 
            trainKey="box_loss" 
            valKey="val_box_loss" 
            color="#f43f5e" 
          />
          <SmallLossChart 
            title="Class Loss" 
            data={history} 
            trainKey="cls_loss" 
            valKey="val_cls_loss" 
            color="#f59e0b" 
          />
          {isDetection && (
            <SmallLossChart 
              title="DFL Loss" 
              data={history} 
              trainKey="dfl_loss" 
              valKey="val_dfl_loss" 
              color="#3b82f6" 
            />
          )}
        </div>
      </div>

      {/* SECTION 3: SYSTEM & LR */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
         <div className="md:col-span-2">
            <ChartCard
              title="Learning Rate Schedule"
              icon={<Zap size={18} className="text-yellow-400" />}
              height={200}
              latestValue={latest.lr != null ? Number(latest.lr).toExponential(2) : '-'}
              trend="neutral"
            >
              <AreaChart data={history} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorLr" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#eab308" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#eab308" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
                <XAxis dataKey="epoch" stroke="#475569" tick={{fontSize: 10}} minTickGap={20} />
                <YAxis stroke="#475569" tick={{fontSize: 10}} tickFormatter={(v) => v.toExponential(0)} scale="log" domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="stepAfter" dataKey="lr" stroke="#eab308" strokeWidth={2} dot={false} fill="url(#colorLr)" />
                <Brush
                  dataKey="epoch"
                  height={20}
                  stroke="#475569"
                  fill="#0f1117"
                  tickFormatter={() => ''}
                  travellerWidth={10}
                />
              </AreaChart>
            </ChartCard>
         </div>
         <div>
            <ChartCard
              title="Time per Epoch (s)"
              icon={<Clock size={18} className="text-slate-400" />}
              height={200}
              latestValue={latest.epoch_time != null ? Number(latest.epoch_time).toFixed(1) : '-'}
              trend="neutral"
            >
              <BarChart data={history} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
                <XAxis dataKey="epoch" stroke="#475569" tick={{fontSize: 10}} minTickGap={20} />
                <YAxis stroke="#475569" tick={{fontSize: 10}} domain={['auto', 'auto']} tickFormatter={(v) => `${Number(v).toFixed(0)}s`} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="epoch_time" name="Epoch time (s)" fill="#475569" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ChartCard>
         </div>
      </div>

      {/* SECTION 4: GPU / RAM RESOURCES (conditional — only when data exists) */}
      {history.some(h => h.gpu_mem_gb != null) && (
        <div>
          <h3 className="text-white font-semibold flex items-center gap-2 mb-4">
            <Cpu size={18} className="text-sky-400" /> System Resources
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* GPU Memory */}
            <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 hover:border-slate-700 transition-colors">
              <div className="flex justify-between items-center mb-2">
                <span className="text-xs font-bold text-slate-400 uppercase">GPU Memory (GB)</span>
                <span className="text-xs text-slate-500 font-mono">
                  {latest.gpu_mem_gb != null ? `${Number(latest.gpu_mem_gb).toFixed(2)} / ${Number(latest.gpu_mem_reserved_gb ?? latest.gpu_mem_gb).toFixed(2)} GB` : '—'}
                </span>
              </div>
              <div className="h-[120px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={history}>
                    <defs>
                      <linearGradient id="colorGpuAlloc" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.4}/>
                        <stop offset="95%" stopColor="#38bdf8" stopOpacity={0}/>
                      </linearGradient>
                      <linearGradient id="colorGpuRsv" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#7dd3fc" stopOpacity={0.15}/>
                        <stop offset="95%" stopColor="#7dd3fc" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
                    <XAxis dataKey="epoch" hide />
                    <YAxis stroke="#475569" tick={{fontSize: 9}} width={30} domain={[0, 'auto']} tickFormatter={(v) => `${Number(v).toFixed(1)}`} />
                    <Tooltip content={<CustomTooltip />} />
                    {history.some(h => h.gpu_mem_reserved_gb != null) && (
                      <Area type="monotone" dataKey="gpu_mem_reserved_gb" name="Reserved (GB)" stroke="#7dd3fc" strokeWidth={1.5} strokeDasharray="4 4" fillOpacity={1} fill="url(#colorGpuRsv)" />
                    )}
                    <Area type="monotone" dataKey="gpu_mem_gb" name="Allocated (GB)" stroke="#38bdf8" strokeWidth={2} fillOpacity={1} fill="url(#colorGpuAlloc)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
            {/* RAM */}
            {history.some(h => h.ram_gb != null) && (
              <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 hover:border-slate-700 transition-colors">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs font-bold text-slate-400 uppercase">RAM Usage (GB)</span>
                  <span className="text-xs text-slate-500 font-mono">
                    {latest.ram_gb != null ? `${Number(latest.ram_gb).toFixed(2)} GB` : '—'}
                  </span>
                </div>
                <div className="h-[120px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={history}>
                      <defs>
                        <linearGradient id="colorRam" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#a78bfa" stopOpacity={0.35}/>
                          <stop offset="95%" stopColor="#a78bfa" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
                      <XAxis dataKey="epoch" hide />
                      <YAxis stroke="#475569" tick={{fontSize: 9}} width={30} domain={[0, 'auto']} tickFormatter={(v) => `${Number(v).toFixed(1)}`} />
                      <Tooltip content={<CustomTooltip />} />
                      <Area type="monotone" dataKey="ram_gb" name="RAM (GB)" stroke="#a78bfa" strokeWidth={2} fillOpacity={1} fill="url(#colorRam)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* SECTION 6: CS²GA DEBUG METRICS (conditional) */}
      {history.some(h => (h.hsg_detr as any)?.['cs2ga/0/ls_p3'] != null) && (() => {
        const cs2gaHistory = history.map(h => {
          const d = (h.hsg_detr || {}) as Record<string, number>;
          return {
            epoch: h.epoch,
            ls_p3: d['cs2ga/0/ls_p3'],
            ls_p4: d['cs2ga/0/ls_p4'],
            ls_p5: d['cs2ga/0/ls_p5'],
            delta_p3: d['cs2ga/0/delta_abs_p3'],
            delta_p4: d['cs2ga/0/delta_abs_p4'],
            delta_p5: d['cs2ga/0/delta_abs_p5'],
            attn_cross: d['cs2ga/0/attn_cross_frac'],
            attn_within: d['cs2ga/0/attn_within_frac'],
            attn_entropy: d['cs2ga/0/attn_entropy'],
            grad_backbone: d['grad/backbone_norm'],
            grad_neck: d['grad/neck_norm'],
            grad_sgb: d['grad/sgb_norm'],
            grad_sgb_sparse: d['grad/sgb_sparse_norm'],
            grad_sgb_gamma: d['grad/sgb_gamma_norm'],
          };
        }).filter(h => h.ls_p3 != null);
        const latestCs2ga = cs2gaHistory[cs2gaHistory.length - 1];
        return (
          <div>
            <h3 className="text-white font-semibold flex items-center gap-2 mb-4">
              <Eye size={18} className="text-violet-400" /> CS²GA Attention Debug
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">

              {/* LayerScale */}
              <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs font-bold text-slate-400 uppercase">LayerScale (ls_p*)</span>
                  <span className="text-xs text-violet-400 font-mono">
                    p3:{latestCs2ga?.ls_p3?.toFixed(3)} p4:{latestCs2ga?.ls_p4?.toFixed(3)} p5:{latestCs2ga?.ls_p5?.toFixed(3)}
                  </span>
                </div>
                <div className="h-[120px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={cs2gaHistory} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
                      <XAxis dataKey="epoch" stroke="#475569" tick={{fontSize: 9}} minTickGap={20} />
                      <YAxis stroke="#475569" tick={{fontSize: 9}} width={40} tickFormatter={v => v.toFixed(3)} domain={[0, 'auto']} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend iconType="circle" wrapperStyle={{fontSize: '10px'}} />
                      <Line type="monotone" dataKey="ls_p3" name="ls_p3" stroke="#a78bfa" strokeWidth={1.5} dot={false} />
                      <Line type="monotone" dataKey="ls_p4" name="ls_p4" stroke="#7c3aed" strokeWidth={1.5} dot={false} />
                      <Line type="monotone" dataKey="ls_p5" name="ls_p5" stroke="#5b21b6" strokeWidth={1.5} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <p className="text-xs text-slate-500 mt-1">Should grow over epochs if attention is useful</p>
              </div>

              {/* Delta abs */}
              <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs font-bold text-slate-400 uppercase">Delta Magnitude (ls × Δ)</span>
                  <span className="text-xs text-cyan-400 font-mono">
                    p5:{((latestCs2ga?.ls_p5 ?? 0) * (latestCs2ga?.delta_p5 ?? 0)).toExponential(2)}
                  </span>
                </div>
                <div className="h-[120px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={cs2gaHistory} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
                      <XAxis dataKey="epoch" stroke="#475569" tick={{fontSize: 9}} minTickGap={20} />
                      <YAxis stroke="#475569" tick={{fontSize: 9}} width={45} tickFormatter={v => v.toExponential(1)} domain={[0, 'auto']} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend iconType="circle" wrapperStyle={{fontSize: '10px'}} />
                      <Line type="monotone" dataKey="delta_p3" name="Δ_p3" stroke="#22d3ee" strokeWidth={1.5} dot={false} />
                      <Line type="monotone" dataKey="delta_p4" name="Δ_p4" stroke="#0891b2" strokeWidth={1.5} dot={false} />
                      <Line type="monotone" dataKey="delta_p5" name="Δ_p5" stroke="#164e63" strokeWidth={1.5} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <p className="text-xs text-slate-500 mt-1">Should grow as attention contributes more</p>
              </div>

              {/* Cross-scale attention fraction */}
              <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs font-bold text-slate-400 uppercase">Attention Mix</span>
                  <span className="text-xs text-emerald-400 font-mono">
                    cross: {(latestCs2ga?.attn_cross != null ? (latestCs2ga.attn_cross * 100).toFixed(1) : '—')}%
                  </span>
                </div>
                <div className="h-[120px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={cs2gaHistory} margin={{ top: 5, right: 5, left: 0, bottom: 0 }} stackOffset="expand">
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
                      <XAxis dataKey="epoch" stroke="#475569" tick={{fontSize: 9}} minTickGap={20} />
                      <YAxis stroke="#475569" tick={{fontSize: 9}} width={30} tickFormatter={v => `${(v*100).toFixed(0)}%`} domain={[0, 1]} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend iconType="square" wrapperStyle={{fontSize: '10px'}} />
                      <Area type="monotone" dataKey="attn_cross" name="Cross-scale" stackId="1" stroke="#10b981" fill="#10b981" fillOpacity={0.4} />
                      <Area type="monotone" dataKey="attn_within" name="Within-scale" stackId="1" stroke="#6b7280" fill="#6b7280" fillOpacity={0.3} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                <p className="text-xs text-slate-500 mt-1">Cross &gt; 50% = healthy mixing. If drops below 40% = collapse</p>
              </div>

              {/* Gradient norms comparison */}
              <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 md:col-span-2 xl:col-span-3">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs font-bold text-slate-400 uppercase">Gradient Norms — Backbone vs Neck vs CS²GA</span>
                  <span className="text-xs text-slate-500 font-mono">
                    ratio sgb/backbone: {latestCs2ga?.grad_backbone && latestCs2ga?.grad_sgb
                      ? (latestCs2ga.grad_sgb / latestCs2ga.grad_backbone).toFixed(3)
                      : '—'}×
                  </span>
                </div>
                <div className="h-[160px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={cs2gaHistory} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
                      <XAxis dataKey="epoch" stroke="#475569" tick={{fontSize: 9}} minTickGap={20} />
                      <YAxis stroke="#475569" tick={{fontSize: 9}} width={40} tickFormatter={v => v.toFixed(3)} domain={[0, 'auto']} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend iconType="circle" wrapperStyle={{fontSize: '10px'}} />
                      <Line type="monotone" dataKey="grad_backbone" name="Backbone" stroke="#f59e0b" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="grad_neck" name="Neck" stroke="#3b82f6" strokeWidth={1.5} dot={false} />
                      <Line type="monotone" dataKey="grad_sgb" name="CS²GA (total)" stroke="#a78bfa" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="grad_sgb_sparse" name="CS²GA proj" stroke="#7c3aed" strokeWidth={1} strokeDasharray="3 3" dot={false} />
                      <Line type="monotone" dataKey="grad_sgb_gamma" name="CS²GA ls" stroke="#ec4899" strokeWidth={1} strokeDasharray="3 3" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <p className="text-xs text-slate-500 mt-1">
                  Goal: CS²GA gradient should grow to be within 5× of neck (not 50×). After LR fix expect ratio to increase.
                </p>
              </div>

            </div>
          </div>
        );
      })()}

      {/* SECTION 5: INFERENCE LATENCY (conditional — only when data exists) */}
      {history.some(h => h.inference_latency_ms != null) && (
        <div>
          <h3 className="text-white font-semibold flex items-center gap-2 mb-4">
            <Timer size={18} className="text-orange-400" /> Inference Latency (ms)
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Latency stacked bar */}
            <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 hover:border-slate-700 transition-colors col-span-full">
              <div className="flex justify-between items-center mb-2">
                <span className="text-xs font-bold text-slate-400 uppercase">Pipeline Breakdown (ms / image)</span>
                <span className="text-xs text-slate-500 font-mono">
                  Latest total: {latest.total_latency_ms != null ? `${Number(latest.total_latency_ms).toFixed(1)} ms` : '—'}
                </span>
              </div>
              <div className="h-[140px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={history.filter(h => h.inference_latency_ms != null)} margin={{ top: 5, right: 10, left: 0, bottom: 0 }} stackOffset="none">
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
                    <XAxis dataKey="epoch" stroke="#475569" tick={{fontSize: 10}} minTickGap={20} />
                    <YAxis stroke="#475569" tick={{fontSize: 9}} width={35} tickFormatter={(v) => `${Number(v).toFixed(0)}`} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend iconType="square" wrapperStyle={{fontSize: '11px'}} />
                    <Bar dataKey="preprocess_latency_ms" name="Preprocess" stackId="lat" fill="#6366f1" />
                    <Bar dataKey="inference_latency_ms" name="Inference" stackId="lat" fill="#f59e0b" />
                    <Bar dataKey="postprocess_latency_ms" name="Postprocess" stackId="lat" fill="#ec4899" radius={[2,2,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default JobCharts;
