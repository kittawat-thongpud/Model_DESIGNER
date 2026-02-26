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
} from 'recharts';
import {
  TrendingUp,
  Activity,
  Zap,
  Clock,
  Target,
  Crosshair,
} from 'lucide-react';
import { EpochMetrics } from '../types';

interface JobChartsProps {
  history: EpochMetrics[];
  isDetection: boolean;
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
        <span className="text-xs font-bold text-slate-400 uppercase">{title}</span>
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

const JobCharts: React.FC<JobChartsProps> = ({ history, isDetection }) => {
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
      {isDetection && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* mAP Chart */}
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
              <YAxis stroke="#475569" tick={{fontSize: 10}} domain={[0, 1]} tickFormatter={(v) => v.toFixed(2)} />
              <Tooltip content={<CustomTooltip />} />
              <Legend iconType="circle" wrapperStyle={{fontSize: '12px'}} />
              <Area type="monotone" dataKey="mAP50" name="mAP@50" stroke="#10b981" strokeWidth={2} fillOpacity={1} fill="url(#colormAP)" />
              <Area type="monotone" dataKey="mAP50_95" name="mAP@50-95" stroke="#34d399" strokeWidth={2} fillOpacity={0} strokeDasharray="4 4" />
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
              <YAxis stroke="#475569" tick={{fontSize: 10}} domain={[0, 1]} tickFormatter={(v) => v.toFixed(2)} />
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
              latestValue={Number(latest.lr).toExponential(2)}
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
              latestValue={Number(latest.epoch_time).toFixed(1)}
              trend="neutral"
            >
              <BarChart data={history} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
                <XAxis dataKey="epoch" stroke="#475569" tick={{fontSize: 10}} minTickGap={20} />
                <YAxis stroke="#475569" tick={{fontSize: 10}} domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="epoch_time" fill="#475569" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ChartCard>
         </div>
      </div>
    </div>
  );
};

export default JobCharts;
