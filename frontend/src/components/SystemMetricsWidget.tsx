import { useState, useEffect } from 'react';
import { Cpu, HardDrive, Activity, Zap } from 'lucide-react';

interface SystemMetrics {
  gpu_memory_used?: number;
  gpu_memory_total?: number;
  gpu_utilization?: number;
  cpu_percent?: number;
  ram_used?: number;
  ram_total?: number;
  disk_used?: number;
  disk_total?: number;
}

interface Props {
  jobId: string;
  isRunning: boolean;
}

export default function SystemMetricsWidget({ jobId, isRunning }: Props) {
  const [metrics, setMetrics] = useState<SystemMetrics>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!isRunning) {
      setLoading(false);
      return;
    }

    // Poll metrics every 2 seconds
    const fetchMetrics = async () => {
      try {
        // Use global metrics endpoint if no jobId provided
        const endpoint = jobId ? `/api/jobs/${jobId}/metrics` : `/api/system/metrics`;
        const response = await fetch(endpoint);
        if (response.ok) {
          const data = await response.json();
          setMetrics(data);
          setLoading(false);
        } else {
          // API failed, keep previous values
          setLoading(false);
        }
      } catch (error) {
        console.error('Failed to fetch metrics:', error);
        // Keep previous metrics values on error
        setLoading(false);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 2000);
    return () => clearInterval(interval);
  }, [jobId, isRunning]);

  if (!isRunning && !metrics.gpu_memory_used) {
    return null;
  }

  const gpuMemPercent = metrics.gpu_memory_total 
    ? (metrics.gpu_memory_used! / metrics.gpu_memory_total) * 100 
    : 0;
  const ramPercent = metrics.ram_total 
    ? (metrics.ram_used! / metrics.ram_total) * 100 
    : 0;

  const formatBytes = (bytes?: number) => {
    if (bytes === undefined || bytes === null) return '-';
    const gb = bytes / (1024 ** 3);
    return `${gb.toFixed(1)} GB`;
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2">
          <Activity size={16} className="text-emerald-400" />
          System Metrics
        </h3>
        {isRunning && (
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* GPU Memory */}
        {metrics.gpu_memory_total && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-400 flex items-center gap-1.5">
                <Zap size={12} className="text-amber-400" />
                GPU Memory
              </span>
              <span className="text-xs text-white font-mono">
                {formatBytes(metrics.gpu_memory_used)} / {formatBytes(metrics.gpu_memory_total)}
              </span>
            </div>
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
              <div 
                className={`h-full rounded-full transition-all duration-300 ${
                  gpuMemPercent > 90 ? 'bg-red-500' : 
                  gpuMemPercent > 70 ? 'bg-amber-500' : 
                  'bg-emerald-500'
                }`}
                style={{ width: `${gpuMemPercent}%` }}
              />
            </div>
            <span className="text-[10px] text-slate-500 font-mono">
              {gpuMemPercent.toFixed(1)}% used
            </span>
          </div>
        )}

        {/* GPU Utilization */}
        {metrics.gpu_utilization !== undefined && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-400 flex items-center gap-1.5">
                <Zap size={12} className="text-indigo-400" />
                GPU Util
              </span>
              <span className="text-xs text-white font-mono">
                {metrics.gpu_utilization !== undefined ? metrics.gpu_utilization.toFixed(0) : '-'}%
              </span>
            </div>
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
              <div 
                className="h-full bg-indigo-500 rounded-full transition-all duration-300"
                style={{ width: `${metrics.gpu_utilization}%` }}
              />
            </div>
            <span className="text-[10px] text-slate-500 font-mono">
              {metrics.gpu_utilization > 80 ? 'High' : metrics.gpu_utilization > 50 ? 'Medium' : 'Low'}
            </span>
          </div>
        )}

        {/* CPU */}
        {metrics.cpu_percent !== undefined && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-400 flex items-center gap-1.5">
                <Cpu size={12} className="text-blue-400" />
                CPU
              </span>
              <span className="text-xs text-white font-mono">
                {metrics.cpu_percent !== undefined ? metrics.cpu_percent.toFixed(0) : '-'}%
              </span>
            </div>
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
              <div 
                className="h-full bg-blue-500 rounded-full transition-all duration-300"
                style={{ width: `${metrics.cpu_percent}%` }}
              />
            </div>
            <span className="text-[10px] text-slate-500 font-mono">
              {metrics.cpu_percent > 80 ? 'High' : metrics.cpu_percent > 50 ? 'Medium' : 'Low'}
            </span>
          </div>
        )}

        {/* RAM */}
        {metrics.ram_total && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-400 flex items-center gap-1.5">
                <HardDrive size={12} className="text-purple-400" />
                RAM
              </span>
              <span className="text-xs text-white font-mono">
                {formatBytes(metrics.ram_used)} / {formatBytes(metrics.ram_total)}
              </span>
            </div>
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
              <div 
                className={`h-full rounded-full transition-all duration-300 ${
                  ramPercent > 90 ? 'bg-red-500' : 
                  ramPercent > 70 ? 'bg-amber-500' : 
                  'bg-purple-500'
                }`}
                style={{ width: `${ramPercent}%` }}
              />
            </div>
            <span className="text-[10px] text-slate-500 font-mono">
              {ramPercent.toFixed(1)}% used
            </span>
          </div>
        )}
      </div>

    </div>
  );
}
