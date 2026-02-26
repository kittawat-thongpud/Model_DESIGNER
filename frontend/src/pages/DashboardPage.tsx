import { useState, useEffect } from 'react';
import { api } from '../services/api';
import type { DashboardStats, LogEntry, PageName } from '../types';
import { Box, Activity, Weight, Database, RefreshCw, Network, ArrowRight } from 'lucide-react';
import SystemMetricsWidget from '../components/SystemMetricsWidget';

interface Props { onNavigate: (page: PageName) => void; }

export default function DashboardPage({ onNavigate }: Props) {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);

  const load = () => {
    api.getStats().then(setStats).catch(() => {});
    api.getLogs({ limit: 15, days: 1 }).then(setLogs).catch(() => {});
  };

  useEffect(() => { load(); }, []);

  const cards = stats ? [
    { label: 'Models', value: stats.total_models, icon: <Box size={20} />, color: 'text-indigo-400', bg: 'bg-indigo-500/10' },
    { label: 'Train Jobs', value: stats.total_jobs, icon: <Activity size={20} />, color: 'text-emerald-400', bg: 'bg-emerald-500/10' },
    { label: 'Active Jobs', value: stats.active_jobs, icon: <Activity size={20} />, color: 'text-amber-400', bg: 'bg-amber-500/10' },
    { label: 'Weights', value: stats.total_weights, icon: <Weight size={20} />, color: 'text-cyan-400', bg: 'bg-cyan-500/10' },
  ] : [];

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-6xl mx-auto p-8 space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">Dashboard</h1>
            <p className="text-slate-500 text-sm mt-1">Overview of your Model DESIGNER workspace</p>
          </div>
          <button onClick={load} className="flex items-center gap-2 px-3 py-1.5 text-sm text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors cursor-pointer">
            <RefreshCw size={14} /> Refresh
          </button>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {cards.map((c) => (
            <div key={c.label} className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex items-center justify-between">
              <div>
                <p className="text-slate-500 text-xs font-medium uppercase tracking-wider">{c.label}</p>
                <p className="text-2xl font-bold text-white mt-1">{c.value}</p>
              </div>
              <div className={`w-10 h-10 rounded-lg ${c.bg} flex items-center justify-center ${c.color}`}>{c.icon}</div>
            </div>
          ))}
        </div>

        {/* System Metrics - Global */}
        <SystemMetricsWidget jobId="" isRunning={true} />

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Recent Activity */}
          <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-white mb-4">Recent Activity</h3>
            <div className="space-y-2 max-h-80 overflow-y-auto">
              {logs.length === 0 ? (
                <p className="text-slate-600 text-sm">No recent activity</p>
              ) : logs.map((log, i) => (
                <div key={i} className="flex items-center gap-3 text-sm py-1.5">
                  <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${log.level === 'ERROR' ? 'bg-red-400' : log.level === 'WARNING' ? 'bg-amber-400' : 'bg-emerald-400'}`} />
                  <span className="text-slate-500 text-xs font-mono shrink-0">{new Date(log.timestamp).toLocaleTimeString()}</span>
                  <span className="text-slate-600 text-xs font-mono px-1.5 py-0.5 bg-slate-800 rounded shrink-0">{log.level}</span>
                  <span className="text-slate-300 truncate">{log.message}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Quick Actions */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-white mb-4">Quick Actions</h3>
            <div className="space-y-2">
              {[
                { label: 'Model Designer', icon: <Network size={16} />, page: 'model-designer' as PageName, primary: true },
                { label: 'View Jobs', icon: <Activity size={16} />, page: 'jobs' as PageName },
                { label: 'View Weights', icon: <Weight size={16} />, page: 'weights' as PageName },
                { label: 'Datasets', icon: <Database size={16} />, page: 'datasets' as PageName },
              ].map((a) => (
                <button
                  key={a.label}
                  onClick={() => onNavigate(a.page)}
                  className={`flex items-center gap-3 w-full px-4 py-2.5 rounded-lg text-sm font-medium transition-colors cursor-pointer ${
                    a.primary
                      ? 'bg-indigo-600 hover:bg-indigo-500 text-white'
                      : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                  }`}
                >
                  {a.icon}
                  <span className="flex-1 text-left">{a.label}</span>
                  <ArrowRight size={14} className="text-slate-500" />
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
