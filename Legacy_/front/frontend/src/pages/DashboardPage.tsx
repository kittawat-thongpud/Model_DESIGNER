/**
 * Dashboard Page — overview stats, recent activity, quick actions.
 */
import { useEffect, useState } from 'react';
import { api } from '../services/api';
import type { DashboardStats, LogEntry, PageName } from '../types';
import {
  Network,
  Activity,
  Zap,
  HardDrive,
  RefreshCw,
  ArrowRight,
} from 'lucide-react';
import type { ReactNode } from 'react';

interface Props {
  onNavigate: (page: PageName) => void;
}

export default function DashboardPage({ onNavigate }: Props) {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [recentLogs, setRecentLogs] = useState<LogEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const [s, logs] = await Promise.all([
          api.getStats(),
          api.getLogs({ limit: 15 }),
        ]);
        setStats(s);
        setRecentLogs(logs);
      } catch (e) {
        console.error('Dashboard load error:', e);
      }
      setLoading(false);
    }
    load();
    const interval = setInterval(load, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading && !stats) {
    return <div className="flex-1 flex items-center justify-center text-slate-500">Loading dashboard...</div>;
  }

  const statCards: { icon: ReactNode; label: string; value: number; color: string; page: PageName }[] = [
    { icon: <Network className="text-indigo-400" size={20} />, label: 'Models', value: stats?.total_models ?? 0, color: 'indigo', page: 'designer' },
    { icon: <Activity className="text-cyan-400" size={20} />, label: 'Train Jobs', value: stats?.total_jobs ?? 0, color: 'cyan', page: 'jobs' },
    { icon: <Zap className="text-emerald-400" size={20} />, label: 'Active Jobs', value: stats?.active_jobs ?? 0, color: 'emerald', page: 'jobs' },
    { icon: <HardDrive className="text-orange-400" size={20} />, label: 'Weights', value: stats?.total_weights ?? 0, color: 'orange', page: 'weights' },
  ];

  const logDotColor: Record<string, string> = {
    info: 'bg-blue-400',
    warning: 'bg-amber-400',
    error: 'bg-red-400',
    debug: 'bg-slate-500',
  };

  return (
    <div className="flex-1 overflow-y-auto p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">Dashboard</h1>
            <p className="text-slate-400 text-sm mt-1">Overview of your Model DESIGNER workspace</p>
          </div>
          <button
            onClick={() => setLoading(true)}
            className="px-4 py-2 text-sm font-medium text-slate-300 hover:text-white hover:bg-slate-800 rounded-lg transition-colors flex items-center gap-2 cursor-pointer"
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} /> Refresh
          </button>
        </div>

        {/* Stat Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {statCards.map((card) => (
            <button
              key={card.label}
              onClick={() => onNavigate(card.page)}
              className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex items-start justify-between hover:border-slate-600 transition-all cursor-pointer text-left group"
            >
              <div>
                <p className="text-sm font-medium text-slate-400 mb-1">{card.label}</p>
                <h3 className="text-2xl font-bold text-white">{card.value}</h3>
              </div>
              <div className="w-10 h-10 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center group-hover:scale-110 transition-transform">
                {card.icon}
              </div>
            </button>
          ))}
        </div>

        {/* Two column grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Recent Activity */}
          <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl shadow-lg overflow-hidden">
            <div className="p-5 border-b border-slate-800">
              <h2 className="text-lg font-semibold text-white">Recent Activity</h2>
            </div>
            <div className="divide-y divide-slate-800/50 max-h-96 overflow-y-auto">
              {recentLogs.length === 0 && (
                <p className="text-sm text-slate-500 text-center py-8">No recent activity</p>
              )}
              {recentLogs.map((log, i) => (
                <div key={i} className="flex items-center gap-3 px-5 py-3 text-sm hover:bg-slate-800/30 transition-colors">
                  <span className={`w-2 h-2 rounded-full shrink-0 ${logDotColor[log.level.toLowerCase()] || 'bg-slate-500'}`} />
                  <span className="text-slate-500 text-xs font-mono w-20 shrink-0">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </span>
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-slate-800 text-slate-400 border border-slate-700 uppercase shrink-0">
                    {log.category}
                  </span>
                  <span className="text-slate-300 truncate">{log.message}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Quick Actions */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl shadow-lg overflow-hidden">
            <div className="p-5 border-b border-slate-800">
              <h2 className="text-lg font-semibold text-white">Quick Actions</h2>
            </div>
            <div className="p-4 space-y-2">
              {[
                { label: 'New Model', page: 'designer' as PageName, icon: <Network size={16} />, primary: true },
                { label: 'View Jobs', page: 'jobs' as PageName, icon: <Activity size={16} /> },
                { label: 'View Weights', page: 'weights' as PageName, icon: <HardDrive size={16} /> },
                { label: 'Datasets', page: 'datasets' as PageName, icon: <HardDrive size={16} /> },
              ].map((action) => (
                <button
                  key={action.label}
                  onClick={() => onNavigate(action.page)}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all cursor-pointer group ${
                    action.primary
                      ? 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-600/20'
                      : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                  }`}
                >
                  {action.icon}
                  <span className="flex-1 text-left">{action.label}</span>
                  <ArrowRight size={14} className="opacity-0 group-hover:opacity-100 transition-opacity" />
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
