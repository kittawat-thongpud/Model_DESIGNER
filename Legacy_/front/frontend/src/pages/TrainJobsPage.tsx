/**
 * Train Jobs Page — lists all training jobs with modern Tailwind table.
 */
import { useEffect, useState, useCallback } from 'react';
import { api } from '../services/api';
import CopyButton from '../components/CopyButton';
import type { JobRecord } from '../types';
import {
  Activity,
  CheckCircle2,
  XCircle,
  Clock,
  RefreshCw,
  MoreVertical,
  Square,
  Trash2,
} from 'lucide-react';

interface Props {
  onOpenJob: (jobId: string) => void;
}

export default function TrainJobsPage({ onOpenJob }: Props) {
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<string>('');
  const [refreshing, setRefreshing] = useState(false);

  const loadJobs = useCallback(async () => {
    try {
      const params = filter ? { status: filter } : undefined;
      const data = await api.listJobs(params);
      setJobs(data);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
    setRefreshing(false);
  }, [filter]);

  const handleRefresh = () => {
    setRefreshing(true);
    loadJobs();
  };

  useEffect(() => {
    loadJobs();
    const interval = setInterval(loadJobs, 3000);
    return () => clearInterval(interval);
  }, [filter, loadJobs]);

  const handleDelete = async (jobId: string) => {
    try {
      await api.deleteJob(jobId);
      setJobs((prev) => prev.filter((j) => j.job_id !== jobId));
    } catch (e) {
      console.error(e);
    }
  };

  const handleStop = async (jobId: string) => {
    try {
      await api.stopTraining(jobId);
      loadJobs();
    } catch (e) {
      console.error(e);
    }
  };

  const activeCount = jobs.filter((j) => j.status === 'running').length;
  const completedCount = jobs.filter((j) => j.status === 'completed').length;

  const statusBadge = (status: string) => {
    const styles: Record<string, string> = {
      running: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
      completed: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
      failed: 'bg-red-500/10 text-red-400 border-red-500/20',
      stopped: 'bg-slate-500/10 text-slate-400 border-slate-500/20',
      pending: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    };
    const icons: Record<string, React.ReactNode> = {
      running: <Activity size={12} className="animate-pulse" />,
      completed: <CheckCircle2 size={12} />,
      failed: <XCircle size={12} />,
    };
    return (
      <span className={`px-2.5 py-1 rounded-full text-xs font-medium border capitalize flex items-center gap-1.5 w-fit ${styles[status] || styles.stopped}`}>
        {icons[status]}
        {status}
      </span>
    );
  };

  const progressPct = (job: JobRecord) =>
    job.total_epochs ? Math.round((job.epoch / job.total_epochs) * 100) : 0;

  const progressBarColor = (status: string) => {
    if (status === 'failed') return 'bg-red-500';
    if (status === 'running') return 'bg-blue-500';
    if (status === 'completed') return 'bg-emerald-500';
    return 'bg-slate-600';
  };

  return (
    <div className="flex-1 overflow-y-auto p-8">
      <div className="max-w-6xl w-full mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">Train Jobs</h1>
            <p className="text-slate-400 text-sm mt-1">Manage and monitor your neural network training sessions.</p>
          </div>
          <button
            onClick={handleRefresh}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-medium shadow-lg shadow-indigo-600/20 transition-colors flex items-center gap-2 cursor-pointer"
          >
            <RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} /> Refresh
          </button>
        </div>

        {/* Stats Row */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex items-start justify-between">
            <div>
              <p className="text-sm font-medium text-slate-400 mb-1">Active Jobs</p>
              <h3 className="text-2xl font-bold text-white">{activeCount}</h3>
            </div>
            <div className="w-10 h-10 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center">
              <Activity className="text-blue-400" size={20} />
            </div>
          </div>
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex items-start justify-between">
            <div>
              <p className="text-sm font-medium text-slate-400 mb-1">Completed</p>
              <h3 className="text-2xl font-bold text-white">{completedCount}</h3>
            </div>
            <div className="w-10 h-10 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center">
              <CheckCircle2 className="text-emerald-400" size={20} />
            </div>
          </div>
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex items-start justify-between">
            <div>
              <p className="text-sm font-medium text-slate-400 mb-1">Total Jobs</p>
              <h3 className="text-2xl font-bold text-white">{jobs.length}</h3>
            </div>
            <div className="w-10 h-10 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center">
              <Clock className="text-purple-400" size={20} />
            </div>
          </div>
        </div>

        {/* Table */}
        <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden shadow-xl">
          <div className="p-5 border-b border-slate-800 flex justify-between items-center bg-slate-900/50">
            <h2 className="text-lg font-semibold text-white">Recent Jobs</h2>
            <select
              className="bg-slate-950 border border-slate-700 text-slate-300 text-sm rounded-md px-3 py-1.5 focus:outline-none focus:border-indigo-500 appearance-none cursor-pointer"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            >
              <option value="">All Status</option>
              <option value="pending">Pending</option>
              <option value="running">Running</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
              <option value="stopped">Stopped</option>
            </select>
          </div>

          {loading ? (
            <div className="text-center text-slate-500 py-16">Loading jobs...</div>
          ) : jobs.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <Activity size={40} className="text-slate-700 mb-3" />
              <p className="text-slate-400 text-sm">No training jobs yet</p>
              <p className="text-slate-600 text-xs mt-1">Start training from the Designer page</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm text-slate-400 table-fixed">
                <thead className="bg-slate-950/50 text-xs uppercase font-semibold text-slate-500 border-b border-slate-800">
                  <tr>
                    <th className="px-6 py-4 w-48">Job ID</th>
                    <th className="px-6 py-4">Model</th>
                    <th className="px-6 py-4 w-28">Status</th>
                    <th className="px-6 py-4 w-40">Progress</th>
                    <th className="px-6 py-4 w-24">Accuracy</th>
                    <th className="px-6 py-4 w-40">Created</th>
                    <th className="px-6 py-4 w-24 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/50">
                  {jobs.map((job) => {
                    const pct = progressPct(job);
                    return (
                      <tr
                        key={job.job_id}
                        onClick={() => onOpenJob(job.job_id)}
                        className="hover:bg-slate-800/30 transition-colors group cursor-pointer"
                      >
                        <td className="px-6 py-4 font-mono text-xs text-slate-300">
                          <div className="flex items-center gap-2">
                            <div className={`w-1.5 h-1.5 rounded-full shrink-0 ${job.status === 'running' ? 'bg-blue-500 animate-pulse' : 'bg-transparent'}`} />
                            <span className="truncate">{job.job_id}</span>
                            <span onClick={(e) => e.stopPropagation()}>
                              <CopyButton text={job.job_id} label="Copy Job ID" />
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4 font-medium text-white truncate">{job.model_name}</td>
                        <td className="px-6 py-4">{statusBadge(job.status)}</td>
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-3">
                            <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                              <div
                                className={`h-1.5 rounded-full transition-all ${progressBarColor(job.status)}`}
                                style={{ width: `${pct}%` }}
                              />
                            </div>
                            <span className="text-xs font-medium w-8 text-right shrink-0">{pct}%</span>
                          </div>
                        </td>
                        <td className="px-6 py-4 font-medium text-slate-300">
                          {job.val_accuracy != null ? `${job.val_accuracy}%` : '—'}
                        </td>
                        <td className="px-6 py-4 text-slate-500 text-xs">
                          {new Date(job.created_at).toLocaleString()}
                        </td>
                        <td className="px-6 py-4 text-right">
                          <div className="flex items-center justify-end gap-1 opacity-0 group-hover:opacity-100 transition-opacity" onClick={(e) => e.stopPropagation()}>
                            {job.status === 'running' && (
                              <button className="p-1.5 text-amber-400 hover:bg-slate-700 rounded transition-colors cursor-pointer" onClick={() => handleStop(job.job_id)} title="Stop">
                                <Square size={14} />
                              </button>
                            )}
                            <button className="p-1.5 text-slate-500 hover:text-red-400 hover:bg-slate-700 rounded transition-colors cursor-pointer" onClick={() => handleDelete(job.job_id)} title="Delete">
                              <Trash2 size={14} />
                            </button>
                            <button className="p-1.5 text-slate-500 hover:text-white hover:bg-slate-700 rounded transition-colors cursor-pointer" title="More">
                              <MoreVertical size={14} />
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
