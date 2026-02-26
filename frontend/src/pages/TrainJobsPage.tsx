import { useState, useEffect } from 'react';
import { api } from '../services/api';
import type { JobRecord } from '../types';
import { Activity, Trash2, RefreshCw, Eye, Plus, Play } from 'lucide-react';
import ConfirmDialog from '../components/ConfirmDialog';
import CreateTrainJobModal from '../components/CreateTrainJobModal';
import { JOB_STATUS_COLORS } from '../constants';
import { useJobsStore } from '../store/jobsStore';

interface Props { onOpenJob: (jobId: string) => void; }

export default function TrainJobsPage({ onOpenJob }: Props) {
  const jobs = useJobsStore((s) => s.jobs);
  const loading = useJobsStore((s) => s.loading);
  const loadJobs = useJobsStore((s) => s.load);
  const startPolling = useJobsStore((s) => s.startPolling);
  const stopPolling = useJobsStore((s) => s.stopPolling);
  const invalidateJobs = useJobsStore((s) => s.invalidate);

  const [showCreateModal, setShowCreateModal] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<JobRecord | null>(null);

  const load = () => { loadJobs(true); };
  useEffect(() => { startPolling(); return () => stopPolling(); }, []);

  const handleDeleteConfirm = async () => {
    if (!confirmDelete) return;
    const id = confirmDelete.job_id;
    setConfirmDelete(null);
    await api.deleteJob(id);
    invalidateJobs();
    load();
  };

  const handleJobCreated = (jobId: string) => {
    setShowCreateModal(false);
    load();
    onOpenJob(jobId);
  };

  const handleResume = async (jobId: string) => {
    // Resume not yet implemented in new Ultralytics trainer — just open the job
    onOpenJob(jobId);
  };

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-6xl mx-auto p-8 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">Training Jobs</h1>
            <p className="text-slate-500 text-sm mt-1">Monitor and manage your training runs.</p>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={load} className="flex items-center gap-2 px-3 py-1.5 text-sm text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors cursor-pointer">
              <RefreshCw size={14} /> Refresh
            </button>
            <button
              onClick={() => setShowCreateModal(true)}
              className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition-colors cursor-pointer shadow-lg shadow-indigo-500/20"
            >
              <Plus size={14} /> New Job
            </button>
          </div>
        </div>

        {loading && jobs.length === 0 ? (
          <p className="text-slate-500 text-center py-20">Loading...</p>
        ) : jobs.length === 0 ? (
          <div className="text-center py-20">
            <Activity size={48} className="mx-auto text-slate-700 mb-4" />
            <h3 className="text-lg font-semibold text-white mb-2">No training jobs</h3>
            <p className="text-slate-500 mb-4">Create a training job to get started.</p>
            <button
              onClick={() => setShowCreateModal(true)}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm rounded-lg transition-colors cursor-pointer"
            >
              Create your first job
            </button>
          </div>
        ) : (
          <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
            <table className="w-full text-left text-sm table-fixed">
              <thead>
                <tr className="border-b border-slate-800 text-slate-500 text-xs uppercase tracking-wider">
                  <th className="px-6 py-3 w-36">Job ID</th>
                  <th className="px-6 py-3">Model</th>
                  <th className="px-6 py-3 w-28">Status</th>
                  <th className="px-6 py-3 w-40">Progress</th>
                  <th className="px-6 py-3 w-24">Accuracy</th>
                  <th className="px-6 py-3 w-36">Created</th>
                  <th className="px-6 py-3 w-24 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((j) => (
                  <tr key={j.job_id} className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
                    <td className="px-6 py-4 font-mono text-xs text-slate-400 truncate">
                      <button onClick={() => onOpenJob(j.job_id)} className="hover:text-indigo-400 cursor-pointer transition-colors">{j.job_id.slice(0, 8)}</button>
                    </td>
                    <td className="px-6 py-4 text-white font-medium truncate">
                      <button onClick={() => onOpenJob(j.job_id)} className="hover:text-indigo-400 cursor-pointer transition-colors">{j.model_name}</button>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`px-2 py-0.5 rounded-full text-xs font-medium border ${JOB_STATUS_COLORS[j.status] || JOB_STATUS_COLORS.pending}`}>
                        {j.status}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-indigo-500 rounded-full transition-all"
                            style={{ width: j.total_epochs ? `${(j.epoch / j.total_epochs) * 100}%` : '0%' }}
                          />
                        </div>
                        <span className="text-xs text-slate-500 shrink-0">{j.epoch}/{j.total_epochs}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-emerald-400 text-sm font-medium">
                      {j.val_accuracy != null ? `${j.val_accuracy.toFixed(1)}%` : '—'}
                    </td>
                    <td className="px-6 py-4 text-slate-500 text-xs">{new Date(j.created_at).toLocaleDateString()}</td>
                    <td className="px-6 py-4 text-right">
                      <div className="flex items-center justify-end gap-2">
                        {(j.status === 'stopped' || j.status === 'failed') && (
                          <button onClick={() => handleResume(j.job_id)} className="p-1.5 text-slate-500 hover:text-emerald-400 cursor-pointer" title="Resume"><Play size={14} /></button>
                        )}
                        <button onClick={() => onOpenJob(j.job_id)} className="p-1.5 text-slate-500 hover:text-indigo-400 cursor-pointer" title="View Details"><Eye size={14} /></button>
                        <button onClick={() => setConfirmDelete(j)} className="p-1.5 text-slate-500 hover:text-red-400 cursor-pointer" title="Delete"><Trash2 size={14} /></button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <CreateTrainJobModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onJobCreated={handleJobCreated}
      />

      {confirmDelete && (
        <ConfirmDialog
          title="Delete Job"
          message={`Are you sure you want to delete job "${confirmDelete.job_id.slice(0, 8)}..." (${confirmDelete.model_name})? This action cannot be undone.`}
          confirmLabel="Delete"
          danger
          onConfirm={handleDeleteConfirm}
          onCancel={() => setConfirmDelete(null)}
        />
      )}
    </div>
  );
}
