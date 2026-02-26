/**
 * Train Jobs Page — lists all training jobs.
 * Click a row to open full detail page.
 */
import { useEffect, useState } from 'react';
import { api } from '../services/api';
import type { JobRecord } from '../types';

interface Props {
  onOpenJob: (jobId: string) => void;
}

export default function TrainJobsPage({ onOpenJob }: Props) {
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<string>('');

  const loadJobs = async () => {
    try {
      const params = filter ? { status: filter } : undefined;
      const data = await api.listJobs(params);
      setJobs(data);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  useEffect(() => {
    loadJobs();
    const interval = setInterval(loadJobs, 3000);
    return () => clearInterval(interval);
  }, [filter]);

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

  const statusColor: Record<string, string> = {
    pending: '#fb923c',
    running: '#4ade80',
    completed: '#6366f1',
    failed: '#f87171',
    stopped: '#9090b0',
  };

  return (
    <div className="page-container">
      <div className="page-header">
        <h1>🏋️ Train Jobs</h1>
        <div className="page-actions">
          <select className="filter-select" value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option value="">All Status</option>
            <option value="pending">Pending</option>
            <option value="running">Running</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="stopped">Stopped</option>
          </select>
        </div>
      </div>

      <div className="jobs-layout-full">
        <div className="jobs-table-container">
          {loading ? (
            <div className="page-loading">Loading jobs...</div>
          ) : jobs.length === 0 ? (
            <div className="empty-state-page">
              <span className="empty-icon">🏋️</span>
              <p>No training jobs yet</p>
              <p className="text-muted">Start training a model from the Designer page</p>
            </div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Job ID</th>
                  <th>Model</th>
                  <th>Status</th>
                  <th>Progress</th>
                  <th>Accuracy</th>
                  <th>Created</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => (
                  <tr
                    key={job.job_id}
                    onClick={() => onOpenJob(job.job_id)}
                    className="clickable-row"
                  >
                    <td className="mono">{job.job_id}</td>
                    <td>{job.model_name}</td>
                    <td>
                      <span className="status-badge" style={{ color: statusColor[job.status] || '#9090b0' }}>
                        ● {job.status}
                      </span>
                    </td>
                    <td>
                      <div className="progress-bar-container">
                        <div
                          className="progress-bar-fill"
                          style={{ width: `${job.total_epochs ? (job.epoch / job.total_epochs) * 100 : 0}%` }}
                        />
                        <span className="progress-text">{job.epoch}/{job.total_epochs}</span>
                      </div>
                    </td>
                    <td>{job.val_accuracy != null ? `${job.val_accuracy}%` : '—'}</td>
                    <td className="text-muted">{new Date(job.created_at).toLocaleString()}</td>
                    <td>
                      <div className="action-btns" onClick={(e) => e.stopPropagation()}>
                        {job.status === 'running' && (
                          <button className="btn btn-sm btn-danger" onClick={() => handleStop(job.job_id)}>Stop</button>
                        )}
                        <button className="btn btn-sm btn-ghost" onClick={() => handleDelete(job.job_id)}>🗑</button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
