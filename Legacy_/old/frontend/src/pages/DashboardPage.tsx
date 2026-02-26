/**
 * Dashboard Page — overview stats, recent activity, quick actions.
 */
import { useEffect, useState } from 'react';
import { api } from '../services/api';
import type { DashboardStats, LogEntry, PageName } from '../types';

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
    return <div className="page-container"><div className="page-loading">Loading dashboard...</div></div>;
  }

  const statCards = [
    { icon: '🧠', label: 'Models', value: stats?.total_models ?? 0, color: '#6366f1', page: 'designer' as PageName },
    { icon: '🏋️', label: 'Train Jobs', value: stats?.total_jobs ?? 0, color: '#22d3ee', page: 'jobs' as PageName },
    { icon: '⚡', label: 'Active Jobs', value: stats?.active_jobs ?? 0, color: '#4ade80', page: 'jobs' as PageName },
    { icon: '💾', label: 'Weights', value: stats?.total_weights ?? 0, color: '#fb923c', page: 'weights' as PageName },
  ];

  return (
    <div className="page-container">
      <div className="page-header">
        <h1>Dashboard</h1>
        <p className="page-subtitle">Overview of your Model DESIGNER workspace</p>
      </div>

      <div className="stats-grid">
        {statCards.map((card) => (
          <button
            key={card.label}
            className="stat-card"
            style={{ '--card-accent': card.color } as React.CSSProperties}
            onClick={() => onNavigate(card.page)}
          >
            <div className="stat-icon">{card.icon}</div>
            <div className="stat-info">
              <div className="stat-value">{card.value}</div>
              <div className="stat-label">{card.label}</div>
            </div>
          </button>
        ))}
      </div>

      <div className="dashboard-grid">
        <div className="dashboard-section">
          <h2>Recent Activity</h2>
          <div className="activity-list">
            {recentLogs.length === 0 && <p className="empty-text">No recent activity</p>}
            {recentLogs.map((log, i) => (
              <div key={i} className="activity-item">
                <span className={`activity-dot level-${log.level.toLowerCase()}`} />
                <span className="activity-time">
                  {new Date(log.timestamp).toLocaleTimeString()}
                </span>
                <span className="activity-category">{log.category}</span>
                <span className="activity-message">{log.message}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="dashboard-section">
          <h2>Quick Actions</h2>
          <div className="quick-actions">
            <button className="btn btn-primary" onClick={() => onNavigate('designer')}>
              🧠 New Model
            </button>
            <button className="btn btn-secondary" onClick={() => onNavigate('jobs')}>
              🏋️ View Jobs
            </button>
            <button className="btn btn-secondary" onClick={() => onNavigate('weights')}>
              💾 View Weights
            </button>
            <button className="btn btn-secondary" onClick={() => onNavigate('datasets')}>
              📦 Datasets
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
