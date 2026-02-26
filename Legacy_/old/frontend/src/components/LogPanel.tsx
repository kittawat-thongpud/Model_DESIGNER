/**
 * Log Panel — displays structured logs at the bottom of the screen.
 */
import { useDesignerStore } from '../store/designerStore';

const LEVEL_COLORS: Record<string, string> = {
  DEBUG: '#888',
  INFO: '#4CAF50',
  WARNING: '#FF9800',
  ERROR: '#f44336',
};

export default function LogPanel() {
  const logs = useDesignerStore((s) => s.logs);
  const showLogs = useDesignerStore((s) => s.showLogs);
  const refreshLogs = useDesignerStore((s) => s.refreshLogs);

  if (!showLogs) return null;

  return (
    <div className="log-panel">
      <div className="log-header">
        <h3>📋 Logs</h3>
        <button className="btn btn-ghost btn-sm" onClick={refreshLogs}>🔄 Refresh</button>
      </div>
      <div className="log-list">
        {logs.length === 0 && <p className="log-empty">No logs yet.</p>}
        {logs.map((log, i) => (
          <div key={i} className="log-entry">
            <span className="log-time">{new Date(log.timestamp).toLocaleTimeString()}</span>
            <span className="log-level" style={{ color: LEVEL_COLORS[log.level] || '#888' }}>
              {log.level}
            </span>
            <span className="log-category">[{log.category}]</span>
            <span className="log-message">{log.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
