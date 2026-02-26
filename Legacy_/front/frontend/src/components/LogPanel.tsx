/**
 * Log Panel — displays structured logs at the bottom of the screen.
 */
import { useDesignerStore } from '../store/designerStore';
import { RefreshCw } from 'lucide-react';

const LEVEL_CLS: Record<string, string> = {
  DEBUG: 'text-slate-500',
  INFO: 'text-emerald-400',
  WARNING: 'text-amber-400',
  ERROR: 'text-red-400',
};

export default function LogPanel() {
  const logs = useDesignerStore((s) => s.logs);
  const showLogs = useDesignerStore((s) => s.showLogs);
  const refreshLogs = useDesignerStore((s) => s.refreshLogs);

  if (!showLogs) return null;

  return (
    <div className="h-40 border-t border-slate-800 bg-slate-950/80 flex flex-col shrink-0">
      <div className="flex items-center justify-between px-4 py-2 border-b border-slate-800 shrink-0">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Logs</h3>
        <button className="p-1 text-slate-500 hover:text-white rounded transition-colors cursor-pointer" onClick={refreshLogs} title="Refresh">
          <RefreshCw size={12} />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto px-4 py-1 font-mono text-[11px] leading-5">
        {logs.length === 0 && <p className="text-slate-600 py-2">No logs yet.</p>}
        {logs.map((log, i) => (
          <div key={i} className="flex items-baseline gap-2">
            <span className="text-slate-600 w-16 shrink-0">{new Date(log.timestamp).toLocaleTimeString()}</span>
            <span className={`w-14 shrink-0 font-semibold ${LEVEL_CLS[log.level] || 'text-slate-500'}`}>
              {log.level}
            </span>
            <span className="text-slate-500 shrink-0">[{log.category}]</span>
            <span className="text-slate-300 truncate">{log.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
