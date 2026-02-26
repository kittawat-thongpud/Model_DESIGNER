import React from 'react';

interface ProgressData {
  type: 'progress';
  epoch: string;
  batch: string;
  percent: number;
  losses?: {
    box: number;
    cls: number;
    dfl: number;
  };
}

interface ProgressLogItemProps {
  timestamp: string;
  message: string;
  data: ProgressData;
}

export function ProgressLogItem({ timestamp, message, data }: ProgressLogItemProps) {
  const { epoch, batch, percent, losses } = data;
  
  return (
    <div className="flex items-center gap-4 px-4 py-3 bg-gradient-to-r from-slate-800/40 to-slate-800/20 rounded-lg border border-slate-700/50 hover:border-slate-600/50 transition-colors">
      {/* Epoch Badge */}
      <div className="flex-shrink-0">
        <div className="px-2 py-1 bg-blue-500/10 border border-blue-500/30 rounded text-xs font-mono text-blue-400">
          Epoch {epoch}
        </div>
      </div>
      
      {/* Progress Bar Section */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-xs text-slate-400 font-medium">
            Batch {batch}
          </span>
          <span className="text-xs font-bold text-cyan-400">
            {percent}%
          </span>
        </div>
        <div className="relative h-2 bg-slate-900/50 rounded-full overflow-hidden border border-slate-700/30">
          <div 
            className="absolute inset-y-0 left-0 bg-gradient-to-r from-blue-500 via-cyan-500 to-teal-400 transition-all duration-500 ease-out"
            style={{ width: `${percent}%` }}
          >
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer" />
          </div>
        </div>
      </div>
      
      {/* Losses Display */}
      {losses && (
        <div className="flex gap-4 flex-shrink-0">
          <div className="flex flex-col items-center">
            <span className="text-[10px] text-slate-500 uppercase tracking-wider mb-0.5">Box</span>
            <span className="text-xs font-mono font-semibold text-orange-400">
              {losses.box.toFixed(3)}
            </span>
          </div>
          <div className="flex flex-col items-center">
            <span className="text-[10px] text-slate-500 uppercase tracking-wider mb-0.5">Cls</span>
            <span className="text-xs font-mono font-semibold text-purple-400">
              {losses.cls.toFixed(3)}
            </span>
          </div>
          <div className="flex flex-col items-center">
            <span className="text-[10px] text-slate-500 uppercase tracking-wider mb-0.5">DFL</span>
            <span className="text-xs font-mono font-semibold text-pink-400">
              {losses.dfl.toFixed(3)}
            </span>
          </div>
        </div>
      )}
      
      {/* Timestamp */}
      <div className="flex-shrink-0 text-[10px] text-slate-600 font-mono">
        {new Date(timestamp).toLocaleTimeString('en-US', { 
          hour12: false,
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit'
        })}
      </div>
    </div>
  );
}

// Add shimmer animation to global CSS or tailwind config
// @keyframes shimmer {
//   0% { transform: translateX(-100%); }
//   100% { transform: translateX(100%); }
// }
