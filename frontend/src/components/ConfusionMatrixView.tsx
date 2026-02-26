/**
 * ConfusionMatrixView — heatmap visualization of a confusion matrix.
 * Extracted from WeightDetailPage.tsx.
 */

interface Props {
  matrix: number[][];
  classNames: string[];
}

export default function ConfusionMatrixView({ matrix, classNames }: Props) {
  const n = matrix.length;
  if (n === 0) return null;

  // Find max value for color scaling
  const maxVal = Math.max(...matrix.flat(), 1);

  // Limit display for very large matrices
  const maxDisplay = 15;
  const display = n > maxDisplay ? matrix.slice(0, maxDisplay).map(r => r.slice(0, maxDisplay)) : matrix;
  const displayNames = classNames.slice(0, Math.min(n, maxDisplay));
  const truncated = n > maxDisplay;

  return (
    <div>
      <div className="overflow-x-auto">
        <div className="inline-block">
          {/* Column headers */}
          <div className="flex">
            <div className="w-16 shrink-0" />
            {displayNames.map((name, i) => (
              <div key={i} className="w-10 h-12 flex items-end justify-center pb-1">
                <span className="text-[9px] text-slate-500 rotate-[-45deg] origin-bottom-left whitespace-nowrap">
                  {name.length > 6 ? name.slice(0, 6) + '..' : name}
                </span>
              </div>
            ))}
          </div>
          {/* Rows */}
          {display.map((row, ri) => (
            <div key={ri} className="flex">
              <div className="w-16 shrink-0 flex items-center justify-end pr-2">
                <span className="text-[9px] text-slate-500 truncate">
                  {displayNames[ri] ? (displayNames[ri].length > 8 ? displayNames[ri].slice(0, 8) + '..' : displayNames[ri]) : ri}
                </span>
              </div>
              {row.map((val, ci) => {
                const intensity = val / maxVal;
                const isDiag = ri === ci;
                return (
                  <div
                    key={ci}
                    className="w-10 h-10 flex items-center justify-center text-[9px] font-mono rounded-sm m-px cursor-default transition-transform hover:scale-110"
                    style={{
                      backgroundColor: isDiag
                        ? `rgba(99, 102, 241, ${Math.max(0.08, intensity)})`
                        : val > 0
                          ? `rgba(239, 68, 68, ${Math.max(0.05, intensity * 0.6)})`
                          : 'rgba(30, 41, 59, 0.3)',
                    }}
                    title={`${displayNames[ri] ?? ri} → ${displayNames[ci] ?? ci}: ${val}`}
                  >
                    <span className={val > 0 ? 'text-white/80' : 'text-slate-700'}>{val}</span>
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
      {truncated && (
        <p className="text-xs text-slate-600 mt-2">Showing {maxDisplay}×{maxDisplay} of {n}×{n} matrix</p>
      )}
      <div className="flex justify-between text-xs text-slate-500 mt-3">
        <span>↓ Actual</span>
        <span>→ Predicted</span>
      </div>
    </div>
  );
}
