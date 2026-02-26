import { useMemo } from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface EpochMetrics {
  epoch: number;
  box_loss?: number;
  cls_loss?: number;
  dfl_loss?: number;
  precision?: number;
  recall?: number;
  mAP50?: number;
  mAP50_95?: number;
  fitness?: number;
  lr?: number;
  epoch_time?: number;
  gpu_memory_mb?: number;
  map?: number;
  map50?: number;
  map75?: number;
  mp?: number;
  mr?: number;
}

interface MetricsDashboardProps {
  history: EpochMetrics[];
  bestMetrics?: {
    best_fitness?: number;
    best_mAP50?: number;
    best_mAP50_95?: number;
  };
}

export default function MetricsDashboard({ history, bestMetrics }: MetricsDashboardProps) {
  const latestMetrics = useMemo(() => {
    if (!history || history.length === 0) return null;
    return history[history.length - 1];
  }, [history]);

  const metricTrend = (key: keyof EpochMetrics) => {
    if (!history || history.length < 2) return 'stable';
    const current = history[history.length - 1][key] as number;
    const previous = history[history.length - 2][key] as number;
    if (current === undefined || previous === undefined) return 'stable';
    if (current > previous) return 'up';
    if (current < previous) return 'down';
    return 'stable';
  };

  const getTrendIcon = (trend: string, isLoss: boolean = false) => {
    if (trend === 'stable') return <Minus className="w-4 h-4 text-gray-400" />;
    const isGood = isLoss ? trend === 'down' : trend === 'up';
    if (trend === 'up') {
      return <TrendingUp className={`w-4 h-4 ${isGood ? 'text-green-400' : 'text-red-400'}`} />;
    }
    return <TrendingDown className={`w-4 h-4 ${isGood ? 'text-green-400' : 'text-red-400'}`} />;
  };

  const formatValue = (value: number | undefined, decimals: number = 4): string => {
    if (value === undefined || value === null) return 'N/A';
    return value.toFixed(decimals);
  };

  if (!latestMetrics) {
    return (
      <div className="text-center py-8 text-gray-400">
        No metrics available yet
      </div>
    );
  }

  const metricCards = [
    {
      title: 'Loss Metrics',
      metrics: [
        { label: 'Box Loss', key: 'box_loss' as keyof EpochMetrics, isLoss: true, decimals: 4, best: undefined },
        { label: 'Class Loss', key: 'cls_loss' as keyof EpochMetrics, isLoss: true, decimals: 4, best: undefined },
        { label: 'DFL Loss', key: 'dfl_loss' as keyof EpochMetrics, isLoss: true, decimals: 4, best: undefined },
      ],
    },
    {
      title: 'Validation Metrics',
      metrics: [
        { label: 'mAP@0.5', key: 'mAP50' as keyof EpochMetrics, best: bestMetrics?.best_mAP50, decimals: 4, isLoss: false },
        { label: 'mAP@0.5:0.95', key: 'mAP50_95' as keyof EpochMetrics, best: bestMetrics?.best_mAP50_95, decimals: 4, isLoss: false },
        { label: 'Fitness', key: 'fitness' as keyof EpochMetrics, best: bestMetrics?.best_fitness, decimals: 4, isLoss: false },
      ],
    },
    {
      title: 'Precision & Recall',
      metrics: [
        { label: 'Precision', key: 'precision' as keyof EpochMetrics, decimals: 4, best: undefined, isLoss: false },
        { label: 'Recall', key: 'recall' as keyof EpochMetrics, decimals: 4, best: undefined, isLoss: false },
        { label: 'Mean Precision', key: 'mp' as keyof EpochMetrics, decimals: 4, best: undefined, isLoss: false },
        { label: 'Mean Recall', key: 'mr' as keyof EpochMetrics, decimals: 4, best: undefined, isLoss: false },
      ],
    },
    {
      title: 'System',
      metrics: [
        { label: 'Learning Rate', key: 'lr' as keyof EpochMetrics, decimals: 6, best: undefined, isLoss: false },
        { label: 'Epoch Time (s)', key: 'epoch_time' as keyof EpochMetrics, decimals: 2, best: undefined, isLoss: false },
        { label: 'GPU Memory (MB)', key: 'gpu_memory_mb' as keyof EpochMetrics, decimals: 0, best: undefined, isLoss: false },
      ],
    },
  ];

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {metricCards.map((card) => (
          <div key={card.title} className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-gray-300 mb-3">{card.title}</h3>
            <div className="space-y-2">
              {card.metrics.map((metric) => {
                const value = latestMetrics[metric.key] as number;
                const trend = metricTrend(metric.key);
                const decimals = metric.decimals ?? 4;

                return (
                  <div key={metric.label} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {getTrendIcon(trend, metric.isLoss)}
                      <span className="text-xs text-gray-400">{metric.label}</span>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-mono text-white">
                        {formatValue(value, decimals)}
                      </div>
                      {metric.best !== undefined && (
                        <div className="text-xs text-green-400">
                          Best: {formatValue(metric.best, decimals)}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Metrics Table */}
      <div className="bg-gray-800 rounded-lg overflow-hidden">
        <div className="p-4 border-b border-gray-700">
          <h3 className="font-semibold text-white">Training History</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-900 text-gray-400">
              <tr>
                <th className="px-4 py-2 text-left">Epoch</th>
                <th className="px-4 py-2 text-right">Box Loss</th>
                <th className="px-4 py-2 text-right">Cls Loss</th>
                <th className="px-4 py-2 text-right">DFL Loss</th>
                <th className="px-4 py-2 text-right">Precision</th>
                <th className="px-4 py-2 text-right">Recall</th>
                <th className="px-4 py-2 text-right">mAP@0.5</th>
                <th className="px-4 py-2 text-right">mAP@0.5:0.95</th>
                <th className="px-4 py-2 text-right">Fitness</th>
                <th className="px-4 py-2 text-right">LR</th>
                <th className="px-4 py-2 text-right">Time (s)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              {history.slice().reverse().map((epoch) => (
                <tr
                  key={epoch.epoch}
                  className="hover:bg-gray-750 transition-colors"
                >
                  <td className="px-4 py-2 text-white font-medium">{epoch.epoch}</td>
                  <td className="px-4 py-2 text-right font-mono text-gray-300">
                    {formatValue(epoch.box_loss)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-gray-300">
                    {formatValue(epoch.cls_loss)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-gray-300">
                    {formatValue(epoch.dfl_loss)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-gray-300">
                    {formatValue(epoch.precision)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-gray-300">
                    {formatValue(epoch.recall)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-gray-300">
                    {formatValue(epoch.mAP50 || epoch.map50)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-gray-300">
                    {formatValue(epoch.mAP50_95 || epoch.map)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-gray-300">
                    {formatValue(epoch.fitness)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-gray-300 text-xs">
                    {formatValue(epoch.lr, 6)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-gray-300">
                    {formatValue(epoch.epoch_time, 2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
