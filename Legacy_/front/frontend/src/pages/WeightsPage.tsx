/**
 * Weights Page — lists all saved weights with parent model lineage.
 */
import { useEffect, useState } from 'react';
import { api } from '../services/api';
import CopyButton from '../components/CopyButton';
import type { WeightRecord } from '../types';
import { HardDrive, RefreshCw, Trash2 } from 'lucide-react';

export default function WeightsPage() {
  const [weights, setWeights] = useState<WeightRecord[]>([]);
  const [loading, setLoading] = useState(true);

  const loadWeights = async () => {
    try {
      const data = await api.listWeights();
      setWeights(data);
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  useEffect(() => {
    loadWeights();
  }, []);

  const handleDelete = async (weightId: string) => {
    try {
      await api.deleteWeight(weightId);
      setWeights((prev) => prev.filter((w) => w.weight_id !== weightId));
    } catch (e) { console.error(e); }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  return (
    <div className="flex-1 overflow-y-auto p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">Weights</h1>
            <p className="text-slate-400 text-sm mt-1">Trained model weights with parent model tracking</p>
          </div>
          <button onClick={loadWeights} className="px-4 py-2 text-sm font-medium text-slate-300 hover:text-white hover:bg-slate-800 rounded-lg transition-colors flex items-center gap-2 cursor-pointer">
            <RefreshCw size={14} /> Refresh
          </button>
        </div>

        <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden shadow-xl">
          {loading ? (
            <div className="text-center text-slate-500 py-16">Loading weights...</div>
          ) : weights.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <HardDrive size={40} className="text-slate-700 mb-3" />
              <p className="text-slate-400 text-sm">No trained weights yet</p>
              <p className="text-slate-600 text-xs mt-1">Train a model to generate weights</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm text-slate-400">
                <thead className="bg-slate-950/50 text-xs uppercase font-semibold text-slate-500 border-b border-slate-800">
                  <tr>
                    <th className="px-6 py-4">Weight ID</th>
                    <th className="px-6 py-4">Parent Model</th>
                    <th className="px-6 py-4">Dataset</th>
                    <th className="px-6 py-4">Epochs</th>
                    <th className="px-6 py-4">Accuracy</th>
                    <th className="px-6 py-4">Loss</th>
                    <th className="px-6 py-4">Size</th>
                    <th className="px-6 py-4">Created</th>
                    <th className="px-6 py-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/50">
                  {weights.map((w) => (
                    <tr key={w.weight_id} className="hover:bg-slate-800/30 transition-colors group">
                      <td className="px-6 py-4 font-mono text-xs text-slate-300 flex items-center gap-1">
                        {w.weight_id}
                        <CopyButton text={w.weight_id} label="Copy Weight ID" />
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex flex-col">
                          <span className="text-white font-medium text-xs">{w.model_name}</span>
                          <span className="text-slate-600 text-[10px] font-mono">{w.model_id}</span>
                        </div>
                      </td>
                      <td className="px-6 py-4">{w.dataset}</td>
                      <td className="px-6 py-4">{w.epochs_trained}</td>
                      <td className="px-6 py-4 font-mono">{w.final_accuracy != null ? `${w.final_accuracy}%` : '—'}</td>
                      <td className="px-6 py-4 font-mono">{w.final_loss != null ? w.final_loss.toFixed(4) : '—'}</td>
                      <td className="px-6 py-4">{formatBytes(w.file_size_bytes)}</td>
                      <td className="px-6 py-4 text-slate-500 text-xs">{new Date(w.created_at).toLocaleString()}</td>
                      <td className="px-6 py-4 text-right">
                        <div className="flex items-center justify-end gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                          {w.job_id && (
                            <span className="text-[10px] font-mono text-indigo-400 mr-2" title={`From job ${w.job_id}`}>
                              {w.job_id.slice(0, 8)}
                            </span>
                          )}
                          <button className="p-1.5 text-slate-500 hover:text-red-400 hover:bg-slate-700 rounded transition-colors cursor-pointer" onClick={() => handleDelete(w.weight_id)} title="Delete">
                            <Trash2 size={14} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
