import { useState, useEffect } from 'react';
import { api } from '../services/api';
import type { WeightRecord, ModelSummary } from '../types';
import { Weight, Trash2, RefreshCw, Eye, Plus, Loader2, X, Network } from 'lucide-react';
import ConfirmDialog from '../components/ConfirmDialog';
import { fmtSize } from '../utils/format';
import { useWeightsStore } from '../store/weightsStore';
import { useModelsStore } from '../store/modelsStore';

interface Props { onOpenWeight?: (weightId: string) => void; }

export default function WeightsPage({ onOpenWeight }: Props) {
  const weights = useWeightsStore((s) => s.weights);
  const loading = useWeightsStore((s) => s.loading);
  const loadWeights = useWeightsStore((s) => s.load);
  const invalidateWeights = useWeightsStore((s) => s.invalidate);
  const models = useModelsStore((s) => s.models);
  const loadModels = useModelsStore((s) => s.load);

  const [deleteTarget, setDeleteTarget] = useState<WeightRecord | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState('');
  const [weightName, setWeightName] = useState('');
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);

  const load = () => { loadWeights(null, true); };
  useEffect(() => { loadWeights(); }, []);

  const handleDelete = async () => {
    if (!deleteTarget) return;
    await api.deleteWeight(deleteTarget.weight_id);
    setDeleteTarget(null);
    invalidateWeights();
    load();
  };

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-6xl mx-auto p-8 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">Trained Weights</h1>
            <p className="text-slate-500 text-sm mt-1">Model weights saved from training jobs.</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                setShowCreateModal(true);
                setCreateError(null);
                loadModels(true);
              }}
              className="flex items-center gap-2 px-3 py-1.5 text-sm bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-colors cursor-pointer"
            >
              <Plus size={14} /> Create Empty Weight
            </button>
            <button onClick={load} className="flex items-center gap-2 px-3 py-1.5 text-sm text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors cursor-pointer">
              <RefreshCw size={14} /> Refresh
            </button>
          </div>
        </div>

        {loading ? (
          <p className="text-slate-500 text-center py-20">Loading...</p>
        ) : weights.length === 0 ? (
          <div className="text-center py-20">
            <Weight size={48} className="mx-auto text-slate-700 mb-4" />
            <h3 className="text-lg font-semibold text-white mb-2">No weights yet</h3>
            <p className="text-slate-500">Train a model to save weights here.</p>
          </div>
        ) : (
          <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
            <table className="w-full text-left text-sm table-fixed">
              <thead>
                <tr className="border-b border-slate-800 text-slate-500 text-xs uppercase tracking-wider">
                  <th className="px-6 py-3 w-36">Weight ID</th>
                  <th className="px-6 py-3">Model</th>
                  <th className="px-6 py-3 w-28">Dataset</th>
                  <th className="px-6 py-3 w-24">Epochs</th>
                  <th className="px-6 py-3 w-24">Accuracy</th>
                  <th className="px-6 py-3 w-24">Size</th>
                  <th className="px-6 py-3 w-36">Created</th>
                  <th className="px-6 py-3 w-20 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {weights.map((w) => (
                  <tr key={w.weight_id} className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
                    <td className="px-6 py-4 font-mono text-xs text-slate-400 truncate">
                      {onOpenWeight ? (
                        <button onClick={() => onOpenWeight(w.weight_id)} className="hover:text-indigo-400 cursor-pointer transition-colors">{w.weight_id.slice(0, 8)}</button>
                      ) : w.weight_id.slice(0, 8)}
                    </td>
                    <td className="px-6 py-4 text-white font-medium truncate">
                      {onOpenWeight ? (
                        <button onClick={() => onOpenWeight(w.weight_id)} className="hover:text-indigo-400 cursor-pointer transition-colors">{w.model_name}</button>
                      ) : w.model_name}
                    </td>
                    <td className="px-6 py-4 text-slate-400">{w.dataset}</td>
                    <td className="px-6 py-4 text-slate-400">{w.epochs_trained}</td>
                    <td className="px-6 py-4 text-emerald-400 font-medium">
                      {w.final_accuracy != null ? `${w.final_accuracy.toFixed(1)}%` : '—'}
                    </td>
                    <td className="px-6 py-4 text-slate-500 text-xs">{fmtSize(w.file_size_bytes)}</td>
                    <td className="px-6 py-4 text-slate-500 text-xs">{new Date(w.created_at).toLocaleDateString()}</td>
                    <td className="px-6 py-4 text-right">
                      <div className="flex items-center justify-end gap-2">
                        {onOpenWeight && (
                          <button onClick={() => onOpenWeight(w.weight_id)} className="p-1.5 text-slate-500 hover:text-indigo-400 cursor-pointer" title="View Details">
                            <Eye size={14} />
                          </button>
                        )}
                        <button onClick={() => setDeleteTarget(w)} className="p-1.5 text-slate-500 hover:text-red-400 cursor-pointer" title="Delete">
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

        {/* Create Empty Weight Modal */}
        {showCreateModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
            <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-[420px] overflow-hidden">
              <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800">
                <h3 className="text-white font-semibold text-sm flex items-center gap-2">
                  <Plus size={16} className="text-indigo-400" />
                  Create Empty Weight
                </h3>
                <button onClick={() => setShowCreateModal(false)} className="text-slate-500 hover:text-white cursor-pointer">
                  <X size={16} />
                </button>
              </div>
              <div className="p-5 space-y-4">
                <p className="text-xs text-slate-400">
                  Generate a randomly-initialised weight file from a model's architecture. No training required.
                </p>
                <div>
                  <label className="block text-[10px] text-slate-500 mb-1">Weight Name</label>
                  <input
                    type="text"
                    value={weightName}
                    onChange={(e) => setWeightName(e.target.value)}
                    placeholder="e.g. MyModel-init"
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:border-indigo-500 outline-none placeholder:text-slate-600"
                  />
                </div>
                <div>
                  <label className="block text-[10px] text-slate-500 mb-1">Select Model</label>
                  <select
                    value={selectedModelId}
                    onChange={(e) => setSelectedModelId(e.target.value)}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:border-indigo-500 outline-none"
                  >
                    <option value="">— Choose a model —</option>
                    {models.map((m) => (
                      <option key={m.model_id} value={m.model_id}>
                        {m.name} ({m.layer_count} layers)
                      </option>
                    ))}
                  </select>
                </div>
                {createError && (
                  <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
                    {createError}
                  </div>
                )}
              </div>
              <div className="flex justify-end gap-2 px-5 py-3 border-t border-slate-800">
                <button
                  onClick={() => setShowCreateModal(false)}
                  className="px-4 py-1.5 text-xs text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  onClick={async () => {
                    if (!selectedModelId) return;
                    setCreating(true);
                    setCreateError(null);
                    try {
                      await api.createEmptyWeight(selectedModelId, weightName);
                      setShowCreateModal(false);
                      setSelectedModelId('');
                      setWeightName('');
                      load();
                    } catch (e) {
                      setCreateError((e as Error).message);
                    } finally {
                      setCreating(false);
                    }
                  }}
                  disabled={!selectedModelId || creating}
                  className="flex items-center gap-1.5 px-4 py-1.5 text-xs bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-default"
                >
                  {creating ? <Loader2 size={12} className="animate-spin" /> : <Network size={12} />}
                  Create
                </button>
              </div>
            </div>
          </div>
        )}

        {deleteTarget && (
          <ConfirmDialog
            title="Delete Weight"
            message={`Delete weight ${deleteTarget.weight_id.slice(0, 8)} (${deleteTarget.model_name}, ${deleteTarget.dataset})? The .pt file will be permanently removed.`}
            confirmLabel="Delete"
            danger
            onConfirm={handleDelete}
            onCancel={() => setDeleteTarget(null)}
          />
        )}
      </div>
    </div>
  );
}
