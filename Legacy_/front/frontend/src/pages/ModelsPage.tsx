/**
 * Models Page — shows built (compiled) models with code and layer details.
 */
import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import CopyButton from '../components/CopyButton';
import type { BuildSummary, BuildRecord } from '../types';
import { Box, RefreshCw, Trash2, Code2 } from 'lucide-react';

export default function ModelsPage() {
  const [builds, setBuilds] = useState<BuildSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedBuild, setSelectedBuild] = useState<BuildRecord | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const fetchBuilds = useCallback(async () => {
    setLoading(true);
    try {
      const list = await api.listBuilds();
      setBuilds(list);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchBuilds();
  }, [fetchBuilds]);

  const handleSelectBuild = async (buildId: string) => {
    if (selectedBuild?.build_id === buildId) {
      setSelectedBuild(null);
      return;
    }
    setDetailLoading(true);
    try {
      const detail = await api.getBuild(buildId);
      setSelectedBuild(detail);
    } catch {
      // ignore
    } finally {
      setDetailLoading(false);
    }
  };

  const handleDelete = async (buildId: string) => {
    try {
      await api.deleteBuild(buildId);
      setBuilds((prev) => prev.filter((b) => b.build_id !== buildId));
      if (selectedBuild?.build_id === buildId) setSelectedBuild(null);
    } catch {
      // ignore
    }
  };

  const formatDate = (d: string) => {
    try {
      return new Date(d).toLocaleDateString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric',
        hour: '2-digit', minute: '2-digit',
      });
    } catch {
      return d;
    }
  };

  return (
    <div className="flex-1 overflow-y-auto p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">Built Models</h1>
            <p className="text-slate-400 text-sm mt-1">{builds.length} build{builds.length !== 1 ? 's' : ''} available</p>
          </div>
          <button onClick={fetchBuilds} className="px-4 py-2 text-sm font-medium text-slate-300 hover:text-white hover:bg-slate-800 rounded-lg transition-colors flex items-center gap-2 cursor-pointer">
            <RefreshCw size={14} /> Refresh
          </button>
        </div>

        {loading ? (
          <div className="text-center text-slate-500 py-20">Loading builds...</div>
        ) : builds.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-24 text-center">
            <Box size={48} className="text-slate-700 mb-4" />
            <h3 className="text-lg font-semibold text-white mb-2">No builds yet</h3>
            <p className="text-slate-500">Create a model in Designer and click Build to generate PyTorch code.</p>
          </div>
        ) : (
          <div className="flex gap-6">
            {/* Build cards list */}
            <div className="w-80 shrink-0 space-y-3">
              {builds.map((b) => (
                <div
                  key={b.build_id}
                  onClick={() => handleSelectBuild(b.build_id)}
                  className={`bg-slate-900 border rounded-xl p-4 cursor-pointer transition-all hover:shadow-lg ${
                    selectedBuild?.build_id === b.build_id ? 'border-indigo-500 shadow-indigo-500/10' : 'border-slate-800 hover:border-slate-600'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-white font-medium text-sm truncate">{b.model_name}</h4>
                    <button
                      className="p-1 text-slate-600 hover:text-red-400 rounded transition-colors cursor-pointer"
                      onClick={(e) => { e.stopPropagation(); handleDelete(b.build_id); }}
                      title="Delete"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">{b.class_name}</span>
                    <span className="text-xs text-slate-500">{b.layer_count} layers</span>
                  </div>
                  <div className="flex flex-wrap gap-1 mb-2">
                    {b.layer_types.slice(0, 6).map((lt, i) => (
                      <span key={i} className="px-1.5 py-0.5 rounded text-[9px] bg-slate-800 text-slate-400 border border-slate-700">{lt}</span>
                    ))}
                    {b.layer_types.length > 6 && <span className="text-[9px] text-slate-500">+{b.layer_types.length - 6}</span>}
                  </div>
                  <p className="text-[10px] text-slate-600">{formatDate(b.created_at)}</p>
                </div>
              ))}
            </div>

            {/* Detail panel */}
            {selectedBuild && (
              <div className="flex-1 bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
                {detailLoading ? (
                  <div className="text-center text-slate-500 py-16">Loading...</div>
                ) : (
                  <>
                    <div className="p-5 border-b border-slate-800">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
                          <Code2 size={20} />
                        </div>
                        <div>
                          <h3 className="text-white font-semibold">{selectedBuild.model_name}</h3>
                          <span className="text-xs font-mono text-indigo-400">{selectedBuild.class_name}</span>
                        </div>
                      </div>
                    </div>

                    <div className="grid grid-cols-3 border-b border-slate-800">
                      {[
                        { label: 'Nodes', value: selectedBuild.node_count },
                        { label: 'Edges', value: selectedBuild.edge_count },
                        { label: 'Layers', value: selectedBuild.layers.length },
                      ].map((s) => (
                        <div key={s.label} className="p-4 text-center border-r border-slate-800 last:border-r-0">
                          <p className="text-xs text-slate-500">{s.label}</p>
                          <p className="text-lg font-bold text-white">{s.value}</p>
                        </div>
                      ))}
                    </div>

                    <div className="p-5 border-b border-slate-800">
                      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Layer Details</h4>
                      <div className="space-y-1 max-h-48 overflow-y-auto">
                        {selectedBuild.layers.map((layer, i) => (
                          <div key={i} className="flex items-center gap-2 text-xs py-1.5 px-2 rounded hover:bg-slate-800/50">
                            <span className="text-indigo-400 font-medium w-24 shrink-0">{layer.layer_type}</span>
                            <span className="text-slate-500 font-mono truncate">
                              {Object.entries(layer.params).map(([k, v]) => `${k}=${v}`).join(', ')}
                            </span>
                          </div>
                        ))}
                        {selectedBuild.layers.length === 0 && (
                          <p className="text-xs text-slate-500">No layers</p>
                        )}
                      </div>
                    </div>

                    <div className="p-5 border-b border-slate-800">
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Generated Code</h4>
                        <CopyButton text={selectedBuild.code} label="Copy code" />
                      </div>
                      <pre className="code-block"><code>{selectedBuild.code}</code></pre>
                    </div>

                    <div className="px-5 py-3 flex items-center justify-between text-[10px] text-slate-600">
                      <span>Built: {formatDate(selectedBuild.created_at)}</span>
                      <span className="flex items-center gap-1 font-mono">
                        {selectedBuild.build_id}
                        <CopyButton text={selectedBuild.build_id} label="Copy Build ID" />
                      </span>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
