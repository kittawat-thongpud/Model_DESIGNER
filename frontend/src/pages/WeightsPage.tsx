import { useState, useEffect, useRef } from 'react';
import { api } from '../services/api';
import type { WeightRecord, ModelSummary } from '../types';
import { Weight, Trash2, RefreshCw, Eye, Plus, Loader2, X, Network, Upload, Download, BarChart2, Pencil, Check, Package, ChevronDown } from 'lucide-react';
import ConfirmDialog from '../components/ConfirmDialog';
import { fmtSize, fmtDataset } from '../utils/format';
import { useWeightsStore } from '../store/weightsStore';
import { useModelsStore } from '../store/modelsStore';
import BenchmarkPanel from '../components/BenchmarkPanel';
import ExportWeightPanel from '../components/ExportWeightPanel';
import ImportPackageModal from '../components/ImportPackageModal';

interface Props { onOpenWeight?: (weightId: string) => void; }

export default function WeightsPage({ onOpenWeight }: Props) {
  const weights = useWeightsStore((s) => s.weights);
  const loading = useWeightsStore((s) => s.loading);
  const loadWeights = useWeightsStore((s) => s.load);
  const invalidateWeights = useWeightsStore((s) => s.invalidate);
  const models = useModelsStore((s) => s.models);
  const loadModels = useModelsStore((s) => s.load);

  const [deleteTarget, setDeleteTarget] = useState<WeightRecord | null>(null);
  const [renameTarget, setRenameTarget] = useState<WeightRecord | null>(null);
  const [renameName, setRenameName] = useState('');
  const [renaming, setRenaming] = useState(false);
  const [benchmarkTarget, setBenchmarkTarget] = useState<WeightRecord | null>(null);
  const [exportTarget, setExportTarget] = useState<WeightRecord | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState('');
  const [weightName, setWeightName] = useState('');
  const [createScale, setCreateScale] = useState<string>('n');
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  // Official YOLO
  const [createMode, setCreateMode] = useState<'custom' | 'official'>('official');
  const [yoloVariant, setYoloVariant] = useState<string>('yolov8');
  const [usePretrained, setUsePretrained] = useState(true);

  // Import state
  const [importing, setImporting] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);
  const importFileRef = useRef<HTMLInputElement>(null);
  const [showImportPkg, setShowImportPkg] = useState(false);
  const [showImportMenu, setShowImportMenu] = useState(false);
  const [exportMenuFor, setExportMenuFor] = useState<string | null>(null);
  const [pkgIncludeJobs, setPkgIncludeJobs] = useState(false);
  const importMenuRef = useRef<HTMLDivElement>(null);
  const exportMenuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (importMenuRef.current && !importMenuRef.current.contains(e.target as Node)) setShowImportMenu(false);
      if (exportMenuRef.current && !exportMenuRef.current.contains(e.target as Node)) setExportMenuFor(null);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const load = () => { loadWeights(null, true); };
  useEffect(() => { loadWeights(); }, []);

  const handleDelete = async () => {
    if (!deleteTarget) return;
    await api.deleteWeight(deleteTarget.weight_id);
    setDeleteTarget(null);
    invalidateWeights();
    load();
  };

  const handleRename = async () => {
    if (!renameTarget || !renameName.trim()) return;
    setRenaming(true);
    try {
      await api.renameWeight(renameTarget.weight_id, renameName.trim());
      setRenameTarget(null);
      invalidateWeights();
      load();
    } catch { /* ignore */ }
    setRenaming(false);
  };

  const handleImportFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImporting(true);
    setImportError(null);
    try {
      const result = await api.importWeight(file, file.name.replace(/\.pt[h]?$/, ''));
      invalidateWeights();
      load();
      if (onOpenWeight) onOpenWeight(result.weight_id);
    } catch (err) {
      setImportError(err instanceof Error ? err.message : 'Import failed');
    } finally {
      setImporting(false);
      if (importFileRef.current) importFileRef.current.value = '';
    }
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
              <Plus size={14} /> Create Empty
            </button>

            {/* Import dropdown */}
            <div className="relative" ref={importMenuRef}>
              <button
                onClick={() => setShowImportMenu(v => !v)}
                className={`flex items-center gap-2 px-3 py-1.5 text-sm bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 rounded-lg transition-colors cursor-pointer ${importing ? 'opacity-60 pointer-events-none' : ''}`}
              >
                {importing ? <RefreshCw size={14} className="animate-spin" /> : <Upload size={14} />}
                {importing ? 'Importing…' : 'Import'}
                {!importing && <ChevronDown size={13} className={`transition-transform ${showImportMenu ? 'rotate-180' : ''}`} />}
              </button>
              {showImportMenu && !importing && (
                <div className="absolute right-0 top-full mt-1 w-52 bg-slate-800 border border-slate-700 rounded-xl shadow-xl z-30 overflow-hidden">
                  <label className="w-full flex items-center gap-3 px-4 py-3 text-sm text-slate-200 hover:bg-slate-700 transition-colors cursor-pointer">
                    <Upload size={14} className="text-emerald-400 shrink-0" />
                    <div className="text-left">
                      <div className="font-medium">Import .pt</div>
                      <div className="text-xs text-slate-500">Upload weight file</div>
                    </div>
                    <input ref={importFileRef} type="file" accept=".pt,.pth" className="hidden" disabled={importing}
                      onChange={(e) => { setShowImportMenu(false); handleImportFile(e); }} />
                  </label>
                  <div className="border-t border-slate-700/50" />
                  <button
                    onClick={() => { setShowImportPkg(true); setShowImportMenu(false); }}
                    className="w-full flex items-center gap-3 px-4 py-3 text-sm text-slate-200 hover:bg-slate-700 transition-colors cursor-pointer"
                  >
                    <Package size={14} className="text-violet-400 shrink-0" />
                    <div className="text-left">
                      <div className="font-medium">Import Package</div>
                      <div className="text-xs text-slate-500">Import .mdpkg with lineage</div>
                    </div>
                  </button>
                </div>
              )}
            </div>

            <button onClick={load} className="flex items-center gap-2 px-3 py-1.5 text-sm text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors cursor-pointer">
              <RefreshCw size={14} /> Refresh
            </button>
          </div>
        </div>

        {importError && (
          <div className="flex items-center gap-2 text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-2">
            <X size={12} /> {importError}
            <button onClick={() => setImportError(null)} className="ml-auto text-slate-500 hover:text-white cursor-pointer"><X size={12} /></button>
          </div>
        )}

        {loading ? (
          <p className="text-slate-500 text-center py-20">Loading...</p>
        ) : weights.length === 0 ? (
          <div className="text-center py-20">
            <Weight size={48} className="mx-auto text-slate-700 mb-4" />
            <h3 className="text-lg font-semibold text-white mb-2">No weights yet</h3>
            <p className="text-slate-500">Train a model to save weights here, or import a .pt file.</p>
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
                  <th className="px-6 py-3 w-28 text-right">Actions</th>
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
                    <td className="px-6 py-4 text-slate-400">{fmtDataset(w.dataset, w.dataset_name)}</td>
                    <td className="px-6 py-4 text-slate-400">{w.epochs_trained}</td>
                    <td className="px-6 py-4 text-emerald-400 font-medium">
                      {w.final_accuracy != null ? `${w.final_accuracy.toFixed(1)}%` : '—'}
                    </td>
                    <td className="px-6 py-4 text-slate-500 text-xs">{fmtSize(w.file_size_bytes)}</td>
                    <td className="px-6 py-4 text-slate-500 text-xs">{new Date(w.created_at).toLocaleDateString()}</td>
                    <td className="px-6 py-4 text-right">
                      <div className="flex items-center justify-end gap-1">
                        {onOpenWeight && (
                          <button
                            onClick={() => onOpenWeight(w.weight_id)}
                            className="p-1.5 text-slate-500 hover:text-indigo-400 cursor-pointer transition-colors"
                            title="View Details"
                          >
                            <Eye size={14} />
                          </button>
                        )}
                        <button
                          onClick={() => setBenchmarkTarget(w)}
                          className="p-1.5 text-slate-500 hover:text-indigo-400 cursor-pointer transition-colors"
                          title="Benchmark"
                        >
                          <BarChart2 size={14} />
                        </button>
                        {/* Export dropdown per-row */}
                        <div className="relative" ref={exportMenuFor === w.weight_id ? exportMenuRef : undefined}>
                          <button
                            onClick={() => setExportMenuFor(exportMenuFor === w.weight_id ? null : w.weight_id)}
                            className="p-1.5 text-slate-500 hover:text-emerald-400 cursor-pointer transition-colors flex items-center gap-0.5"
                            title="Export"
                          >
                            <Download size={14} /><ChevronDown size={11} />
                          </button>
                          {exportMenuFor === w.weight_id && (
                            <div className="absolute right-0 top-full mt-1 w-48 bg-slate-800 border border-slate-700 rounded-xl shadow-xl z-30 overflow-hidden">
                              <button
                                onClick={() => { setExportTarget(w); setExportMenuFor(null); }}
                                className="w-full flex items-center gap-3 px-3 py-2.5 text-sm text-slate-200 hover:bg-slate-700 transition-colors cursor-pointer"
                              >
                                <Download size={13} className="text-emerald-400 shrink-0" />
                                <div className="text-left">
                                  <div className="text-xs font-medium">Export .pt</div>
                                  <div className="text-[10px] text-slate-500">Weight file only</div>
                                </div>
                              </button>
                              <div className="border-t border-slate-700/50" />
                              <button
                                onClick={() => { api.exportWeightPackage(w.weight_id, pkgIncludeJobs); setExportMenuFor(null); }}
                                className="w-full flex items-center gap-3 px-3 py-2.5 text-sm text-slate-200 hover:bg-slate-700 transition-colors cursor-pointer"
                              >
                                <Package size={13} className="text-violet-400 shrink-0" />
                                <div className="text-left flex-1">
                                  <div className="text-xs font-medium">Export Package</div>
                                  <div className="text-[10px] text-slate-500">Lineage (.mdpkg)</div>
                                </div>
                              </button>
                              <div className="border-t border-slate-700/50" />
                              <div className="px-3 py-2 flex items-center justify-between">
                                <span className="text-[10px] text-slate-400">Include Jobs</span>
                                <button
                                  onClick={e => { e.stopPropagation(); setPkgIncludeJobs(v => !v); }}
                                  className={`w-7 h-4 rounded-full relative transition-colors cursor-pointer shrink-0 ${pkgIncludeJobs ? 'bg-violet-500' : 'bg-slate-600'}`}
                                >
                                  <div className={`absolute top-0.5 w-3 h-3 bg-white rounded-full shadow transition-transform ${pkgIncludeJobs ? 'translate-x-3.5' : 'translate-x-0.5'}`} />
                                </button>
                              </div>
                            </div>
                          )}
                        </div>
                        <button
                          onClick={() => { setRenameTarget(w); setRenameName(w.model_name); }}
                          className="p-1.5 text-slate-500 hover:text-amber-400 cursor-pointer transition-colors"
                          title="Rename"
                        >
                          <Pencil size={14} />
                        </button>
                        <button
                          onClick={() => setDeleteTarget(w)}
                          className="p-1.5 text-slate-500 hover:text-red-400 cursor-pointer transition-colors"
                          title="Delete"
                        >
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

        {/* Import Package Modal */}
        {showImportPkg && (
          <ImportPackageModal
            onClose={() => setShowImportPkg(false)}
            onDone={() => { invalidateWeights(); setTimeout(load, 50); }}
          />
        )}

        {/* Rename Weight Dialog */}
        {renameTarget && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => setRenameTarget(null)}>
            <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-96 overflow-hidden" onClick={e => e.stopPropagation()}>
              <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800">
                <h3 className="text-white font-semibold text-sm flex items-center gap-2">
                  <Pencil size={15} className="text-amber-400" /> Rename Weight
                </h3>
                <button onClick={() => setRenameTarget(null)} className="text-slate-500 hover:text-white cursor-pointer"><X size={16} /></button>
              </div>
              <div className="p-5 space-y-4">
                <div>
                  <label className="block text-xs text-slate-400 mb-1.5">Weight Name</label>
                  <input
                    autoFocus
                    type="text"
                    value={renameName}
                    onChange={e => setRenameName(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter') handleRename(); if (e.key === 'Escape') setRenameTarget(null); }}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:border-amber-500 outline-none"
                  />
                </div>
                <div className="flex justify-end gap-2">
                  <button onClick={() => setRenameTarget(null)} className="px-4 py-2 text-sm text-slate-400 hover:text-white cursor-pointer">Cancel</button>
                  <button
                    onClick={handleRename}
                    disabled={renaming || !renameName.trim()}
                    className="flex items-center gap-2 px-4 py-2 text-sm bg-amber-600 hover:bg-amber-500 disabled:opacity-40 text-white rounded-lg cursor-pointer"
                  >
                    {renaming ? <Loader2 size={14} className="animate-spin" /> : <Check size={14} />} Save
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Create Empty Weight Modal */}
        {showCreateModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => setShowCreateModal(false)}>
            <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-[480px] overflow-hidden" onClick={e => e.stopPropagation()}>
              <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800">
                <h3 className="text-white font-semibold text-sm flex items-center gap-2">
                  <Plus size={16} className="text-indigo-400" />
                  Create Weight
                </h3>
                <button onClick={() => setShowCreateModal(false)} className="text-slate-500 hover:text-white cursor-pointer">
                  <X size={16} />
                </button>
              </div>
              <div className="p-5 space-y-4">
                {/* Mode toggle */}
                <div className="flex gap-1 p-1 bg-slate-800 rounded-lg">
                  {(['official', 'custom'] as const).map(mode => (
                    <button
                      key={mode}
                      onClick={() => setCreateMode(mode)}
                      className={`flex-1 py-1.5 text-xs font-medium rounded-md transition-all cursor-pointer ${
                        createMode === mode
                          ? 'bg-indigo-600 text-white shadow'
                          : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      {mode === 'official' ? '⚡ Official YOLO' : '🔧 Custom Model'}
                    </button>
                  ))}
                </div>

                {/* Official YOLO section */}
                {createMode === 'official' && (() => {
                  const ARCH_SCALES: Record<string, string[]> = {
                    yolov8: ['n','s','m','l','x'],
                    yolov9: ['t','s','m','c','e'],
                    yolov10: ['n','s','m','b','l','x'],
                    yolov11: ['n','s','m','l','x'],
                    rtdetr: ['l','x'],
                  };
                  const availableScales = ARCH_SCALES[yoloVariant] ?? ['n','s','m','l','x'];
                  const effectiveScale = availableScales.includes(createScale) ? createScale : availableScales[0];
                  const modelKey = yoloVariant === 'rtdetr' ? `rtdetr-${effectiveScale}` : `${yoloVariant}${effectiveScale}`;
                  return (
                  <>
                    <div>
                      <label className="block text-[10px] text-slate-500 mb-1.5">Architecture</label>
                      <div className="flex flex-wrap gap-1.5">
                        {(['yolov8','yolov9','yolov10','yolov11','rtdetr'] as const).map(v => (
                          <button
                            key={v}
                            onClick={() => {
                              setYoloVariant(v);
                              const sc = ARCH_SCALES[v];
                              if (!sc.includes(createScale)) setCreateScale(sc[0]);
                            }}
                            className={`px-2.5 py-1.5 rounded-lg border text-[10px] font-semibold transition-all cursor-pointer ${
                              yoloVariant === v
                                ? v === 'rtdetr'
                                  ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-400'
                                  : 'bg-amber-500/20 border-amber-500/50 text-amber-400'
                                : 'bg-slate-800 border-slate-700 text-slate-400 hover:border-slate-600'
                            }`}
                          >
                            {v === 'rtdetr' ? 'RT-DETR' : v.replace('yolo', 'YOLO')}
                          </button>
                        ))}
                      </div>
                    </div>
                    <div>
                      <label className="block text-[10px] text-slate-500 mb-1.5">Scale</label>
                      <div className="flex gap-1.5">
                        {availableScales.map(sc => (
                          <button
                            key={sc}
                            onClick={() => setCreateScale(sc)}
                            className={`flex-1 py-1.5 rounded-lg border text-xs font-bold transition-all cursor-pointer ${
                              effectiveScale === sc
                                ? 'bg-indigo-500/20 border-indigo-500/50 text-indigo-400'
                                : 'bg-slate-800 border-slate-700 text-slate-400 hover:border-slate-600'
                            }`}
                          >
                            {sc.toUpperCase()}
                          </button>
                        ))}
                      </div>
                      <p className="text-[10px] text-slate-600 mt-1">
                        Model: <span className="text-amber-400 font-mono">{modelKey}</span>
                      </p>
                    </div>
                    {/* Pretrained toggle */}
                    <div className="flex items-center justify-between bg-slate-800/60 rounded-lg px-3 py-2.5">
                      <div>
                        <div className="text-xs text-white font-medium">Load Pretrained Weights</div>
                        <div className="text-[10px] text-slate-500 mt-0.5">
                          {usePretrained
                            ? 'Download COCO-pretrained checkpoint from Ultralytics'
                            : 'Random initialization — no download needed'}
                        </div>
                      </div>
                      <button
                        onClick={() => setUsePretrained(v => !v)}
                        className={`w-10 h-5.5 rounded-full relative transition-colors cursor-pointer shrink-0 ml-3 ${
                          usePretrained ? 'bg-amber-500' : 'bg-slate-600'
                        }`}
                        style={{ minWidth: '2.5rem', height: '1.375rem' }}
                      >
                        <div className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${
                          usePretrained ? 'translate-x-5' : 'translate-x-0.5'
                        }`} />
                      </button>
                    </div>
                    {usePretrained && (
                      <p className="text-[10px] text-amber-400/80 bg-amber-500/5 border border-amber-500/20 rounded-lg px-3 py-2">
                        ⚡ Pretrained on COCO — ideal for fine-tuning or transfer learning
                      </p>
                    )}
                  </>
                  );
                })()}

                {/* Custom model section */}
                {createMode === 'custom' && (
                  <>
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
                    <div>
                      <label className="block text-[10px] text-slate-500 mb-1">Model Scale</label>
                      <div className="flex gap-1.5">
                        {['n', 's', 'm', 'l', 'x'].map(sc => (
                          <button
                            key={sc}
                            onClick={() => setCreateScale(sc)}
                            className={`flex-1 py-1.5 rounded-lg border text-xs font-medium transition-all cursor-pointer ${
                              createScale === sc
                                ? 'bg-indigo-500/20 border-indigo-500/50 text-indigo-400'
                                : 'bg-slate-800 border-slate-700 text-slate-400 hover:border-slate-600'
                            }`}
                          >
                            {sc.toUpperCase()}
                          </button>
                        ))}
                      </div>
                    </div>
                  </>
                )}

                {/* Weight name (always shown) */}
                <div>
                  <label className="block text-[10px] text-slate-500 mb-1">Weight Name <span className="text-slate-600">(optional)</span></label>
                  <input
                    type="text"
                    value={weightName}
                    onChange={(e) => setWeightName(e.target.value)}
                    placeholder={createMode === 'official' ? `e.g. ${yoloVariant === 'rtdetr' ? `rtdetr-${createScale}` : `${yoloVariant}${createScale}`}-finetune` : 'e.g. MyModel-init'}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:border-indigo-500 outline-none placeholder:text-slate-600"
                  />
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
                    if (createMode === 'custom' && !selectedModelId) return;
                    setCreating(true);
                    setCreateError(null);
                    try {
                      if (createMode === 'official') {
                        const yoloKey = yoloVariant === 'rtdetr' ? `rtdetr-${createScale}` : `${yoloVariant}${createScale}`;
                        await api.createEmptyWeight('', weightName, createScale, yoloKey, usePretrained);
                      } else {
                        await api.createEmptyWeight(selectedModelId, weightName, createScale);
                      }
                      setShowCreateModal(false);
                      setSelectedModelId('');
                      setWeightName('');
                      setCreateScale('n');
                      load();
                    } catch (e) {
                      setCreateError((e as Error).message);
                    } finally {
                      setCreating(false);
                    }
                  }}
                  disabled={(createMode === 'custom' && !selectedModelId) || creating}
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
            message={`Delete weight ${deleteTarget.weight_id.slice(0, 8)} (${deleteTarget.model_name}, ${fmtDataset(deleteTarget.dataset, deleteTarget.dataset_name)})? The .pt file will be permanently removed.`}
            confirmLabel="Delete"
            danger
            onConfirm={handleDelete}
            onCancel={() => setDeleteTarget(null)}
          />
        )}
      </div>

      {benchmarkTarget && (
        <BenchmarkPanel
          weightId={benchmarkTarget.weight_id}
          onClose={() => setBenchmarkTarget(null)}
        />
      )}

      {exportTarget && (
        <ExportWeightPanel
          weightId={exportTarget.weight_id}
          onClose={() => setExportTarget(null)}
        />
      )}
    </div>
  );
}
