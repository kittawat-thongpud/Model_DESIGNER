/**
 * WeightEditorPage — Full-page weight editor with mapping canvas and lineage.
 *
 * Layout:
 *  - Left: Source weight selector + import
 *  - Center: WeightMappingPanel (dual-column mapping canvas)
 *  - Right: Lineage timeline + edit history
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import {
  ArrowLeft, GitBranch, Check, Loader2,
  Clock, Zap, Database, Box,
} from 'lucide-react';
import { api } from '../services/api';
import WeightMappingPanel, {
  type GroupMapping,
  type MappingPanelHandle,
  type AnnotatedWeightGroup,
} from '../components/WeightMappingPanel';
import type { WeightRecord, MappingKey } from '../types';
import { fmtSize, timeAgo } from '../utils/format';

interface Props {
  weightId: string;
  onBack: () => void;
  onOpenWeight?: (weightId: string) => void;
}

export default function WeightEditorPage({ weightId, onBack, onOpenWeight }: Props) {
  const [weight, setWeight] = useState<WeightRecord | null>(null);
  const [lineage, setLineage] = useState<WeightRecord[]>([]);
  const [annotatedGroups, setAnnotatedGroups] = useState<AnnotatedWeightGroup[] | undefined>(undefined);
  const [loading, setLoading] = useState(true);
  const [applying, setApplying] = useState(false);
  const [applyResult, setApplyResult] = useState<string | null>(null);
  const [showLineage, setShowLineage] = useState(true);

  const panelRef = useRef<HTMLDivElement>(null);
  const mappingsRef = useRef<GroupMapping[]>([]);

  // ── Load weight metadata + lineage ──
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.all([
      api.getWeight(weightId),
      api.getWeightLineage(weightId),
    ]).then(([w, lin]) => {
      if (cancelled) return;
      setWeight(w);
      setLineage(lin);
      // Fetch annotated groups if we have a model_id
      if (w.model_id) {
        api.getWeightGroupsAnnotated(weightId, w.model_id)
          .then((g) => { if (!cancelled) setAnnotatedGroups(g as AnnotatedWeightGroup[]); })
          .catch(() => {});
      }
    }).catch(() => {}).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [weightId]);

  const handleMappingsChange = useCallback((m: GroupMapping[]) => {
    mappingsRef.current = m;
    setApplyResult(null);
  }, []);

  // ── Apply mapping ──
  const handleApply = useCallback(async () => {
    const el = panelRef.current?.querySelector('[data-mapping-panel]');
    const handle = el ? (el as unknown as Record<string, MappingPanelHandle>).__mappingState : null;

    const mappings = handle?.getMappings() ?? mappingsRef.current;
    const sourceWeightId = handle?.getSourceWeightId() ?? '';
    const frozenGroups = handle?.getFrozenGroups() ?? new Set<string>();

    if (!sourceWeightId || mappings.length === 0) return;

    const allKeys: MappingKey[] = [];
    const freezeNodeIds: string[] = [];
    for (const m of mappings) {
      for (const k of m.keys) {
        if (k.src_key && k.tgt_key && k.matched) allKeys.push(k);
      }
      if (frozenGroups.has(m.tgtPrefix)) freezeNodeIds.push(m.tgtPrefix);
    }

    setApplying(true);
    setApplyResult(null);
    try {
      const result = await api.applyWeightMap(weightId, sourceWeightId, allKeys, freezeNodeIds);
      setApplyResult(`Applied ${result.applied} keys · ${freezeNodeIds.length} frozen`);
      // Refresh lineage
      api.getWeightLineage(weightId).then(setLineage).catch(() => {});
    } catch (e) {
      setApplyResult(`Error: ${(e as Error).message}`);
    } finally {
      setApplying(false);
    }
  }, [weightId]);

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Loader2 size={24} className="animate-spin text-slate-500" />
      </div>
    );
  }

  if (!weight) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-slate-500 gap-4">
        <p>Weight not found</p>
        <button onClick={onBack} className="text-indigo-400 hover:text-indigo-300 text-sm cursor-pointer">
          ← Back to Weights
        </button>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
      {/* ── Top Header ── */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-slate-800 bg-slate-950 shrink-0">
        <div className="flex items-center gap-4">
          <button onClick={onBack} className="text-slate-500 hover:text-white p-1 cursor-pointer">
            <ArrowLeft size={18} />
          </button>
          <div>
            <h1 className="text-white font-semibold text-sm flex items-center gap-2">
              <span className="text-lg">✏️</span>
              Weight Editor
              <span className="text-slate-500 font-mono text-xs">{weightId.slice(0, 8)}</span>
            </h1>
            <p className="text-[10px] text-slate-500">
              {weight.model_name} · {fmtSize(weight.file_size_bytes)} · {timeAgo(weight.created_at)}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {/* Toggle lineage */}
          <button
            onClick={() => setShowLineage((p) => !p)}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg border transition-colors cursor-pointer ${
              showLineage
                ? 'bg-cyan-500/10 border-cyan-500/30 text-cyan-400'
                : 'bg-slate-800 border-slate-700 text-slate-400 hover:text-white'
            }`}
          >
            <GitBranch size={12} />
            Lineage
          </button>

          {/* Apply */}
          <button
            onClick={handleApply}
            disabled={applying}
            className="flex items-center gap-1.5 px-4 py-1.5 text-xs bg-green-600 hover:bg-green-500 text-white rounded-lg transition-colors cursor-pointer disabled:opacity-40"
          >
            {applying ? <Loader2 size={12} className="animate-spin" /> : <Check size={14} />}
            Apply & Save
          </button>
        </div>
      </div>

      {/* ── Apply result banner ── */}
      {applyResult && (
        <div className={`px-6 py-2 text-xs shrink-0 ${
          applyResult.startsWith('Error') ? 'bg-red-500/10 text-red-400' : 'bg-green-500/10 text-green-400'
        }`}>
          {applyResult}
        </div>
      )}

      {/* ── Main Content ── */}
      <div className="flex-1 flex min-h-0 overflow-hidden">
        {/* Center: Mapping Panel */}
        <div ref={panelRef} className="flex-1 min-h-0 min-w-0">
          <WeightMappingPanel
            targetWeightId={weightId}
            targetGroupsOverride={annotatedGroups}
            targetModelId={weight?.model_id}
            showFreezeControls={true}
            showImport={true}
            onMappingsChange={handleMappingsChange}
          />
        </div>

        {/* Right: Lineage Sidebar */}
        {showLineage && (
          <div className="w-72 border-l border-slate-800 bg-slate-900/50 flex flex-col min-h-0 shrink-0">
            <div className="px-4 py-3 border-b border-slate-800 shrink-0">
              <h3 className="text-xs font-semibold text-white flex items-center gap-2">
                <GitBranch size={14} className="text-cyan-400" />
                Training Lineage
              </h3>
              <p className="text-[9px] text-slate-600 mt-0.5">
                {lineage.length} run{lineage.length !== 1 ? 's' : ''} in history
              </p>
            </div>

            <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
              {lineage.length === 0 ? (
                <p className="text-[10px] text-slate-600 text-center py-4">No lineage data</p>
              ) : (
                lineage.map((entry, i) => {
                  const isCurrent = entry.weight_id === weightId;
                  const runs = entry.training_runs || [];
                  const lastRun = runs[runs.length - 1];
                  return (
                    <div
                      key={entry.weight_id}
                      className={`relative pl-5 ${i < lineage.length - 1 ? 'pb-3' : ''}`}
                    >
                      {/* Timeline line */}
                      {i < lineage.length - 1 && (
                        <div className="absolute left-[7px] top-3 bottom-0 w-px bg-slate-700" />
                      )}
                      {/* Timeline dot */}
                      <div className={`absolute left-0 top-1 w-[15px] h-[15px] rounded-full border-2 ${
                        isCurrent
                          ? 'border-cyan-400 bg-cyan-400/20'
                          : 'border-slate-600 bg-slate-800'
                      }`} />

                      <div
                        className={`rounded-lg border p-2.5 transition-colors ${
                          isCurrent
                            ? 'border-cyan-500/30 bg-cyan-500/5'
                            : 'border-slate-800 bg-slate-900/50 hover:border-slate-700 cursor-pointer'
                        }`}
                        onClick={() => !isCurrent && onOpenWeight?.(entry.weight_id)}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-[10px] font-mono text-slate-300">
                            {entry.weight_id.slice(0, 8)}
                          </span>
                          {isCurrent && (
                            <span className="text-[8px] bg-cyan-500/20 text-cyan-400 px-1.5 py-0.5 rounded-full">
                              CURRENT
                            </span>
                          )}
                        </div>
                        <div className="space-y-0.5 text-[9px]">
                          {lastRun && (
                            <>
                              <div className="flex items-center gap-1 text-slate-500">
                                <Database size={8} />
                                <span>{lastRun.dataset || '—'}</span>
                              </div>
                              <div className="flex items-center gap-1 text-slate-500">
                                <Zap size={8} />
                                <span>{lastRun.epochs || 0} epochs</span>
                                {lastRun.accuracy != null && (
                                  <span className="text-emerald-400 ml-1">
                                    {lastRun.accuracy.toFixed(1)}%
                                  </span>
                                )}
                              </div>
                            </>
                          )}
                          <div className="flex items-center gap-1 text-slate-600">
                            <Clock size={8} />
                            <span>{timeAgo(entry.created_at)}</span>
                          </div>
                          {entry.parent_weight_id && (
                            <div className="flex items-center gap-1 text-slate-600">
                              <Box size={8} />
                              <span>from {entry.parent_weight_id.slice(0, 8)}</span>
                            </div>
                          )}
                        </div>

                        {/* Edit history entries */}
                        {(entry as WeightRecord & { edits?: EditEntry[] }).edits?.map((edit, ei) => (
                          <div key={ei} className="mt-1.5 pt-1.5 border-t border-slate-800/50">
                            <div className="flex items-center gap-1 text-[9px]">
                              <span className="text-amber-400">✏</span>
                              <span className="text-slate-400">
                                {edit.edit_type === 'transfer' ? 'Weight Transfer' : edit.edit_type}
                              </span>
                              {edit.mapping_count != null && (
                                <span className="text-slate-600">· {edit.mapping_count} keys</span>
                              )}
                            </div>
                            {edit.frozen_nodes && edit.frozen_nodes.length > 0 && (
                              <div className="text-[8px] text-cyan-400/60 mt-0.5">
                                Frozen: {edit.frozen_nodes.join(', ')}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// Internal type for edit history entries in weight metadata
interface EditEntry {
  edit_type: string;
  source_weight_id?: string;
  mapping_count?: number;
  frozen_nodes?: string[];
  created_at?: string;
}
