/**
 * WeightMappingPanel — Reusable dual-column weight mapping core.
 *
 * Displays source weight groups (left) ↔ target model weight groups (right)
 * with SVG connection lines. Used by both WeightMapCanvas (modal) and
 * WeightEditorPage (full page).
 *
 * Features:
 *  - Import .pt files via plugin system
 *  - Auto-map via shape+suffix matching
 *  - Drag source groups onto target groups for manual mapping
 *  - Click connection lines to delete mappings
 *  - Per-group freeze toggles
 *  - Freeze-all toggle
 *  - Node label display (when provided via annotations)
 */
import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  Upload, Wand2, Trash2, Check, Loader2,
  Link2, Unlink, Snowflake, AlertTriangle, Database,
  ChevronRight, ChevronDown, Box, ShieldAlert, Eye, XCircle, Download, Globe,
} from 'lucide-react';
import { api } from '../services/api';
import type {
  WeightGroup, MappingKey,
  ImportWeightResult, WeightSourceInfo, WeightRecord,
  CompatCheckResult, PretrainedModelInfo,
} from '../types';
import WeightLayerDetailPanel from './WeightLayerDetailPanel';

// ─── Types ───────────────────────────────────────────────────────────────────

export interface GroupMapping {
  srcPrefix: string;
  tgtPrefix: string;
  keys: MappingKey[];
  status: 'matched' | 'shape_mismatch' | 'manual';
}

export interface AnnotatedWeightGroup extends WeightGroup {
  node_label?: string;
  is_submodel?: boolean;
  children?: {
    prefix: string;
    display_prefix: string;
    module_type: string;
    param_count: number;
    keys: { key: string; shape: number[]; dtype: string }[];
  }[];
}

export interface MappingPanelProps {
  /** Target weight ID */
  targetWeightId: string;
  /** Optional initial source weight ID */
  initialSourceWeightId?: string;
  /** Optional annotated target groups (with node labels) — overrides API fetch */
  targetGroupsOverride?: AnnotatedWeightGroup[];
  /** Called when mappings change */
  onMappingsChange?: (mappings: GroupMapping[]) => void;
  /** Show freeze controls (training mode) */
  showFreezeControls?: boolean;
  /** Show import button */
  showImport?: boolean;
  /** Compact mode (no toolbar stats) */
  compact?: boolean;
  /** Target model_id — used to highlight same-model weights in picker */
  targetModelId?: string;
}

export interface MappingPanelHandle {
  getMappings: () => GroupMapping[];
  getSourceWeightId: () => string;
  getFreezeMatched: () => boolean;
  getFrozenGroups: () => Set<string>;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function shapeStr(shape: number[] | null): string {
  if (!shape) return '—';
  return `[${shape.join('×')}]`;
}

function statusColor(status: string): string {
  switch (status) {
    case 'matched': return '#22c55e';
    case 'shape_mismatch': return '#eab308';
    case 'manual': return '#6366f1';
    default: return '#64748b';
  }
}

function statusIcon(status: string) {
  switch (status) {
    case 'matched': return <Check size={10} className="text-green-400" />;
    case 'shape_mismatch': return <AlertTriangle size={10} className="text-yellow-400" />;
    case 'manual': return <Link2 size={10} className="text-indigo-400" />;
    default: return null;
  }
}

// ─── Component ───────────────────────────────────────────────────────────────

export default function WeightMappingPanel({
  targetWeightId,
  initialSourceWeightId,
  targetGroupsOverride,
  onMappingsChange,
  showFreezeControls = true,
  showImport = true,
  compact = false,
  targetModelId,
}: MappingPanelProps) {
  // ── State ──
  const [sourceWeightId, setSourceWeightId] = useState<string>(initialSourceWeightId || '');
  const [sourceGroups, setSourceGroups] = useState<WeightGroup[]>([]);
  const [targetGroups, setTargetGroups] = useState<AnnotatedWeightGroup[]>([]);
  const [mappings, setMappings] = useState<GroupMapping[]>([]);
  const [freezeMatched, setFreezeMatched] = useState(true);
  const [frozenGroups, setFrozenGroups] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sources, setSources] = useState<WeightSourceInfo[]>([]);
  const [importResult, setImportResult] = useState<ImportWeightResult | null>(null);

  // System weights for "Select from System" picker
  const [systemWeights, setSystemWeights] = useState<WeightRecord[]>([]);
  const [showSystemPicker, setShowSystemPicker] = useState(false);
  const [sameModelOnly, setSameModelOnly] = useState(true);

  // Pretrained model catalog
  const [pretrained, setPretrained] = useState<PretrainedModelInfo[]>([]);
  const [showPretrainedPicker, setShowPretrainedPicker] = useState(false);
  const [downloading, setDownloading] = useState<string | null>(null);

  // Compatibility check dialog
  const [compatResult, setCompatResult] = useState<CompatCheckResult | null>(null);
  const [compatLoading, setCompatLoading] = useState(false);
  const [showCompatDialog, setShowCompatDialog] = useState(false);
  // Pending mapping from drag-drop (held until user confirms compat dialog)
  const [pendingMapping, setPendingMapping] = useState<{ srcPrefix: string; tgtPrefix: string; keys: MappingKey[]; status: 'manual' } | null>(null);

  // Layer detail panel
  const [layerDetailKey, setLayerDetailKey] = useState<{ weightId: string; key: string } | null>(null);

  // Expanded SubModel groups
  const [expandedSubs, setExpandedSubs] = useState<Set<string>>(new Set());
  const toggleExpand = useCallback((prefix: string) => {
    setExpandedSubs((prev) => {
      const next = new Set(prev);
      if (next.has(prefix)) next.delete(prefix); else next.add(prefix);
      return next;
    });
  }, []);

  // Expanded key lists (show all keys instead of top 3)
  const [expandedKeys, setExpandedKeys] = useState<Set<string>>(new Set());
  const toggleExpandKeys = useCallback((prefix: string) => {
    setExpandedKeys((prev) => {
      const next = new Set(prev);
      if (next.has(prefix)) next.delete(prefix); else next.add(prefix);
      return next;
    });
  }, []);

  // For drag-to-map
  const [dragSrcPrefix, setDragSrcPrefix] = useState<string | null>(null);

  // Refs for SVG line positions
  const containerRef = useRef<HTMLDivElement>(null);
  const srcRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const tgtRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const [lineKey, setLineKey] = useState(0);

  // ── Load weight sources + system weights + pretrained catalog on mount ──
  useEffect(() => {
    let cancelled = false;
    api.listWeightSources().then((s) => { if (!cancelled) setSources(s); }).catch(() => {});
    api.listWeights().then((list) => {
      if (!cancelled) setSystemWeights(list.filter((w) => w.weight_id !== targetWeightId));
    }).catch(() => {});
    api.listPretrained().then((p) => { if (!cancelled) setPretrained(p); }).catch(() => {});
    return () => { cancelled = true; };
  }, [targetWeightId]);

  // ── Load target groups ──
  useEffect(() => {
    if (targetGroupsOverride) {
      setTargetGroups(targetGroupsOverride);
      return;
    }
    if (!targetWeightId) return;
    let cancelled = false;
    api.getWeightGroups(targetWeightId)
      .then((g) => { if (!cancelled) setTargetGroups(g as AnnotatedWeightGroup[]); })
      .catch(() => { if (!cancelled) setError('Failed to load target weight groups'); });
    return () => { cancelled = true; };
  }, [targetWeightId, targetGroupsOverride]);

  // ── Load source groups when sourceWeightId changes ──
  useEffect(() => {
    if (!sourceWeightId) {
      setSourceGroups([]);
      setMappings([]);
      return;
    }
    let cancelled = false;
    api.getWeightGroups(sourceWeightId)
      .then((g) => {
        if (cancelled) return;
        setSourceGroups(g);
        setMappings([]);
      })
      .catch(() => { if (!cancelled) setError('Failed to load source weight groups'); });
    return () => { cancelled = true; };
  }, [sourceWeightId]);

  // ── Recalc lines when mappings/groups change ──
  useEffect(() => {
    const timer = setTimeout(() => setLineKey((k) => k + 1), 50);
    return () => clearTimeout(timer);
  }, [mappings, sourceGroups, targetGroups]);

  // ── Notify parent of mapping changes ──
  useEffect(() => {
    onMappingsChange?.(mappings);
  }, [mappings, onMappingsChange]);

  // ── Sync frozenGroups when freezeMatched toggles ──
  useEffect(() => {
    if (freezeMatched) {
      setFrozenGroups(new Set(mappings.map((m) => m.tgtPrefix)));
    } else {
      setFrozenGroups(new Set());
    }
  }, [freezeMatched, mappings]);

  // ── Derived sets ──
  const mappedSrc = useMemo(() => new Set(mappings.map((m) => m.srcPrefix)), [mappings]);
  const mappedTgt = useMemo(() => new Set(mappings.map((m) => m.tgtPrefix)), [mappings]);

  // ── Download pretrained handler ──
  const handleDownloadPretrained = useCallback(async (modelKey: string) => {
    setDownloading(modelKey);
    setError(null);
    try {
      const result = await api.downloadPretrained(modelKey);
      // Refresh system weights + pretrained list
      const [weights, pts] = await Promise.all([api.listWeights(), api.listPretrained()]);
      setSystemWeights(weights.filter((w) => w.weight_id !== targetWeightId));
      setPretrained(pts);
      // Auto-select the downloaded weight as source
      setSourceWeightId(result.weight_id);
      setMappings([]);
      setShowPretrainedPicker(false);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setDownloading(null);
    }
  }, [targetWeightId]);

  // ── Import handler ──
  const handleImport = useCallback(async (file: File) => {
    setImporting(true);
    setError(null);
    try {
      const result = await api.importWeight(file, file.name.replace(/\.[^.]+$/, ''));
      setImportResult(result);
      setSourceWeightId(result.weight_id);
      setSourceGroups(result.groups);
      setMappings([]);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setImporting(false);
    }
  }, []);

  // ── Auto-map handler ──
  const handleAutoMap = useCallback(async () => {
    if (!sourceWeightId || !targetWeightId) return;
    setLoading(true);
    setError(null);
    try {
      const result = await api.autoMapWeights(targetWeightId, sourceWeightId);
      const newMappings: GroupMapping[] = [];
      for (const entry of result.mapping) {
        if (entry.src_prefix && entry.tgt_prefix && entry.status !== 'unmatched') {
          newMappings.push({
            srcPrefix: entry.src_prefix,
            tgtPrefix: entry.tgt_prefix,
            keys: entry.keys,
            status: entry.status === 'matched' ? 'matched' : 'shape_mismatch',
          });
        }
      }
      setMappings(newMappings);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, [sourceWeightId, targetWeightId]);

  // ── Remove mapping ──
  const removeMapping = useCallback((tgtPrefix: string) => {
    setMappings((prev) => prev.filter((m) => m.tgtPrefix !== tgtPrefix));
    setFrozenGroups((prev) => { const next = new Set(prev); next.delete(tgtPrefix); return next; });
  }, []);

  // ── Toggle freeze for individual group ──
  const toggleFreezeGroup = useCallback((prefix: string) => {
    setFrozenGroups((prev) => {
      const next = new Set(prev);
      if (next.has(prefix)) next.delete(prefix);
      else next.add(prefix);
      return next;
    });
  }, []);

  // ── Manual drag handlers ──
  const handleDragStart = useCallback((prefix: string) => {
    setDragSrcPrefix(prefix);
  }, []);

  // Build key-level mapping between two groups
  const buildKeyMapping = useCallback((srcPrefix: string, tgtPrefix: string): MappingKey[] => {
    const srcGroup = sourceGroups.find((g) => g.prefix === srcPrefix);
    const tgtGroup = targetGroups.find((g) => g.prefix === tgtPrefix);
    const keys: MappingKey[] = [];
    if (srcGroup && tgtGroup) {
      const srcByS: Record<string, typeof srcGroup.keys[0]> = {};
      for (const k of srcGroup.keys) {
        const suf = k.key.includes('.') ? k.key.split('.').slice(1).join('.') : k.key;
        srcByS[suf] = k;
      }
      for (const tk of tgtGroup.keys) {
        const suf = tk.key.includes('.') ? tk.key.split('.').slice(1).join('.') : tk.key;
        const sk = srcByS[suf];
        if (sk) {
          const matched = JSON.stringify(sk.shape) === JSON.stringify(tk.shape);
          keys.push({ src_key: sk.key, tgt_key: tk.key, src_shape: sk.shape, tgt_shape: tk.shape, matched });
        } else {
          keys.push({ src_key: null, tgt_key: tk.key, src_shape: null, tgt_shape: tk.shape, matched: false });
        }
      }
    }
    return keys;
  }, [sourceGroups, targetGroups]);

  // Apply a pending or immediate mapping
  const applyMapping = useCallback((srcPrefix: string, tgtPrefix: string, keys: MappingKey[]) => {
    setMappings((prev) => {
      const without = prev.filter(
        (m) => m.srcPrefix !== srcPrefix && m.tgtPrefix !== tgtPrefix,
      );
      return [...without, { srcPrefix, tgtPrefix, keys, status: 'manual' as const }];
    });
    if (freezeMatched) {
      setFrozenGroups((prev) => new Set([...prev, tgtPrefix]));
    }
  }, [freezeMatched]);

  // Confirm pending mapping from compat dialog
  const confirmMapping = useCallback(() => {
    if (pendingMapping) {
      applyMapping(pendingMapping.srcPrefix, pendingMapping.tgtPrefix, pendingMapping.keys);
    }
    setPendingMapping(null);
    setShowCompatDialog(false);
    setCompatResult(null);
  }, [pendingMapping, applyMapping]);

  // Cancel pending mapping
  const cancelMapping = useCallback(() => {
    setPendingMapping(null);
    setShowCompatDialog(false);
    setCompatResult(null);
  }, []);

  const handleDrop = useCallback((tgtPrefix: string) => {
    if (!dragSrcPrefix) return;
    const keys = buildKeyMapping(dragSrcPrefix, tgtPrefix);
    const hasIssues = keys.some((k) => !k.matched);

    if (hasIssues && sourceWeightId && targetWeightId) {
      // Show compat dialog before applying
      const pending = { srcPrefix: dragSrcPrefix, tgtPrefix, keys, status: 'manual' as const };
      setPendingMapping(pending);
      setCompatLoading(true);
      setShowCompatDialog(true);
      api.compatCheck(targetWeightId, sourceWeightId, dragSrcPrefix, tgtPrefix)
        .then((result) => { setCompatResult(result); })
        .catch(() => { /* still show dialog with key-level info */ })
        .finally(() => setCompatLoading(false));
    } else {
      // All matched — apply directly
      applyMapping(dragSrcPrefix, tgtPrefix, keys);
    }
    setDragSrcPrefix(null);
  }, [dragSrcPrefix, sourceWeightId, targetWeightId, buildKeyMapping, applyMapping]);

  // ── SVG Lines ──
  const lines = useMemo(() => {
    if (!containerRef.current) return [];
    const cRect = containerRef.current.getBoundingClientRect();
    return mappings.map((m) => {
      const srcEl = srcRefs.current[m.srcPrefix];
      const tgtEl = tgtRefs.current[m.tgtPrefix];
      if (!srcEl || !tgtEl) return null;
      const sRect = srcEl.getBoundingClientRect();
      const tRect = tgtEl.getBoundingClientRect();
      return {
        x1: sRect.right - cRect.left,
        y1: sRect.top + sRect.height / 2 - cRect.top,
        x2: tRect.left - cRect.left,
        y2: tRect.top + tRect.height / 2 - cRect.top,
        color: statusColor(m.status),
        tgtPrefix: m.tgtPrefix,
      };
    }).filter(Boolean) as { x1: number; y1: number; x2: number; y2: number; color: string; tgtPrefix: string }[];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mappings, lineKey]);

  // ── Stats ──
  const matchedCount = mappings.filter((m) => m.status === 'matched').length;
  const totalMapped = mappings.length;

  // ── File input ref ──
  const fileRef = useRef<HTMLInputElement>(null);

  // ── Public getters (exposed via data attributes for parent to read) ──
  const panelRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (panelRef.current) {
      (panelRef.current as unknown as Record<string, unknown>).__mappingState = {
        getMappings: () => mappings,
        getSourceWeightId: () => sourceWeightId,
        getFreezeMatched: () => freezeMatched,
        getFrozenGroups: () => frozenGroups,
      };
    }
  }, [mappings, sourceWeightId, freezeMatched, frozenGroups]);

  return (
    <div ref={panelRef} className="flex flex-col h-full" data-mapping-panel>
      {/* ── Toolbar ── */}
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-slate-800 shrink-0 bg-slate-900/50">
        {/* Source selection */}
        {showImport && (
          <>
            {/* System weight picker */}
            <div className="relative">
              <button
                onClick={() => setShowSystemPicker((p) => !p)}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-slate-800 border border-slate-700 rounded-lg text-slate-300 hover:text-white hover:bg-slate-700 transition-colors cursor-pointer"
              >
                <Database size={12} />
                System Weight
              </button>
              {showSystemPicker && (
                <div className="absolute top-full left-0 mt-1 w-80 bg-slate-900 border border-slate-700 rounded-lg shadow-2xl z-30 flex flex-col max-h-80">
                  {/* Filter toolbar */}
                  {targetModelId && (
                    <div className="flex items-center gap-2 px-3 py-2 border-b border-slate-800 shrink-0">
                      <button
                        onClick={() => setSameModelOnly((p) => !p)}
                        className={`flex items-center gap-1.5 px-2 py-1 rounded text-[10px] font-medium transition-colors cursor-pointer ${
                          sameModelOnly
                            ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/30'
                            : 'bg-slate-800 text-slate-500 border border-slate-700 hover:text-white'
                        }`}
                      >
                        Same model only
                      </button>
                      <span className="text-[9px] text-slate-600">
                        {(sameModelOnly
                          ? systemWeights.filter((w) => w.model_id === targetModelId)
                          : systemWeights
                        ).length} weights
                      </span>
                    </div>
                  )}
                  <div className="overflow-y-auto flex-1">
                    {(() => {
                      const filtered = (sameModelOnly && targetModelId)
                        ? systemWeights.filter((w) => w.model_id === targetModelId)
                        : systemWeights;
                      if (filtered.length === 0) return (
                        <div className="px-3 py-4 text-[10px] text-slate-600 text-center">
                          {sameModelOnly && targetModelId
                            ? 'No weights from the same model. Toggle filter to see all.'
                            : 'No weights available'}
                        </div>
                      );
                      return filtered.map((w) => {
                        const isMatch = targetModelId && w.model_id === targetModelId;
                        return (
                          <button
                            key={w.weight_id}
                            onClick={() => {
                              setSourceWeightId(w.weight_id);
                              setImportResult(null);
                              setMappings([]);
                              setShowSystemPicker(false);
                            }}
                            className={`w-full text-left px-3 py-2 hover:bg-slate-800 transition-colors cursor-pointer border-b border-slate-800/50 last:border-0 ${
                              sourceWeightId === w.weight_id ? 'bg-indigo-500/10' : ''
                            }`}
                          >
                            <div className="flex items-center justify-between gap-2">
                              <span className="text-xs text-white font-medium truncate">{w.model_name}</span>
                              <div className="flex items-center gap-1 shrink-0">
                                {isMatch && (
                                  <span className="text-[8px] bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 rounded px-1 py-0.5">same model</span>
                                )}
                                <span className="text-[9px] text-slate-500 font-mono">{w.weight_id.slice(0, 8)}</span>
                              </div>
                            </div>
                            <div className="text-[9px] text-slate-500 mt-0.5">
                              {w.dataset} · {w.epochs_trained} epochs
                              {w.final_accuracy != null && <span className="text-emerald-400"> · {w.final_accuracy.toFixed(1)}%</span>}
                            </div>
                          </button>
                        );
                      });
                    })()}
                  </div>
                </div>
              )}
            </div>

            {/* Pretrained model picker */}
            {pretrained.length > 0 && (
              <div className="relative">
                <button
                  onClick={() => { setShowPretrainedPicker((p) => !p); setShowSystemPicker(false); }}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-slate-800 border border-slate-700 rounded-lg text-slate-300 hover:text-white hover:bg-slate-700 transition-colors cursor-pointer"
                >
                  <Globe size={12} />
                  Pretrained
                </button>
                {showPretrainedPicker && (
                  <div className="absolute top-full left-0 mt-1 w-80 bg-slate-900 border border-slate-700 rounded-lg shadow-2xl z-30 max-h-72 overflow-y-auto">
                    <div className="px-3 py-2 border-b border-slate-800 sticky top-0 bg-slate-900">
                      <span className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider">Pretrained Models</span>
                    </div>
                    {pretrained.map((m) => (
                      <div
                        key={m.model_key}
                        className="flex items-center justify-between px-3 py-2 hover:bg-slate-800 transition-colors border-b border-slate-800/50 last:border-0"
                      >
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-1.5">
                            <span className="text-xs text-white font-medium truncate">{m.display_name}</span>
                            <span className={`text-[9px] px-1 py-0.5 rounded ${
                              m.task === 'detection' ? 'bg-orange-500/20 text-orange-400' :
                              m.task === 'classification' ? 'bg-blue-500/20 text-blue-400' :
                              'bg-purple-500/20 text-purple-400'
                            }`}>{m.task}</span>
                          </div>
                          <div className="text-[9px] text-slate-500 mt-0.5">{m.description}</div>
                        </div>
                        {m.downloaded ? (
                          <span className="text-[9px] text-emerald-400 flex items-center gap-1 ml-2 shrink-0">
                            <Check size={10} /> Ready
                          </span>
                        ) : (
                          <button
                            onClick={() => handleDownloadPretrained(m.model_key)}
                            disabled={downloading !== null}
                            className="ml-2 shrink-0 flex items-center gap-1 px-2 py-1 text-[10px] bg-indigo-600 hover:bg-indigo-500 text-white rounded transition-colors cursor-pointer disabled:opacity-40"
                          >
                            {downloading === m.model_key ? (
                              <Loader2 size={10} className="animate-spin" />
                            ) : (
                              <Download size={10} />
                            )}
                            {downloading === m.model_key ? 'Downloading...' : 'Download'}
                          </button>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Divider */}
            <div className="w-px h-5 bg-slate-700" />

            {/* Import .pt file */}
            <input
              ref={fileRef}
              type="file"
              accept=".pt,.pth,.bin"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleImport(f);
                e.target.value = '';
              }}
            />
            <button
              onClick={() => fileRef.current?.click()}
              disabled={importing}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-slate-800 border border-slate-700 rounded-lg text-slate-300 hover:text-white hover:bg-slate-700 transition-colors cursor-pointer disabled:opacity-50"
            >
              {importing ? <Loader2 size={12} className="animate-spin" /> : <Upload size={12} />}
              Import .pt
            </button>
          </>
        )}

        {/* Auto Map */}
        <button
          onClick={handleAutoMap}
          disabled={!sourceWeightId || !targetWeightId || loading}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-default"
        >
          {loading ? <Loader2 size={12} className="animate-spin" /> : <Wand2 size={12} />}
          Auto Map
        </button>

        {/* Clear All */}
        <button
          onClick={() => setMappings([])}
          disabled={mappings.length === 0}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-slate-800 border border-slate-700 rounded-lg text-slate-400 hover:text-red-400 hover:border-red-700 transition-colors cursor-pointer disabled:opacity-30"
        >
          <Trash2 size={12} />
          Clear
        </button>

        <div className="flex-1" />

        {/* Stats */}
        {!compact && (
          <div className="text-[10px] text-slate-500 space-x-3">
            <span>Source: <span className="text-slate-300">{sourceGroups.length}</span></span>
            <span>Target: <span className="text-slate-300">{targetGroups.length}</span></span>
            <span>Mapped: <span className="text-green-400">{matchedCount}</span>/{totalMapped}</span>
          </div>
        )}

        {/* Import result badge */}
        {importResult && (
          <span className="text-[10px] text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded">
            {importResult.source_display_name} · {importResult.key_count} keys
          </span>
        )}

        {/* Freeze all toggle */}
        {showFreezeControls && (
          <label className="flex items-center gap-1.5 text-[11px] text-slate-400 cursor-pointer select-none">
            <Snowflake size={12} className={freezeMatched ? 'text-cyan-400' : 'text-slate-600'} />
            <input
              type="checkbox"
              checked={freezeMatched}
              onChange={(e) => setFreezeMatched(e.target.checked)}
              className="accent-cyan-500"
            />
            Freeze all
          </label>
        )}
      </div>

      {/* ── Error ── */}
      {error && (
        <div className="mx-4 mt-2 px-3 py-2 bg-red-500/10 border border-red-500/30 rounded-lg text-xs text-red-400">
          {error}
          <button onClick={() => setError(null)} className="ml-2 text-red-500 hover:text-red-300 cursor-pointer">✕</button>
        </div>
      )}

      {/* ── Main Canvas ── */}
      <div ref={containerRef} className="flex-1 flex relative overflow-hidden min-h-0">
        {/* SVG overlay */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none z-10">
          {lines.map((line, i) => {
            const midX = (line.x1 + line.x2) / 2;
            return (
              <g key={`line-${i}`}>
                <path
                  d={`M ${line.x1} ${line.y1} C ${midX} ${line.y1}, ${midX} ${line.y2}, ${line.x2} ${line.y2}`}
                  stroke={line.color}
                  strokeWidth={2}
                  fill="none"
                  strokeDasharray={line.color === statusColor('manual') ? '4,4' : undefined}
                  opacity={0.7}
                  className="pointer-events-auto cursor-pointer hover:opacity-100"
                  onClick={() => removeMapping(line.tgtPrefix)}
                />
                <circle
                  cx={midX}
                  cy={(line.y1 + line.y2) / 2}
                  r={4}
                  fill={line.color}
                  opacity={0.5}
                  className="pointer-events-auto cursor-pointer"
                  onClick={() => removeMapping(line.tgtPrefix)}
                >
                  <title>Click to remove</title>
                </circle>
              </g>
            );
          })}
        </svg>

        {/* Source Column (Left) */}
        <div className="w-[42%] border-r border-slate-800 flex flex-col min-h-0">
          <div className="px-4 py-2 border-b border-slate-800 bg-slate-800/30 shrink-0">
            <div className="text-[10px] font-semibold text-amber-400 uppercase tracking-wider">
              ◀ Source Weight
            </div>
            {!sourceWeightId && (
              <p className="text-[10px] text-slate-600 mt-0.5">Select a system weight or import .pt</p>
            )}
          </div>
          <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1">
            {sourceGroups.map((g) => {
              const isMapped = mappedSrc.has(g.prefix);
              const keysExpanded = expandedKeys.has(`src:${g.prefix}`);
              const visibleKeys = keysExpanded ? g.keys : g.keys.slice(0, 3);
              const hiddenCount = g.keys.length - 3;
              return (
                <div
                  key={g.prefix}
                  ref={(el) => { srcRefs.current[g.prefix] = el; }}
                  draggable
                  onDragStart={() => handleDragStart(g.prefix)}
                  onDragEnd={() => setDragSrcPrefix(null)}
                  className={`px-3 py-2 rounded-lg border transition-all cursor-grab active:cursor-grabbing ${
                    isMapped
                      ? 'border-green-500/30 bg-green-500/5'
                      : dragSrcPrefix === g.prefix
                        ? 'border-indigo-500 bg-indigo-500/10'
                        : 'border-slate-700/50 bg-slate-800/30 hover:border-slate-600 hover:bg-slate-800/60'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="text-xs font-mono font-bold text-amber-300">{g.prefix}</span>
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-slate-700/50 text-slate-400">{g.module_type}</span>
                    </div>
                    <span className="text-[9px] text-slate-600 shrink-0">{g.param_count} params</span>
                  </div>
                  <div className="mt-1 space-y-0.5">
                    {visibleKeys.map((k) => (
                      <div
                        key={k.key}
                        className="text-[9px] text-slate-500 font-mono truncate flex items-center gap-1 group/key hover:text-amber-300 cursor-pointer"
                        onClick={(e) => { e.stopPropagation(); setLayerDetailKey({ weightId: sourceWeightId, key: k.key }); }}
                      >
                        <Eye size={8} className="opacity-0 group-hover/key:opacity-100 text-amber-400 shrink-0" />
                        <span className="truncate">{k.key}</span>
                        <span className="text-slate-600 shrink-0">{shapeStr(k.shape)}</span>
                      </div>
                    ))}
                    {hiddenCount > 0 && (
                      <button
                        className="text-[9px] text-amber-400/70 hover:text-amber-300 cursor-pointer hover:underline"
                        onClick={(e) => { e.stopPropagation(); toggleExpandKeys(`src:${g.prefix}`); }}
                      >
                        {keysExpanded ? '▲ collapse' : `▼ +${hiddenCount} more`}
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
            {sourceGroups.length === 0 && sourceWeightId && (
              <div className="text-xs text-slate-600 text-center py-8">No groups found</div>
            )}
          </div>
        </div>

        {/* Gap zone */}
        <div className="w-[16%] flex items-center justify-center relative">
          {dragSrcPrefix && (
            <div className="text-[10px] text-indigo-400 text-center animate-pulse">
              Drop on a target →
            </div>
          )}
          {!dragSrcPrefix && mappings.length === 0 && sourceGroups.length > 0 && (
            <div className="text-[10px] text-slate-600 text-center px-2">
              Drag layers or<br />click "Auto Map"
            </div>
          )}
        </div>

        {/* Target Column (Right) */}
        <div className="w-[42%] border-l border-slate-800 flex flex-col min-h-0">
          <div className="px-4 py-2 border-b border-slate-800 bg-slate-800/30 shrink-0">
            <div className="text-[10px] font-semibold text-cyan-400 uppercase tracking-wider">
              Target Model ▶
            </div>
          </div>
          <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1">
            {targetGroups.map((g) => {
              const mapping = mappings.find((m) => m.tgtPrefix === g.prefix);
              const isMapped = !!mapping;
              const isFrozen = frozenGroups.has(g.prefix);
              const isSubmodel = g.is_submodel && g.children && g.children.length > 0;
              const isExpanded = expandedSubs.has(g.prefix);
              return (
                <div key={g.prefix}>
                  {/* Main group row */}
                  <div
                    ref={(el) => { tgtRefs.current[g.prefix] = el; }}
                    onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'link'; }}
                    onDrop={(e) => { e.preventDefault(); handleDrop(g.prefix); }}
                    className={`px-3 py-2 rounded-lg border transition-all ${
                      isMapped
                        ? 'border-green-500/30 bg-green-500/5'
                        : dragSrcPrefix
                          ? 'border-dashed border-indigo-500/50 bg-indigo-500/5'
                          : 'border-slate-700/50 bg-slate-800/30'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 min-w-0">
                        {/* SubModel expand/collapse toggle */}
                        {isSubmodel && (
                          <button
                            onClick={() => toggleExpand(g.prefix)}
                            className="text-slate-400 hover:text-white cursor-pointer p-0.5 -ml-1"
                          >
                            {isExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                          </button>
                        )}
                        <span className="text-xs font-mono font-bold text-cyan-300">{g.prefix}</span>
                        {g.node_label && (
                          <span className="text-[9px] text-cyan-200/70 truncate max-w-[80px]" title={g.node_label}>
                            {g.node_label}
                          </span>
                        )}
                        {isSubmodel ? (
                          <span className="text-[9px] px-1.5 py-0.5 rounded bg-violet-500/20 text-violet-300 flex items-center gap-1">
                            <Box size={8} /> SubModel
                          </span>
                        ) : (
                          <span className="text-[9px] px-1.5 py-0.5 rounded bg-slate-700/50 text-slate-400">{g.module_type}</span>
                        )}
                        {mapping && (
                          <span className="flex items-center gap-0.5 text-[9px]">
                            {statusIcon(mapping.status)}
                            <span style={{ color: statusColor(mapping.status) }}>← {mapping.srcPrefix}</span>
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-1">
                        {showFreezeControls && isMapped && (
                          <button
                            onClick={() => toggleFreezeGroup(g.prefix)}
                            className={`p-0.5 cursor-pointer ${isFrozen ? 'text-cyan-400' : 'text-slate-600 hover:text-slate-400'}`}
                            title={isFrozen ? 'Unfreeze' : 'Freeze'}
                          >
                            <Snowflake size={10} />
                          </button>
                        )}
                        {mapping && (
                          <button
                            onClick={() => removeMapping(g.prefix)}
                            className="text-slate-600 hover:text-red-400 cursor-pointer p-0.5"
                            title="Remove mapping"
                          >
                            <Unlink size={10} />
                          </button>
                        )}
                        <span className="text-[9px] text-slate-600">{g.param_count}</span>
                      </div>
                    </div>
                    {/* Show keys only for non-submodel or collapsed submodel */}
                    {(!isSubmodel || !isExpanded) && (() => {
                      const tgtKeysExpanded = expandedKeys.has(`tgt:${g.prefix}`);
                      const tgtVisible = tgtKeysExpanded ? g.keys : g.keys.slice(0, 3);
                      const tgtHidden = g.keys.length - 3;
                      return (
                        <div className="mt-1 space-y-0.5">
                          {tgtVisible.map((k) => (
                            <div
                              key={k.key}
                              className="text-[9px] text-slate-500 font-mono truncate flex items-center gap-1 group/key hover:text-cyan-300 cursor-pointer"
                              onClick={(e) => { e.stopPropagation(); setLayerDetailKey({ weightId: targetWeightId, key: k.key }); }}
                            >
                              <Eye size={8} className="opacity-0 group-hover/key:opacity-100 text-cyan-400 shrink-0" />
                              <span className="truncate">{k.key}</span>
                              <span className="text-slate-600 shrink-0">{shapeStr(k.shape)}</span>
                            </div>
                          ))}
                          {tgtHidden > 0 && (
                            <button
                              className="text-[9px] text-cyan-400/70 hover:text-cyan-300 cursor-pointer hover:underline"
                              onClick={(e) => { e.stopPropagation(); toggleExpandKeys(`tgt:${g.prefix}`); }}
                            >
                              {tgtKeysExpanded ? '▲ collapse' : `▼ +${tgtHidden} more`}
                            </button>
                          )}
                        </div>
                      );
                    })()}
                  </div>
                  {/* SubModel children (expanded) */}
                  {isSubmodel && isExpanded && g.children!.map((child) => {
                    const childMapping = mappings.find((m) => m.tgtPrefix === child.prefix);
                    const childMapped = !!childMapping;
                    const childFrozen = frozenGroups.has(child.prefix);
                    return (
                      <div
                        key={child.prefix}
                        ref={(el) => { tgtRefs.current[child.prefix] = el; }}
                        onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'link'; }}
                        onDrop={(e) => { e.preventDefault(); handleDrop(child.prefix); }}
                        className={`ml-5 mt-1.5 px-3 py-2 rounded-lg border transition-all ${
                          childMapped
                            ? 'border-green-500/30 bg-green-500/5'
                            : dragSrcPrefix
                              ? 'border-dashed border-indigo-500/50 bg-indigo-500/5'
                              : 'border-slate-700/50 bg-slate-800/30'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2 min-w-0">
                            <span className="text-[10px] font-mono font-bold text-cyan-200">{child.display_prefix}</span>
                            {child.module_type && child.module_type !== 'unknown' && (
                              <span className="text-[9px] px-1.5 py-0.5 rounded bg-slate-700/50 text-slate-400">{child.module_type}</span>
                            )}
                            <span className="text-[9px] text-slate-600">{child.param_count}</span>
                            {childMapping && (
                              <span className="flex items-center gap-0.5 text-[9px]">
                                {statusIcon(childMapping.status)}
                                <span style={{ color: statusColor(childMapping.status) }}>← {childMapping.srcPrefix}</span>
                              </span>
                            )}
                          </div>
                          <div className="flex items-center gap-1">
                            {showFreezeControls && childMapped && (
                              <button
                                onClick={() => toggleFreezeGroup(child.prefix)}
                                className={`p-0.5 cursor-pointer ${childFrozen ? 'text-cyan-400' : 'text-slate-600 hover:text-slate-400'}`}
                              >
                                <Snowflake size={9} />
                              </button>
                            )}
                            {childMapping && (
                              <button
                                onClick={() => removeMapping(child.prefix)}
                                className="text-slate-600 hover:text-red-400 cursor-pointer p-0.5"
                              >
                                <Unlink size={9} />
                              </button>
                            )}
                          </div>
                        </div>
                        {(() => {
                          const childKeysExpanded = expandedKeys.has(`child:${child.prefix}`);
                          const childVisible = childKeysExpanded ? child.keys : child.keys.slice(0, 2);
                          const childHidden = child.keys.length - 2;
                          return (
                            <div className="mt-0.5 space-y-0.5">
                              {childVisible.map((k) => (
                                <div
                                  key={k.key}
                                  className="text-[8px] text-slate-600 font-mono truncate flex items-center gap-1 group/key hover:text-cyan-300 cursor-pointer"
                                  onClick={(e) => { e.stopPropagation(); setLayerDetailKey({ weightId: targetWeightId, key: k.key }); }}
                                >
                                  <Eye size={7} className="opacity-0 group-hover/key:opacity-100 text-cyan-400 shrink-0" />
                                  <span className="truncate">{k.key}</span>
                                  <span className="text-slate-700 shrink-0">{shapeStr(k.shape)}</span>
                                </div>
                              ))}
                              {childHidden > 0 && (
                                <button
                                  className="text-[8px] text-cyan-400/70 hover:text-cyan-300 cursor-pointer hover:underline"
                                  onClick={(e) => { e.stopPropagation(); toggleExpandKeys(`child:${child.prefix}`); }}
                                >
                                  {childKeysExpanded ? '▲ collapse' : `▼ +${childHidden} more`}
                                </button>
                              )}
                            </div>
                          );
                        })()}
                      </div>
                    );
                  })}
                </div>
              );
            })}
            {targetGroups.length === 0 && (
              <div className="text-xs text-slate-600 text-center py-8">No target groups</div>
            )}
          </div>
        </div>
      </div>

      {/* ── Bottom Status ── */}
      <div className="flex items-center justify-between px-4 py-2 border-t border-slate-800 shrink-0 text-[10px] text-slate-500">
        <span>
          {sources.length > 0 && `Formats: ${sources.map((s) => s.display_name).join(', ')}`}
        </span>
        <span>
          {totalMapped > 0 && (
            <>
              {matchedCount} matched / {totalMapped} total
              {showFreezeControls && ` · ${frozenGroups.size} frozen`}
            </>
          )}
        </span>
      </div>

      {/* ── Compatibility Warning Dialog ── */}
      {showCompatDialog && pendingMapping && (
        <div className="absolute inset-0 z-40 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-full max-w-lg mx-4 max-h-[80%] flex flex-col overflow-hidden"
            onClick={(e) => e.stopPropagation()}>
            {/* Header */}
            <div className="px-4 py-3 border-b border-slate-800 flex items-center gap-3 shrink-0">
              <ShieldAlert size={18} className={
                compatResult?.overall === 'incompatible' ? 'text-red-400'
                  : compatResult?.overall === 'partial' ? 'text-yellow-400'
                    : 'text-green-400'
              } />
              <div className="min-w-0 flex-1">
                <h3 className="text-sm font-semibold text-white">
                  Compatibility Check
                </h3>
                <div className="text-[10px] text-slate-500">
                  {pendingMapping.srcPrefix} → {pendingMapping.tgtPrefix}
                </div>
              </div>
              <button onClick={cancelMapping} className="p-1 text-slate-500 hover:text-white cursor-pointer">
                <XCircle size={16} />
              </button>
            </div>

            {/* Body */}
            <div className="flex-1 overflow-y-auto px-4 py-3 min-h-0">
              {compatLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 size={20} className="animate-spin text-indigo-400" />
                  <span className="ml-2 text-xs text-slate-400">Checking compatibility...</span>
                </div>
              ) : compatResult ? (
                <div className="space-y-3">
                  {/* Overall badge */}
                  <div className={`px-3 py-2 rounded-lg border text-xs ${
                    compatResult.overall === 'incompatible'
                      ? 'bg-red-500/10 border-red-500/30 text-red-300'
                      : compatResult.overall === 'partial'
                        ? 'bg-yellow-500/10 border-yellow-500/30 text-yellow-300'
                        : 'bg-green-500/10 border-green-500/30 text-green-300'
                  }`}>
                    {compatResult.summary}
                  </div>

                  {/* Stats row */}
                  <div className="flex gap-2 text-[10px]">
                    <span className="px-2 py-1 rounded bg-green-500/10 text-green-400">
                      {compatResult.ok_count} OK
                    </span>
                    {compatResult.error_count > 0 && (
                      <span className="px-2 py-1 rounded bg-red-500/10 text-red-400">
                        {compatResult.error_count} errors
                      </span>
                    )}
                    {compatResult.warning_count > 0 && (
                      <span className="px-2 py-1 rounded bg-yellow-500/10 text-yellow-400">
                        {compatResult.warning_count} warnings
                      </span>
                    )}
                  </div>

                  {/* Issues list */}
                  {compatResult.issues.length > 0 && (
                    <div className="space-y-1.5">
                      <div className="text-[10px] text-slate-400 font-semibold uppercase">Issues</div>
                      {compatResult.issues.map((issue, i) => (
                        <div key={i} className={`px-3 py-2 rounded-md border text-[10px] ${
                          issue.severity === 'error'
                            ? 'bg-red-500/5 border-red-500/20 text-red-300'
                            : issue.severity === 'warning'
                              ? 'bg-yellow-500/5 border-yellow-500/20 text-yellow-300'
                              : 'bg-slate-800/50 border-slate-700/30 text-slate-400'
                        }`}>
                          <span className="font-mono font-bold">{issue.suffix}</span>
                          <span className="ml-2">{issue.message}</span>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Key-level detail table */}
                  <div className="space-y-1">
                    <div className="text-[10px] text-slate-400 font-semibold uppercase">Parameter Details</div>
                    <div className="border border-slate-700/30 rounded-lg overflow-hidden">
                      <table className="w-full text-[9px]">
                        <thead>
                          <tr className="bg-slate-800/50">
                            <th className="text-left px-2 py-1.5 text-slate-500">Parameter</th>
                            <th className="text-left px-2 py-1.5 text-slate-500">Source Shape</th>
                            <th className="text-left px-2 py-1.5 text-slate-500">Target Shape</th>
                            <th className="text-center px-2 py-1.5 text-slate-500">Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {compatResult.keys.map((k, i) => (
                            <tr key={i} className="border-t border-slate-800/50">
                              <td className="px-2 py-1 font-mono text-slate-300">{k.suffix}</td>
                              <td className="px-2 py-1 font-mono text-amber-300">
                                {k.src_shape ? `[${k.src_shape.join('×')}]` : '—'}
                              </td>
                              <td className="px-2 py-1 font-mono text-cyan-300">
                                {k.tgt_shape ? `[${k.tgt_shape.join('×')}]` : '—'}
                              </td>
                              <td className="px-2 py-1 text-center">
                                {k.status === 'ok' ? (
                                  <Check size={10} className="inline text-green-400" />
                                ) : k.status === 'error' ? (
                                  <XCircle size={10} className="inline text-red-400" />
                                ) : (
                                  <AlertTriangle size={10} className="inline text-yellow-400" />
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              ) : (
                /* Fallback: show key-level info from pending mapping */
                <div className="space-y-2">
                  <div className="px-3 py-2 rounded-lg bg-yellow-500/10 border border-yellow-500/30 text-xs text-yellow-300">
                    Some parameters have mismatched shapes. Mismatched keys will be skipped during transfer.
                  </div>
                  {pendingMapping.keys.map((k, i) => (
                    <div key={i} className={`flex items-center justify-between px-2 py-1 rounded text-[9px] ${
                      k.matched ? 'text-green-400' : 'text-yellow-400'
                    }`}>
                      <span className="font-mono">{k.tgt_key || '—'}</span>
                      <span>{k.matched ? '✓' : '⚠ shape mismatch'}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="px-4 py-3 border-t border-slate-800 flex items-center justify-end gap-2 shrink-0">
              <button
                onClick={cancelMapping}
                className="px-3 py-1.5 text-xs text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors cursor-pointer"
              >
                Cancel
              </button>
              <button
                onClick={confirmMapping}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg transition-colors cursor-pointer ${
                  compatResult?.overall === 'incompatible'
                    ? 'bg-red-600/80 hover:bg-red-500 text-white'
                    : 'bg-indigo-600 hover:bg-indigo-500 text-white'
                }`}
              >
                {compatResult?.overall === 'incompatible' ? (
                  <>Apply Anyway (partial)</>
                ) : (
                  <>Apply Mapping</>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Layer Detail Modal ── */}
      {layerDetailKey && (
        <div className="absolute inset-0 z-30 flex items-center justify-center bg-black/50 backdrop-blur-sm"
          onClick={() => setLayerDetailKey(null)}>
          <div
            className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-full max-w-2xl mx-4 overflow-hidden"
            style={{ maxHeight: '85%' }}
            onClick={(e) => e.stopPropagation()}
          >
            <WeightLayerDetailPanel
              weightId={layerDetailKey.weightId}
              keyName={layerDetailKey.key}
              onClose={() => setLayerDetailKey(null)}
            />
          </div>
        </div>
      )}
    </div>
  );
}
