import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { usePersistedState } from '../hooks/usePersistedState';
import { api } from '../services/api';
import type { DatasetInfo, SplitResponse, SampleData, Annotation, PartitionSummary, PartitionEntry } from '../types';
import {
  ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight,
  Database, Loader2, Image as ImageIcon, X,
  ZoomIn, ZoomOut, RotateCcw, Tag,
  Settings2, Save, Check, Shuffle, Eye, EyeOff,
  Plus, Trash2, Scissors, Layers,
} from 'lucide-react';

interface Props {
  datasetName: string;
  onBack: () => void;
}

const MIN_PAGE = 12;
const MAX_PAGE = 200;
const TARGET_PAYLOAD_BYTES = 1.5 * 1024 * 1024; // ~1.5 MB target per page
const THUMB_SIZE = 192; // px — grid thumbnail size

/**
 * Compute optimal page_size based on image dimensions and viewport.
 * Small images (MNIST 28×28) → larger pages (up to 200).
 * Large images (COCO 640×640) → smaller pages (~20).
 */
function computePageSize(inputShape: number[]): number {
  const [channels, h, w] = inputShape.length === 3 ? inputShape : [1, 28, 28];
  // Estimate thumbnail payload after resize + encoding
  const thumbDim = Math.min(THUMB_SIZE, Math.max(w, h));
  const ratio = thumbDim / Math.max(w, h, 1);
  const thumbW = Math.round(w * ratio);
  const thumbH = Math.round(h * ratio);
  // JPEG compression ~0.15 for color, PNG ~0.40 for grayscale
  const compressionFactor = channels >= 3 ? 0.15 : 0.40;
  const estimatedBytes = thumbW * thumbH * channels * compressionFactor;
  // base64 overhead ~1.37×
  const payloadPerImage = estimatedBytes * 1.37;
  const count = Math.floor(TARGET_PAYLOAD_BYTES / Math.max(payloadPerImage, 1));
  return Math.max(MIN_PAGE, Math.min(MAX_PAGE, count));
}

/** Compute grid columns based on native image size */
function computeGridCols(inputShape: number[]): string {
  const maxDim = Math.max(inputShape[1] || 28, inputShape[2] || 28);
  if (maxDim <= 32) return 'grid-cols-8 sm:grid-cols-10 md:grid-cols-12 lg:grid-cols-16';
  if (maxDim <= 64) return 'grid-cols-6 sm:grid-cols-8 md:grid-cols-10 lg:grid-cols-12';
  if (maxDim <= 128) return 'grid-cols-4 sm:grid-cols-5 md:grid-cols-6 lg:grid-cols-8';
  return 'grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6';
}

type SplitInfo = SplitResponse;

// ── Split Editor ─────────────────────────────────────────────────────────────

function TransferSlider({ label, value, max, color, onChange }: {
  label: string; value: number; max: number; color: string; onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center gap-2 min-w-0">
      <span className="text-[11px] text-slate-500 whitespace-nowrap w-16 text-right">{label}</span>
      <input type="range" min={0} max={max} step={1} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className={`flex-1 h-1.5 cursor-pointer accent-${color}`}
        style={{ accentColor: color === 'amber' ? '#f59e0b' : color === 'emerald' ? '#059669' : '#4f46e5' }} />
      <span className="text-[11px] tabular-nums text-slate-300 w-8 text-right">{value}%</span>
    </div>
  );
}

const PRESETS: { label: string; t: number; v: number; te: number }[] = [
  { label: '80 / 10 / 10', t: 80, v: 10, te: 10 },
  { label: '70 / 15 / 15', t: 70, v: 15, te: 15 },
  { label: '60 / 20 / 20', t: 60, v: 20, te: 20 },
  { label: '90 / 5 / 5',   t: 90, v: 5,  te: 5 },
];

function computePresetTransfers(
  origTrain: number, origTest: number, origVal: number,
  targetTrainPct: number, targetValPct: number, targetTestPct: number,
): { t2v: number; t2te: number; te2tr: number; te2v: number; v2tr: number; v2te: number } {
  const total = origTrain + origTest + origVal;
  if (total === 0) return { t2v: 0, t2te: 0, te2tr: 0, te2v: 0, v2tr: 0, v2te: 0 };

  const tgtTr = Math.round(total * targetTrainPct / 100);
  const tgtV  = Math.round(total * targetValPct / 100);
  const tgtTe = Math.round(total * targetTestPct / 100);

  // Per-split delta: positive = needs more, negative = has surplus
  const dTr = tgtTr - origTrain;
  const dV  = tgtV  - origVal;
  const dTe = tgtTe - origTest;

  // 6 transfer counts (all >= 0)
  let t2v = 0, t2te = 0, te2tr = 0, te2v = 0, v2tr = 0, v2te = 0;

  // Train has surplus → give to val/test that need it
  if (dTr < 0) {
    const surplus = -dTr;
    if (dV > 0)  t2v  = Math.min(dV, surplus);
    if (dTe > 0) t2te = Math.min(dTe, surplus - t2v);
  }
  // Val has surplus → give to train/test that need it
  if (dV < 0) {
    const surplus = -dV;
    if (dTr > 0) v2tr = Math.min(dTr, surplus);
    if (dTe > 0) v2te = Math.min(dTe, surplus - v2tr);
  }
  // Test has surplus → give to train/val that need it
  if (dTe < 0) {
    const surplus = -dTe;
    if (dTr > 0) te2tr = Math.min(dTr - v2tr, surplus);
    if (dV > 0)  te2v  = Math.min(dV - t2v, surplus - te2tr);
  }

  // Convert counts → percentages of source
  return {
    t2v:   origTrain > 0 ? Math.round(t2v / origTrain * 100) : 0,
    t2te:  origTrain > 0 ? Math.round(t2te / origTrain * 100) : 0,
    te2tr: origTest > 0  ? Math.round(te2tr / origTest * 100) : 0,
    te2v:  origTest > 0  ? Math.round(te2v / origTest * 100) : 0,
    v2tr:  origVal > 0   ? Math.round(v2tr / origVal * 100) : 0,
    v2te:  origVal > 0   ? Math.round(v2te / origVal * 100) : 0,
  };
}

function SplitEditor({ datasetName, splitInfo, onSaved }: {
  datasetName: string;
  splitInfo: SplitInfo;
  onSaved: (s: SplitInfo) => void;
}) {
  const [t2v, setT2v] = useState(splitInfo.train_to_val);
  const [t2te, setT2te] = useState(splitInfo.train_to_test);
  const [te2tr, setTe2tr] = useState(splitInfo.test_to_train);
  const [te2v, setTe2v] = useState(splitInfo.test_to_val);
  const [v2tr, setV2tr] = useState(splitInfo.val_to_train);
  const [v2te, setV2te] = useState(splitInfo.val_to_test);
  const [seed, setSeed] = useState(splitInfo.seed);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  const origTrain = splitInfo.orig_train;
  const origTest = splitInfo.orig_test;
  const origVal = splitInfo.orig_val;

  const applyPreset = (tp: number, vp: number, tep: number) => {
    const p = computePresetTransfers(origTrain, origTest, origVal, tp, vp, tep);
    setT2v(p.t2v); setT2te(p.t2te); setTe2tr(p.te2tr); setTe2v(p.te2v); setV2tr(p.v2tr); setV2te(p.v2te);
  };

  // Compute effective counts
  const t2vN = Math.floor(origTrain * t2v / 100);
  const t2teN = Math.floor(origTrain * t2te / 100);
  const te2trN = Math.floor(origTest * te2tr / 100);
  const te2vN = Math.floor(origTest * te2v / 100);
  const v2trN = Math.floor(origVal * v2tr / 100);
  const v2teN = Math.floor(origVal * v2te / 100);

  const finalTrain = origTrain - t2vN - t2teN + te2trN + v2trN;
  const finalVal = origVal - v2trN - v2teN + t2vN + te2vN;
  const finalTest = origTest - te2trN - te2vN + t2teN + v2teN;
  const totalAll = finalTrain + finalVal + finalTest;

  const trainPct = totalAll > 0 ? (finalTrain / totalAll) * 100 : 0;
  const valPct = totalAll > 0 ? (finalVal / totalAll) * 100 : 0;
  const testPct = totalAll > 0 ? (finalTest / totalAll) * 100 : 0;

  const dirty = t2v !== splitInfo.train_to_val || t2te !== splitInfo.train_to_test
    || te2tr !== splitInfo.test_to_train || te2v !== splitInfo.test_to_val
    || v2tr !== splitInfo.val_to_train || v2te !== splitInfo.val_to_test
    || seed !== splitInfo.seed;

  const handleSave = async () => {
    setSaving(true);
    try {
      const res = await api.saveDatasetSplits(datasetName, {
        seed, train_to_val: t2v, train_to_test: t2te,
        test_to_train: te2tr, test_to_val: te2v,
        val_to_train: v2tr, val_to_test: v2te,
      });
      onSaved(res);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch { /* ignore */ }
    setSaving(false);
  };

  return (
    <div className="space-y-4">
      {/* Result bar */}
      <div className="space-y-2">
        <div className="flex rounded-lg overflow-hidden h-8 border border-slate-700">
          {trainPct > 0 && (
            <div className="bg-indigo-600 flex items-center justify-center text-[10px] text-white font-medium transition-all duration-300" style={{ width: `${trainPct}%` }}>
              {finalTrain.toLocaleString()}
            </div>
          )}
          {valPct > 0 && (
            <div className="bg-amber-500 flex items-center justify-center text-[10px] text-slate-900 font-medium transition-all duration-300" style={{ width: `${Math.max(valPct, 2)}%` }}>
              {finalVal.toLocaleString()}
            </div>
          )}
          {testPct > 0 && (
            <div className="bg-emerald-600 flex items-center justify-center text-[10px] text-white font-medium transition-all duration-300" style={{ width: `${testPct}%` }}>
              {finalTest.toLocaleString()}
            </div>
          )}
        </div>
        <div className="flex items-center gap-4 text-[11px]">
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-indigo-600" /> Train: {finalTrain.toLocaleString()} ({trainPct.toFixed(1)}%)</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-amber-500" /> Val: {finalVal.toLocaleString()} ({valPct.toFixed(1)}%)</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-emerald-600" /> Test: {finalTest.toLocaleString()} ({testPct.toFixed(1)}%)</span>
        </div>
      </div>

      {/* Presets */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-[10px] text-slate-500 uppercase tracking-wider">Presets</span>
        <button onClick={() => { setT2v(0); setT2te(0); setTe2tr(0); setTe2v(0); setV2tr(0); setV2te(0); }}
          className={`px-2.5 py-1 text-[11px] rounded-md border transition-colors cursor-pointer ${
            t2v === 0 && t2te === 0 && te2tr === 0 && te2v === 0 && v2tr === 0 && v2te === 0
              ? 'bg-indigo-600/20 text-indigo-400 border-indigo-500/40'
              : 'bg-slate-800/50 text-slate-400 border-slate-700 hover:border-slate-600'
          }`}>
          Original
        </button>
        {PRESETS.map((p) => {
          const cp = computePresetTransfers(origTrain, origTest, origVal, p.t, p.v, p.te);
          const active = t2v === cp.t2v && t2te === cp.t2te && te2tr === cp.te2tr && te2v === cp.te2v && v2tr === cp.v2tr && v2te === cp.v2te;
          return (
            <button key={p.label} onClick={() => applyPreset(p.t, p.v, p.te)}
              className={`px-2.5 py-1 text-[11px] rounded-md border transition-colors cursor-pointer ${
                active
                  ? 'bg-indigo-600/20 text-indigo-400 border-indigo-500/40'
                  : 'bg-slate-800/50 text-slate-400 border-slate-700 hover:border-slate-600'
              }`}>
              {p.label}
            </button>
          );
        })}
      </div>

      {/* Transfer controls */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* From Original Train */}
        {origTrain > 0 && (
          <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50 space-y-2">
            <div className="flex items-center gap-2 text-xs font-medium">
              <span className="w-2.5 h-2.5 rounded-sm bg-indigo-600" />
              <span className="text-white">Original Train</span>
              <span className="text-slate-500">({origTrain.toLocaleString()})</span>
            </div>
            <TransferSlider label="→ Val" value={t2v} max={100 - t2te} color="amber"
              onChange={(v) => setT2v(v)} />
            <TransferSlider label="→ Test" value={t2te} max={100 - t2v} color="emerald"
              onChange={(v) => setT2te(v)} />
            <div className="text-[10px] text-slate-500 pl-[72px]">Keeps {(100 - t2v - t2te)}% = {(origTrain - t2vN - t2teN).toLocaleString()}</div>
          </div>
        )}

        {/* From Original Val */}
        {origVal > 0 && (
          <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50 space-y-2">
            <div className="flex items-center gap-2 text-xs font-medium">
              <span className="w-2.5 h-2.5 rounded-sm bg-amber-500" />
              <span className="text-white">Original Val</span>
              <span className="text-slate-500">({origVal.toLocaleString()})</span>
            </div>
            <TransferSlider label="→ Train" value={v2tr} max={100 - v2te} color="indigo"
              onChange={(v) => setV2tr(v)} />
            <TransferSlider label="→ Test" value={v2te} max={100 - v2tr} color="emerald"
              onChange={(v) => setV2te(v)} />
            <div className="text-[10px] text-slate-500 pl-[72px]">Keeps {(100 - v2tr - v2te)}% = {(origVal - v2trN - v2teN).toLocaleString()}</div>
          </div>
        )}

        {/* From Original Test */}
        {origTest > 0 && (
          <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50 space-y-2">
            <div className="flex items-center gap-2 text-xs font-medium">
              <span className="w-2.5 h-2.5 rounded-sm bg-emerald-600" />
              <span className="text-white">Original Test</span>
              <span className="text-slate-500">({origTest.toLocaleString()})</span>
            </div>
            <TransferSlider label="→ Train" value={te2tr} max={100 - te2v} color="indigo"
              onChange={(v) => setTe2tr(v)} />
            <TransferSlider label="→ Val" value={te2v} max={100 - te2tr} color="amber"
              onChange={(v) => setTe2v(v)} />
            <div className="text-[10px] text-slate-500 pl-[72px]">Keeps {(100 - te2tr - te2v)}% = {(origTest - te2trN - te2vN).toLocaleString()}</div>
          </div>
        )}
      </div>

      {/* Seed + Save */}
      <div className="flex items-end gap-4">
        <div className="space-y-1.5">
          <label className="text-xs text-slate-400 flex items-center gap-1"><Shuffle size={11} /> Seed</label>
          <input type="number" min={0} value={seed}
            onChange={(e) => setSeed(Math.max(0, Number(e.target.value)))}
            className="w-24 bg-slate-800 border border-slate-700 rounded-md px-2.5 py-1.5 text-xs text-white tabular-nums focus:outline-none focus:border-indigo-500" />
        </div>
        <div className="flex-1" />
        <button
          onClick={handleSave}
          disabled={!dirty || saving}
          className={`flex items-center gap-1.5 px-4 py-1.5 text-xs rounded-lg transition-colors cursor-pointer border ${
            saved
              ? 'bg-emerald-600/20 text-emerald-400 border-emerald-600/40'
              : dirty
                ? 'bg-indigo-600 hover:bg-indigo-500 text-white border-indigo-500'
                : 'bg-slate-800 text-slate-500 border-slate-700 cursor-default'
          }`}
        >
          {saving ? <Loader2 size={12} className="animate-spin" /> : saved ? <Check size={12} /> : <Save size={12} />}
          {saved ? 'Saved' : 'Save'}
        </button>
      </div>
    </div>
  );
}

// ── Partition Editor ──────────────────────────────────────────────────────────

const PART_COLORS = [
  '#6366f1', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6',
  '#06b6d4', '#f97316', '#ec4899', '#14b8a6', '#84cc16',
];

function PartitionEditor({ datasetName, onUpdated }: { datasetName: string; onUpdated?: (ps: PartitionSummary) => void }) {
  const [summary, setSummary] = useState<PartitionSummary | null>(null);
  const [loading, setLoading] = useState(true);

  // Create form
  const [newName, setNewName] = useState('');
  const [newPct, setNewPct] = useState(10);
  const [creating, setCreating] = useState(false);

  // Split form
  const [splitTarget, setSplitTarget] = useState<PartitionEntry | null>(null);
  const [splitChildren, setSplitChildren] = useState<{ name: string; percent: number }[]>([]);

  useEffect(() => {
    setLoading(true);
    api.getDatasetPartitions(datasetName)
      .then(setSummary)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [datasetName]);

  const handleCreate = async () => {
    if (!newName.trim() || newPct < 1) return;
    setCreating(true);
    try {
      const res = await api.createPartition(datasetName, { name: newName.trim(), percent: newPct });
      setSummary(res);
      onUpdated?.(res);
      setNewName('');
      setNewPct(10);
    } catch { /* ignore */ }
    setCreating(false);
  };

  const handleDelete = async (id: string) => {
    try {
      const res = await api.deletePartition(datasetName, id);
      setSummary(res);
      onUpdated?.(res);
    } catch { /* ignore */ }
  };

  const handleMethodChange = async (method: string) => {
    try {
      const res = await api.updatePartitionMethod(datasetName, method);
      setSummary(res);
      onUpdated?.(res);
    } catch { /* ignore */ }
  };

  const openSplit = (entry: PartitionEntry) => {
    const half = Math.floor(entry.percent / 2);
    setSplitTarget(entry);
    setSplitChildren([
      { name: `${entry.name} A`, percent: half },
      { name: `${entry.name} B`, percent: entry.percent - half },
    ]);
  };

  const handleSplit = async () => {
    if (!splitTarget) return;
    const total = splitChildren.reduce((s, c) => s + c.percent, 0);
    if (total !== splitTarget.percent) return;
    try {
      const res = await api.splitPartition(datasetName, splitTarget.id, splitChildren);
      setSummary(res);
      onUpdated?.(res);
      setSplitTarget(null);
      setSplitChildren([]);
    } catch { /* ignore */ }
  };

  if (loading) return <div className="flex items-center gap-2 py-3 text-xs text-slate-500"><Loader2 size={12} className="animate-spin" /> Loading partitions...</div>;
  if (!summary) return null;

  const allEntries: PartitionEntry[] = [summary.master, ...summary.partitions];
  const masterPct = summary.master.percent;

  return (
    <div className="space-y-4 mt-6 pt-5 border-t border-slate-700/50">
      <div className="flex items-center gap-2 text-xs font-medium text-white">
        <Layers size={13} className="text-indigo-400" />
        Partitions
        <span className="text-slate-500 font-normal">({summary.partitions.length + 1} total)</span>
      </div>

      {/* Method selector */}
      {summary.available_methods && summary.available_methods.length > 1 && (
        <div className="flex items-center gap-3 bg-slate-800/30 rounded-lg px-3 py-2.5 border border-slate-700/30">
          <span className="text-[11px] text-slate-400 shrink-0">Partition method</span>
          <select
            value={summary.method}
            onChange={(e) => handleMethodChange(e.target.value)}
            className="flex-1 max-w-xs bg-slate-800 border border-slate-700 rounded-md px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500 cursor-pointer"
          >
            {summary.available_methods.map((m) => (
              <option key={m} value={m}>
                {{ random: 'Random Shuffle', stratified: 'Stratified', round_robin: 'Round-Robin Interleaving', iterative: 'Iterative Stratification' }[m] ?? m}
              </option>
            ))}
          </select>
          <span className="text-[10px] text-slate-500">
            {{ random: 'No class balancing', stratified: 'Proportional class distribution per partition', round_robin: 'Cycle samples class-by-class across partitions', iterative: 'Greedy balancing, best for imbalanced datasets' }[summary.method] ?? ''}
          </span>
        </div>
      )}

      {/* Visual bar */}
      <div className="flex rounded-lg overflow-hidden h-7 border border-slate-700">
        {allEntries.map((e, i) => {
          const pct = e.percent;
          if (pct <= 0) return null;
          const color = e.id === 'master' ? '#475569' : PART_COLORS[i % PART_COLORS.length];
          return (
            <div key={e.id} className="flex items-center justify-center text-[10px] text-white font-medium transition-all duration-300"
              style={{ width: `${Math.max(pct, 2)}%`, backgroundColor: color }}>
              {pct >= 5 ? `${e.name} ${pct}%` : pct >= 2 ? `${pct}%` : ''}
            </div>
          );
        })}
      </div>

      {/* Partition list */}
      <div className="space-y-1.5">
        {allEntries.map((e, i) => {
          const color = e.id === 'master' ? '#475569' : PART_COLORS[i % PART_COLORS.length];
          return (
            <div key={e.id} className="flex items-center gap-3 bg-slate-800/40 rounded-lg px-3 py-2 border border-slate-700/40 text-xs">
              <span className="w-2.5 h-2.5 rounded-sm shrink-0" style={{ backgroundColor: color }} />
              <span className="text-white font-medium min-w-[100px]">{e.name}</span>
              <span className="text-slate-400 tabular-nums">{e.percent}%</span>
              <span className="text-slate-500 tabular-nums">train:{e.train_count.toLocaleString()}</span>
              <span className="text-slate-500 tabular-nums">val:{e.val_count.toLocaleString()}</span>
              <span className="text-slate-500 tabular-nums">test:{e.test_count.toLocaleString()}</span>
              <div className="flex-1" />
              {e.percent >= 2 && (
                <button onClick={() => openSplit(e)} className="p-1 rounded hover:bg-slate-700 text-slate-500 hover:text-amber-400 transition-colors cursor-pointer" title="Split partition">
                  <Scissors size={12} />
                </button>
              )}
              {e.id !== 'master' && (
                <button onClick={() => handleDelete(e.id)} className="p-1 rounded hover:bg-slate-700 text-slate-500 hover:text-red-400 transition-colors cursor-pointer" title="Delete (return to master)">
                  <Trash2 size={12} />
                </button>
              )}
            </div>
          );
        })}
      </div>

      {/* Create from master */}
      {masterPct > 1 && (
        <div className="flex items-end gap-3 bg-slate-800/30 rounded-lg p-3 border border-slate-700/30">
          <div className="space-y-1">
            <label className="text-[10px] text-slate-500">Name</label>
            <input type="text" value={newName} onChange={(e) => setNewName(e.target.value)}
              placeholder="Partition name"
              className="w-40 bg-slate-800 border border-slate-700 rounded-md px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500" />
          </div>
          <div className="space-y-1">
            <label className="text-[10px] text-slate-500">Percent (max {masterPct - 1}%)</label>
            <input type="number" min={1} max={masterPct - 1} value={newPct}
              onChange={(e) => setNewPct(Math.max(1, Math.min(masterPct - 1, Number(e.target.value))))}
              className="w-20 bg-slate-800 border border-slate-700 rounded-md px-2.5 py-1.5 text-xs text-white tabular-nums focus:outline-none focus:border-indigo-500" />
          </div>
          <button onClick={handleCreate} disabled={creating || !newName.trim() || newPct < 1}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white border border-indigo-500 transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-default">
            {creating ? <Loader2 size={12} className="animate-spin" /> : <Plus size={12} />}
            Create from Master
          </button>
        </div>
      )}

      {/* Split dialog */}
      {splitTarget && (
        <div className="bg-slate-800/60 rounded-lg p-4 border border-amber-500/30 space-y-3">
          <div className="flex items-center gap-2 text-xs font-medium text-amber-400">
            <Scissors size={13} />
            Split "{splitTarget.name}" ({splitTarget.percent}%)
          </div>
          <div className="space-y-2">
            {splitChildren.map((child, ci) => (
              <div key={ci} className="flex items-center gap-2">
                <input type="text" value={child.name}
                  onChange={(e) => { const c = [...splitChildren]; c[ci] = { ...c[ci], name: e.target.value }; setSplitChildren(c); }}
                  className="flex-1 bg-slate-800 border border-slate-700 rounded-md px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-amber-500" />
                <input type="number" min={1} max={splitTarget.percent - 1} value={child.percent}
                  onChange={(e) => {
                    const c = [...splitChildren];
                    c[ci] = { ...c[ci], percent: Math.max(1, Number(e.target.value)) };
                    setSplitChildren(c);
                  }}
                  className="w-16 bg-slate-800 border border-slate-700 rounded-md px-2.5 py-1.5 text-xs text-white tabular-nums focus:outline-none focus:border-amber-500" />
                <span className="text-[10px] text-slate-500">%</span>
                {splitChildren.length > 2 && (
                  <button onClick={() => setSplitChildren(splitChildren.filter((_, j) => j !== ci))}
                    className="p-1 rounded hover:bg-slate-700 text-slate-500 hover:text-red-400 cursor-pointer">
                    <X size={12} />
                  </button>
                )}
              </div>
            ))}
          </div>
          <div className="flex items-center gap-2">
            <button onClick={() => setSplitChildren([...splitChildren, { name: `Part ${splitChildren.length + 1}`, percent: 1 }])}
              className="text-[11px] text-slate-400 hover:text-white cursor-pointer flex items-center gap-1">
              <Plus size={11} /> Add child
            </button>
            <div className="flex-1" />
            {(() => {
              const total = splitChildren.reduce((s, c) => s + c.percent, 0);
              const valid = total === splitTarget.percent && splitChildren.every(c => c.name.trim() && c.percent >= 1);
              return (
                <>
                  <span className={`text-[11px] tabular-nums ${total === splitTarget.percent ? 'text-emerald-400' : 'text-red-400'}`}>
                    {total}/{splitTarget.percent}%
                  </span>
                  <button onClick={() => { setSplitTarget(null); setSplitChildren([]); }}
                    className="px-3 py-1.5 text-xs rounded-lg bg-slate-700 text-slate-300 hover:bg-slate-600 border border-slate-600 cursor-pointer transition-colors">
                    Cancel
                  </button>
                  <button onClick={handleSplit} disabled={!valid}
                    className="px-3 py-1.5 text-xs rounded-lg bg-amber-600 hover:bg-amber-500 text-white border border-amber-500 cursor-pointer transition-colors disabled:opacity-40 disabled:cursor-default">
                    Split
                  </button>
                </>
              );
            })()}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Annotation color palette ─────────────────────────────────────────────────
const ANN_COLORS = [
  '#f43f5e', '#3b82f6', '#22c55e', '#eab308', '#a855f7',
  '#06b6d4', '#f97316', '#ec4899', '#14b8a6', '#8b5cf6',
];
function annColor(catId: number): string { return ANN_COLORS[catId % ANN_COLORS.length]; }

// ── Image cache ──────────────────────────────────────────────────────────────
const imageCache = new Map<string, string>();

function LazyImage({ sample, cacheKey, showAnnotations, onClick }: {
  sample: SampleData; cacheKey: string; showAnnotations?: boolean; onClick?: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const cached = imageCache.get(cacheKey);
  const src = `data:${sample.mime || 'image/png'};base64,${sample.image_base64}`;

  useEffect(() => {
    if (cached) { setVisible(true); setLoaded(true); return; }
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { setVisible(true); observer.disconnect(); } },
      { rootMargin: '200px' },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [cached]);

  useEffect(() => { if (visible && src) imageCache.set(cacheKey, src); }, [visible, src, cacheKey]);

  const imgSrc = cached || (visible ? src : undefined);

  // Scale annotations from original image coords to thumbnail display
  const anns = showAnnotations && sample.annotations ? sample.annotations : [];

  return (
    <div
      ref={ref}
      className="aspect-square bg-slate-800/50 rounded-lg overflow-hidden border border-slate-800 relative cursor-pointer hover:border-indigo-500/50 transition-colors"
      onClick={onClick}
    >
      {imgSrc ? (
        <>
          <img src={imgSrc} alt={`${sample.class_name} #${sample.index}`} loading="lazy" onLoad={() => setLoaded(true)}
            className={`w-full h-full object-contain transition-opacity duration-200 ${loaded ? 'opacity-100' : 'opacity-0'}`} />
          {loaded && anns.length > 0 && (
            <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox={`0 0 ${sample.orig_w} ${sample.orig_h}`} preserveAspectRatio="xMidYMid meet">
              {anns.map((a, i) => (
                <rect key={i} x={a.bbox[0]} y={a.bbox[1]} width={a.bbox[2]} height={a.bbox[3]}
                  fill="none" stroke={annColor(a.category_id)} strokeWidth={Math.max(2, sample.orig_w / 150)}
                  opacity={0.8} />
              ))}
            </svg>
          )}
        </>
      ) : (
        <div className="w-full h-full flex items-center justify-center"><ImageIcon size={16} className="text-slate-700" /></div>
      )}
      {visible && !loaded && !cached && (
        <div className="absolute inset-0 flex items-center justify-center"><Loader2 size={14} className="text-slate-600 animate-spin" /></div>
      )}
    </div>
  );
}

// ── Image Detail Modal with zoom/pan/nav + full-res loading ──────────────────

function ImageDetailModal({ sample, samples, datasetName, split, showAnnotations: initialShowAnn, onClose, onNavigate }: {
  sample: SampleData;
  samples: SampleData[];
  datasetName: string;
  split: string;
  showAnnotations: boolean;
  onClose: () => void;
  onNavigate: (s: SampleData) => void;
}) {
  const [localShowAnn, setLocalShowAnn] = useState(initialShowAnn);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const didDrag = useRef(false);
  const dragStart = useRef({ x: 0, y: 0 });
  const panStart = useRef({ x: 0, y: 0 });
  const [fullRes, setFullRes] = useState<SampleData | null>(null);
  const [loadingFull, setLoadingFull] = useState(false);

  const idx = samples.findIndex((s) => s.index === sample.index);
  const hasPrev = idx > 0;
  const hasNext = idx < samples.length - 1;

  const resetView = () => { setZoom(1); setPan({ x: 0, y: 0 }); };

  // Load full-resolution image when sample changes — always fetch annotations
  // for detection datasets so toggling is purely visual (no re-fetch)
  useEffect(() => {
    setLoadingFull(true);
    setFullRes(null);
    api.getDatasetSampleDetail(datasetName, sample.index, {
      split,
      include_annotations: true,
    }).then((res) => {
      setFullRes(res);
    }).catch(() => {}).finally(() => setLoadingFull(false));
  }, [datasetName, sample.index, split]);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.stopPropagation();
    const delta = e.deltaY > 0 ? -0.15 : 0.15;
    setZoom((z) => Math.min(10, Math.max(0.5, z + delta)));
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (zoom <= 1) return;
    e.preventDefault();
    setDragging(true);
    dragStart.current = { x: e.clientX, y: e.clientY };
    panStart.current = { ...pan };
  }, [zoom, pan]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragging) return;
    setPan({
      x: panStart.current.x + (e.clientX - dragStart.current.x),
      y: panStart.current.y + (e.clientY - dragStart.current.y),
    });
  }, [dragging]);

  const handleMouseUp = useCallback(() => setDragging(false), []);

  const goNav = useCallback((dir: -1 | 1) => {
    const ni = idx + dir;
    if (ni >= 0 && ni < samples.length) {
      onNavigate(samples[ni]);
      resetView();
    }
  }, [idx, samples, onNavigate]);

  // Keyboard nav
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      else if (e.key === 'ArrowLeft') goNav(-1);
      else if (e.key === 'ArrowRight') goNav(1);
      else if (e.key === '+' || e.key === '=') setZoom((z) => Math.min(10, z + 0.25));
      else if (e.key === '-') setZoom((z) => Math.max(0.5, z - 0.25));
      else if (e.key === '0') resetView();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose, goNav]);

  // Use full-res image if loaded, else fallback to thumb
  const displaySample = fullRes || sample;
  const imgSrc = `data:${displaySample.mime || 'image/png'};base64,${displaySample.image_base64}`;
  const hasAnnotations = !!(displaySample.annotations && displaySample.annotations.length > 0);
  const anns = localShowAnn && displaySample.annotations ? displaySample.annotations : [];

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex flex-col" onClick={onClose}>
      {/* Top bar */}
      <div className="shrink-0 flex items-center justify-between px-6 py-3 bg-slate-900/80 border-b border-slate-800" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-3">
          <span className="text-sm text-white font-medium">#{sample.index}</span>
          <span className="px-2 py-0.5 text-xs bg-indigo-500/20 text-indigo-300 rounded-full">{sample.class_name}</span>
          <span className="text-xs text-slate-500">class {sample.label}</span>
          {displaySample.orig_w > 0 && (
            <span className="text-xs text-slate-600">{displaySample.orig_w}×{displaySample.orig_h}</span>
          )}
          {loadingFull && <Loader2 size={12} className="text-indigo-400 animate-spin" />}
        </div>
        <div className="flex items-center gap-2">
          {hasAnnotations && (
            <button
              onClick={() => setLocalShowAnn((v) => !v)}
              className={`flex items-center gap-1 px-2 py-1 text-xs rounded-md transition-colors cursor-pointer border ${
                localShowAnn ? 'bg-amber-500/20 text-amber-300 border-amber-500/40' : 'text-slate-400 hover:text-white border-slate-700 hover:border-slate-600'
              }`}
              title={localShowAnn ? 'Hide annotations' : 'Show annotations'}
            >
              {localShowAnn ? <Eye size={13} /> : <EyeOff size={13} />}
              <span>{displaySample.annotations!.length}</span>
            </button>
          )}
          <div className="w-px h-5 bg-slate-700 mx-1" />
          <button onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))} className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded-md cursor-pointer" title="Zoom out">
            <ZoomOut size={16} />
          </button>
          <span className="text-xs text-slate-400 tabular-nums w-12 text-center">{Math.round(zoom * 100)}%</span>
          <button onClick={() => setZoom((z) => Math.min(10, z + 0.25))} className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded-md cursor-pointer" title="Zoom in">
            <ZoomIn size={16} />
          </button>
          <button onClick={resetView} className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded-md cursor-pointer" title="Reset view">
            <RotateCcw size={16} />
          </button>
          <div className="w-px h-5 bg-slate-700 mx-1" />
          <button onClick={onClose} className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded-md cursor-pointer">
            <X size={16} />
          </button>
        </div>
      </div>

      {/* Image area — click empty space to close */}
      <div
        className="flex-1 flex items-center justify-center overflow-hidden relative select-none"
        onClick={() => { if (zoom <= 1 && !didDrag.current) onClose(); }}
        onWheel={handleWheel}
        onMouseDown={(e) => { didDrag.current = false; handleMouseDown(e); }}
        onMouseMove={(e) => { if (dragging) didDrag.current = true; handleMouseMove(e); }}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ cursor: zoom > 1 ? (dragging ? 'grabbing' : 'grab') : 'default' }}
      >
        <div className="relative" onClick={(e) => e.stopPropagation()} style={{
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          transition: dragging ? 'none' : 'transform 0.15s ease-out',
        }}>
          <img
            src={imgSrc}
            alt={`${sample.class_name} #${sample.index}`}
            className="max-w-none"
            draggable={false}
            style={{ imageRendering: zoom >= 2 ? 'pixelated' : 'auto' }}
          />
          {anns.length > 0 && (
            <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox={`0 0 ${displaySample.orig_w} ${displaySample.orig_h}`} preserveAspectRatio="xMidYMid meet">
              {anns.map((a, i) => {
                const color = annColor(a.category_id);
                return (
                  <g key={i}>
                    <rect x={a.bbox[0]} y={a.bbox[1]} width={a.bbox[2]} height={a.bbox[3]}
                      fill="none" stroke={color} strokeWidth={Math.max(2, displaySample.orig_w / 300)} opacity={0.9} />
                    <rect x={a.bbox[0]} y={Math.max(0, a.bbox[1] - 18)} width={a.category_name.length * 7 + 8} height={18}
                      fill={color} opacity={0.8} rx={2} />
                    <text x={a.bbox[0] + 4} y={Math.max(0, a.bbox[1] - 18) + 13}
                      fill="white" fontSize={11} fontFamily="monospace">{a.category_name}</text>
                  </g>
                );
              })}
            </svg>
          )}
        </div>

        {/* Prev / Next arrows */}
        {hasPrev && (
          <button
            onClick={(e) => { e.stopPropagation(); goNav(-1); }}
            className="absolute left-4 top-1/2 -translate-y-1/2 p-3 bg-slate-900/80 border border-slate-700 rounded-full text-slate-300 hover:text-white hover:bg-slate-800 transition-colors cursor-pointer shadow-lg"
          >
            <ChevronLeft size={20} />
          </button>
        )}
        {hasNext && (
          <button
            onClick={(e) => { e.stopPropagation(); goNav(1); }}
            className="absolute right-4 top-1/2 -translate-y-1/2 p-3 bg-slate-900/80 border border-slate-700 rounded-full text-slate-300 hover:text-white hover:bg-slate-800 transition-colors cursor-pointer shadow-lg"
          >
            <ChevronRight size={20} />
          </button>
        )}
      </div>

      {/* Bottom bar */}
      <div className="shrink-0 flex items-center justify-center gap-4 px-6 py-2 bg-slate-900/80 border-t border-slate-800 text-xs text-slate-500" onClick={(e) => e.stopPropagation()}>
        <span>{idx + 1} / {samples.length} on this page</span>
        {anns.length > 0 && <span className="text-indigo-400">{anns.length} annotations</span>}
        <span>Scroll to zoom · Drag to pan · Arrow keys to navigate · Esc to close</span>
      </div>
    </div>
  );
}

// ── Main Page ────────────────────────────────────────────────────────────────

export default function DatasetDetailPage({ datasetName, onBack }: Props) {
  const [info, setInfo] = useState<DatasetInfo | null>(null);
  const [samples, setSamples] = useState<SampleData[]>([]);
  const [page, setPage] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [samplesLoading, setSamplesLoading] = useState(false);
  const [enabledClasses, setEnabledClasses] = useState<Set<number>>(new Set());
  const [split, setSplit] = usePersistedState<'train' | 'test' | 'val'>(`dataset.${datasetName}.split`, 'train');
  const [detailSample, setDetailSample] = useState<SampleData | null>(null);
  const [showAnnotations, setShowAnnotations] = usePersistedState(`dataset.${datasetName}.showAnnotations`, false);
  const [taskType, setTaskType] = useState<string>('classification');
  const [splitInfo, setSplitInfo] = useState<SplitInfo | null>(null);
  const [editorOpen, setEditorOpen] = usePersistedState(`dataset.${datasetName}.editorOpen`, false);
  const [partSummary, setPartSummary] = useState<PartitionSummary | null>(null);
  const [selectedPartition, setSelectedPartition] = usePersistedState<string>(`dataset.${datasetName}.partition`, '');
  const scrollRef = useRef<HTMLDivElement>(null);

  // Load dataset info + split config + partitions
  useEffect(() => {
    setLoading(true);
    Promise.all([
      api.getDatasetInfo(datasetName),
      api.getDatasetSplits(datasetName).catch(() => null),
      api.getDatasetPartitions(datasetName).catch(() => null),
    ]).then(([d, sp, ps]) => {
      setInfo(d);
      setEnabledClasses(new Set((d.class_names ?? d.classes ?? []).map((_c: string, i: number) => i)));
      if (sp) setSplitInfo(sp);
      if (ps) setPartSummary(ps);
    }).catch(() => {}).finally(() => setLoading(false));
  }, [datasetName]);

  // Compute class_indices param from enabled set
  const classIndicesParam = useCallback((): string | undefined => {
    if (!info) return undefined;
    // If all enabled, no filter
    if (enabledClasses.size === (info.class_names ?? info.classes ?? []).length || enabledClasses.size === 0) return undefined;
    return Array.from(enabledClasses).sort((a, b) => a - b).join(',');
  }, [info, enabledClasses]);

  // Dynamic page size based on image dimensions
  const pageSize = useMemo(() => info ? computePageSize(info.input_shape) : 50, [info]);
  const gridCols = useMemo(() => info ? computeGridCols(info.input_shape) : 'grid-cols-5 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-10', [info]);
  const isDetection = taskType === 'detection';

  // Load samples — include annotations when toggle is on for detection datasets
  const loadSamples = useCallback(async (p: number, s: string) => {
    setSamplesLoading(true);
    try {
      const ci = classIndicesParam();
      const res = await api.getDatasetSamples(datasetName, {
        page: p,
        page_size: pageSize,
        class_indices: ci,
        split: s,
        thumb_size: THUMB_SIZE,
        include_annotations: showAnnotations,
        partition_id: selectedPartition || undefined,
      });
      setSamples(res.samples);
      setTotalPages(res.total_pages);
      setTotal(res.total);
      if (res.task_type) setTaskType(res.task_type);
    } catch {
      setSamples([]);
    }
    setSamplesLoading(false);
  }, [datasetName, classIndicesParam, pageSize, showAnnotations, selectedPartition]);

  // Gate on info loaded — prevents double-fetch with stale default pageSize
  // Also re-fetch when showAnnotations or partition changes
  useEffect(() => {
    if (!info) return;
    loadSamples(page, split);
  }, [info, page, split, loadSamples, showAnnotations, selectedPartition]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
  }, [page]);

  const goToPage = (p: number) => { if (p >= 0 && p < totalPages) setPage(p); };

  const toggleClass = (idx: number) => {
    setEnabledClasses((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx); else next.add(idx);
      return next;
    });
    setPage(0);
  };

  const selectAll = () => {
    if (!info) return;
    setEnabledClasses(new Set((info.class_names ?? info.classes ?? []).map((_, i) => i)));
    setPage(0);
  };

  const selectNone = () => {
    setEnabledClasses(new Set());
    setPage(0);
  };

  const handleSplitChange = (s: 'train' | 'test' | 'val') => { setSplit(s); setPage(0); };
  const handlePartitionChange = (pid: string) => { setSelectedPartition(pid); setPage(0); };

  // Compute per-class counts for the current split + selected partition
  const classCountsMap = useMemo((): Record<string, number> => {
    if (!partSummary) return {};
    const allEntries = [partSummary.master, ...partSummary.partitions];
    const entry = selectedPartition
      ? allEntries.find((e) => e.id === selectedPartition)
      : null;
    if (entry?.class_counts) {
      return entry.class_counts[split] || {};
    }
    // No partition selected → sum all partitions
    const merged: Record<string, number> = {};
    for (const e of allEntries) {
      const cc = e.class_counts?.[split];
      if (cc) {
        for (const [k, v] of Object.entries(cc)) {
          merged[k] = (merged[k] || 0) + v;
        }
      }
    }
    return merged;
  }, [partSummary, selectedPartition, split]);

  const hasVal = splitInfo != null && splitInfo.val_count > 0;

  if (loading) {
    return <div className="flex-1 flex items-center justify-center"><Loader2 size={24} className="text-slate-500 animate-spin" /></div>;
  }

  if (!info) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-4">
        <Database size={48} className="text-slate-700" />
        <p className="text-slate-500">Dataset not found</p>
        <button onClick={onBack} className="text-indigo-400 hover:text-indigo-300 text-sm cursor-pointer">← Back</button>
      </div>
    );
  }

  const allEnabled = enabledClasses.size === (info.class_names ?? info.classes ?? []).length;
  const noneEnabled = enabledClasses.size === 0;
  const startIdx = page * pageSize + 1;
  const endIdx = Math.min((page + 1) * pageSize, total);

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
      {/* Header */}
      <div className="shrink-0 border-b border-slate-800 px-8 py-4">
        <div className="max-w-7xl mx-auto flex items-center gap-4">
          <button onClick={onBack} className="flex items-center gap-1 text-sm text-slate-400 hover:text-white transition-colors cursor-pointer">
            <ChevronLeft size={16} /> Datasets
          </button>
          <div className="w-px h-6 bg-slate-800" />
          <div className="flex-1 min-w-0">
            <h1 className="text-lg font-bold text-white truncate">{info.display_name}</h1>
            <div className="flex items-center gap-4 text-xs text-slate-500 mt-0.5">
              <span>{info.task_type}</span>
              <span>{info.input_shape.join('×')}</span>
              <span>{info.num_classes} classes</span>
              <span>
                {info.train_size.toLocaleString()} train
                {splitInfo && splitInfo.train_count !== info.train_size && (
                  <span className="text-indigo-400"> ({splitInfo.train_count.toLocaleString()})</span>
                )}
                {' / '}
                {(info.val_size || 0).toLocaleString()} val
                {splitInfo && splitInfo.val_count !== (info.val_size || 0) && (
                  <span className="text-amber-400"> ({splitInfo.val_count.toLocaleString()})</span>
                )}
                {' / '}
                {info.test_size.toLocaleString()} test
                {splitInfo && splitInfo.test_count !== info.test_size && (
                  <span className="text-emerald-400"> ({splitInfo.test_count.toLocaleString()})</span>
                )}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Toolbar: split + pagination */}
      <div className="shrink-0 border-b border-slate-800 px-8 py-3">
        <div className="max-w-7xl mx-auto flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-1 bg-slate-900 rounded-lg p-0.5 border border-slate-800">
            <button onClick={() => handleSplitChange('train')}
              className={`px-3 py-1 text-xs rounded-md transition-colors cursor-pointer ${split === 'train' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`}>
              Train {info.train_size.toLocaleString()}
              {splitInfo && splitInfo.train_count !== info.train_size && (
                <span className="opacity-70"> → {splitInfo.train_count.toLocaleString()}</span>
              )}
            </button>
            {hasVal && (
              <button onClick={() => handleSplitChange('val')}
                className={`px-3 py-1 text-xs rounded-md transition-colors cursor-pointer ${split === 'val' ? 'bg-amber-500 text-slate-900' : 'text-slate-400 hover:text-white'}`}>
                Val {(info.val_size || 0).toLocaleString()}
                {splitInfo && splitInfo.val_count !== (info.val_size || 0) && (
                  <span className="opacity-70"> → {splitInfo.val_count.toLocaleString()}</span>
                )}
              </button>
            )}
            <button onClick={() => handleSplitChange('test')}
              className={`px-3 py-1 text-xs rounded-md transition-colors cursor-pointer ${split === 'test' ? 'bg-emerald-600 text-white' : 'text-slate-400 hover:text-white'}`}>
              Test {info.test_size.toLocaleString()}
              {splitInfo && splitInfo.test_count !== info.test_size && (
                <span className="opacity-70"> → {splitInfo.test_count.toLocaleString()}</span>
              )}
            </button>
          </div>
          <button
            onClick={() => setEditorOpen(!editorOpen)}
            className={`flex items-center gap-1.5 px-3 py-1 text-xs rounded-md transition-colors cursor-pointer border ${
              editorOpen ? 'bg-slate-800 text-white border-slate-600' : 'text-slate-400 hover:text-white border-slate-800 hover:border-slate-700'
            }`}
          >
            <Settings2 size={13} /> Split Editor
          </button>
          {isDetection && (
            <button
              onClick={() => setShowAnnotations(!showAnnotations)}
              className={`flex items-center gap-1.5 px-3 py-1 text-xs rounded-md transition-colors cursor-pointer border ${
                showAnnotations ? 'bg-amber-500/20 text-amber-300 border-amber-500/40' : 'text-slate-400 hover:text-white border-slate-800 hover:border-slate-700'
              }`}
            >
              {showAnnotations ? <Eye size={13} /> : <EyeOff size={13} />} Annotations
            </button>
          )}
          {partSummary && partSummary.partitions.length > 0 && (
            <select
              value={selectedPartition}
              onChange={(e) => handlePartitionChange(e.target.value)}
              className="bg-slate-900 border border-slate-800 rounded-md px-2.5 py-1 text-xs text-white focus:outline-none focus:border-indigo-500 cursor-pointer"
            >
              <option value="">All partitions</option>
              <option value="master">Master ({partSummary.master.percent}%)</option>
              {partSummary.partitions.map((p) => (
                <option key={p.id} value={p.id}>{p.name} ({p.percent}%)</option>
              ))}
            </select>
          )}
          <div className="flex-1" />
          <span className="text-xs text-slate-500">
            {total > 0 ? `${startIdx}–${endIdx} of ${total.toLocaleString()}` : 'No samples'}
          </span>
          <div className="flex items-center gap-1">
            <button onClick={() => goToPage(0)} disabled={page === 0} className="p-1.5 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 disabled:opacity-30 disabled:cursor-default cursor-pointer transition-colors" title="First page"><ChevronsLeft size={14} /></button>
            <button onClick={() => goToPage(page - 1)} disabled={page === 0} className="p-1.5 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 disabled:opacity-30 disabled:cursor-default cursor-pointer transition-colors" title="Previous page"><ChevronLeft size={14} /></button>
            <span className="text-xs text-slate-300 px-2 tabular-nums">{page + 1} / {totalPages}</span>
            <button onClick={() => goToPage(page + 1)} disabled={page >= totalPages - 1} className="p-1.5 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 disabled:opacity-30 disabled:cursor-default cursor-pointer transition-colors" title="Next page"><ChevronRight size={14} /></button>
            <button onClick={() => goToPage(totalPages - 1)} disabled={page >= totalPages - 1} className="p-1.5 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 disabled:opacity-30 disabled:cursor-default cursor-pointer transition-colors" title="Last page"><ChevronsRight size={14} /></button>
          </div>
        </div>
      </div>

      {/* Split Editor panel */}
      {editorOpen && splitInfo && (
        <div className="shrink-0 border-b border-slate-800 px-8 py-4 bg-slate-900/50 max-h-[60vh] overflow-y-auto">
          <div className="max-w-7xl mx-auto">
            <SplitEditor
              datasetName={datasetName}
              splitInfo={splitInfo}
              onSaved={(s) => {
                setSplitInfo(s);
                if (split === 'val' && s.val_count <= 0) setSplit('train');
                setPage(0);
                loadSamples(0, split);
              }}
            />
            <PartitionEditor datasetName={datasetName} onUpdated={(ps) => setPartSummary(ps)} />
          </div>
        </div>
      )}

      {/* Class filter tags */}
      <div className="shrink-0 border-b border-slate-800 px-8 py-2.5">
        <div className="max-w-7xl mx-auto flex items-center gap-2 flex-wrap">
          <Tag size={13} className="text-slate-500 shrink-0" />
          <button
            onClick={allEnabled || noneEnabled ? selectNone : selectAll}
            className={`px-2.5 py-1 text-[11px] rounded-full border transition-colors cursor-pointer ${
              allEnabled || noneEnabled
                ? 'bg-indigo-600 text-white border-indigo-500'
                : 'bg-slate-800/50 text-slate-400 border-slate-700 hover:border-slate-600'
            }`}
          >
            All
          </button>
          {(info.class_names ?? info.classes ?? []).map((c, i) => {
            const on = enabledClasses.has(i);
            const cnt = classCountsMap[String(i)];
            return (
              <button
                key={i}
                onClick={() => toggleClass(i)}
                className={`inline-flex items-center gap-1 max-w-[180px] px-2.5 py-1 text-[11px] rounded-full border transition-colors cursor-pointer whitespace-nowrap ${
                  on
                    ? 'bg-indigo-500/20 text-indigo-300 border-indigo-500/40'
                    : 'bg-slate-800/30 text-slate-600 border-slate-800 hover:text-slate-400 hover:border-slate-700'
                }`}
                title={cnt != null ? `${c} (${cnt.toLocaleString()})` : c}
              >
                <span className="truncate">{c}</span>{cnt != null && <span className="shrink-0 opacity-60">{cnt.toLocaleString()}</span>}
              </button>
            );
          })}
        </div>
      </div>

      {/* Image Grid */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="max-w-7xl mx-auto p-8">
          {samplesLoading ? (
            <div className="flex items-center justify-center py-20"><Loader2 size={24} className="text-slate-500 animate-spin" /></div>
          ) : samples.length === 0 ? (
            <div className="text-center py-20">
              <ImageIcon size={48} className="mx-auto text-slate-700 mb-4" />
              <p className="text-slate-500 text-sm">No samples found</p>
            </div>
          ) : (
            <>
              <div className={`grid ${gridCols} gap-2`}>
                {samples.map((s) => (
                  <div key={`${split}-${s.index}`} className="group relative">
                    <LazyImage
                      sample={s}
                      cacheKey={`${datasetName}-${split}-${s.index}`}
                      showAnnotations={showAnnotations}
                      onClick={() => setDetailSample(s)}
                    />
                    <div className="mt-1 text-center">
                      <p className="text-[10px] text-slate-500 truncate">{s.class_name}</p>
                    </div>
                    <div className="absolute -top-10 left-1/2 -translate-x-1/2 bg-slate-800 border border-slate-700 rounded-md px-2 py-1 text-[10px] text-white whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-10 shadow-lg">
                      #{s.index} · {s.class_name} (class {s.label})
                    </div>
                  </div>
                ))}
              </div>

              {totalPages > 1 && (
                <div className="flex items-center justify-center gap-2 mt-8">
                  <button onClick={() => goToPage(page - 1)} disabled={page === 0}
                    className="px-3 py-1.5 text-xs bg-slate-900 border border-slate-800 rounded-lg text-slate-400 hover:text-white hover:border-slate-700 disabled:opacity-30 disabled:cursor-default cursor-pointer transition-colors">← Previous</button>
                  <div className="flex items-center gap-1">
                    {generatePageNumbers(page, totalPages).map((p, i) =>
                      p === -1 ? (
                        <span key={`ellipsis-${i}`} className="text-slate-600 px-1">…</span>
                      ) : (
                        <button key={p} onClick={() => goToPage(p)}
                          className={`w-8 h-8 text-xs rounded-md cursor-pointer transition-colors ${p === page ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white hover:bg-slate-800'}`}>{p + 1}</button>
                      )
                    )}
                  </div>
                  <button onClick={() => goToPage(page + 1)} disabled={page >= totalPages - 1}
                    className="px-3 py-1.5 text-xs bg-slate-900 border border-slate-800 rounded-lg text-slate-400 hover:text-white hover:border-slate-700 disabled:opacity-30 disabled:cursor-default cursor-pointer transition-colors">Next →</button>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Image Detail Modal */}
      {detailSample && (
        <ImageDetailModal
          sample={detailSample}
          samples={samples}
          datasetName={datasetName}
          split={split}
          showAnnotations={showAnnotations}
          onClose={() => setDetailSample(null)}
          onNavigate={(s) => setDetailSample(s)}
        />
      )}
    </div>
  );
}

function generatePageNumbers(current: number, total: number): number[] {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i);
  const pages: number[] = [];
  pages.push(0);
  if (current > 3) pages.push(-1);
  const start = Math.max(1, current - 1);
  const end = Math.min(total - 2, current + 1);
  for (let i = start; i <= end; i++) pages.push(i);
  if (current < total - 4) pages.push(-1);
  pages.push(total - 1);
  return pages;
}
