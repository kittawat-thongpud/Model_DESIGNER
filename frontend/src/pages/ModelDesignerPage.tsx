/**
 * Model Designer Page — Ultralytics YAML architecture editor.
 *
 * Pin-based manual linking system:
 * - Sequential pins (top/bottom): one-in, one-out — defines layer order
 * - Skip pins (left/right): multi-in, multi-out — skip connections
 * - First node of each chain marked as backbone-start or head-start
 * - Layer order derived by walking sequential links from start nodes
 * - Validation on save: both start nodes required, valid chains
 */
import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { api } from '../services/api';
import type { ModelSummary, CatalogModule, ModelYAML, LayerDef } from '../types';
import {
  ArrowLeft, Plus, Save, ChevronDown, ChevronRight, X, Loader2,
  ZoomIn, ZoomOut, Maximize2, Trash2, Box, Layers, GitBranch,
  Cpu, Eye, Sparkles, Puzzle, Cog, Undo2, Redo2, Copy,
  AlertTriangle, Check, FilePlus2, Upload, Target, ChevronUp,
  Settings2,
} from 'lucide-react';
import ImportYAMLModal from '../components/ImportYAMLModal';

/* ═══════════════════════════════════════════════════════════════════════════
   Types & Constants
   ═══════════════════════════════════════════════════════════════════════════ */
interface Props { onBack: () => void; }
type EdgeStyle = 'bezier' | 'step' | 'straight';
type NodeRole = 'normal' | 'backbone-start' | 'head-start' | 'detector';
type LinkType = 'seq' | 'skip';
type PinKind = 'seq-in' | 'seq-out' | 'skip-in' | 'skip-out';

interface GNode {
  id: string;
  module: string;
  repeats: number;
  args: unknown[];
  role: NodeRole;
  x: number;
  y: number;
}

interface GLink {
  id: string;
  from: string;   // node id
  to: string;     // node id
  type: LinkType;
}

interface CtxMenu { x: number; y: number; nodeId?: string; }
interface Snapshot { nodes: GNode[]; links: GLink[]; nc: number; }

const MAX_HISTORY = 50;
const NODE_W = 240;
const NODE_H = 64;
const NODE_GAP_Y = 24;
const INIT_X = 80;
const INIT_Y = 40;
const ZOOM_STEP = 0.1;
const ZOOM_MIN = 0.3;
const ZOOM_MAX = 2.0;

let _idCounter = 0;
function uid(): string { return `n${Date.now().toString(36)}_${(++_idCounter).toString(36)}`; }
function linkId(): string { return `l${Date.now().toString(36)}_${(++_idCounter).toString(36)}`; }

/* ── Detector modules — singleton blocks, always placed last in head ──── */
const DETECTOR_MODULES = new Set(['Detect', 'Segment', 'Pose', 'Classify', 'OBB']);

/* ── Scale presets (depth_multiple, width_multiple, max_channels) ───────── */
type ScaleKey = 'n' | 's' | 'm' | 'l' | 'x';
const SCALE_PRESETS: Record<ScaleKey, [number, number, number]> = {
  n: [0.33, 0.25, 1024],
  s: [0.33, 0.50, 1024],
  m: [0.67, 0.75,  768],
  l: [1.00, 1.00,  512],
  x: [1.00, 1.25,  512],
};

/* ── Category visual config ─────────────────────────────────────────────── */
const CAT_CFG: Record<string, { gradient: string; border: string; text: string; iconBg: string; icon: React.ReactNode }> = {
  basic:       { gradient: 'from-blue-500/20 to-blue-600/5',      border: 'border-blue-500/30',    text: 'text-blue-400',    iconBg: 'bg-blue-500/20',    icon: <Box size={14} /> },
  composite:   { gradient: 'from-violet-500/20 to-violet-600/5',  border: 'border-violet-500/30',  text: 'text-violet-400',  iconBg: 'bg-violet-500/20',  icon: <Layers size={14} /> },
  attention:   { gradient: 'from-amber-500/20 to-amber-600/5',    border: 'border-amber-500/30',   text: 'text-amber-400',   iconBg: 'bg-amber-500/20',   icon: <Eye size={14} /> },
  head:        { gradient: 'from-emerald-500/20 to-emerald-600/5',border: 'border-emerald-500/30', text: 'text-emerald-400', iconBg: 'bg-emerald-500/20', icon: <Sparkles size={14} /> },
  detector:    { gradient: 'from-fuchsia-500/20 to-fuchsia-600/5',border: 'border-fuchsia-500/40', text: 'text-fuchsia-400', iconBg: 'bg-fuchsia-500/20', icon: <Target size={14} /> },
  specialized: { gradient: 'from-cyan-500/20 to-cyan-600/5',      border: 'border-cyan-500/30',    text: 'text-cyan-400',    iconBg: 'bg-cyan-500/20',    icon: <Puzzle size={14} /> },
  torch_nn:    { gradient: 'from-orange-500/20 to-orange-600/5',  border: 'border-orange-500/30',  text: 'text-orange-400',  iconBg: 'bg-orange-500/20',  icon: <Cog size={14} /> },
};
const DEF_CFG = { gradient: 'from-slate-500/20 to-slate-600/5', border: 'border-slate-500/30', text: 'text-slate-400', iconBg: 'bg-slate-500/20', icon: <Cpu size={14} /> };

function getCfg(catalog: CatalogModule[], modName: string) {
  const m = catalog.find(c => c.name === modName);
  return CAT_CFG[m?.category ?? ''] ?? DEF_CFG;
}

const CAT_HEX: Record<string, string> = {
  basic: '#3b82f6', composite: '#8b5cf6', attention: '#f59e0b',
  head: '#10b981', specialized: '#06b6d4', torch_nn: '#f97316',
};

function safeArgs(node: GNode): unknown[] {
  return Array.isArray(node.args) ? node.args : [];
}

/* ── Param estimation (approximate, no channel inference) ────────────── */
function estimateParams(mod: string, args: unknown[], repeats: number): number {
  const a = args.map(v => (typeof v === 'number' ? v : 0));
  const out = a[0] || 0;
  const k = a[1] || 3;
  const cin = out; // rough: assume cin ≈ cout for estimation
  let p = 0;
  switch (mod) {
    case 'Conv':         p = cin * out * k * k + out; break;           // Conv+BN: weight + bias(BN)
    case 'DWConv':       p = out * k * k + out; break;
    case 'ConvTranspose':p = cin * out * k * k + out; break;
    case 'C2f':          p = out * out * 2 + out * out * k * k * 2; break; // rough
    case 'C3':           p = out * out * 2 + out * out * k * k * 2; break;
    case 'C3k2':         p = out * out * 3; break;
    case 'SPPF':         p = out * out * 2; break;
    case 'SPP':          p = out * out * 2; break;
    case 'Bottleneck':   p = out * out * 2 * k * k; break;
    case 'Concat':       p = 0; break;
    case 'nn.Upsample':  p = 0; break;
    case 'nn.Identity':  p = 0; break;
    case 'Detect':       p = out * 85 * 3; break; // rough: 3 scales
    default:             p = out * out; break; // generic fallback
  }
  return Math.round(p * Math.max(repeats, 1));
}

function formatParams(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

/* ── Edge path generators ───────────────────────────────────────────────── */
function seqEdgePath(style: EdgeStyle, x1: number, y1: number, x2: number, y2: number): string {
  switch (style) {
    case 'bezier': {
      const dy = y2 - y1;
      const cpOff = Math.min(Math.abs(dy) * 0.4, 120);
      return `M ${x1} ${y1} C ${x1} ${y1 + cpOff}, ${x2} ${y2 - cpOff}, ${x2} ${y2}`;
    }
    case 'step': {
      const midY = (y1 + y2) / 2;
      return `M ${x1} ${y1} L ${x1} ${midY} L ${x2} ${midY} L ${x2} ${y2}`;
    }
    default: return `M ${x1} ${y1} L ${x2} ${y2}`;
  }
}

function skipEdgePath(style: EdgeStyle, x1: number, y1: number, x2: number, y2: number): string {
  switch (style) {
    case 'bezier': {
      const dx = x2 - x1;
      const cpOff = Math.max(Math.abs(dx) * 0.3, 60);
      return `M ${x1} ${y1} C ${x1 + cpOff} ${y1}, ${x2 - cpOff} ${y2}, ${x2} ${y2}`;
    }
    case 'step': {
      const midX = (x1 + x2) / 2;
      return `M ${x1} ${y1} L ${midX} ${y1} L ${midX} ${y2} L ${x2} ${y2}`;
    }
    default: return `M ${x1} ${y1} L ${x2} ${y2}`;
  }
}

/* ── Walk sequential chain from a start node ──────────────────────────── */
function walkSeqChain(startId: string, nodes: GNode[], links: GLink[], stopAt?: Set<string>): string[] {
  const chain: string[] = [startId];
  const visited = new Set<string>([startId]);
  let cur = startId;
  for (let i = 0; i < nodes.length + 1; i++) {
    const next = links.find(l => l.type === 'seq' && l.from === cur);
    if (!next || visited.has(next.to)) break;
    if (stopAt && stopAt.has(next.to)) break;
    chain.push(next.to);
    visited.add(next.to);
    cur = next.to;
  }
  return chain;
}

/* ── Build YAML from graph ────────────────────────────────────────────── */
interface ValidationResult { ok: boolean; errors: string[]; yaml?: ModelYAML; }

function buildYaml(
  nodes: GNode[],
  links: GLink[],
  nc: number,
  scales?: Record<string, number[]>,
): ValidationResult {
  const errors: string[] = [];
  const bbStart = nodes.find(n => n.role === 'backbone-start');
  const hdStart = nodes.find(n => n.role === 'head-start');
  let detNode = nodes.find(n => n.role === 'detector');

  // Fallback: if no node has explicit 'detector' role (stale state), check known detector modules
  if (!detNode) {
    detNode = nodes.find(n => DETECTOR_MODULES.has(n.module)); 
  }

  if (!bbStart) errors.push('No backbone start node set');
  if (!hdStart) errors.push('No head start node set');
  if (!detNode) errors.push('No Detector block added (drag Detect/Segment/Pose/Classify/OBB from the panel)');
  if (errors.length > 0) return { ok: false, errors };

  // Walk chains — stop at head-start boundary to avoid cross-contamination
  const bbChain = walkSeqChain(bbStart!.id, nodes, links, new Set(detNode ? [detNode.id] : []));
  // Head chain excludes the detector node; it will be appended last
  const hdChainRaw = walkSeqChain(hdStart!.id, nodes, links);
  const hdChain = hdChainRaw.filter(id => id !== detNode?.id);

  if (bbChain.length === 0) errors.push('Backbone chain is empty');
  if (hdChain.length === 0 && !detNode) errors.push('Head chain is empty');

  // Orphan check: every node must be in backbone, head, or be the detector
  const allInChain = new Set([...bbChain, ...hdChain, ...(detNode ? [detNode.id] : [])]);
  const orphans = nodes.filter(n => !allInChain.has(n.id));
  if (orphans.length > 0) errors.push(`${orphans.length} node(s) not in any chain`);

  if (errors.length > 0) return { ok: false, errors };

  // Final ordered list: backbone → head (without detector) → detector
  const ordered = [...bbChain, ...hdChain, ...(detNode ? [detNode.id] : [])];
  const idxMap = new Map<string, number>();
  ordered.forEach((id, i) => idxMap.set(id, i));

  const buildLayers = (chain: string[]): LayerDef[] => {
    return chain.map((nodeId, i) => {
      const node = nodes.find(n => n.id === nodeId)!;
      const skipIn = links.filter(l => l.type === 'skip' && l.to === nodeId);
      const hasSeqInput = i > 0 && links.some(l => l.type === 'seq' && l.to === nodeId);
      let from: number | number[];
      if (skipIn.length === 0) {
        from = -1;
      } else if (!hasSeqInput) {
        // Only skip inputs (Detect receiving P3/P4/P5)
        const indices = skipIn.map(l => idxMap.get(l.from) ?? -1);
        from = indices.length === 1 ? indices[0] : indices;
      } else {
        // Both sequential + skip (Concat)
        from = [-1, ...skipIn.map(l => idxMap.get(l.from) ?? -1)];
        if (from.length === 1) from = from[0];
      }
      return { from, repeats: node.repeats, module: node.module, args: [...(node.args ?? [])] };
    });
  };

  const backbone = buildLayers(bbChain);
  // Build head layers: sequential head nodes + detector appended at the end
  const headBodyLayers = buildLayers(hdChain);
  const detectorLayer = detNode ? buildLayers([detNode.id]) : [];
  const head = [...headBodyLayers, ...detectorLayer];

  return { ok: true, errors: [], yaml: { nc, scales: scales && Object.keys(scales).length ? scales : undefined, backbone, head } };
}

/* ── Normalize layer: handle both array [from, repeats, module, args] and object formats ── */
function normalizeLayer(raw: unknown): LayerDef {
  if (Array.isArray(raw)) {
    // Ultralytics YAML format: [from, repeats, module, args]
    return { from: raw[0] ?? -1, repeats: raw[1] ?? 1, module: String(raw[2] ?? 'Conv'), args: Array.isArray(raw[3]) ? raw[3] : [] };
  }
  const obj = raw as Record<string, unknown>;
  return {
    from: (obj.from ?? obj.from_ ?? -1) as number | number[],
    repeats: (obj.repeats ?? 1) as number,
    module: String(obj.module ?? 'Conv'),
    args: Array.isArray(obj.args) ? obj.args : [],
  };
}

/* ── Build graph from YAML (for loading saved models) ───────────────── */
function yamlToGraph(yaml: ModelYAML, savedPositions?: { x: number; y: number }[]): { nodes: GNode[]; links: GLink[] } {
  const nodes: GNode[] = [];
  const links: GLink[] = [];
  const rawBB = (yaml.backbone ?? []) as unknown[];
  const rawHD = (yaml.head ?? []) as unknown[];
  const all = [...rawBB, ...rawHD].map(normalizeLayer);
  const bbLen = rawBB.length;

  // Create nodes
  all.forEach((layer, i) => {
    let role: NodeRole;
    if (i === 0) role = 'backbone-start';
    else if (i === bbLen) role = 'head-start';
    else if (DETECTOR_MODULES.has(layer.module)) role = 'detector';
    else role = 'normal';
    const pos = savedPositions?.[i] ?? { x: INIT_X, y: INIT_Y + i * (NODE_H + NODE_GAP_Y) };
    nodes.push({ id: uid(), module: layer.module, repeats: layer.repeats, args: layer.args, role, x: pos.x, y: pos.y });
  });

  // Create links from `from` field
  all.forEach((layer, toIdx) => {
    const froms = Array.isArray(layer.from) ? layer.from : [layer.from];
    froms.forEach(f => {
      let fromIdx: number;
      if (f === -1) { fromIdx = toIdx - 1; } else { fromIdx = f; }
      if (fromIdx < 0 || fromIdx >= all.length || fromIdx === toIdx) return;
      // Determine if this is a sequential or skip link
      // Sequential = from previous node in same section chain
      const isSeq = f === -1;
      links.push({ id: linkId(), from: nodes[fromIdx].id, to: nodes[toIdx].id, type: isSeq ? 'seq' : 'skip' });
    });
  });

  return { nodes, links };
}

/* ── Default graph ────────────────────────────────────────────────────── */
const DEFAULT_YAML: ModelYAML = {
  nc: 80,
  backbone: [
    { from: -1, repeats: 1, module: 'Conv', args: [64, 3, 2] },
    { from: -1, repeats: 1, module: 'Conv', args: [128, 3, 2] },
    { from: -1, repeats: 3, module: 'C2f', args: [128, true] },
    { from: -1, repeats: 1, module: 'Conv', args: [256, 3, 2] },
    { from: -1, repeats: 6, module: 'C2f', args: [256, true] },
    { from: -1, repeats: 1, module: 'Conv', args: [512, 3, 2] },
    { from: -1, repeats: 6, module: 'C2f', args: [512, true] },
    { from: -1, repeats: 1, module: 'Conv', args: [1024, 3, 2] },
    { from: -1, repeats: 3, module: 'C2f', args: [1024, true] },
    { from: -1, repeats: 1, module: 'SPPF', args: [1024, 5] },
  ],
  head: [
    { from: -1, repeats: 1, module: 'nn.Upsample', args: ['None', 2, 'nearest'] },
    { from: [-1, 6], repeats: 1, module: 'Concat', args: [1] },
    { from: -1, repeats: 3, module: 'C2f', args: [512] },
    { from: -1, repeats: 1, module: 'nn.Upsample', args: ['None', 2, 'nearest'] },
    { from: [-1, 4], repeats: 1, module: 'Concat', args: [1] },
    { from: -1, repeats: 3, module: 'C2f', args: [256] },
    { from: -1, repeats: 1, module: 'Conv', args: [256, 3, 2] },
    { from: [-1, 12], repeats: 1, module: 'Concat', args: [1] },
    { from: -1, repeats: 3, module: 'C2f', args: [512] },
    { from: -1, repeats: 1, module: 'Conv', args: [512, 3, 2] },
    { from: [-1, 9], repeats: 1, module: 'Concat', args: [1] },
    { from: -1, repeats: 3, module: 'C2f', args: [1024] },
    { from: [15, 18, 21], repeats: 1, module: 'Detect', args: ['nc'] },
  ],
};

const defaultGraph = yamlToGraph(DEFAULT_YAML);

/* ═══════════════════════════════════════════════════════════════════════════
   Save Modal — validates graph, builds YAML, saves with _graph metadata
   ═══════════════════════════════════════════════════════════════════════════ */
function SaveModal({ nodes, links, nc, scales, existingId, onSaved, onClose }: {
  nodes: GNode[]; links: GLink[]; nc: number;
  scales: Record<string, number[]>; existingId: string | null;
  onSaved: (id: string, name: string) => void; onClose: () => void;
}) {
  const [name, setName] = useState('');
  const [desc, setDesc] = useState('');
  const [task, setTask] = useState('detect');
  const [saving, setSaving] = useState(false);
  const [result, setResult] = useState<{ ok: boolean; modelId: string; message: string; params?: number; gradients?: number; flops?: number; layers?: number } | null>(null);
  const [localScales, setLocalScales] = useState<Record<string, number[]>>(() => ({ ...scales }));
  const [showScales, setShowScales] = useState(Object.keys(scales).length > 0);
  const validation = useMemo(() => buildYaml(nodes, links, nc, showScales ? localScales : {}), [nodes, links, nc, localScales, showScales]);

  const applyPreset = (key: ScaleKey) => {
    setLocalScales(prev => ({ ...prev, [key]: [...SCALE_PRESETS[key]] }));
  };
  const removeScale = (key: string) => {
    setLocalScales(prev => { const n = { ...prev }; delete n[key]; return n; });
  };
  const updateScaleVal = (key: string, idx: number, val: number) => {
    setLocalScales(prev => ({ ...prev, [key]: prev[key].map((v, i) => i === idx ? val : v) }));
  };

  const handleSave = async () => {
    if (!name.trim() || !validation.ok || !validation.yaml) return;
    setSaving(true);
    setResult(null);
    try {
      const graphMeta = { _nodes: nodes, _links: links };
      const payload = { ...validation.yaml, _graph: graphMeta } as unknown as Record<string, unknown>;
      const res = await api.saveModel({ name: name.trim(), description: desc, task, yaml_def: payload }, false);
      // Validate with Ultralytics YOLO
      try {
        const v = await api.validateModel(res.model_id);
        setResult({ ok: v.valid, modelId: res.model_id, message: v.message, params: v.params, gradients: v.gradients, flops: v.flops, layers: v.layers });
        if (v.valid) onSaved(res.model_id, name.trim());
      } catch {
        setResult({ ok: true, modelId: res.model_id, message: 'Saved (validation skipped)' });
        onSaved(res.model_id, name.trim());
      }
    } catch (err) {
      setResult({ ok: false, modelId: '', message: `Save failed: ${err}` });
    }
    setSaving(false);
  };

  const scaleKeys = Object.keys(localScales);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={result ? undefined : onClose}>
      <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-full max-w-lg mx-4 p-6 space-y-4 max-h-[90vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
        <h3 className="text-lg font-semibold text-white">{existingId ? 'Save As New Model' : 'Save Model'}</h3>

        {/* Result banner */}
        {result && (
          <div className={`${result.ok ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-red-500/10 border-red-500/30'} border rounded-lg p-3 space-y-1`}>
            <div className={`flex items-center gap-2 text-xs font-semibold ${result.ok ? 'text-emerald-400' : 'text-red-400'}`}>
              {result.ok ? <Check size={14} /> : <AlertTriangle size={14} />}
              {result.ok ? 'Build Check Passed' : 'Build Check Failed'}
            </div>
            <p className={`text-xs ${result.ok ? 'text-emerald-300' : 'text-red-300'} ml-5`}>{result.message}</p>
            {result.ok && result.params != null && (
              <div className="text-xs text-emerald-300/70 ml-5">
                <p>
                  {result.params.toLocaleString()} params · {result.gradients?.toLocaleString() || result.params.toLocaleString()} gradients · {result.layers} layers
                  {result.flops != null && ` · ${result.flops.toFixed(1)} GFLOPs`}
                </p>
              </div>
            )}
          </div>
        )}

        {/* Graph validation */}
        {!result && !validation.ok && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 space-y-1">
            <div className="flex items-center gap-2 text-red-400 text-xs font-semibold"><AlertTriangle size={14} /> Validation Errors</div>
            {validation.errors.map((e, i) => <p key={i} className="text-xs text-red-300 ml-5">• {e}</p>)}
          </div>
        )}
        {!result && validation.ok && (
          <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg p-3 flex items-center gap-2 text-emerald-400 text-xs font-semibold">
            <Check size={14} /> Graph valid — {validation.yaml!.backbone.length}B + {validation.yaml!.head.length}H layers
          </div>
        )}

        {/* Form (hidden after result) */}
        {!result && (
          <>
            <div className="space-y-3">
              <div className="space-y-1">
                <label className="text-xs text-slate-400">Name</label>
                <input value={name} onChange={e => setName(e.target.value)} placeholder="My YOLOv8 Model"
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500" />
              </div>
              <div className="space-y-1">
                <label className="text-xs text-slate-400">Task</label>
                <select value={task} onChange={e => setTask(e.target.value)}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500">
                  <option value="detect">Detection</option><option value="classify">Classification</option>
                  <option value="segment">Segmentation</option><option value="pose">Pose</option><option value="obb">OBB</option>
                </select>
              </div>
              <div className="space-y-1">
                <label className="text-xs text-slate-400">Description</label>
                <textarea value={desc} onChange={e => setDesc(e.target.value)} rows={2} placeholder="Optional description..."
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500 resize-none" />
              </div>

              {/* ───── Scale Config ──────────────────────────────── */}
              <div className="border border-slate-700/60 rounded-lg overflow-hidden">
                <button
                  type="button"
                  onClick={() => setShowScales(s => !s)}
                  className="w-full flex items-center justify-between px-3 py-2.5 text-xs font-medium text-slate-300 hover:bg-slate-800/50 transition-colors cursor-pointer"
                >
                  <div className="flex items-center gap-2">
                    <Settings2 size={13} className="text-indigo-400" />
                    Scale Config <span className="text-slate-500">(n / s / m / l / x)</span>
                    {showScales && scaleKeys.length > 0 && (
                      <span className="ml-1 px-1.5 py-0.5 text-[10px] bg-indigo-500/20 text-indigo-400 rounded-full">{scaleKeys.length} scale{scaleKeys.length > 1 ? 's' : ''}</span>
                    )}
                  </div>
                  {showScales ? <ChevronUp size={13} className="text-slate-500" /> : <ChevronDown size={13} className="text-slate-500" />}
                </button>

                {showScales && (
                  <div className="px-3 pb-3 pt-1 space-y-3 border-t border-slate-700/60">
                    {/* Preset insert buttons */}
                    <div>
                      <p className="text-[10px] text-slate-500 mb-1.5 uppercase tracking-wider">Insert preset</p>
                      <div className="flex gap-1.5 flex-wrap">
                        {(Object.keys(SCALE_PRESETS) as ScaleKey[]).map(k => (
                          <button key={k} type="button" onClick={() => applyPreset(k)}
                            className={`px-2.5 py-1 text-xs rounded-md border font-mono transition-colors cursor-pointer ${
                              localScales[k] ? 'bg-indigo-600/30 border-indigo-500/50 text-indigo-300' : 'bg-slate-800 border-slate-700 text-slate-300 hover:border-slate-600'
                            }`}>
                            {k}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Editable table */}
                    {scaleKeys.length > 0 && (
                      <div className="overflow-x-auto">
                        <table className="w-full text-[11px]">
                          <thead>
                            <tr className="text-slate-500">
                              <th className="text-left pb-1 font-medium w-8">Scale</th>
                              <th className="text-right pb-1 font-medium px-2">depth</th>
                              <th className="text-right pb-1 font-medium px-2">width</th>
                              <th className="text-right pb-1 font-medium px-2">max_ch</th>
                              <th className="w-6" />
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-800">
                            {scaleKeys.map(k => (
                              <tr key={k}>
                                <td className="py-1 font-mono text-indigo-400 font-semibold">{k}</td>
                                {[0, 1, 2].map(idx => (
                                  <td key={idx} className="py-1 px-2">
                                    <input
                                      type="number" step="0.01"
                                      value={localScales[k]?.[idx] ?? ''}
                                      onChange={e => updateScaleVal(k, idx, parseFloat(e.target.value) || 0)}
                                      className="w-16 bg-slate-800 border border-slate-700 rounded px-1.5 py-0.5 text-right text-white font-mono focus:outline-none focus:border-indigo-500"
                                    />
                                  </td>
                                ))}
                                <td className="py-1 pl-1">
                                  <button type="button" onClick={() => removeScale(k)}
                                    className="text-slate-600 hover:text-red-400 transition-colors cursor-pointer"><X size={12} /></button>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                    {scaleKeys.length === 0 && (
                      <p className="text-[11px] text-slate-600 italic">No scales defined — click a preset above to add one</p>
                    )}
                  </div>
                )}
              </div>
              {/* ───────────────────────────────────────── */}
            </div>
            <div className="flex gap-3 justify-end">
              <button onClick={onClose} className="px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors cursor-pointer">Cancel</button>
              <button onClick={handleSave} disabled={saving || !name.trim() || !validation.ok}
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-sm rounded-lg transition-colors cursor-pointer flex items-center gap-2">
                {saving && <Loader2 size={14} className="animate-spin" />} Save &amp; Validate
              </button>
            </div>
          </>
        )}

        {/* Close button after result */}
        {result && (
          <div className="flex gap-3 justify-end">
            <button onClick={onClose} className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white text-sm rounded-lg transition-colors cursor-pointer">
              Close
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   Properties Panel (right sidebar) — edits node module, repeats, args
   ═══════════════════════════════════════════════════════════════════════════ */
function PropertiesPanel({ node, catalog, onChange, onClose }: {
  node: GNode; catalog: CatalogModule[]; onChange: (n: Partial<GNode>) => void; onClose: () => void;
}) {
  const [argsStr, setArgsStr] = useState(JSON.stringify(safeArgs(node)));
  const [repeats, setRepeats] = useState(node.repeats);
  const [module, setModule] = useState(node.module);
  const cfg = getCfg(catalog, module);
  const catMod = catalog.find(m => m.name === module);

  useEffect(() => {
    setArgsStr(JSON.stringify(safeArgs(node)));
    setRepeats(node.repeats);
    setModule(node.module);
  }, [node]);

  const apply = () => {
    let parsedArgs: unknown[];
    try { const raw = JSON.parse(argsStr); parsedArgs = Array.isArray(raw) ? raw : [raw]; }
    catch { parsedArgs = safeArgs(node); }
    onChange({ module, repeats, args: parsedArgs });
  };

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={`w-7 h-7 rounded-lg ${cfg.iconBg} flex items-center justify-center ${cfg.text}`}>{cfg.icon}</div>
          <div>
            <div className="text-sm font-semibold text-white">{module}</div>
            <div className="text-[10px] text-slate-500">{node.id.slice(0, 8)}</div>
          </div>
        </div>
        <button onClick={onClose} className="text-slate-500 hover:text-white cursor-pointer"><X size={16} /></button>
      </div>

      {/* Role badge */}
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] text-slate-500 uppercase tracking-wider font-medium">Role:</span>
        <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
          node.role === 'backbone-start' ? 'bg-indigo-500/20 text-indigo-400' :
          node.role === 'head-start' ? 'bg-emerald-500/20 text-emerald-400' :
          node.role === 'detector' ? 'bg-fuchsia-500/20 text-fuchsia-400' :
          'bg-slate-700/50 text-slate-500'
        }`}>
          {node.role === 'normal' ? 'Normal' :
           node.role === 'backbone-start' ? 'Backbone Start' :
           node.role === 'head-start' ? 'Head Start' :
           'Detector (Singleton)'}
        </span>
      </div>

      {catMod?.description && <p className="text-xs text-slate-400 bg-slate-800/50 rounded-lg px-3 py-2">{catMod.description}</p>}

      <div className="space-y-3">
        <div className="space-y-1">
          <label className="text-[10px] text-slate-500 uppercase tracking-wider font-medium">Module</label>
          <select value={module} onChange={e => setModule(e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-2 text-xs text-white font-mono focus:outline-none focus:border-indigo-500">
            {catalog.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}
          </select>
        </div>
        <div className="space-y-1">
          <label className="text-[10px] text-slate-500 uppercase tracking-wider font-medium">Repeats</label>
          <input type="number" min={1} max={99} value={repeats}
            onChange={e => setRepeats(Math.max(1, Number(e.target.value)))}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-2 text-xs text-white font-mono focus:outline-none focus:border-indigo-500" />
        </div>
        <div className="space-y-1">
          <label className="text-[10px] text-slate-500 uppercase tracking-wider font-medium">Args</label>
          <input value={argsStr} onChange={e => setArgsStr(e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-2 text-xs text-white font-mono focus:outline-none focus:border-indigo-500" />
          {catMod && catMod.args.length > 0 && (
            <div className="mt-1.5 bg-slate-800/30 rounded-lg p-2 space-y-0.5">
              {catMod.args.map((a, i) => (
                <div key={i} className="text-[10px] flex items-center gap-1">
                  <span className="text-slate-400 font-mono">{a.name}</span>
                  <span className="text-slate-700">:</span>
                  <span className="text-slate-500">{a.type}</span>
                  {a.default !== undefined && <span className="text-slate-600 ml-auto">= {JSON.stringify(a.default)}</span>}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <button onClick={apply}
        className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium rounded-lg transition-colors cursor-pointer">
        Apply Changes
      </button>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   Context Menu — delete, duplicate, set role
   ═══════════════════════════════════════════════════════════════════════════ */
function ContextMenuPopup({ menu, onDelete, onDuplicate, onSetRole, onClose, isDetector }: {
  menu: CtxMenu; onDelete: () => void; onDuplicate: () => void;
  onSetRole: (role: NodeRole) => void; onClose: () => void; isDetector?: boolean;
}) {
  useEffect(() => {
    const handler = () => onClose();
    window.addEventListener('click', handler);
    return () => window.removeEventListener('click', handler);
  }, [onClose]);

  return (
    <div className="fixed z-50 bg-slate-900 border border-slate-700 rounded-lg shadow-xl py-1 min-w-[160px]"
      style={{ left: menu.x, top: menu.y }} onClick={e => e.stopPropagation()}>
      {menu.nodeId !== undefined && (
        <>
          {/* Hide role-change buttons for detector nodes */}
          {!isDetector && (
            <>
              <button onClick={() => { onSetRole('backbone-start'); onClose(); }}
                className="w-full text-left px-3 py-1.5 text-xs text-indigo-400 hover:bg-slate-800 flex items-center gap-2 cursor-pointer">
                <Box size={12} /> Set as Backbone Start
              </button>
              <button onClick={() => { onSetRole('head-start'); onClose(); }}
                className="w-full text-left px-3 py-1.5 text-xs text-emerald-400 hover:bg-slate-800 flex items-center gap-2 cursor-pointer">
                <Sparkles size={12} /> Set as Head Start
              </button>
              <button onClick={() => { onSetRole('normal'); onClose(); }}
                className="w-full text-left px-3 py-1.5 text-xs text-slate-400 hover:bg-slate-800 flex items-center gap-2 cursor-pointer">
                <Cpu size={12} /> Set as Normal
              </button>
              <div className="border-t border-slate-700/50 my-1" />
            </>
          )}
          <button onClick={() => { onDuplicate(); onClose(); }}
            className="w-full text-left px-3 py-1.5 text-xs text-slate-300 hover:bg-slate-800 flex items-center gap-2 cursor-pointer">
            <Copy size={12} /> Duplicate
          </button>
          <button onClick={() => { onDelete(); onClose(); }}
            className="w-full text-left px-3 py-1.5 text-xs text-red-400 hover:bg-slate-800 flex items-center gap-2 cursor-pointer">
            <Trash2 size={12} /> Delete
          </button>
        </>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   Main Page
   ═══════════════════════════════════════════════════════════════════════════ */
export default function ModelDesignerPage({ onBack }: Props) {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [catalog, setCatalog] = useState<CatalogModule[]>([]);
  const [loading, setLoading] = useState(true);

  const [nodes, setNodes] = useState<GNode[]>(defaultGraph.nodes);
  const [links, setLinks] = useState<GLink[]>(defaultGraph.links);
  const [nc, setNc] = useState(DEFAULT_YAML.nc);
  const [scales, setScales] = useState<Record<string, number[]>>({});
  const [modelId, setModelId] = useState<string | null>(null);
  const [modelName, setModelName] = useState('');
  const [selectedNodes, setSelectedNodes] = useState<Set<string>>(new Set());
  const [selectedLink, setSelectedLink] = useState<string | null>(null);
  const [showSave, setShowSave] = useState(false);
  const [showImport, setShowImport] = useState(false);
  const [catalogSearch, setCatalogSearch] = useState('');
  const [collapsedCats, setCollapsedCats] = useState<Set<string>>(new Set());
  const [edgeStyle, setEdgeStyle] = useState<EdgeStyle>('bezier');
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [showMinimap, setShowMinimap] = useState(true);
  const [ctxMenu, setCtxMenu] = useState<CtxMenu | null>(null);

  // Drag state
  const [dragging, setDragging] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });

  // Selection rectangle
  const [selRect, setSelRect] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null);
  const [selRectStart, setSelRectStart] = useState<{ x: number; y: number } | null>(null);

  // Linking state: user clicked a pin, waiting for second pin
  const [linkingFrom, setLinkingFrom] = useState<{ nodeId: string; pin: PinKind; x: number; y: number } | null>(null);
  const [linkingMouse, setLinkingMouse] = useState<{ x: number; y: number } | null>(null);

  // Dirty tracking: snapshot of last saved state
  const [savedSnap, setSavedSnap] = useState<string>('');

  // Toast notification
  const [toast, setToast] = useState<{ ok: boolean; message: string } | null>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const showToast = useCallback((ok: boolean, message: string) => {
    setToast({ ok, message });
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 5000);
  }, []);

  // Undo/redo
  const mkSnap = useCallback((): Snapshot => ({
    nodes: JSON.parse(JSON.stringify(nodes)),
    links: JSON.parse(JSON.stringify(links)),
    nc,
  }), [nodes, links, nc]);

  const [history, setHistory] = useState<Snapshot[]>(() => [{ nodes: JSON.parse(JSON.stringify(defaultGraph.nodes)), links: JSON.parse(JSON.stringify(defaultGraph.links)), nc: DEFAULT_YAML.nc }]);
  const [historyIdx, setHistoryIdx] = useState(0);

  const canvasRef = useRef<HTMLDivElement>(null);
  const pinClickedRef = useRef(false);
  const selRectDoneRef = useRef(false);

  const pushHistory = useCallback((snap: Snapshot) => {
    setHistory(prev => {
      const trimmed = prev.slice(0, historyIdx + 1);
      const next = [...trimmed, JSON.parse(JSON.stringify(snap))];
      if (next.length > MAX_HISTORY) next.shift();
      return next;
    });
    setHistoryIdx(prev => Math.min(prev + 1, MAX_HISTORY - 1));
  }, [historyIdx]);

  const commit = useCallback((newNodes: GNode[], newLinks: GLink[], newNc?: number) => {
    setNodes(newNodes); setLinks(newLinks);
    if (newNc !== undefined) setNc(newNc);
    pushHistory({ nodes: JSON.parse(JSON.stringify(newNodes)), links: JSON.parse(JSON.stringify(newLinks)), nc: newNc ?? nc });
  }, [pushHistory, nc]);

  const undo = useCallback(() => {
    if (historyIdx <= 0) return;
    const snap = history[historyIdx - 1];
    if (!snap) return;
    setNodes(JSON.parse(JSON.stringify(snap.nodes)));
    setLinks(JSON.parse(JSON.stringify(snap.links)));
    setNc(snap.nc);
    setHistoryIdx(historyIdx - 1);
    setSelectedNodes(new Set()); setSelectedLink(null);
  }, [history, historyIdx]);

  const redo = useCallback(() => {
    if (historyIdx >= history.length - 1) return;
    const snap = history[historyIdx + 1];
    if (!snap) return;
    setNodes(JSON.parse(JSON.stringify(snap.nodes)));
    setLinks(JSON.parse(JSON.stringify(snap.links)));
    setNc(snap.nc);
    setHistoryIdx(historyIdx + 1);
    setSelectedNodes(new Set()); setSelectedLink(null);
  }, [history, historyIdx]);

  useEffect(() => {
    Promise.all([api.listModels(), api.getModuleCatalog()])
      .then(([m, c]) => { setModels(m ?? []); setCatalog(c ?? []); })
      .finally(() => setLoading(false));
  }, []);

  const refreshModels = () => api.listModels().then(m => setModels(m ?? [])).catch(() => {});

  const removeModel = async (id: string) => {
    try {
      await api.deleteModel(id);
      if (modelId === id) { setModelId(null); setModelName(''); }
      refreshModels();
    } catch { /* ignore */ }
  };

  const handleImported = async (importedModelId: string) => {
    try {
      const rec = await api.loadModel(importedModelId);
      const g = yamlToGraph(rec.yaml_def as ModelYAML);
      setNodes(g.nodes);
      setLinks(g.links);
      setNc(rec.yaml_def.nc || 80);
      setScales((rec.yaml_def.scales as Record<string, number[]>) ?? {});
      setModelId(importedModelId);
      setModelName(rec.name);
      setSavedSnap(JSON.stringify({ nodes: g.nodes, links: g.links, nc: rec.yaml_def.nc || 80 }));
      refreshModels();
    } catch (err) {
      console.error('Failed to load imported model:', err);
    }
  };

  const newModel = () => {
    const g = yamlToGraph(DEFAULT_YAML);
    setNodes(g.nodes); setLinks(g.links); setNc(DEFAULT_YAML.nc); setScales({});
    setModelId(null); setModelName('');
    setSavedSnap('');
    setSelectedNodes(new Set()); setSelectedLink(null);
    pushHistory({ nodes: JSON.parse(JSON.stringify(g.nodes)), links: JSON.parse(JSON.stringify(g.links)), nc: DEFAULT_YAML.nc });
  };

  const quickSave = async () => {
    if (!modelId || !modelName) { setShowSave(true); return; }
    const v = buildYaml(nodes, links, nc, scales);
    if (!v.ok || !v.yaml) { setShowSave(true); return; }
    try {
      const graphMeta = { _nodes: nodes, _links: links };
      const payload = { ...v.yaml, _graph: graphMeta } as unknown as Record<string, unknown>;
      await api.saveModel({ name: modelName, yaml_def: payload }, true);
      setSavedSnap(JSON.stringify({ nodes, links, nc }));
      refreshModels();
      // Validate with YOLO
      try {
        const val = await api.validateModel(modelId);
        if (val.valid) {
          showToast(true, `Saved ✓ ${val.params.toLocaleString()} params · ${val.layers} layers`);
        } else {
          showToast(false, `Saved but build failed: ${val.message}`);
        }
      } catch {
        showToast(true, 'Saved (validation skipped)');
      }
    } catch (err) {
      showToast(false, `Save failed: ${err}`);
    }
  };

  /* ── Derived ────────────────────────────────────────────────────────── */
  const nodeMap = useMemo(() => new Map(nodes.map(n => [n.id, n])), [nodes]);
  const singleSel = selectedNodes.size === 1 ? nodeMap.get([...selectedNodes][0]!) ?? null : null;
  const selLinkObj = selectedLink ? links.find(l => l.id === selectedLink) ?? null : null;

  // Validation status for toolbar indicator
  const validation = useMemo(() => buildYaml(nodes, links, nc, scales), [nodes, links, nc, scales]);
  const bbStart = nodes.find(n => n.role === 'backbone-start');
  const hdStart = nodes.find(n => n.role === 'head-start');

  // Check which pins are occupied
  const hasSeqIn = useMemo(() => new Set(links.filter(l => l.type === 'seq').map(l => l.to)), [links]);
  const hasSeqOut = useMemo(() => new Set(links.filter(l => l.type === 'seq').map(l => l.from)), [links]);

  // Backbone / Head chain node sets (for coloring)
  // Backbone stops at head-start; head stops at backbone-start
  const bbChainSet = useMemo(() => {
    if (!bbStart) return new Set<string>();
    const stop = hdStart ? new Set([hdStart.id]) : undefined;
    return new Set(walkSeqChain(bbStart.id, nodes, links, stop));
  }, [bbStart, hdStart, nodes, links]);
  const hdChainSet = useMemo(() => {
    if (!hdStart) return new Set<string>();
    const stop = bbStart ? new Set([bbStart.id]) : undefined;
    return new Set(walkSeqChain(hdStart.id, nodes, links, stop));
  }, [hdStart, bbStart, nodes, links]);

  /* ── Node operations ────────────────────────────────────────────────── */
  const updateNode = useCallback((id: string, patch: Partial<GNode>) => {
    const newNodes = nodes.map(n => n.id === id ? { ...n, ...patch } : n);
    commit(newNodes, links);
  }, [nodes, links, commit]);

  const deleteNodeIds = useCallback((ids: string[]) => {
    if (ids.length === 0) return;
    const idSet = new Set(ids);
    const newNodes = nodes.filter(n => !idSet.has(n.id));
    const newLinks = links.filter(l => !idSet.has(l.from) && !idSet.has(l.to));
    commit(newNodes, newLinks);
    setSelectedNodes(new Set()); setSelectedLink(null);
  }, [nodes, links, commit]);

  const duplicateNodeIds = useCallback((ids: string[]) => {
    if (ids.length === 0) return;
    const newNodes = [...nodes];
    for (const id of ids) {
      const orig = nodeMap.get(id);
      if (!orig) continue;
      newNodes.push({ ...JSON.parse(JSON.stringify(orig)), id: uid(), role: 'normal', x: orig.x + 30, y: orig.y + 30 });
    }
    commit(newNodes, links);
  }, [nodes, links, commit, nodeMap]);

  const addNode = useCallback((moduleName: string, dropPos?: { x: number; y: number }) => {
    const catMod = catalog.find(m => m.name === moduleName);
    // Detector singleton: only one detector block allowed
    if (catMod?.category === 'detector') {
      const existing = nodes.find(n => n.role === 'detector');
      if (existing) {
        showToast(false, `Only one Detector block allowed. Remove the existing "${existing.module}" first.`);
        return;
      }
    }
    const defaultArgs = catMod?.args.filter(a => a.default !== undefined).map(a => a.default) ?? [];
    const lastY = nodes.length > 0 ? Math.max(...nodes.map(n => n.y)) : 0;
    const pos = dropPos ?? { x: INIT_X, y: lastY + NODE_H + NODE_GAP_Y };
    const role: NodeRole = catMod?.category === 'detector' ? 'detector' : 'normal';
    const newNode: GNode = { id: uid(), module: moduleName, repeats: 1, args: defaultArgs, role, x: pos.x, y: pos.y };
    commit([...nodes, newNode], links);
  }, [catalog, nodes, links, commit, showToast]);

  const setNodeRole = useCallback((id: string, role: NodeRole) => {
    const target = nodes.find(n => n.id === id);
    // Prevent changing detector node role (it's permanently 'detector')
    if (target?.role === 'detector') return;
    const newNodes = nodes.map(n => {
      if (n.id === id) return { ...n, role };
      // Clear same role from other nodes (only one backbone-start, one head-start)
      if (role !== 'normal' && n.role === role) return { ...n, role: 'normal' as NodeRole };
      return n;
    });
    commit(newNodes, links);
  }, [nodes, links, commit]);

  const deleteLink = useCallback((linkIdVal: string) => {
    commit(nodes, links.filter(l => l.id !== linkIdVal));
    setSelectedLink(null);
  }, [nodes, links, commit]);

  /* ── Pin click → create link ────────────────────────────────────────── */
  const canConnect = useCallback((fromNodeId: string, fromPin: PinKind, toNodeId: string, toPin: PinKind): boolean => {
    if (fromNodeId === toNodeId) return false;
    // seq-out → seq-in
    if (fromPin === 'seq-out' && toPin === 'seq-in') {
      if (hasSeqOut.has(fromNodeId) || hasSeqIn.has(toNodeId)) return false;
      return true;
    }
    // skip-out → skip-in
    if (fromPin === 'skip-out' && toPin === 'skip-in') return true;
    // Also allow reverse: seq-in clicked first, then seq-out
    if (fromPin === 'seq-in' && toPin === 'seq-out') {
      if (hasSeqIn.has(fromNodeId) || hasSeqOut.has(toNodeId)) return false;
      return true;
    }
    if (fromPin === 'skip-in' && toPin === 'skip-out') return true;
    return false;
  }, [hasSeqIn, hasSeqOut]);

  const onPinClick = useCallback((e: React.MouseEvent, nodeId: string, pin: PinKind) => {
    e.stopPropagation();
    pinClickedRef.current = true;
    const node = nodeMap.get(nodeId);
    if (!node) return;

    // Compute pin world position
    let px: number, py: number;
    if (pin === 'seq-in') { px = node.x + NODE_W / 2; py = node.y; }
    else if (pin === 'seq-out') { px = node.x + NODE_W / 2; py = node.y + NODE_H; }
    else if (pin === 'skip-in') { px = node.x; py = node.y + NODE_H / 2; }
    else { px = node.x + NODE_W; py = node.y + NODE_H / 2; }

    if (!linkingFrom) {
      setLinkingFrom({ nodeId, pin, x: px, y: py });
      setLinkingMouse({ x: px, y: py });
      return;
    }

    // Second click — try to connect
    if (canConnect(linkingFrom.nodeId, linkingFrom.pin, nodeId, pin)) {
      let fromId = linkingFrom.nodeId, toId = nodeId;
      let type: LinkType = 'seq';
      if (linkingFrom.pin === 'seq-out' && pin === 'seq-in') { type = 'seq'; }
      else if (linkingFrom.pin === 'seq-in' && pin === 'seq-out') { type = 'seq'; fromId = nodeId; toId = linkingFrom.nodeId; }
      else if (linkingFrom.pin === 'skip-out' && pin === 'skip-in') { type = 'skip'; }
      else if (linkingFrom.pin === 'skip-in' && pin === 'skip-out') { type = 'skip'; fromId = nodeId; toId = linkingFrom.nodeId; }
      // Check duplicate
      const exists = links.some(l => l.from === fromId && l.to === toId && l.type === type);
      if (!exists) {
        commit(nodes, [...links, { id: linkId(), from: fromId, to: toId, type }]);
      }
    }
    setLinkingFrom(null); setLinkingMouse(null);
  }, [linkingFrom, canConnect, links, nodes, commit, nodeMap]);

  /* ── Load model ─────────────────────────────────────────────────────── */
  const loadModel = async (id: string) => {
    try {
      const m = await api.loadModel(id);
      const yamlDef = m.yaml_def as ModelYAML & { _graph?: { _nodes?: GNode[]; _links?: GLink[] } };
      let loadedNodes: GNode[];
      let loadedLinks: GLink[];
      // New format: _graph._nodes has full node data (module field present)
      const hasFullGraph = yamlDef._graph?._nodes?.length && yamlDef._graph?._links
        && yamlDef._graph._nodes[0]?.module;
      if (hasFullGraph) {
        loadedNodes = yamlDef._graph!._nodes!.map(n => ({
          id: n.id ?? uid(),
          module: n.module,
          repeats: n.repeats ?? 1,
          args: Array.isArray(n.args) ? n.args : [],
          role: n.role ?? 'normal',
          x: n.x ?? INIT_X,
          y: n.y ?? INIT_Y,
        }));
        const nodeIdSet = new Set(loadedNodes.map(n => n.id));
        loadedLinks = yamlDef._graph!._links!.filter(l => nodeIdSet.has(l.from) && nodeIdSet.has(l.to));
      } else {
        // Fallback: rebuild graph from YAML (old format or no _graph)
        // Use old _graph positions if available
        const oldPositions = yamlDef._graph?._nodes?.map(n => ({ x: n.x ?? INIT_X, y: n.y ?? INIT_Y }));
        const graph = yamlToGraph(yamlDef, oldPositions);
        loadedNodes = graph.nodes;
        loadedLinks = graph.links;
      }
      setNodes(loadedNodes); setLinks(loadedLinks); setNc(yamlDef.nc);
      setScales((yamlDef.scales as Record<string, number[]>) ?? {});
      setSavedSnap(JSON.stringify({ nodes: loadedNodes, links: loadedLinks, nc: yamlDef.nc }));
      pushHistory({ nodes: JSON.parse(JSON.stringify(loadedNodes)), links: JSON.parse(JSON.stringify(loadedLinks)), nc: yamlDef.nc });
      setModelId(m.model_id); setModelName(m.name);
      setSelectedNodes(new Set()); setSelectedLink(null);
    } catch (err) { console.error('loadModel failed:', err); }
  };

  /* ── Keyboard handler ───────────────────────────────────────────────── */
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        if (selectedLink) deleteLink(selectedLink);
        else if (selectedNodes.size > 0) deleteNodeIds([...selectedNodes]);
      }
      if (e.key === 'Escape') { setSelectedNodes(new Set()); setSelectedLink(null); setCtxMenu(null); setLinkingFrom(null); setLinkingMouse(null); }
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) { e.preventDefault(); undo(); }
      if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) { e.preventDefault(); redo(); }
      if ((e.ctrlKey || e.metaKey) && e.key === 'a') { e.preventDefault(); setSelectedNodes(new Set(nodes.map(n => n.id))); }
      if ((e.ctrlKey || e.metaKey) && e.key === 'd' && selectedNodes.size > 0) { e.preventDefault(); duplicateNodeIds([...selectedNodes]); }
      if ((e.ctrlKey || e.metaKey) && e.key === 's' && !e.shiftKey) { e.preventDefault(); quickSave(); }
      if ((e.ctrlKey || e.metaKey) && e.key === 's' && e.shiftKey) { e.preventDefault(); setShowSave(true); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [selectedLink, selectedNodes, deleteLink, deleteNodeIds, undo, redo, nodes, duplicateNodeIds, quickSave]);

  /* ── Mouse handlers ─────────────────────────────────────────────────── */
  const toCanvas = useCallback((clientX: number, clientY: number) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    return { x: (clientX - rect.left - pan.x) / zoom, y: (clientY - rect.top - pan.y) / zoom };
  }, [zoom, pan]);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button === 2) return;
    setCtxMenu(null);
    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      setIsPanning(true); setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y }); e.preventDefault();
    } else if (e.button === 0 && e.shiftKey && !linkingFrom) {
      const pos = toCanvas(e.clientX, e.clientY);
      setSelRectStart(pos); setSelRect({ x1: pos.x, y1: pos.y, x2: pos.x, y2: pos.y });
    }
  }, [pan, toCanvas, linkingFrom]);

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (isPanning) { setPan({ x: e.clientX - panStart.x, y: e.clientY - panStart.y }); return; }
    if (linkingFrom) { setLinkingMouse(toCanvas(e.clientX, e.clientY)); return; }
    if (selRectStart) { const pos = toCanvas(e.clientX, e.clientY); setSelRect({ x1: selRectStart.x, y1: selRectStart.y, x2: pos.x, y2: pos.y }); return; }
    if (dragging !== null) {
      const pos = toCanvas(e.clientX, e.clientY);
      const dragNode = nodeMap.get(dragging);
      if (!dragNode) return;
      const dx = pos.x - dragOffset.x - dragNode.x;
      const dy = pos.y - dragOffset.y - dragNode.y;
      if (selectedNodes.has(dragging) && selectedNodes.size > 1) {
        setNodes(prev => prev.map(n => selectedNodes.has(n.id) ? { ...n, x: n.x + dx, y: n.y + dy } : n));
        setDragOffset({ x: pos.x - dragNode.x - dx, y: pos.y - dragNode.y - dy });
      } else {
        setNodes(prev => prev.map(n => n.id === dragging ? { ...n, x: pos.x - dragOffset.x, y: pos.y - dragOffset.y } : n));
      }
    }
  }, [isPanning, panStart, dragging, dragOffset, toCanvas, selRectStart, linkingFrom, nodeMap, selectedNodes]);

  const onMouseUp = useCallback(() => {
    if (dragging !== null) pushHistory(mkSnap());
    if (selRect) {
      const x1 = Math.min(selRect.x1, selRect.x2), x2 = Math.max(selRect.x1, selRect.x2);
      const y1 = Math.min(selRect.y1, selRect.y2), y2 = Math.max(selRect.y1, selRect.y2);
      const sel = new Set<string>();
      nodes.forEach(n => { if (n.x + NODE_W > x1 && n.x < x2 && n.y + NODE_H > y1 && n.y < y2) sel.add(n.id); });
      setSelectedNodes(sel);
      selRectDoneRef.current = true;
    }
    setDragging(null); setIsPanning(false); setSelRect(null); setSelRectStart(null);
  }, [dragging, selRect, nodes, mkSnap, pushHistory]);

  const onNodeMouseDown = useCallback((e: React.MouseEvent, nodeId: string) => {
    if (e.button !== 0 || e.altKey || linkingFrom) return;
    e.stopPropagation();
    setCtxMenu(null);
    const node = nodeMap.get(nodeId);
    if (!node) return;
    const pos = toCanvas(e.clientX, e.clientY);
    setDragOffset({ x: pos.x - node.x, y: pos.y - node.y });
    setDragging(nodeId);
    if (e.shiftKey) {
      setSelectedNodes(prev => { const n = new Set(prev); n.has(nodeId) ? n.delete(nodeId) : n.add(nodeId); return n; });
    } else if (!selectedNodes.has(nodeId)) {
      setSelectedNodes(new Set([nodeId]));
    }
    setSelectedLink(null);
  }, [nodeMap, toCanvas, selectedNodes, linkingFrom]);

  const onNodeContextMenu = useCallback((e: React.MouseEvent, nodeId: string) => {
    e.preventDefault(); e.stopPropagation();
    if (!selectedNodes.has(nodeId)) setSelectedNodes(new Set([nodeId]));
    setCtxMenu({ x: e.clientX, y: e.clientY, nodeId });
  }, [selectedNodes]);

  /* ── Zoom ────────────────────────────────────────────────────────────── */
  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    setZoom(z => Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, z + (e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP))));
  }, []);
  const zoomIn = () => setZoom(z => Math.min(ZOOM_MAX, z + ZOOM_STEP));
  const zoomOut = () => setZoom(z => Math.max(ZOOM_MIN, z - ZOOM_STEP));
  const fitView = () => {
    if (nodes.length === 0) return;
    const xs = nodes.map(n => n.x), ys = nodes.map(n => n.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs) + NODE_W;
    const minY = Math.min(...ys), maxY = Math.max(...ys) + NODE_H;
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const newZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, Math.min(rect.width / (maxX - minX + 80), rect.height / (maxY - minY + 80))));
    setZoom(newZoom); setPan({ x: -minX * newZoom + 40, y: -minY * newZoom + 40 });
  };

  /* ── Catalog filtering ──────────────────────────────────────────────── */
  const filteredCatalog = useMemo(() => {
    if (!catalogSearch) return catalog;
    const q = catalogSearch.toLowerCase();
    return catalog.filter(m => m.name.toLowerCase().includes(q) || m.category.toLowerCase().includes(q));
  }, [catalog, catalogSearch]);
  const categories = useMemo(() => [...new Set(filteredCatalog.map(m => m.category))].sort(), [filteredCatalog]);
  const toggleCat = (cat: string) => { setCollapsedCats(prev => { const n = new Set(prev); n.has(cat) ? n.delete(cat) : n.add(cat); return n; }); };

  /* ── Bounds for minimap ─────────────────────────────────────────────── */
  const bounds = useMemo(() => {
    if (nodes.length === 0) return { x: 0, y: 0, w: 400, h: 400 };
    const xs = nodes.map(n => n.x), ys = nodes.map(n => n.y);
    return { x: Math.min(...xs) - 20, y: Math.min(...ys) - 20, w: Math.max(...xs) + NODE_W - Math.min(...xs) + 40, h: Math.max(...ys) + NODE_H - Math.min(...ys) + 40 };
  }, [nodes]);

  /* ── Pin position helpers ───────────────────────────────────────────── */
  const pinPos = (n: GNode, pin: PinKind) => {
    if (pin === 'seq-in') return { x: n.x + NODE_W / 2, y: n.y };
    if (pin === 'seq-out') return { x: n.x + NODE_W / 2, y: n.y + NODE_H };
    if (pin === 'skip-in') return { x: n.x, y: n.y + NODE_H / 2 };
    return { x: n.x + NODE_W, y: n.y + NODE_H / 2 };
  };

  if (loading) {
    return <div className="flex-1 flex items-center justify-center"><Loader2 size={32} className="text-blue-500 animate-spin" /></div>;
  }

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-hidden select-none" onContextMenu={e => e.preventDefault()}>
      {/* ── Toolbar ─────────────────────────────────────────────────────── */}
      <div className="shrink-0 h-11 border-b border-slate-800 bg-slate-950/80 backdrop-blur-sm flex items-center px-4 gap-2">
        <button onClick={onBack} className="text-slate-400 hover:text-white transition-colors cursor-pointer mr-1"><ArrowLeft size={18} /></button>
        <h1 className="text-sm font-semibold text-white truncate">
          {modelName || 'Untitled Model'}
        </h1>
        {modelId && <span className="text-[10px] text-slate-600 font-mono truncate ml-1">({modelId.slice(0, 8)})</span>}

        <div className="w-px h-5 bg-slate-700/50 mx-1" />

        <button onClick={newModel} title="New Model"
          className="flex items-center gap-1 px-2 py-1 text-[11px] text-slate-400 hover:text-white hover:bg-slate-800 rounded-md transition-colors cursor-pointer">
          <FilePlus2 size={13} /> New
        </button>

        <button onClick={() => setShowImport(true)} title="Import YAML"
          className="flex items-center gap-1 px-2 py-1 text-[11px] text-slate-400 hover:text-white hover:bg-slate-800 rounded-md transition-colors cursor-pointer">
          <Upload size={13} /> Import
        </button>

        <div className="flex-1" />

        {/* Undo / Redo */}
        <button onClick={undo} disabled={historyIdx <= 0} title="Undo (Ctrl+Z)"
          className="text-slate-400 hover:text-white p-1 cursor-pointer disabled:opacity-30 disabled:cursor-default"><Undo2 size={15} /></button>
        <button onClick={redo} disabled={historyIdx >= history.length - 1} title="Redo (Ctrl+Y)"
          className="text-slate-400 hover:text-white p-1 cursor-pointer disabled:opacity-30 disabled:cursor-default"><Redo2 size={15} /></button>

        <div className="w-px h-5 bg-slate-700/50 mx-1" />

        {/* Edge style */}
        <div className="flex items-center bg-slate-800/60 rounded-lg border border-slate-700/50 p-0.5 gap-0.5">
          {(['bezier', 'step', 'straight'] as EdgeStyle[]).map(s => (
            <button key={s} onClick={() => setEdgeStyle(s)}
              className={`px-2 py-1 text-[10px] rounded-md transition-colors cursor-pointer capitalize
                ${edgeStyle === s ? 'bg-indigo-600/30 text-indigo-300' : 'text-slate-500 hover:text-slate-300'}`}>
              {s}
            </button>
          ))}
        </div>

        <div className="w-px h-5 bg-slate-700/50 mx-1" />

        <button onClick={zoomOut} className="text-slate-400 hover:text-white p-1 cursor-pointer"><ZoomOut size={15} /></button>
        <span className="text-[10px] text-slate-500 w-10 text-center tabular-nums">{Math.round(zoom * 100)}%</span>
        <button onClick={zoomIn} className="text-slate-400 hover:text-white p-1 cursor-pointer"><ZoomIn size={15} /></button>
        <button onClick={fitView} className="text-slate-400 hover:text-white p-1 cursor-pointer" title="Fit view"><Maximize2 size={15} /></button>

        <div className="w-px h-5 bg-slate-700/50 mx-1" />

        <button onClick={() => setShowMinimap(p => !p)}
          className={`px-2 py-1 text-[10px] rounded-md transition-colors cursor-pointer ${showMinimap ? 'text-indigo-400' : 'text-slate-500'}`}>Map</button>

        <div className="flex items-center gap-1 ml-2">
          <label className="text-[10px] text-slate-500">nc</label>
          <input type="number" min={1} value={nc} onChange={e => setNc(Math.max(1, Number(e.target.value)))}
            className="w-12 bg-slate-800 border border-slate-700 rounded px-1.5 py-0.5 text-[11px] text-white font-mono text-center focus:outline-none focus:border-indigo-500" />
        </div>

        {/* Status indicators */}
        {modelId && <span className={`text-[10px] ml-1 ${savedSnap === JSON.stringify({ nodes, links, nc }) ? 'text-green-400' : 'text-amber-400'}`}>
          {savedSnap === JSON.stringify({ nodes, links, nc }) ? '● Saved' : '● Unsaved'}
        </span>}
        {!modelId && <span className="text-[10px] text-slate-500 ml-1">New model</span>}
        <span className="text-[10px] text-slate-600 ml-1">{nodes.length} nodes</span>
        {!bbStart && <span className="text-[10px] text-amber-500 ml-1">⚠ No BB start</span>}
        {!hdStart && <span className="text-[10px] text-amber-500 ml-1">⚠ No Head start</span>}
        {selectedNodes.size > 1 && <span className="text-[10px] text-indigo-400 ml-1">{selectedNodes.size} selected</span>}
        {linkingFrom && <span className="text-[10px] text-cyan-400 ml-1 animate-pulse">Linking…</span>}

        <button onClick={quickSave} title={modelId ? 'Save (overwrite)' : 'Save new'}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white text-[11px] font-medium rounded-lg transition-colors cursor-pointer ml-2">
          <Save size={12} /> Save
        </button>
        <button onClick={() => setShowSave(true)} title="Save as new model"
          className="flex items-center gap-1.5 px-2.5 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 text-[11px] font-medium rounded-lg transition-colors cursor-pointer">
          Save As
        </button>

        {selectedLink && (
          <button onClick={() => deleteLink(selectedLink)}
            className="flex items-center gap-1 px-2.5 py-1.5 bg-red-600/20 hover:bg-red-600/40 text-red-400 text-[11px] rounded-lg transition-colors cursor-pointer ml-1">
            <Trash2 size={12} /> Delete Link
          </button>
        )}
      </div>

      {/* ── Body ────────────────────────────────────────────────────────── */}
      <div className="flex-1 flex min-h-0 overflow-hidden">

        {/* ── Left: Palette ─────────────────────────────────────────────── */}
        <div className="w-56 shrink-0 border-r border-slate-800/70 bg-slate-950/50 flex flex-col min-h-0">
          <div className="px-3 py-2 border-b border-slate-800/40">
            <input value={catalogSearch} onChange={e => setCatalogSearch(e.target.value)} placeholder="Search modules..."
              className="w-full bg-slate-800/40 border border-slate-700/40 rounded-lg px-2.5 py-1.5 text-[11px] text-white placeholder-slate-600 focus:outline-none focus:border-indigo-500" />
          </div>
          <div className="flex-1 overflow-y-auto p-2 space-y-0.5">
            {categories.map(cat => {
              const mods = filteredCatalog.filter(m => m.category === cat);
              const collapsed = collapsedCats.has(cat);
              const cc = CAT_CFG[cat] ?? DEF_CFG;
              return (
                <div key={cat}>
                  <button onClick={() => toggleCat(cat)}
                    className="flex items-center gap-1.5 w-full px-2 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-slate-500 hover:text-slate-300 cursor-pointer">
                    {collapsed ? <ChevronRight size={10} /> : <ChevronDown size={10} />}
                    {cat} <span className="text-slate-700">({mods.length})</span>
                  </button>
                  {!collapsed && mods.map(mod => (
                    <div key={mod.name} draggable onDragStart={e => e.dataTransfer.setData('module', mod.name)}
                      className={`flex items-center gap-2 px-2.5 py-2 rounded-lg cursor-grab active:cursor-grabbing transition-all mb-0.5
                        bg-gradient-to-r ${cc.gradient} border ${cc.border} hover:brightness-125`}>
                      <div className={`w-6 h-6 rounded-md ${cc.iconBg} flex items-center justify-center ${cc.text} shrink-0`}>{cc.icon}</div>
                      <div className="min-w-0 flex-1">
                        <div className={`text-[11px] font-mono font-medium ${cc.text} truncate`}>{mod.name}</div>
                        {mod.description && <p className="text-[9px] text-slate-500 truncate">{mod.description}</p>}
                      </div>
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        </div>

        {/* ── Center: Canvas ────────────────────────────────────────────── */}
        <div className="flex-1 relative overflow-hidden bg-[#0a0d17]"
          ref={canvasRef}
          onMouseDown={onMouseDown} onMouseMove={onMouseMove} onMouseUp={onMouseUp} onMouseLeave={onMouseUp}
          onWheel={onWheel} onDragOver={e => e.preventDefault()}
          onDrop={e => { const mod = e.dataTransfer.getData('module'); if (!mod) return; const pos = toCanvas(e.clientX, e.clientY); addNode(mod, { x: pos.x - NODE_W / 2, y: pos.y - NODE_H / 2 }); }}
          onContextMenu={e => e.preventDefault()}
          onClick={() => {
            if (pinClickedRef.current) { pinClickedRef.current = false; return; }
            if (selRectDoneRef.current) { selRectDoneRef.current = false; return; }
            if (dragging === null && !linkingFrom) { setSelectedNodes(new Set()); setSelectedLink(null); setCtxMenu(null); }
            if (linkingFrom) { setLinkingFrom(null); setLinkingMouse(null); }
          }}
          style={{ cursor: isPanning ? 'grabbing' : dragging !== null ? 'grabbing' : linkingFrom ? 'crosshair' : selRectStart ? 'crosshair' : 'default' }}
        >
          {/* Dot grid */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none">
            <defs>
              <pattern id="dotgrid" width={20 * zoom} height={20 * zoom} patternUnits="userSpaceOnUse" x={pan.x % (20 * zoom)} y={pan.y % (20 * zoom)}>
                <circle cx={1} cy={1} r={0.8} fill="#1e293b" />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#dotgrid)" />
          </svg>

          {/* Transformed layer */}
          <div style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, transformOrigin: '0 0' }}>
            {/* SVG edges */}
            <svg className="absolute inset-0 pointer-events-none" style={{ width: bounds.w + bounds.x + 200, height: bounds.h + bounds.y + 200, overflow: 'visible' }}>
              <defs>
                <marker id="arrowSkip" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="6" markerHeight="6" orient="auto">
                  <path d="M 0 0 L 8 4 L 0 8 z" fill="#06b6d4" opacity="0.8" />
                </marker>
              </defs>
              {links.map(link => {
                const fromNode = nodeMap.get(link.from);
                const toNode = nodeMap.get(link.to);
                if (!fromNode || !toNode) return null;
                const isSeq = link.type === 'seq';
                const fp = isSeq ? pinPos(fromNode, 'seq-out') : pinPos(fromNode, 'skip-out');
                const tp = isSeq ? pinPos(toNode, 'seq-in') : pinPos(toNode, 'skip-in');
                const pathFn = isSeq ? seqEdgePath : skipEdgePath;
                const d = pathFn(edgeStyle, fp.x, fp.y, tp.x, tp.y);
                const isSel = selectedLink === link.id;
                // Color seq links by chain: purple=backbone, green=head
                const seqColor = bbChainSet.has(link.from) ? '#a855f7' : hdChainSet.has(link.from) ? '#22c55e' : '#334155';
                const color = isSeq ? seqColor : '#06b6d4';
                return (
                  <g key={link.id}>
                    <path d={d} fill="none" stroke="transparent" strokeWidth={12} className="cursor-pointer pointer-events-auto"
                      onClick={e => { e.stopPropagation(); setSelectedLink(link.id); setSelectedNodes(new Set()); }} />
                    <path d={d} fill="none" stroke={isSel ? '#818cf8' : color}
                      strokeWidth={isSel ? 2.5 : isSeq ? 1.8 : 1.2}
                      strokeDasharray={isSeq ? undefined : '6 4'}
                      opacity={isSel ? 1 : isSeq ? 0.7 : 0.7}
                      markerEnd={isSeq ? undefined : 'url(#arrowSkip)'}
                      className="pointer-events-none transition-all duration-150" />
                    {isSel && <circle cx={(fp.x + tp.x) / 2} cy={(fp.y + tp.y) / 2} r={4} fill="#818cf8" className="pointer-events-none" />}
                  </g>
                );
              })}
              {/* Linking preview line */}
              {linkingFrom && linkingMouse && (
                <line x1={linkingFrom.x} y1={linkingFrom.y} x2={linkingMouse.x} y2={linkingMouse.y}
                  stroke="#06b6d4" strokeWidth={1.5} strokeDasharray="4 4" opacity={0.6} className="pointer-events-none" />
              )}
            </svg>

            {/* Nodes */}
            {nodes.map(node => {
              const cfg = getCfg(catalog, node.module);
              const isSel = selectedNodes.has(node.id);
              const inBB = bbChainSet.has(node.id);
              const inHD = hdChainSet.has(node.id);
              const roleBorder = inBB ? 'border-purple-500/60' : inHD ? 'border-green-500/60' : '';
              const nodeParams = estimateParams(node.module, safeArgs(node), node.repeats);

              return (
                <div key={node.id}
                  onMouseDown={e => onNodeMouseDown(e, node.id)}
                  onContextMenu={e => onNodeContextMenu(e, node.id)}
                  onClick={e => e.stopPropagation()}
                  className={`absolute rounded-xl border-2 select-none transition-shadow duration-150
                    bg-gradient-to-br ${cfg.gradient} backdrop-blur-sm
                    ${isSel ? `${cfg.border} shadow-lg shadow-indigo-500/10` : roleBorder || 'border-slate-700/40 hover:border-slate-600/60'}
                  `}
                  style={{ left: node.x, top: node.y, width: NODE_W, height: NODE_H, overflow: 'visible', cursor: dragging === node.id ? 'grabbing' : 'grab' }}
                >
                  <div className="flex items-center h-full px-3 gap-2.5">
                    <div className={`w-8 h-8 rounded-lg ${cfg.iconBg} flex items-center justify-center ${cfg.text} shrink-0`}>{cfg.icon}</div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span className={`text-xs font-bold font-mono ${cfg.text}`}>{node.module}</span>
                        {node.repeats > 1 && <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-white/5 text-slate-400 font-medium">x{node.repeats}</span>}
                      </div>
                      <div className="text-[10px] text-slate-500 truncate mt-0.5 font-mono">
                        [{safeArgs(node).map(a => JSON.stringify(a)).join(', ')}]
                        {nodeParams > 0 && <span className="text-slate-600 ml-1">~{formatParams(nodeParams)}</span>}
                      </div>
                    </div>
                    <div className="text-right shrink-0">
                      {(inBB || inHD) && (
                        <div className={`text-[8px] font-bold uppercase tracking-wider ${inBB ? 'text-purple-400' : 'text-green-400'}`}>
                          {inBB ? (node.role === 'backbone-start' ? 'BB' : 'BB·C') : (node.role === 'head-start' ? 'HD' : 'HD·C')}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Sequential pins (top/bottom center) — colored by chain */}
                  {(() => {
                    const seqFill = inBB ? 'bg-purple-500 border-purple-400' : inHD ? 'bg-green-500 border-green-400' : 'bg-slate-700 border-slate-500';
                    const seqHover = inBB ? 'hover:border-purple-400 hover:bg-purple-500/30' : inHD ? 'hover:border-green-400 hover:bg-green-500/30' : 'hover:border-purple-400 hover:bg-purple-500/30';
                    return <>
                      <div className={`absolute -top-2 left-1/2 -translate-x-1/2 w-4 h-4 rounded-full border-2 cursor-pointer z-10 transition-colors
                        ${hasSeqIn.has(node.id) ? seqFill : `bg-slate-700 border-slate-500 ${seqHover}`}
                        ${linkingFrom?.pin === 'seq-out' ? 'ring-2 ring-cyan-400/50' : ''}`}
                        onMouseDown={e => { e.stopPropagation(); e.preventDefault(); onPinClick(e, node.id, 'seq-in'); }}
                        onClick={e => e.stopPropagation()}
                        title="Sequential In" />
                      <div className={`absolute -bottom-2 left-1/2 -translate-x-1/2 w-4 h-4 rounded-full border-2 cursor-pointer z-10 transition-colors
                        ${hasSeqOut.has(node.id) ? seqFill : `bg-slate-700 border-slate-500 ${seqHover}`}
                        ${linkingFrom?.pin === 'seq-in' ? 'ring-2 ring-cyan-400/50' : ''}`}
                        onMouseDown={e => { e.stopPropagation(); e.preventDefault(); onPinClick(e, node.id, 'seq-out'); }}
                        onClick={e => e.stopPropagation()}
                        title="Sequential Out" />
                    </>;
                  })()}

                  {/* Skip pins (left/right sides) */}
                  <div className={`absolute top-1/2 -left-2 -translate-y-1/2 w-3.5 h-3.5 rounded-sm border-2 cursor-pointer z-10 transition-colors rotate-45
                    bg-slate-700 border-slate-500 hover:border-cyan-400 hover:bg-cyan-500/30
                    ${linkingFrom?.pin === 'skip-out' ? 'ring-2 ring-cyan-400/50' : ''}`}
                    onMouseDown={e => { e.stopPropagation(); e.preventDefault(); onPinClick(e, node.id, 'skip-in'); }}
                    onClick={e => e.stopPropagation()}
                    title="Skip In" />
                  <div className={`absolute top-1/2 -right-2 -translate-y-1/2 w-3.5 h-3.5 rounded-sm border-2 cursor-pointer z-10 transition-colors rotate-45
                    bg-slate-700 border-slate-500 hover:border-cyan-400 hover:bg-cyan-500/30
                    ${linkingFrom?.pin === 'skip-in' ? 'ring-2 ring-cyan-400/50' : ''}`}
                    onMouseDown={e => { e.stopPropagation(); e.preventDefault(); onPinClick(e, node.id, 'skip-out'); }}
                    onClick={e => e.stopPropagation()}
                    title="Skip Out" />
                </div>
              );
            })}

            {/* Selection rectangle */}
            {selRect && (
              <div className="absolute border border-indigo-500/50 bg-indigo-500/10 pointer-events-none rounded"
                style={{ left: Math.min(selRect.x1, selRect.x2), top: Math.min(selRect.y1, selRect.y2),
                  width: Math.abs(selRect.x2 - selRect.x1), height: Math.abs(selRect.y2 - selRect.y1) }} />
            )}
          </div>

          {/* Minimap */}
          {/* Total params overlay — pinned to canvas viewport */}
          {(() => {
            const total = nodes.reduce((s, n) => s + estimateParams(n.module, safeArgs(n), n.repeats), 0);
            return total > 0 ? (
              <div className="absolute top-3 right-3 bg-slate-900/80 backdrop-blur-sm border border-slate-700/50 rounded-lg px-3 py-1.5 pointer-events-none z-20">
                <div className="text-[9px] text-slate-500 uppercase tracking-wider">Est. Params</div>
                <div className="text-sm font-bold text-white font-mono">{formatParams(total)}</div>
              </div>
            ) : null;
          })()}

          {showMinimap && nodes.length > 0 && (
            <div className="absolute bottom-3 right-3 w-40 h-28 bg-slate-900/90 border border-slate-700/50 rounded-lg overflow-hidden pointer-events-none">
              <svg viewBox={`${bounds.x} ${bounds.y} ${bounds.w} ${bounds.h}`} className="w-full h-full">
                {links.map(link => {
                  const fn = nodeMap.get(link.from), tn = nodeMap.get(link.to);
                  if (!fn || !tn) return null;
                  return <line key={link.id} x1={fn.x + NODE_W / 2} y1={fn.y + NODE_H} x2={tn.x + NODE_W / 2} y2={tn.y} stroke={link.type === 'seq' ? '#334155' : '#06b6d4'} strokeWidth={2} />;
                })}
                {nodes.map(n => {
                  const hex = CAT_HEX[catalog.find(c => c.name === n.module)?.category ?? ''] ?? '#64748b';
                  return <rect key={n.id} x={n.x} y={n.y} width={NODE_W} height={NODE_H} rx={4} fill={hex} opacity={0.5} />;
                })}
              </svg>
            </div>
          )}
        </div>

        {/* ── Right: Properties / Models ────────────────────────────────── */}
        <div className="w-72 shrink-0 border-l border-slate-800/70 bg-slate-950/50 flex flex-col min-h-0">
          {singleSel ? (
            <div className="flex-1 overflow-y-auto">
              <PropertiesPanel
                node={singleSel}
                catalog={catalog}
                onChange={patch => updateNode(singleSel.id, patch)}
                onClose={() => setSelectedNodes(new Set())}
              />
            </div>
          ) : selLinkObj ? (
            <div className="p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <GitBranch size={14} className="text-indigo-400" />
                  <span className="text-sm font-semibold text-white">Link</span>
                </div>
                <button onClick={() => setSelectedLink(null)} className="text-slate-500 hover:text-white cursor-pointer"><X size={16} /></button>
              </div>
              <div className="bg-slate-800/50 rounded-lg p-3 space-y-2 text-xs">
                <div className="flex justify-between"><span className="text-slate-500">Type</span><span className={`font-medium ${selLinkObj.type === 'seq' ? 'text-indigo-400' : 'text-cyan-400'}`}>{selLinkObj.type === 'seq' ? 'Sequential' : 'Skip'}</span></div>
                <div className="flex justify-between"><span className="text-slate-500">From</span><span className="text-white font-mono">{nodeMap.get(selLinkObj.from)?.module ?? '?'}</span></div>
                <div className="flex justify-between"><span className="text-slate-500">To</span><span className="text-white font-mono">{nodeMap.get(selLinkObj.to)?.module ?? '?'}</span></div>
              </div>
              <button onClick={() => deleteLink(selLinkObj.id)}
                className="w-full py-2 bg-red-600/20 hover:bg-red-600/30 text-red-400 text-xs font-medium rounded-lg transition-colors cursor-pointer flex items-center justify-center gap-1.5">
                <Trash2 size={12} /> Delete Link
              </button>
            </div>
          ) : (
            <div className="flex-1 overflow-y-auto">
              <div className="p-3 border-b border-slate-800/40">
                <h2 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Saved Models ({models.length})</h2>
              </div>
              <div className="p-2 space-y-1">
                {models.length === 0 ? (
                  <p className="text-xs text-slate-600 text-center py-8">No models yet</p>
                ) : models.map(m => (
                  <div key={m.model_id}
                    className={`flex items-center gap-1 px-3 py-2.5 rounded-lg transition-colors
                      ${modelId === m.model_id ? 'bg-indigo-600/15 border border-indigo-500/30' : 'bg-slate-800/20 hover:bg-slate-800/50 border border-transparent'}`}>
                    <button onClick={() => loadModel(m.model_id)} className="flex-1 min-w-0 text-left cursor-pointer">
                      <div className="text-xs text-slate-300 font-medium truncate">{m.name}</div>
                      <div className="flex items-center gap-2 mt-0.5">
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400">{m.task}</span>
                        <span className="text-[10px] text-slate-600">{m.layer_count} layers</span>
                        <span className="text-[10px] text-slate-700 font-mono">{m.model_id.slice(0, 8)}</span>
                      </div>
                    </button>
                    <button onClick={e => { e.stopPropagation(); removeModel(m.model_id); }}
                      className="shrink-0 p-1 text-slate-600 hover:text-red-400 transition-colors cursor-pointer rounded hover:bg-red-500/10"
                      title="Delete model">
                      <Trash2 size={12} />
                    </button>
                  </div>
                ))}
              </div>
              <div className="p-3 border-t border-slate-800/40">
                <div className="text-[10px] text-slate-600 text-center space-y-1">
                  <p>Drag modules from palette to canvas</p>
                  <p>Click pins to link nodes (● seq, ◆ skip)</p>
                  <p>Right-click to set Backbone/Head start</p>
                  <p className="text-slate-700">Ctrl+S save, Ctrl+Shift+S save as</p>
                  <p className="text-slate-700">Ctrl+Z undo, Ctrl+Y redo, Ctrl+A select all</p>
                  <p className="text-slate-700">Ctrl+D duplicate, Alt+Drag pan, Scroll zoom</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Context Menu ────────────────────────────────────────────────── */}
      {ctxMenu && ctxMenu.nodeId !== undefined && (
        <ContextMenuPopup menu={ctxMenu}
          onDelete={() => deleteNodeIds(ctxMenu.nodeId ? [...selectedNodes] : [])}
          onDuplicate={() => duplicateNodeIds(ctxMenu.nodeId ? [...selectedNodes] : [])}
          onSetRole={(role) => { if (ctxMenu.nodeId) setNodeRole(ctxMenu.nodeId, role); }}
          onClose={() => setCtxMenu(null)}
          isDetector={ctxMenu.nodeId ? nodes.find(n => n.id === ctxMenu.nodeId)?.role === 'detector' : false}
        />
      )}

      {/* ── Save Modal ──────────────────────────────────────────────────── */}
      {showSave && (
        <SaveModal nodes={nodes} links={links} nc={nc} scales={scales} existingId={modelId}
          onSaved={(id, savedName) => { setModelId(id); setModelName(savedName); setShowSave(false); refreshModels(); setSavedSnap(JSON.stringify({ nodes, links, nc })); }}
          onClose={() => setShowSave(false)}
        />
      )}

      {/* ── Import YAML Modal ───────────────────────────────────────────── */}
      {showImport && (
        <ImportYAMLModal
          onClose={() => setShowImport(false)}
          onImported={handleImported}
        />
      )}

      {/* ── Toast notification ────────────────────────────────────────── */}
      {toast && (
        <div className={`fixed bottom-6 left-1/2 -translate-x-1/2 z-50 px-4 py-2.5 rounded-lg border shadow-xl backdrop-blur-sm flex items-center gap-2 text-sm font-medium transition-all
          ${toast.ok ? 'bg-emerald-900/90 border-emerald-500/40 text-emerald-300' : 'bg-red-900/90 border-red-500/40 text-red-300'}`}>
          {toast.ok ? <Check size={14} /> : <AlertTriangle size={14} />}
          {toast.message}
          <button onClick={() => setToast(null)} className="ml-2 text-slate-400 hover:text-white cursor-pointer"><X size={12} /></button>
        </div>
      )}
    </div>
  );
}
