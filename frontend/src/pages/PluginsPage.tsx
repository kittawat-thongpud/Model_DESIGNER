import { useState, useEffect, useRef, useCallback } from 'react';
import { api } from '../services/api';
import type { ArchFamily } from '../types';
import {
  Puzzle, Download, CheckCircle2, XCircle, Loader2, RefreshCw,
  ChevronDown, ChevronUp, Terminal, GitBranch, Clock, AlertTriangle,
  Layers, Cpu,
} from 'lucide-react';

// ── Types ──────────────────────────────────────────────────────────────────────

interface InstallStatus {
  ok: boolean;
  status: 'idle' | 'installing' | 'installed' | 'failed' | 'unavailable';
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  log_tail: string[];
}

// ── Helpers ────────────────────────────────────────────────────────────────────

const STATUS_BADGE: Record<string, { cls: string; icon: React.ReactNode; label: string }> = {
  installed:   { cls: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20', icon: <CheckCircle2 size={12} />, label: 'Installed' },
  installing:  { cls: 'bg-blue-500/10 text-blue-400 border-blue-500/20',          icon: <Loader2 size={12} className="animate-spin" />, label: 'Installing…' },
  failed:      { cls: 'bg-red-500/10 text-red-400 border-red-500/20',             icon: <XCircle size={12} />, label: 'Failed' },
  idle:        { cls: 'bg-slate-500/10 text-slate-400 border-slate-500/20',       icon: <Clock size={12} />, label: 'Not installed' },
  unavailable: { cls: 'bg-slate-500/10 text-slate-400 border-slate-500/20',       icon: <AlertTriangle size={12} />, label: 'Unavailable' },
};

const TASK_BADGE: Record<string, string> = {
  detect:   'bg-violet-500/10 text-violet-400 border-violet-500/20',
  segment:  'bg-pink-500/10 text-pink-400 border-pink-500/20',
  classify: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
  pose:     'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
};

function fmtDate(iso: string | null) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString();
}

// ── Mamba-YOLO install panel ───────────────────────────────────────────────────

interface InstallPanelProps {
  status: InstallStatus | null;
  loading: boolean;
  onInstall: () => void;
  onRebuild: () => void;
  onRefresh: () => void;
}

function InstallPanel({ status, loading, onInstall, onRebuild, onRefresh }: InstallPanelProps) {
  const [logOpen, setLogOpen] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logOpen && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [status?.log_tail, logOpen]);

  const s = status?.status ?? 'unavailable';
  const badge = STATUS_BADGE[s] ?? STATUS_BADGE.unavailable;
  const canInstall = s === 'idle' || s === 'failed';
  const canRebuild = s !== 'installing' && s !== 'unavailable';
  const isInstalling = s === 'installing';

  return (
    <div className="mt-4 rounded-xl border border-slate-700/60 bg-slate-800/50 overflow-hidden">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700/60">
        <div className="flex items-center gap-3">
          <GitBranch size={15} className="text-orange-400" />
          <span className="text-sm font-medium text-white">HZAI-ZJNU/Mamba-YOLO</span>
          <span className="text-slate-500 text-xs">Clone + install selective_scan</span>
        </div>
        <div className="flex items-center gap-2">
          {status && (
            <span className={`flex items-center gap-1.5 text-xs font-medium px-2 py-0.5 rounded-full border ${badge.cls}`}>
              {badge.icon} {badge.label}
            </span>
          )}
          <button
            onClick={onRefresh}
            disabled={loading}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700 transition-colors cursor-pointer disabled:opacity-40"
            title="Refresh status"
          >
            <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="p-4 space-y-4">
        {/* Steps */}
        <div className="grid grid-cols-3 gap-3 text-xs text-slate-400">
          {[
            { n: 1, label: 'git clone --depth=1', sub: 'Full repo to data/vendor/' },
            { n: 2, label: 'pip install selective_scan/', sub: 'CUDA extension from source' },
            { n: 3, label: 'pip install einops timm', sub: 'Python dependencies' },
          ].map((step) => (
            <div key={step.n} className="flex items-start gap-2 bg-slate-900/60 rounded-lg p-2.5">
              <span className="w-5 h-5 rounded-full bg-slate-700 text-slate-300 flex items-center justify-center text-xs font-bold shrink-0">
                {step.n}
              </span>
              <div>
                <p className="font-mono text-slate-300">{step.label}</p>
                <p className="text-slate-500 mt-0.5">{step.sub}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Timestamps */}
        {status && (s === 'installed' || s === 'failed') && (
          <div className="flex gap-6 text-xs text-slate-500">
            {status.started_at && (
              <span>Started: <span className="text-slate-400">{fmtDate(status.started_at)}</span></span>
            )}
            {status.finished_at && (
              <span>Finished: <span className="text-slate-400">{fmtDate(status.finished_at)}</span></span>
            )}
          </div>
        )}

        {/* Error */}
        {s === 'failed' && status?.error && (
          <div className="flex items-start gap-2 text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg p-3">
            <XCircle size={14} className="shrink-0 mt-0.5" />
            <span>{status.error}</span>
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center gap-3">
          <button
            onClick={onInstall}
            disabled={!canInstall || loading}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed ${
              canInstall
                ? 'bg-orange-500 hover:bg-orange-400 text-white'
                : 'bg-slate-700 text-slate-400'
            }`}
          >
            {isInstalling ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <Download size={14} />
            )}
            {s === 'failed' ? 'Retry Install' : s === 'installed' ? 'Already Installed' : 'Install'}
          </button>

          <button
            onClick={onRebuild}
            disabled={!canRebuild || loading}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed bg-slate-700 text-slate-200 hover:bg-slate-600"
            title="Rebuild selective_scan and dependencies"
          >
            <RefreshCw size={14} className={isInstalling ? 'animate-spin' : ''} />
            Rebuild
          </button>

          {(status?.log_tail?.length ?? 0) > 0 && (
            <button
              onClick={() => setLogOpen((v) => !v)}
              className="flex items-center gap-1.5 px-3 py-2 text-sm text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-colors cursor-pointer"
            >
              <Terminal size={14} />
              {logOpen ? 'Hide Log' : 'Show Log'}
              {logOpen ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
            </button>
          )}
        </div>

        {/* Log viewer */}
        {logOpen && (status?.log_tail?.length ?? 0) > 0 && (
          <div
            ref={logRef}
            className="h-56 overflow-y-auto rounded-lg bg-slate-950 border border-slate-700 p-3 font-mono text-xs text-slate-300 space-y-0.5"
          >
            {status!.log_tail.map((line, i) => (
              <div
                key={i}
                className={`leading-relaxed ${
                  line.includes('FAILED') || line.includes('ERROR')
                    ? 'text-red-400'
                    : line.includes('SUCCESS') || line.includes('successfully')
                    ? 'text-emerald-400'
                    : line.includes('WARNING')
                    ? 'text-amber-400'
                    : 'text-slate-400'
                }`}
              >
                {line}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Arch Plugin Card ───────────────────────────────────────────────────────────

interface ArchCardProps {
  family: ArchFamily;
  mambaStatus: InstallStatus | null;
  mambaLoading: boolean;
  onMambaInstall: () => void;
  onMambaRebuild: () => void;
  onMambaRefresh: () => void;
}

function ArchCard({ family, mambaStatus, mambaLoading, onMambaInstall, onMambaRebuild, onMambaRefresh }: ArchCardProps) {
  const isMamba = family.family === 'mamba_yolo';
  const taskBadgeCls = TASK_BADGE[family.task_type] ?? 'bg-slate-500/10 text-slate-400 border-slate-500/20';

  return (
    <div className={`bg-slate-900 border rounded-xl p-5 ${isMamba ? 'border-orange-500/20' : 'border-slate-800'}`}>
      {/* Card header */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-3">
          <div className={`w-9 h-9 rounded-lg flex items-center justify-center shrink-0 ${isMamba ? 'bg-orange-500/10' : 'bg-indigo-500/10'}`}>
            <Puzzle size={17} className={isMamba ? 'text-orange-400' : 'text-indigo-400'} />
          </div>
          <div>
            <h3 className="text-base font-semibold text-white">{family.display_name}</h3>
            <div className="flex items-center gap-2 mt-0.5">
              <span className={`text-xs px-2 py-0.5 rounded-full border font-medium ${taskBadgeCls}`}>
                {family.task_type}
              </span>
              {!isMamba && (
                <span className="flex items-center gap-1 text-xs text-emerald-400">
                  <CheckCircle2 size={11} /> Built-in
                </span>
              )}
              {isMamba && mambaStatus && (
                <span className={`flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border font-medium ${STATUS_BADGE[mambaStatus.status]?.cls ?? ''}`}>
                  {STATUS_BADGE[mambaStatus.status]?.icon}
                  {STATUS_BADGE[mambaStatus.status]?.label}
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Scale pills */}
        {family.supported_scales.length > 0 && (
          <div className="flex items-center gap-1.5 flex-wrap">
            {family.supported_scales.map((s) => (
              <span
                key={s.scale}
                className="text-xs px-2.5 py-1 rounded-full bg-slate-800 border border-slate-700 text-slate-300 font-mono"
                title={s.plugin_name}
              >
                {s.label || s.scale.toUpperCase()}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Description */}
      <p className="text-slate-500 text-sm mt-3 leading-relaxed">{family.description}</p>

      {/* Mamba-YOLO install panel */}
      {isMamba && (
        <InstallPanel
          status={mambaStatus}
          loading={mambaLoading}
          onInstall={onMambaInstall}
          onRebuild={onMambaRebuild}
          onRefresh={onMambaRefresh}
        />
      )}
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────────

export default function PluginsPage() {
  const [families, setFamilies] = useState<ArchFamily[]>([]);
  const [familiesLoading, setFamiliesLoading] = useState(true);

  const [mambaStatus, setMambaStatus] = useState<InstallStatus | null>(null);
  const [mambaLoading, setMambaLoading] = useState(false);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Loaders ──────────────────────────────────────────────────────────────────

  const loadFamilies = useCallback(() => {
    setFamiliesLoading(true);
    api.listArchPlugins()
      .then(setFamilies)
      .catch(() => {})
      .finally(() => setFamiliesLoading(false));
  }, []);

  const loadMambaStatus = useCallback(() => {
    setMambaLoading(true);
    api.getMambaInstallStatus()
      .then((s) => setMambaStatus(s as InstallStatus))
      .catch(() => {})
      .finally(() => setMambaLoading(false));
  }, []);

  // ── Polling ───────────────────────────────────────────────────────────────────

  const stopPoll = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const startPoll = useCallback(() => {
    stopPoll();
    pollRef.current = setInterval(() => {
      api.getMambaInstallStatus()
        .then((s) => {
          const st = s as InstallStatus;
          setMambaStatus(st);
          if (st.status !== 'installing') {
            stopPoll();
          }
        })
        .catch(() => {});
    }, 3000);
  }, [stopPoll]);

  // ── Mount ─────────────────────────────────────────────────────────────────────

  useEffect(() => {
    loadFamilies();
    loadMambaStatus();
    return () => stopPoll();
  }, [loadFamilies, loadMambaStatus, stopPoll]);

  // Start polling if currently installing
  useEffect(() => {
    if (mambaStatus?.status === 'installing' && !pollRef.current) {
      startPoll();
    }
  }, [mambaStatus?.status, startPoll]);

  // ── Handlers ──────────────────────────────────────────────────────────────────

  const handleInstall = useCallback(() => {
    api.triggerMambaInstall()
      .then(() => {
        loadMambaStatus();
        startPoll();
      })
      .catch(() => {});
  }, [loadMambaStatus, startPoll]);

  const handleRebuild = useCallback(() => {
    api.triggerMambaRebuild()
      .then(() => {
        loadMambaStatus();
        startPoll();
      })
      .catch(() => {});
  }, [loadMambaStatus, startPoll]);

  const handleRefresh = useCallback(() => {
    loadFamilies();
    loadMambaStatus();
  }, [loadFamilies, loadMambaStatus]);

  // ── Render ────────────────────────────────────────────────────────────────────

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-4xl mx-auto p-8 space-y-8">

        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">Plugins</h1>
            <p className="text-slate-500 text-sm mt-1">
              Manage model architecture plugins and dependencies
            </p>
          </div>
          <button
            onClick={handleRefresh}
            className="flex items-center gap-2 px-3 py-1.5 text-sm text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors cursor-pointer"
          >
            <RefreshCw size={14} className={familiesLoading || mambaLoading ? 'animate-spin' : ''} />
            Refresh
          </button>
        </div>

        {/* Summary row */}
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-indigo-500/10 flex items-center justify-center">
              <Puzzle size={17} className="text-indigo-400" />
            </div>
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wider font-medium">Arch Plugins</p>
              <p className="text-xl font-bold text-white">{families.length}</p>
            </div>
          </div>
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-emerald-500/10 flex items-center justify-center">
              <Layers size={17} className="text-emerald-400" />
            </div>
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wider font-medium">Scale Variants</p>
              <p className="text-xl font-bold text-white">
                {families.reduce((acc, f) => acc + f.supported_scales.length, 0)}
              </p>
            </div>
          </div>
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-orange-500/10 flex items-center justify-center">
              <Cpu size={17} className="text-orange-400" />
            </div>
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wider font-medium">Mamba-YOLO</p>
              <p className="text-xl font-bold text-white capitalize">
                {mambaStatus?.status ?? '…'}
              </p>
            </div>
          </div>
        </div>

        {/* Plugin list */}
        {familiesLoading ? (
          <div className="flex items-center gap-3 text-slate-500 py-10 justify-center">
            <Loader2 size={18} className="animate-spin" />
            <span className="text-sm">Loading plugins…</span>
          </div>
        ) : families.length === 0 ? (
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-10 text-center">
            <Puzzle size={32} className="text-slate-600 mx-auto mb-3" />
            <p className="text-slate-500 text-sm">No arch plugins registered.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {families.map((f) => (
              <ArchCard
                key={f.family}
                family={f}
                mambaStatus={mambaStatus}
                mambaLoading={mambaLoading}
                onMambaInstall={handleInstall}
                onMambaRebuild={handleRebuild}
                onMambaRefresh={loadMambaStatus}
              />
            ))}
          </div>
        )}

      </div>
    </div>
  );
}
