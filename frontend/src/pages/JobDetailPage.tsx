import React, { useState, useEffect, useRef, useCallback } from 'react';
import { usePersistedState } from '../hooks/usePersistedState';
import { api } from '../services/api';
import { useSSE } from '../hooks/useSSE';
import type { JobRecord, EpochMetrics } from '../types';
import { 
  ArrowLeft, Square, Play, RefreshCw, ScrollText, Activity, 
  Timer, ChevronDown, AlertTriangle, Target, ImageIcon, Layers,
  Zap, HardDrive, PlusCircle, Download
} from 'lucide-react';
import { useWeightsStore } from '../store/weightsStore';
import { useJobsStore } from '../store/jobsStore';
import JobCharts from '../components/JobCharts';
import JobConfiguration from '../components/JobConfiguration';
import PlotsGallery from '../components/PlotsGallery';
import ClassSamplesGallery from '../components/ClassSamplesGallery';

interface Props { jobId: string; onBack: () => void; }

interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
  data?: Record<string, unknown>;
}

const INTERVAL_OPTIONS = [
  { label: 'Off', value: 0 },
  { label: '3s', value: 3000 },
  { label: '5s', value: 5000 },
  { label: '10s', value: 10000 },
  { label: '30s', value: 30000 },
  { label: '60s', value: 60000 },
];

const LOG_INTERVAL_OPTIONS = [
  { label: 'Off', value: 0 },
  { label: '1s', value: 1000 },
  { label: '2s', value: 2000 },
  { label: '3s', value: 3000 },
  { label: '5s', value: 5000 },
  { label: '10s', value: 10000 },
];

const StatBadge = ({ label, value, color, subtext }: { label: string, value: string | number, color: string, subtext?: string }) => (
  <div className="bg-slate-900 px-4 py-2 rounded border border-slate-800 min-w-[100px]">
    <span className="text-[10px] text-slate-500 block uppercase tracking-wider">{label}</span>
    <span className={`text-lg font-bold ${color}`}>
      {value} {subtext && <span className="text-xs text-slate-500 font-normal">{subtext}</span>}
    </span>
  </div>
);

export default function JobDetailPage({ jobId, onBack }: Props) {
  const [job, setJob] = useState<JobRecord | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [trainProgress, setTrainProgress] = useState<{
    phase: string;
    epoch: number;
    total_epochs: number;
    batch: number;
    total_batches: number;
    percent: number;
    losses: { box?: number | null; cls?: number | null; dfl?: number | null };
    // device / resource
    device?: string | null;
    ram_gb?: number | null;
    ram_total_gb?: number | null;
    gpu_mem_gb?: number | null;
    gpu_mem_reserved_gb?: number | null;
    // time
    epoch_elapsed_s?: number | null;
    total_elapsed_s?: number | null;
    avg_epoch_s?: number | null;
    eta_s?: number | null;
    imgs_per_sec?: number | null;
    // val metrics (from validation_done phase)
    val_map50?: number | null;
    val_map?: number | null;
    val_precision?: number | null;
    val_recall?: number | null;
    val_time_s?: number | null;
    message?: string;
  } | null>(null);

  const [showAppendModal, setShowAppendModal] = useState(false);
  const [appendEpochs, setAppendEpochs] = useState(50);
  const [appending, setAppending] = useState(false);
  const [appendError, setAppendError] = useState<string | null>(null);

  // Auto-refresh intervals (0 = off) — persisted across page visits
  const [mainInterval, setMainInterval] = usePersistedState('job.mainInterval', 0);
  const [logInterval, setLogInterval] = usePersistedState('job.logInterval', 0);
  const [refreshing, setRefreshing] = useState(false);
  const [showRefreshMenu, setShowRefreshMenu] = useState(false);
  const refreshMenuRef = useRef<HTMLDivElement>(null);

  // Close menu on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (refreshMenuRef.current && !refreshMenuRef.current.contains(e.target as Node)) {
        setShowRefreshMenu(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const refreshLogs = useCallback(() => {
    api.getJobLogs(jobId).then((rawLogs) => {
      const l = rawLogs as LogEntry[];
      setLogs(l);

      // Parse latest PROGRESS log → update trainProgress
      // API returns newest-first, so index [0] is the most recent entry
      const progressLogs = l.filter(log => log.level === 'PROGRESS' && log.data?.type === 'progress');
      if (progressLogs.length > 0) {
        const latest = progressLogs[0];
        const d = latest.data as Record<string, unknown>;
        const phase = (d.phase as string) || 'train';
        const epochStr = (d.epoch as string) ?? '';
        const batchStr = (d.batch as string) ?? '';
        const epochNum = parseInt(epochStr.split('/')[0]) || 0;
        const totalEpochs = parseInt(epochStr.split('/')[1]) || 0;
        const batchNum = parseInt(batchStr.split('/')[0]) || 0;
        const totalBatches = parseInt(batchStr.split('/')[1]) || 0;

        // For val metrics: use latest validation_done entry (newest-first, so find() gives most recent)
        const valLog = progressLogs.find(log => (log.data as Record<string,unknown>)?.phase === 'validation_done');
        const vd = valLog ? valLog.data as Record<string, unknown> : {};

        setTrainProgress(prev => ({
          phase,
          epoch: epochNum || prev?.epoch || 0,
          total_epochs: totalEpochs || prev?.total_epochs || 0,
          batch: batchNum,
          total_batches: totalBatches,
          percent: (d.percent as number) ?? 0,
          losses: (d.losses as { box?: number; cls?: number; dfl?: number }) ?? prev?.losses ?? {},
          // resource — prefer latest, fallback to prev
          device: (d.device as string | null) ?? prev?.device,
          ram_gb: (d.ram_gb as number | null) ?? prev?.ram_gb,
          ram_total_gb: (d.ram_total_gb as number | null) ?? prev?.ram_total_gb,
          gpu_mem_gb: (d.gpu_mem_gb as number | null) ?? prev?.gpu_mem_gb,
          gpu_mem_reserved_gb: (d.gpu_mem_reserved_gb as number | null) ?? prev?.gpu_mem_reserved_gb,
          // time
          epoch_elapsed_s: (d.epoch_elapsed_s as number | null) ?? prev?.epoch_elapsed_s,
          total_elapsed_s: (d.total_elapsed_s as number | null) ?? prev?.total_elapsed_s,
          avg_epoch_s: (d.avg_epoch_s as number | null) ?? prev?.avg_epoch_s,
          eta_s: (d.eta_s as number | null) ?? prev?.eta_s,
          imgs_per_sec: (d.imgs_per_sec as number | null) ?? prev?.imgs_per_sec,
          // val metrics — from latest validation_done log
          val_map50: (vd.val_map50 as number | null) ?? prev?.val_map50,
          val_map: (vd.val_map as number | null) ?? prev?.val_map,
          val_precision: (vd.val_precision as number | null) ?? prev?.val_precision,
          val_recall: (vd.val_recall as number | null) ?? prev?.val_recall,
          val_time_s: (vd.val_time_s as number | null) ?? prev?.val_time_s,
          message: latest.message,
        }));
      }
    }).catch(() => {});
  }, [jobId]);

  const refreshAll = useCallback(() => {
    setRefreshing(true);
    Promise.all([
      api.loadJob(jobId).then(setJob),
      new Promise<void>((resolve) => { refreshLogs(); resolve(); }),
    ]).catch(() => {}).finally(() => setRefreshing(false));
  }, [jobId, refreshLogs]);

  // Initial load
  useEffect(() => {
    refreshAll();
  }, [jobId, refreshAll]);

  // Auto-refresh main data
  useEffect(() => {
    if (mainInterval <= 0) return;
    const id = setInterval(refreshAll, mainInterval);
    return () => clearInterval(id);
  }, [mainInterval, refreshAll]);

  // Auto-refresh logs (separate, can be more frequent)
  useEffect(() => {
    if (logInterval <= 0) return;
    const id = setInterval(refreshLogs, logInterval);
    return () => clearInterval(id);
  }, [logInterval, refreshLogs]);

  // Auto-enable log refresh when job is running, stop when done
  useEffect(() => {
    if (job?.status === 'running') {
      if (logInterval <= 0) setLogInterval(5000);
      if (mainInterval <= 0) setMainInterval(30000);
    } else if (job && job.status !== 'running') {
      if (mainInterval > 0) setMainInterval(0);
      if (logInterval > 0) setLogInterval(0);
      setTrainProgress(null);
      useWeightsStore.getState().invalidate();
      useJobsStore.getState().invalidate();
    }
  }, [job?.status]);

  // SSE — real-time updates for progress, epoch summary, and terminal events
  const sseUrl = job?.status === 'running' ? `/api/train/${jobId}/stream` : null;
  useSSE(sseUrl, (_event, data) => {
    const d = data as Record<string, unknown>;
    if (d.type === 'progress') {
      const phase = (d.phase as string) || 'train';
      const epochStr = (d.epoch as string) ?? '';
      const batchStr = (d.batch as string) ?? '';
      const epochNum = parseInt(epochStr.split('/')[0]) || (d.epoch as number) || 0;
      const totalEpochs = parseInt(epochStr.split('/')[1]) || (d.total_epochs as number) || 0;
      const batchNum = parseInt(batchStr.split('/')[0]) || (d.batch as number) || 0;
      const totalBatches = parseInt(batchStr.split('/')[1]) || (d.total_batches as number) || 0;
      setTrainProgress(prev => ({
        phase,
        epoch: epochNum || prev?.epoch || 0,
        total_epochs: totalEpochs || prev?.total_epochs || 0,
        batch: batchNum,
        total_batches: totalBatches,
        percent: (d.percent as number) ?? prev?.percent ?? 0,
        losses: (d.losses as { box?: number; cls?: number; dfl?: number }) ?? prev?.losses ?? {},
        // resource — always update from train phase, keep val metrics from validation_done
        device: (d.device as string | null) ?? prev?.device,
        ram_gb: (d.ram_gb as number | null) ?? prev?.ram_gb,
        ram_total_gb: (d.ram_total_gb as number | null) ?? prev?.ram_total_gb,
        gpu_mem_gb: (d.gpu_mem_gb as number | null) ?? prev?.gpu_mem_gb,
        gpu_mem_reserved_gb: (d.gpu_mem_reserved_gb as number | null) ?? prev?.gpu_mem_reserved_gb,
        // time
        epoch_elapsed_s: (d.epoch_elapsed_s as number | null) ?? prev?.epoch_elapsed_s,
        total_elapsed_s: (d.total_elapsed_s as number | null) ?? prev?.total_elapsed_s,
        avg_epoch_s: (d.avg_epoch_s as number | null) ?? prev?.avg_epoch_s,
        eta_s: (d.eta_s as number | null) ?? prev?.eta_s,
        imgs_per_sec: (d.imgs_per_sec as number | null) ?? prev?.imgs_per_sec,
        // val metrics — only update from validation_done phase, keep previous otherwise
        val_map50: phase === 'validation_done' ? (d.val_map50 as number | null) : prev?.val_map50,
        val_map: phase === 'validation_done' ? (d.val_map as number | null) : prev?.val_map,
        val_precision: phase === 'validation_done' ? (d.val_precision as number | null) : prev?.val_precision,
        val_recall: phase === 'validation_done' ? (d.val_recall as number | null) : prev?.val_recall,
        val_time_s: phase === 'validation_done' ? (d.val_time_s as number | null) : prev?.val_time_s,
      }));
    } else if (d.type === 'epoch') {
      setJob((prev) => {
        if (!prev) return prev;
        const metrics = d as unknown as EpochMetrics;
        return { ...prev, epoch: metrics.epoch, history: [...prev.history, metrics] };
      });
      refreshLogs();
    } else if (d.type === 'complete' || d.type === 'stopped' || d.type === 'error') {
      api.loadJob(jobId).then(setJob).catch(() => {});
      refreshLogs();
    }
  });

  const handleStop = async () => {
    await api.stopTraining(jobId);
  };

  const handleResumeConfirmNew = async () => {
    try {
      await api.resumeTraining(jobId);
      refreshAll();
    } catch (e: unknown) {
      console.error('Resume failed', e);
    }
  };

  const handleAppendConfirm = async () => {
    setAppending(true);
    setAppendError(null);
    try {
      await api.appendTraining(jobId, appendEpochs);
      setShowAppendModal(false);
      refreshAll();
    } catch (e: unknown) {
      setAppendError(e instanceof Error ? e.message : 'Failed to append training');
    } finally {
      setAppending(false);
    }
  };

  if (!job) return <div className="flex-1 flex items-center justify-center text-slate-500">Loading...</div>;

  const progress = job.total_epochs ? (job.epoch / job.total_epochs) * 100 : 0;
  const isDetection = job.task === 'detect' || true; // Default to true for now
  const bestMap = job.best_mAP50 ?? (job.history.length > 0 ? Math.max(...job.history.map(h => Number(h.mAP50 || 0))) : 0);
  const timeElapsed = job.total_time ? `${(job.total_time / 60).toFixed(0)}m` : (job.history.length > 0 ? `${(job.history.reduce((s, h) => s + h.epoch_time, 0) / 60).toFixed(0)}m` : '-');

  return (
    <div className="flex-1 overflow-y-auto bg-[#0f1117]">
      <div className="max-w-7xl mx-auto p-6 space-y-6">
        
        {/* Header */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-slate-800 pb-6">
          <div>
            <div className="flex items-center gap-3">
              <button onClick={onBack} className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors cursor-pointer mr-1">
                <ArrowLeft size={18} />
              </button>
              <h1 className="text-2xl font-bold text-white tracking-tight">{job.model_name}</h1>
              <span className={`text-xs px-2 py-0.5 rounded border uppercase font-medium ${
                job.status === 'failed' ? 'bg-red-500/10 text-red-400 border-red-500/20' : 
                job.status === 'completed' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' :
                job.status === 'running' ? 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20' :
                'bg-slate-500/10 text-slate-400 border-slate-500/20'
              }`}>
                {job.status}
              </span>
            </div>
            <div className="flex items-center gap-4 mt-1">
              <p className="text-slate-500 text-sm font-mono flex items-center gap-2">
                <Activity size={14} /> Job ID: {job.job_id.slice(0, 12)}
              </p>
              
              {/* Actions */}
              <div className="flex items-center gap-2">
                {job.status === 'running' && (
                  <button onClick={handleStop} className="flex items-center gap-1.5 px-2.5 py-1 bg-red-600/20 hover:bg-red-600/30 text-red-400 border border-red-600/30 rounded text-xs font-medium transition-colors cursor-pointer">
                    <Square size={12} /> Stop
                  </button>
                )}
                {(job.status === 'stopped' || job.status === 'failed') && (
                  <button onClick={handleResumeConfirmNew} className="flex items-center gap-1.5 px-2.5 py-1 bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-400 border border-emerald-600/30 rounded text-xs font-medium transition-colors cursor-pointer">
                    <Play size={12} /> Resume
                  </button>
                )}
                {(job.status === 'completed' || job.status === 'stopped') && (
                  <button onClick={() => { setAppendError(null); setShowAppendModal(true); }} className="flex items-center gap-1.5 px-2.5 py-1 bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-400 border border-indigo-600/30 rounded text-xs font-medium transition-colors cursor-pointer">
                    <PlusCircle size={12} /> Append
                  </button>
                )}
                {job.weight_id && (
                  <button
                    onClick={() => {
                      const fn = `${job.model_name}_${job.weight_id!.slice(0, 8)}.pt`.replace(/\s+/g, '_');
                      api.downloadWeight(job.weight_id!, fn);
                    }}
                    className="flex items-center gap-1.5 px-2.5 py-1 bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-400 border border-emerald-600/30 rounded text-xs font-medium transition-colors cursor-pointer"
                    title="Download trained weight (.pt)"
                  >
                    <Download size={12} /> Export .pt
                  </button>
                )}
                <div className="relative" ref={refreshMenuRef}>
                    <button
                      onClick={refreshAll}
                      className={`p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded transition-colors cursor-pointer ${refreshing ? 'animate-spin' : ''}`}
                      title="Refresh"
                    >
                      <RefreshCw size={14} />
                    </button>
                </div>
              </div>
            </div>
          </div>
          
          <div className="flex gap-4 text-right">
             <StatBadge label="Best mAP50" value={`${(bestMap * 100).toFixed(2)}%`} color="text-emerald-400" />
             <StatBadge label="Progress" value={`${job.epoch} / ${job.total_epochs}`} color="text-white" subtext="Epochs" />
             <StatBadge label="Time Elapsed" value={timeElapsed} color="text-blue-400" />
          </div>
        </div>

        {/* Failed Message */}
        {job.status === 'failed' && (
          <div className="bg-red-500/5 border border-red-500/20 rounded-lg p-4 flex items-center gap-3 text-red-400">
            <AlertTriangle size={20} />
            <span className="text-sm">Training stopped unexpectedly: {job.message}</span>
          </div>
        )}

        {/* Active Progress Widget (Running) */}
        {job.status === 'running' && (() => {
          const tp = trainProgress;
          const fmt = (v: number | null | undefined, digits = 1) =>
            v != null ? v.toFixed(digits) : '—';
          const fmtMin = (s: number | null | undefined) =>
            s != null ? `${(s / 60).toFixed(1)}m` : '—';

          return (
            <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
              {/* Header row */}
              <div className="px-4 py-3 flex items-center justify-between border-b border-slate-800 bg-slate-900/80">
                <span className="text-sm text-slate-300 flex items-center gap-2">
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                  </span>
                  {tp?.phase === 'validation' ? (
                    <span className="text-amber-400 font-medium">Validating…</span>
                  ) : tp?.phase === 'validation_done' ? (
                    <span className="text-sky-400 font-medium">Validation done</span>
                  ) : (
                    <span className="text-emerald-400 font-medium">Training</span>
                  )}
                </span>
                <span className="text-sm text-white font-mono">
                  Epoch <span className="text-emerald-400">{tp?.epoch ?? job.epoch}</span> / {tp?.total_epochs ?? job.total_epochs}
                </span>
              </div>

              <div className="p-4 space-y-4">
                {/* Epoch progress bar */}
                <div>
                  <div className="flex justify-between text-[10px] text-slate-500 font-mono mb-1">
                    <span>Epoch progress</span>
                    <span>{progress.toFixed(1)}%</span>
                  </div>
                  <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-indigo-500 rounded-full transition-all duration-500" style={{ width: `${progress}%` }} />
                  </div>
                </div>

                {/* Batch progress bar — always shown, greyed when no data */}
                <div>
                  <div className="flex justify-between text-[10px] text-slate-500 font-mono mb-1">
                    <span>Batch {tp ? `${tp.batch} / ${tp.total_batches}` : '— / —'}</span>
                    <span>{tp ? `${tp.percent.toFixed(1)}%` : '—'}</span>
                  </div>
                  <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-emerald-500 rounded-full transition-all duration-300"
                      style={{ width: tp ? `${tp.percent}%` : '0%' }}
                    />
                  </div>
                </div>

                {/* Losses row — always shown */}
                <div className="flex items-center gap-5">
                  {(['box', 'cls', 'dfl'] as const).map((k, i) => {
                    const colors = ['text-amber-400', 'text-purple-400', 'text-rose-400'];
                    const v = tp?.losses?.[k];
                    return (
                      <span key={k} className="text-xs">
                        <span className="text-slate-500 uppercase">{k} </span>
                        <span className={`font-mono ${v != null ? colors[i] : 'text-slate-600'}`}>
                          {v != null ? v.toFixed(3) : '—'}
                        </span>
                      </span>
                    );
                  })}
                </div>

                {/* Info grid — always shown, null = — */}
                <div className="grid grid-cols-2 gap-x-6 gap-y-2 pt-2 border-t border-slate-800 text-xs">
                  {/* Left column: device / memory */}
                  <div className="space-y-1.5">
                    <InfoRow icon={<Zap size={11} className="text-amber-400" />} label="Device" value={tp?.device ?? '—'} mono />
                    <InfoRow icon={<HardDrive size={11} className="text-purple-400" />} label="RAM"
                      value={tp?.ram_gb != null ? `${fmt(tp.ram_gb)} / ${fmt(tp.ram_total_gb)} GB` : '—'} mono />
                    <InfoRow icon={<Zap size={11} className="text-emerald-400" />} label="GPU mem"
                      value={tp?.gpu_mem_gb != null ? `${fmt(tp.gpu_mem_gb)} GB (rsv ${fmt(tp.gpu_mem_reserved_gb)} GB)` : '—'} mono />
                    <InfoRow icon={<Activity size={11} className="text-blue-400" />} label="Speed"
                      value={tp?.imgs_per_sec != null ? `${fmt(tp.imgs_per_sec, 0)} img/s` : '—'} mono />
                  </div>
                  {/* Right column: time */}
                  <div className="space-y-1.5">
                    <InfoRow icon={<Timer size={11} className="text-blue-400" />} label="Epoch time"
                      value={tp?.epoch_elapsed_s != null ? `${fmt(tp.epoch_elapsed_s, 0)}s` : '—'} mono />
                    <InfoRow icon={<Timer size={11} className="text-indigo-400" />} label="Avg epoch"
                      value={tp?.avg_epoch_s != null ? `${fmt(tp.avg_epoch_s, 0)}s` : '—'} mono />
                    <InfoRow icon={<Timer size={11} className="text-slate-400" />} label="Total"
                      value={fmtMin(tp?.total_elapsed_s)} mono />
                    <InfoRow icon={<Timer size={11} className="text-orange-400" />} label="ETA"
                      value={fmtMin(tp?.eta_s)} mono highlight={tp?.eta_s != null} />
                  </div>
                </div>

                {/* Val metrics — shown when available (persists from last validation_done) */}
                <div className="pt-2 border-t border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">Last Validation</div>
                  <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-xs">
                    <div className="space-y-1.5">
                      <InfoRow icon={<Target size={11} className="text-emerald-400" />} label="mAP@0.5"
                        value={tp?.val_map50 != null ? tp.val_map50.toFixed(4) : '—'} mono highlight={tp?.val_map50 != null} />
                      <InfoRow icon={<Target size={11} className="text-sky-400" />} label="mAP@0.5:0.95"
                        value={tp?.val_map != null ? tp.val_map.toFixed(4) : '—'} mono highlight={tp?.val_map != null} />
                    </div>
                    <div className="space-y-1.5">
                      <InfoRow icon={<Activity size={11} className="text-violet-400" />} label="Precision"
                        value={tp?.val_precision != null ? tp.val_precision.toFixed(4) : '—'} mono />
                      <InfoRow icon={<Activity size={11} className="text-rose-400" />} label="Recall"
                        value={tp?.val_recall != null ? tp.val_recall.toFixed(4) : '—'} mono />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          );
        })()}

        {/* Charts */}
        <JobCharts history={job.history} isDetection={isDetection} />

        {/* Configuration */}
        {job.config && (
          <JobConfiguration 
            config={job.config as any} 
            partitions={job.partitions}
            modelScale={job.model_scale}
          />
        )}

        {/* Plots Gallery */}
        <div className="mt-6">
           <h3 className="text-white font-semibold flex items-center gap-2 mb-4">
              <ImageIcon size={18} className="text-slate-400" /> Visualization Plots
           </h3>
           <PlotsGallery jobId={jobId} />
        </div>

        {/* Class Samples Gallery */}
        <div className="mt-6">
           <h3 className="text-white font-semibold flex items-center gap-2 mb-4">
              <Layers size={18} className="text-slate-400" /> Class Samples
           </h3>
           <ClassSamplesGallery jobId={jobId} />
        </div>

        {/* Training Logs */}
        {logs.length > 0 && (
          <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden mt-6">
            <div 
              className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/30"
            >
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <ScrollText size={16} className="text-slate-400" /> Training Logs
              </h3>
              <div className="flex items-center gap-4">
                 <span className="text-[10px] text-slate-500 uppercase tracking-wider">{logs.length} Lines</span>
                 <div className="flex gap-1">
                    {LOG_INTERVAL_OPTIONS.map((opt) => (
                      <button
                        key={opt.value}
                        onClick={() => setLogInterval(opt.value)}
                        className={`px-2 py-0.5 text-[10px] rounded transition-colors ${
                          logInterval === opt.value
                            ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/40'
                            : 'bg-slate-800 text-slate-500 border border-slate-700 hover:border-slate-600'
                        }`}
                      >
                        {opt.label}
                      </button>
                    ))}
                 </div>
              </div>
            </div>
            <div className="max-h-96 overflow-y-auto space-y-1 p-4 font-mono text-[11px] bg-[#0d0e12]">
              {logs.filter(log => log.level !== 'PROGRESS').map((log, i) => {
                const levelColor =
                  log.level === 'ERROR' ? 'text-red-400' :
                  log.level === 'WARNING' ? 'text-amber-400' :
                  log.level === 'DEBUG' ? 'text-slate-600' :
                  'text-slate-500';
                return (
                  <div key={i} className="flex gap-3 hover:bg-slate-800/30 px-2 rounded -mx-2">
                    <span className="text-slate-700 shrink-0 w-16 select-none">
                      {new Date(log.timestamp).toLocaleTimeString([], {hour12: false})}
                    </span>
                    <span className={`shrink-0 w-12 ${levelColor} font-bold select-none`}>{log.level}</span>
                    <span className="text-slate-300 flex-1 break-all whitespace-pre-wrap">{log.message}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

      </div>
      
      {/* Append Training Modal */}
      {showAppendModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => setShowAppendModal(false)}>
          <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-full max-w-sm mx-4 p-6" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-lg font-semibold text-white mb-1">Append Training</h3>
            <p className="text-sm text-slate-400 mb-5">
              Continues on the same job from <span className="font-mono text-slate-300">last.pt</span> — logs and history are appended.
            </p>
            <div className="space-y-4">
              <div>
                <label className="block text-xs text-slate-400 mb-1.5">Additional epochs</label>
                <input
                  type="number"
                  min={1}
                  max={10000}
                  value={appendEpochs}
                  onChange={(e) => setAppendEpochs(Math.max(1, parseInt(e.target.value) || 1))}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-indigo-500"
                />
              </div>
              {appendError && (
                <p className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded px-3 py-2">{appendError}</p>
              )}
              <div className="flex justify-end gap-2 pt-1">
                <button
                  onClick={() => setShowAppendModal(false)}
                  className="px-4 py-2 text-sm text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  onClick={handleAppendConfirm}
                  disabled={appending}
                  className="px-4 py-2 text-sm text-white bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition-colors cursor-pointer flex items-center gap-2"
                >
                  {appending ? <RefreshCw size={13} className="animate-spin" /> : <PlusCircle size={13} />}
                  {appending ? 'Starting…' : `Append ${appendEpochs} epochs`}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

function InfoRow({
  icon, label, value, mono, highlight,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  mono?: boolean;
  highlight?: boolean;
}) {
  const isDash = value === '—';
  return (
    <div className="flex items-center gap-1.5 text-slate-400">
      {icon}
      <span className="shrink-0">{label}:</span>
      <span className={`${mono ? 'font-mono' : ''} ${isDash ? 'text-slate-600' : highlight ? 'text-orange-400' : 'text-white'}`}>
        {value}
      </span>
    </div>
  );
}
