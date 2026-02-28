import { useState, useEffect, useRef, useCallback } from 'react';
import { api } from '../services/api';
import type { DatasetInfo, DatasetMeta } from '../types';
import {
  Database, CheckCircle, XCircle, ArrowRight, Search,
  Download, Loader2, AlertCircle, Trash2, HardDrive, Upload, Link, FolderSearch,
} from 'lucide-react';
import { useDatasetsStore } from '../store/datasetsStore';

interface Props {
  onOpenDataset?: (name: string) => void;
}

interface FileProgress {
  name: string;
  label: string;
  status: string;
  progress: number;
  bytes_downloaded: number;
  bytes_total: number;
  rate_bps: number;
  eta_seconds: number;
  message: string;
}

interface DownloadState {
  status: string;
  progress: number;
  message: string;
  rate_bps?: number;
  eta_seconds?: number;
  files?: Record<string, FileProgress>;
  total_files?: number;
  completed_files?: number;
}

interface UploadState {
  status: 'idle' | 'uploading' | 'extracting' | 'complete' | 'error' | 'indexing';
  progress: number;
  message: string;
  bytesReceived?: number;
}

type InstallMethod = 'choose' | 'workspace' | 'upload' | 'url';

interface WorkspaceScan {
  scanning: boolean;
  found: boolean;
  file_count: number;
  dir_count: number;
  size_bytes: number;
  pending_archive: { path: string; name: string; size_bytes: number } | null;
}

interface StatusInfo {
  available: boolean;
  manual?: boolean;
  instructions?: string;
  meta?: DatasetMeta;
}

export default function DatasetsPage({ onOpenDataset }: Props) {
  const datasets = useDatasetsStore((s) => s.datasets);
  const dsLoading = useDatasetsStore((s) => s.loading);
  const loadDatasets = useDatasetsStore((s) => s.load);
  const [statuses, setStatuses] = useState<Record<string, StatusInfo>>({});
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');

  // Download dialog
  const [downloadTarget, setDownloadTarget] = useState<DatasetInfo | null>(null);
  const [dlState, setDlState] = useState<DownloadState>({ status: 'idle', progress: 0, message: '' });
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Upload state
  const [uploadState, setUploadState] = useState<UploadState>({ status: 'idle', progress: 0, message: '' });
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [installMethod, setInstallMethod] = useState<InstallMethod>('choose');
  const [urlInput, setUrlInput] = useState('');
  const [wsScan, setWsScan] = useState<WorkspaceScan>({ scanning: false, found: false, file_count: 0, dir_count: 0, size_bytes: 0, pending_archive: null });

  // Delete dialog
  const [deleteTarget, setDeleteTarget] = useState<DatasetInfo | null>(null);
  const [deleting, setDeleting] = useState(false);

  const loadStatuses = useCallback(async (list: DatasetInfo[]) => {
    const entries = await Promise.all(
      list.map(async (ds) => {
        try {
          const s = await api.getDatasetStatus(ds.name);
          return [ds.name, { available: s.available, manual: s.manual_download, instructions: s.instructions, meta: s.meta }] as const;
        } catch {
          return [ds.name, { available: false }] as const;
        }
      })
    );
    setStatuses(Object.fromEntries(entries));
  }, []);

  useEffect(() => {
    loadDatasets().then(() => {
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  // Load statuses whenever dataset list changes
  useEffect(() => {
    if (datasets.length > 0) {
      loadStatuses(datasets);
    }
  }, [datasets, loadStatuses]);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  const handleCardClick = async (ds: DatasetInfo) => {
    const st = statuses[ds.name];
    if (st?.available) {
      onOpenDataset?.(ds.name);
      return;
    }
    setDownloadTarget(ds);
    setDlState({ status: 'idle', progress: 0, message: '' });
    setUploadState({ status: 'idle', progress: 0, message: '' });
    setInstallMethod('choose');
    setUrlInput('');
    // Always scan workspace — detect both existing files and pending _download_ archives
    setWsScan({ scanning: true, found: false, file_count: 0, dir_count: 0, size_bytes: 0, pending_archive: null });
    try {
      const r = await api.workspaceScan(ds.name);
      setWsScan({
        scanning: false,
        found: r.found,
        file_count: r.file_count,
        dir_count: r.dir_count ?? 0,
        size_bytes: r.size_bytes ?? 0,
        pending_archive: r.pending_archive ?? null,
      });
      if (r.pending_archive) {
        setInstallMethod('workspace');
      } else if (r.found && statuses[ds.name]?.manual) {
        setInstallMethod('workspace');
      }
    } catch {
      setWsScan({ scanning: false, found: false, file_count: 0, dir_count: 0, size_bytes: 0, pending_archive: null });
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      await api.deleteDatasetData(deleteTarget.name);
      setStatuses((prev) => ({
        ...prev,
        [deleteTarget.name]: { ...prev[deleteTarget.name], available: false, meta: prev[deleteTarget.name]?.meta ? { ...prev[deleteTarget.name]!.meta!, available: false, disk_size_bytes: 0, disk_size_human: '0 B', splits: {} } : undefined },
      }));
    } catch { /* ignore */ }
    setDeleting(false);
    setDeleteTarget(null);
  };

  const isManual = downloadTarget ? statuses[downloadTarget.name]?.manual : false;
  const manualInstructions = downloadTarget ? statuses[downloadTarget.name]?.instructions : undefined;

  const handleResumeExtract = useCallback(async () => {
    if (!downloadTarget) return;
    const name = downloadTarget.name;
    setUploadState({ status: 'extracting', progress: 50, message: 'Resuming extraction...' });
    try {
      await api.resumeExtract(name);
    } catch (e: unknown) {
      setUploadState({ status: 'error', progress: 0, message: e instanceof Error ? e.message : 'Resume extract failed' });
      return;
    }
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const s = await api.getDatasetUploadStatus(name);
        setUploadState({ status: s.status as UploadState['status'], progress: s.progress, message: s.message });
        if (s.status === 'complete' || s.status === 'error') {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
          if (s.status === 'complete') {
            const fresh = await api.getDatasetStatus(name).catch(() => null);
            if (fresh) setStatuses(prev => ({ ...prev, [name]: { ...prev[name], available: true, meta: fresh.meta } }));
          }
        }
      } catch { /* ignore */ }
    }, 800);
  }, [downloadTarget]);

  const handleImportLocal = useCallback(async () => {
    if (!downloadTarget) return;
    const name = downloadTarget.name;
    setUploadState({ status: 'indexing', progress: 10, message: 'Indexing workspace files...' });
    try {
      await api.importLocal(name);
    } catch (e: unknown) {
      setUploadState({ status: 'error', progress: 0, message: e instanceof Error ? e.message : 'Import failed' });
      return;
    }
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const s = await api.getDatasetUploadStatus(name);
        setUploadState({ status: s.status as UploadState['status'], progress: s.progress, message: s.message });
        if (s.status === 'complete' || s.status === 'error') {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
          if (s.status === 'complete') {
            const fresh = await api.getDatasetStatus(name).catch(() => null);
            if (fresh) setStatuses(prev => ({ ...prev, [name]: { ...prev[name], available: true, meta: fresh.meta } }));
          }
        }
      } catch { /* ignore */ }
    }, 800);
  }, [downloadTarget]);

  const handleUrlDownload = useCallback(async () => {
    if (!downloadTarget || !urlInput.trim()) return;
    const name = downloadTarget.name;
    try {
      await api.downloadFromUrl(name, urlInput.trim());
    } catch (e: unknown) {
      setDlState({ status: 'error', progress: 0, message: e instanceof Error ? e.message : 'Failed to start download' });
      return;
    }
    setDlState({ status: 'downloading', progress: 0, message: 'Starting download...' });
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const s = await api.getDatasetDownloadStatus(name) as DownloadState & Record<string, unknown>;
        setDlState({ status: s.status, progress: s.progress, message: s.message, rate_bps: s.rate_bps as number | undefined, eta_seconds: s.eta_seconds as number | undefined });
        if (s.status === 'complete') {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
          const fresh = await api.getDatasetStatus(name).catch(() => null);
          if (fresh) setStatuses(prev => ({ ...prev, [name]: { ...prev[name], available: true, meta: fresh.meta } }));
        } else if (s.status === 'error') {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
        }
      } catch { /* ignore */ }
    }, 600);
  }, [downloadTarget, urlInput]);

  const handleUpload = useCallback(async (file: File) => {
    if (!downloadTarget) return;
    const name = downloadTarget.name;
    setUploadState({ status: 'uploading', progress: 0, message: 'Starting upload...' });
    try {
      await api.uploadDataset(name, file, (pct, msg) => {
        setUploadState({ status: 'uploading', progress: pct, message: msg });
      });
    } catch (e: unknown) {
      setUploadState({ status: 'error', progress: 0, message: e instanceof Error ? e.message : 'Upload failed' });
      return;
    }
    // Poll extract progress
    setUploadState({ status: 'extracting', progress: 50, message: 'Extracting archive...' });
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const s = await api.getDatasetUploadStatus(name);
        setUploadState({ status: s.status as UploadState['status'], progress: s.progress, message: s.message, bytesReceived: s.bytes_received });
        if (s.status === 'complete') {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
          try {
            const fresh = await api.getDatasetStatus(name);
            setStatuses(prev => ({ ...prev, [name]: { ...prev[name], available: true, meta: fresh.meta } }));
          } catch { /* ignore */ }
        } else if (s.status === 'error') {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
        }
      } catch { /* ignore */ }
    }, 800);
  }, [downloadTarget]);

  const startDownload = useCallback(async () => {
    if (!downloadTarget) return;
    const name = downloadTarget.name;
    setDlState({ status: 'downloading', progress: 0, message: 'Starting download...' });

    try {
      await api.startDatasetDownload(name);
    } catch {
      setDlState({ status: 'error', progress: 0, message: 'Failed to start download' });
      return;
    }

    // Poll for progress
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const s = await api.getDatasetDownloadStatus(name) as DownloadState & Record<string, unknown>;
        setDlState({
          status: s.status, progress: s.progress, message: s.message,
          rate_bps: s.rate_bps as number | undefined,
          eta_seconds: s.eta_seconds as number | undefined,
          files: s.files as Record<string, FileProgress> | undefined,
          total_files: s.total_files as number | undefined,
          completed_files: s.completed_files as number | undefined,
        });

        if (s.status === 'complete') {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
          // Update status cache
          // Refresh status with meta
          try {
            const fresh = await api.getDatasetStatus(name);
            setStatuses((prev) => ({ ...prev, [name]: { ...prev[name], available: true, meta: fresh.meta } }));
          } catch {
            setStatuses((prev) => ({ ...prev, [name]: { ...prev[name], available: true } }));
          }
        } else if (s.status === 'error') {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
        }
      } catch {
        // ignore transient poll errors
      }
    }, 500);
  }, [downloadTarget]);

  const closeDialog = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = null;
    setDownloadTarget(null);
    setDlState({ status: 'idle', progress: 0, message: '' });
    setUploadState({ status: 'idle', progress: 0, message: '' });
  };

  const filtered = datasets.filter((ds) => {
    if (!search) return true;
    const q = search.toLowerCase();
    return ds.display_name.toLowerCase().includes(q)
      || ds.name.toLowerCase().includes(q)
      || ds.task_type.toLowerCase().includes(q)
      || (ds.class_names ?? ds.classes ?? []).some((c) => c.toLowerCase().includes(q));
  });

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-6xl mx-auto p-8 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">Datasets</h1>
            <p className="text-slate-500 text-sm mt-1">Available datasets for training and validation.</p>
          </div>
          <div className="relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              placeholder="Search datasets..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-9 pr-3 py-2 text-sm bg-slate-900 border border-slate-800 rounded-lg text-white placeholder-slate-600 focus:border-indigo-500 outline-none w-64"
            />
          </div>
        </div>

        {loading ? (
          <p className="text-slate-500 text-center py-20">Loading...</p>
        ) : filtered.length === 0 ? (
          <div className="text-center py-20">
            <Database size={48} className="mx-auto text-slate-700 mb-4" />
            <h3 className="text-lg font-semibold text-white mb-2">No datasets found</h3>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filtered.map((ds) => {
              const st = statuses[ds.name];
              const statusKnown = ds.name in statuses;
              const available = st?.available;
              const meta = st?.meta;
              const trainSplit = meta?.splits?.train;
              const testSplit = meta?.splits?.test;
              const valSplit = meta?.splits?.val;
              return (
                <div
                  key={ds.name}
                  className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3 hover:border-slate-700 transition-colors group cursor-pointer relative"
                  onClick={() => handleCardClick(ds)}
                >
                  {/* Delete button */}
                  {available && (
                    <button
                      onClick={(e) => { e.stopPropagation(); setDeleteTarget(ds); }}
                      className="absolute top-3 right-3 p-1.5 text-slate-600 hover:text-red-400 hover:bg-red-500/10 rounded-md transition-colors opacity-0 group-hover:opacity-100 cursor-pointer"
                      title="Delete dataset"
                    >
                      <Trash2 size={14} />
                    </button>
                  )}
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 shrink-0">
                      <Database size={20} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <h4 className="text-white font-semibold truncate">{ds.display_name}</h4>
                        {statusKnown && (
                          available ? (
                            <span className="shrink-0 flex items-center gap-1 text-[10px] text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded-full">
                              <CheckCircle size={10} /> Ready
                            </span>
                          ) : (
                            <span className="shrink-0 flex items-center gap-1 text-[10px] text-amber-400 bg-amber-500/10 px-1.5 py-0.5 rounded-full">
                              <XCircle size={10} /> Not Downloaded
                            </span>
                          )
                        )}
                      </div>
                      <p className="text-xs text-slate-500">{ds.task_type} · {ds.input_shape.join('×')}</p>
                    </div>
                    <ArrowRight size={16} className="text-slate-700 group-hover:text-indigo-400 transition-colors shrink-0" />
                  </div>

                  {/* Stats from meta or fallback to DatasetInfo */}
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                    <div className="flex justify-between">
                      <span className="text-slate-500">Train:</span>
                      <span className="text-slate-300">
                        {trainSplit ? trainSplit.labeled.toLocaleString() : ds.train_size.toLocaleString()}
                        {trainSplit && trainSplit.effective != null && trainSplit.effective !== trainSplit.labeled && (
                          <span className="text-indigo-400 ml-1">({trainSplit.effective.toLocaleString()})</span>
                        )}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Test:</span>
                      <span className="text-slate-300">
                        {testSplit ? testSplit.labeled.toLocaleString() : ds.test_size.toLocaleString()}
                        {testSplit && testSplit.effective != null && testSplit.effective !== testSplit.labeled && (
                          <span className="text-emerald-400 ml-1">({testSplit.effective.toLocaleString()})</span>
                        )}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Val:</span>
                      <span className="text-slate-300">
                        {valSplit ? valSplit.labeled.toLocaleString() : (ds.val_size || 0).toLocaleString()}
                        {valSplit && valSplit.effective != null && valSplit.effective !== valSplit.labeled && (
                          <span className="text-amber-400 ml-1">({valSplit.effective.toLocaleString()})</span>
                        )}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Classes:</span>
                      <span className="text-slate-300">{ds.num_classes}</span>
                    </div>
                  </div>

                  {/* Disk size + split bar */}
                  {meta && meta.disk_size_bytes > 0 && (
                    <div className="space-y-1.5">
                      <div className="flex items-center gap-1.5 text-[10px] text-slate-500">
                        <HardDrive size={10} /> {meta.disk_size_human}
                      </div>
                      {(trainSplit || testSplit) && (() => {
                        const tEff = trainSplit?.effective ?? trainSplit?.labeled ?? 0;
                        const vEff = valSplit?.effective ?? valSplit?.labeled ?? 0;
                        const teEff = testSplit?.effective ?? testSplit?.labeled ?? 0;
                        const total = tEff + vEff + teEff;
                        if (total === 0) return null;
                        return (
                          <div className="flex rounded-full overflow-hidden h-1.5 bg-slate-800">
                            {tEff > 0 && <div className="bg-indigo-500 transition-all" style={{ width: `${(tEff / total) * 100}%` }} />}
                            {vEff > 0 && <div className="bg-amber-500 transition-all" style={{ width: `${(vEff / total) * 100}%` }} />}
                            {teEff > 0 && <div className="bg-emerald-500 transition-all" style={{ width: `${(teEff / total) * 100}%` }} />}
                          </div>
                        );
                      })()}
                    </div>
                  )}

                  <div className="flex flex-wrap gap-1">
                    {(ds.class_names ?? ds.classes ?? []).slice(0, 6).map((c) => (
                      <span key={c} className="px-2 py-0.5 text-[10px] bg-slate-800 text-slate-400 rounded-full">{c}</span>
                    ))}
                    {(ds.class_names ?? ds.classes ?? []).length > 6 && <span className="px-2 py-0.5 text-[10px] text-slate-500">+{(ds.class_names ?? ds.classes ?? []).length - 6}</span>}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Delete Confirm Dialog */}
      {deleteTarget && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={() => !deleting && setDeleteTarget(null)}>
          <div className="bg-slate-900 border border-slate-800 rounded-xl max-w-sm w-full shadow-2xl p-6 space-y-4" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-red-500/10 border border-red-500/20 flex items-center justify-center text-red-400 shrink-0">
                <Trash2 size={20} />
              </div>
              <div>
                <h3 className="text-white font-semibold">Delete {deleteTarget.display_name}?</h3>
                <p className="text-xs text-slate-500">Downloaded files will be removed. You can re-download later.</p>
              </div>
            </div>
            {statuses[deleteTarget.name]?.meta && (
              <p className="text-xs text-slate-400">
                This will free <span className="text-white font-medium">{statuses[deleteTarget.name]!.meta!.disk_size_human}</span> of disk space.
              </p>
            )}
            <div className="flex items-center justify-end gap-2">
              <button
                onClick={() => setDeleteTarget(null)}
                disabled={deleting}
                className="px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors cursor-pointer"
              >
                Cancel
              </button>
              <button
                onClick={handleDelete}
                disabled={deleting}
                className="flex items-center gap-2 px-4 py-2 text-sm bg-red-600 hover:bg-red-500 text-white rounded-lg transition-colors cursor-pointer"
              >
                {deleting ? <Loader2 size={14} className="animate-spin" /> : <Trash2 size={14} />}
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Download Dialog */}
      {downloadTarget && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={dlState.status === 'downloading' ? undefined : closeDialog}>
          <div className="bg-slate-900 border border-slate-800 rounded-xl max-w-md w-full shadow-2xl" onClick={(e) => e.stopPropagation()}>
            <div className="p-6 space-y-4">
              {/* Header */}
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 shrink-0">
                  <Download size={20} />
                </div>
                <div>
                  <h3 className="text-white font-semibold">{downloadTarget.display_name}</h3>
                  <p className="text-xs text-slate-500">
                    {downloadTarget.task_type} · {downloadTarget.input_shape.join('×')} · {downloadTarget.num_classes} classes
                  </p>
                </div>
              </div>

              {/* ── Workspace scanning spinner ── */}
              {wsScan.scanning && (
                <div className="flex items-center gap-2 py-2 text-slate-400 text-sm">
                  <Loader2 size={14} className="animate-spin shrink-0" /> Scanning workspace...
                </div>
              )}

              {/* ── Pending archive banner (resume extract) ── */}
              {!wsScan.scanning && wsScan.pending_archive && installMethod === 'workspace' && uploadState.status === 'idle' && (
                <div className="space-y-3">
                  <div className="flex items-start gap-2 p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg">
                    <HardDrive size={16} className="text-amber-400 shrink-0 mt-0.5" />
                    <div className="text-sm flex-1 min-w-0">
                      <span className="text-amber-400 font-medium">Downloaded archive found</span>
                      <p className="text-slate-400 text-xs mt-0.5 truncate">{wsScan.pending_archive.name}</p>
                      <p className="text-slate-500 text-xs">{(wsScan.pending_archive.size_bytes / (1024 * 1024 * 1024)).toFixed(2)} GB — ready to extract</p>
                    </div>
                  </div>
                  <p className="text-xs text-slate-500">A previously downloaded archive was found. Click Resume to extract and register the dataset.</p>
                  <div className="flex gap-2 pt-1">
                    <button onClick={() => setInstallMethod('choose')} className="text-xs text-slate-500 hover:text-slate-300 cursor-pointer">Use different method</button>
                  </div>
                </div>
              )}

              {/* ── Workspace found banner (extracted files) ── */}
              {!wsScan.scanning && wsScan.found && !wsScan.pending_archive && installMethod === 'workspace' && uploadState.status === 'idle' && (
                <div className="space-y-3">
                  <div className="flex items-center gap-2 p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
                    <FolderSearch size={16} className="text-emerald-400 shrink-0" />
                    <div className="text-sm">
                      <span className="text-emerald-400 font-medium">Files found in workspace</span>
                      <span className="text-slate-400 ml-2">({wsScan.file_count.toLocaleString()} files, {wsScan.dir_count} dirs)</span>
                    </div>
                  </div>
                  <p className="text-xs text-slate-500">Dataset files already exist in the server workspace. Click Import to build the index.</p>
                  <div className="flex gap-2 pt-1">
                    <button onClick={() => setInstallMethod('choose')} className="text-xs text-slate-500 hover:text-slate-300 cursor-pointer">Use different method</button>
                  </div>
                </div>
              )}

              {/* ── Method selector ── */}
              {!wsScan.scanning && installMethod === 'choose' && uploadState.status === 'idle' && dlState.status === 'idle' && (
                <div className="space-y-3">
                  {manualInstructions && (
                    <pre className="text-xs text-slate-400 bg-slate-800/80 rounded-lg p-3 whitespace-pre-wrap leading-relaxed">{manualInstructions}</pre>
                  )}
                  <p className="text-xs text-slate-500 uppercase tracking-wider font-medium">Choose installation method</p>
                  <div className="grid grid-cols-1 gap-2">
                    <button
                      onClick={() => setInstallMethod('url')}
                      className="flex items-center gap-3 p-3 bg-slate-800 hover:bg-slate-700 border border-slate-700 hover:border-indigo-500 rounded-lg text-left transition-colors cursor-pointer"
                    >
                      <Link size={18} className="text-indigo-400 shrink-0" />
                      <div>
                        <div className="text-sm text-white font-medium">Download from URL</div>
                        <div className="text-xs text-slate-400">Google Drive, direct link — fastest for remote servers</div>
                      </div>
                    </button>
                    <button
                      onClick={() => setInstallMethod('upload')}
                      className="flex items-center gap-3 p-3 bg-slate-800 hover:bg-slate-700 border border-slate-700 hover:border-slate-600 rounded-lg text-left transition-colors cursor-pointer"
                    >
                      <Upload size={18} className="text-slate-400 shrink-0" />
                      <div>
                        <div className="text-sm text-white font-medium">Upload archive</div>
                        <div className="text-xs text-slate-400">Upload .zip / .tar.gz from your computer (slower over internet)</div>
                      </div>
                    </button>
                  </div>
                </div>
              )}

              {/* ── URL input ── */}
              {installMethod === 'url' && dlState.status === 'idle' && (
                <div className="space-y-3">
                  <button onClick={() => setInstallMethod('choose')} className="text-xs text-slate-500 hover:text-slate-300 cursor-pointer">← Back</button>
                  <label className="block text-xs text-slate-400">Download URL</label>
                  <input
                    type="url"
                    value={urlInput}
                    onChange={e => setUrlInput(e.target.value)}
                    placeholder="https://drive.google.com/file/d/... or direct link"
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-indigo-500"
                  />
                  <p className="text-xs text-slate-500">Google Drive share links are resolved automatically. Use File → Share → Copy link.</p>
                </div>
              )}

              {/* ── Upload dropzone ── */}
              {installMethod === 'upload' && uploadState.status === 'idle' && (
                <div className="space-y-3">
                  <button onClick={() => setInstallMethod('choose')} className="text-xs text-slate-500 hover:text-slate-300 cursor-pointer">← Back</button>
                  <label
                    className="flex flex-col items-center justify-center gap-2 border-2 border-dashed border-slate-700 hover:border-indigo-500 rounded-xl p-6 cursor-pointer transition-colors text-slate-500 hover:text-slate-300"
                    onDragOver={e => e.preventDefault()}
                    onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) handleUpload(f); }}
                  >
                    <Upload size={24} />
                    <span className="text-sm font-medium">Drop archive here or click to browse</span>
                    <span className="text-xs">Supports .zip, .tar.gz, .tar.bz2</span>
                    <input ref={fileInputRef} type="file" accept=".zip,.tar,.tar.gz,.tgz,.tar.bz2" className="hidden"
                      onChange={e => { const f = e.target.files?.[0]; if (f) handleUpload(f); }} />
                  </label>
                </div>
              )}

              {/* ── Upload / index progress ── */}
              {(uploadState.status === 'uploading' || uploadState.status === 'extracting' || uploadState.status === 'indexing') && (
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <Loader2 size={14} className="text-indigo-400 animate-spin shrink-0" />
                    <p className="text-sm text-slate-300 truncate">{uploadState.message}</p>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                    <div className={`h-full rounded-full transition-all duration-300 ${
                      uploadState.status === 'extracting' ? 'bg-amber-500' : 'bg-indigo-500'
                    }`} style={{ width: `${Math.max(2, uploadState.progress)}%` }} />
                  </div>
                  <div className="flex justify-between text-xs text-slate-500">
                    <span>{uploadState.status === 'uploading' ? 'Uploading…' : uploadState.status === 'extracting' ? 'Extracting…' : 'Indexing…'}</span>
                    <span>{uploadState.progress}%</span>
                  </div>
                </div>
              )}
              {uploadState.status === 'complete' && (
                <div className="flex items-center gap-2 py-2">
                  <CheckCircle size={16} className="text-emerald-400 shrink-0" />
                  <p className="text-sm text-emerald-400">Dataset imported successfully!</p>
                </div>
              )}
              {uploadState.status === 'error' && (
                <div className="flex items-start gap-2">
                  <AlertCircle size={16} className="text-red-400 shrink-0 mt-0.5" />
                  <p className="text-sm text-red-400">{uploadState.message}</p>
                </div>
              )}
              {dlState.status === 'idle' && !isManual && (
                <>
                  <p className="text-sm text-slate-400">
                    This dataset is not downloaded yet. Download it to <code className="text-xs bg-slate-800 px-1.5 py-0.5 rounded text-slate-300">backend/data/datasets/</code>?
                  </p>
                  <div className="text-xs text-slate-500 space-y-1">
                    <div className="flex justify-between"><span>Train samples:</span><span className="text-slate-300">{downloadTarget.train_size.toLocaleString()}</span></div>
                    <div className="flex justify-between"><span>Test samples:</span><span className="text-slate-300">{downloadTarget.test_size.toLocaleString()}</span></div>
                  </div>
                </>
              )}

              {dlState.status === 'downloading' && (() => {
                const activeFiles = dlState.files ? Object.entries(dlState.files) : [];
                const formatEta = (s: number) => s < 60 ? `${Math.round(s)}s` : s < 3600 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
                return (
                  <div className="space-y-3">
                    {activeFiles.length <= 1 ? (
                      /* Single file: show classic single bar */
                      <>
                        <div className="flex items-center gap-2">
                          <Loader2 size={14} className="text-indigo-400 animate-spin shrink-0" />
                          <p className="text-sm text-slate-300 truncate">
                            {dlState.total_files != null && dlState.total_files > 0 && (
                              <span className="text-indigo-400 font-medium mr-1.5">[{dlState.completed_files ?? 0}/{dlState.total_files}]</span>
                            )}
                            {dlState.message}
                          </p>
                        </div>
                        <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                          <div className="h-full bg-indigo-500 rounded-full transition-all duration-300" style={{ width: `${Math.max(2, dlState.progress)}%` }} />
                        </div>
                        <div className="flex items-center justify-between text-xs text-slate-500 tabular-nums">
                          <span>
                            {dlState.rate_bps != null && dlState.rate_bps > 0 ? `${(dlState.rate_bps / (1024 * 1024)).toFixed(1)} MB/s` : ''}
                            {dlState.eta_seconds != null && dlState.eta_seconds >= 0 ? ` · ${formatEta(dlState.eta_seconds)} left` : ''}
                          </span>
                          <span>{dlState.progress.toFixed(1)}%</span>
                        </div>
                      </>
                    ) : (
                      /* Multiple concurrent files: show per-file bars */
                      <>
                        <div className="flex items-center gap-2 mb-1">
                          <Loader2 size={14} className="text-indigo-400 animate-spin shrink-0" />
                          <p className="text-sm text-slate-300">
                            Downloading{dlState.total_files != null && dlState.total_files > 0 ? ` — ${dlState.completed_files ?? 0}/${dlState.total_files} completed` : ` ${activeFiles.length} files...`}
                          </p>
                        </div>
                        <div className="space-y-2.5">
                          {activeFiles.map(([key, f]) => (
                            <div key={key} className="space-y-1">
                              <div className="flex items-center justify-between text-xs">
                                <span className="text-slate-300 truncate font-medium">{f.name}</span>
                                <span className="text-slate-500 tabular-nums shrink-0 ml-2">
                                  {f.status === 'extracting' ? 'Extracting...' : `${f.progress.toFixed(1)}%`}
                                </span>
                              </div>
                              <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                                <div className={`h-full rounded-full transition-all duration-300 ${
                                  f.status === 'extracting' ? 'bg-amber-500 animate-pulse' : 'bg-indigo-500'
                                }`} style={{ width: `${Math.max(2, f.progress)}%` }} />
                              </div>
                              <div className="flex items-center justify-between text-[10px] text-slate-600 tabular-nums">
                                <span>{f.message}</span>
                                <span>{f.eta_seconds >= 0 ? `${formatEta(f.eta_seconds)} left` : ''}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </>
                    )}
                  </div>
                );
              })()}

              {dlState.status === 'complete' && (
                <div className="flex items-center gap-2 py-2">
                  <CheckCircle size={16} className="text-emerald-400 shrink-0" />
                  <p className="text-sm text-emerald-400">Download complete!</p>
                </div>
              )}

              {dlState.status === 'error' && (
                <div className="space-y-2">
                  <div className="flex items-start gap-2">
                    <AlertCircle size={16} className="text-red-400 shrink-0 mt-0.5" />
                    <p className="text-sm text-red-400">{dlState.message}</p>
                  </div>
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="border-t border-slate-800 px-6 py-4 flex items-center justify-end gap-2">
              {/* In-progress states */}
              {(uploadState.status === 'uploading' || uploadState.status === 'extracting' || uploadState.status === 'indexing' || dlState.status === 'downloading') && (
                <p className="text-xs text-slate-600">Please wait…</p>
              )}

              {/* Upload/index complete */}
              {uploadState.status === 'complete' && (
                <>
                  <button onClick={closeDialog} className="px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors cursor-pointer">Close</button>
                  <button onClick={() => { closeDialog(); onOpenDataset?.(downloadTarget!.name); }}
                    className="flex items-center gap-2 px-4 py-2 text-sm bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-colors cursor-pointer">
                    Open Dataset <ArrowRight size={14} />
                  </button>
                </>
              )}
              {uploadState.status === 'error' && (
                <>
                  <button onClick={() => { setUploadState({ status: 'idle', progress: 0, message: '' }); setInstallMethod('choose'); }} className="px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors cursor-pointer">← Back</button>
                  <button onClick={closeDialog} className="px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors cursor-pointer">Close</button>
                </>
              )}

              {/* URL download complete/error */}
              {dlState.status === 'complete' && (
                <>
                  <button onClick={closeDialog} className="px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors cursor-pointer">Close</button>
                  <button onClick={() => { closeDialog(); onOpenDataset?.(downloadTarget!.name); }}
                    className="flex items-center gap-2 px-4 py-2 text-sm bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-colors cursor-pointer">
                    Open Dataset <ArrowRight size={14} />
                  </button>
                </>
              )}
              {dlState.status === 'error' && (
                <>
                  <button onClick={() => { setDlState({ status: 'idle', progress: 0, message: '' }); setInstallMethod('url'); }} className="px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors cursor-pointer">← Back</button>
                  <button onClick={closeDialog} className="px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors cursor-pointer">Close</button>
                </>
              )}

              {/* Idle states */}
              {uploadState.status === 'idle' && dlState.status === 'idle' && !wsScan.scanning && (
                <>
                  {/* Workspace found → Resume Extract or Import */}
                  {installMethod === 'workspace' && (
                    <>
                      <button onClick={closeDialog} className="px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors cursor-pointer">Cancel</button>
                      {wsScan.pending_archive ? (
                        <button onClick={handleResumeExtract}
                          className="flex items-center gap-2 px-4 py-2 text-sm bg-amber-600 hover:bg-amber-500 text-white rounded-lg transition-colors cursor-pointer">
                          <HardDrive size={14} /> Resume Extract
                        </button>
                      ) : (
                        <button onClick={handleImportLocal}
                          className="flex items-center gap-2 px-4 py-2 text-sm bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg transition-colors cursor-pointer">
                          <FolderSearch size={14} /> Import from workspace
                        </button>
                      )}
                    </>
                  )}
                  {/* Method selector / instructions */}
                  {(installMethod === 'choose' || installMethod === 'upload') && (
                    <button onClick={closeDialog} className="px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors cursor-pointer">Cancel</button>
                  )}
                  {/* URL → Download button */}
                  {installMethod === 'url' && (
                    <>
                      <button onClick={closeDialog} className="px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors cursor-pointer">Cancel</button>
                      <button onClick={handleUrlDownload} disabled={!urlInput.trim()}
                        className="flex items-center gap-2 px-4 py-2 text-sm bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-lg transition-colors cursor-pointer">
                        <Download size={14} /> Download
                      </button>
                    </>
                  )}
                  {/* Auto-downloadable dataset */}
                  {!isManual && (
                    <>
                      <button onClick={closeDialog} className="px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors cursor-pointer">Cancel</button>
                      <button onClick={startDownload}
                        className="flex items-center gap-2 px-4 py-2 text-sm bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-colors cursor-pointer">
                        <Download size={14} /> Download
                      </button>
                    </>
                  )}
                </>
              )}
              {wsScan.scanning && (
                <button onClick={closeDialog} className="px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors cursor-pointer">Cancel</button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
