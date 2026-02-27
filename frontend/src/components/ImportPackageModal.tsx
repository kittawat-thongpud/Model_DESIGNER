import { useState, useRef } from 'react';
import { X, Upload, Package, CheckCircle2, AlertTriangle, Loader2, Pencil } from 'lucide-react';
import { api } from '../services/api';

type Step = 'select' | 'rename' | 'done';

interface WeightInfo {
  id: string;
  model_name: string;
  dataset: string;
  epochs_trained: number;
}

interface ImportResult {
  weights_imported: { old_id: string; new_id: string; name: string }[];
  jobs_imported: { old_id: string; new_id: string }[];
  errors: string[];
}

interface Props {
  onClose: () => void;
  onDone?: () => void;
}

export default function ImportPackageModal({ onClose, onDone }: Props) {
  const [step, setStep] = useState<Step>('select');
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [peeking, setPeeking] = useState(false);
  const [weights, setWeights] = useState<WeightInfo[]>([]);
  const [jobCount, setJobCount] = useState(0);
  const [names, setNames] = useState<Record<string, string>>({});   // {old_id: display_name}
  const [includeJobs, setIncludeJobs] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ImportResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFile = async (f: File) => {
    if (!f.name.endsWith('.mdpkg') && !f.name.endsWith('.zip')) {
      setError('Please select a .mdpkg file');
      return;
    }
    setFile(f);
    setError(null);
    setPeeking(true);
    try {
      const info = await api.peekPackage(f);
      setWeights(info.weights);
      setJobCount(info.jobs.length);
      // initialise name map with original names
      const m: Record<string, string> = {};
      for (const w of info.weights) m[w.id] = w.model_name;
      setNames(m);
      setStep('rename');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to read package');
    } finally {
      setPeeking(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };

  const handleImport = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      // Only send entries that differ from originals (or send all — backend handles it)
      const renameMap: Record<string, string> = {};
      for (const w of weights) {
        if (names[w.id] && names[w.id] !== w.model_name) {
          renameMap[w.id] = names[w.id].trim();
        }
      }
      const res = await api.importPackage(file, renameMap, includeJobs);
      setResult(res);
      setStep('done');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Import failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl w-full max-w-lg mx-4"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Package size={18} className="text-indigo-400" />
            <h3 className="text-white font-semibold">Import Package</h3>
            {step === 'rename' && <span className="text-xs text-slate-500 ml-1">— Review & Rename</span>}
            {step === 'done'   && <span className="text-xs text-emerald-400 ml-1">— Complete</span>}
          </div>
          <button onClick={onClose} className="text-slate-500 hover:text-white cursor-pointer transition-colors">
            <X size={18} />
          </button>
        </div>

        <div className="p-6 space-y-4">

          {/* ── STEP 1: select file ── */}
          {step === 'select' && (
            <div
              className={`border-2 border-dashed rounded-xl p-8 text-center transition-all cursor-pointer ${
                dragging ? 'border-indigo-400 bg-indigo-500/10' : 'border-slate-700 hover:border-slate-600 bg-slate-800/30'
              }`}
              onClick={() => fileRef.current?.click()}
              onDragOver={e => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
            >
              <input ref={fileRef} type="file" accept=".mdpkg,.zip" className="hidden"
                onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
              {peeking ? (
                <div className="flex flex-col items-center gap-2">
                  <Loader2 size={28} className="text-indigo-400 animate-spin" />
                  <p className="text-slate-400 text-sm">Reading package…</p>
                </div>
              ) : (
                <div className="space-y-2">
                  <Upload size={28} className="text-slate-500 mx-auto" />
                  <p className="text-slate-400 text-sm">Drop a <span className="text-indigo-400 font-mono">.mdpkg</span> file here</p>
                  <p className="text-slate-600 text-xs">or click to browse</p>
                </div>
              )}
            </div>
          )}

          {/* ── STEP 2: rename weights ── */}
          {step === 'rename' && (
            <div className="space-y-4">
              <p className="text-xs text-slate-500">
                Package contains <span className="text-white">{weights.length}</span> weight{weights.length !== 1 ? 's' : ''} and <span className="text-white">{jobCount}</span> job{jobCount !== 1 ? 's' : ''}.
                Each will get a <span className="text-indigo-400">new ID</span> — existing data will not be overwritten.
                Optionally rename any weight below.
              </p>

              <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                {weights.map((w) => (
                  <div key={w.id} className="bg-slate-800/60 border border-slate-700/50 rounded-lg p-3 space-y-1.5">
                    <div className="flex items-center gap-2 text-xs text-slate-500">
                      <span className="font-mono text-slate-600">{w.id.slice(0, 10)}</span>
                      <span>·</span>
                      <span>{w.dataset || '—'}</span>
                      {w.epochs_trained > 0 && <><span>·</span><span>{w.epochs_trained} ep</span></>}
                    </div>
                    <div className="flex items-center gap-2">
                      <Pencil size={13} className="text-slate-500 shrink-0" />
                      <input
                        type="text"
                        value={names[w.id] ?? w.model_name}
                        onChange={e => setNames(prev => ({ ...prev, [w.id]: e.target.value }))}
                        placeholder={w.model_name}
                        className="flex-1 bg-slate-900 border border-slate-700 focus:border-indigo-500 rounded-lg px-3 py-1.5 text-sm text-white outline-none transition-colors"
                      />
                    </div>
                  </div>
                ))}
              </div>

              {/* Include Jobs toggle */}
              <div className="flex items-center justify-between pt-1 border-t border-slate-800">
                <div>
                  <div className="text-xs text-slate-300 font-medium">Include Jobs</div>
                  <div className="text-[10px] text-slate-500">Import training job records (larger file)</div>
                </div>
                <button
                  onClick={() => setIncludeJobs(v => !v)}
                  className={`w-9 h-5 rounded-full relative transition-colors cursor-pointer shrink-0 ${includeJobs ? 'bg-violet-500' : 'bg-slate-600'}`}
                >
                  <div className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${includeJobs ? 'translate-x-4' : 'translate-x-0.5'}`} />
                </button>
              </div>

              <p className="text-[10px] text-slate-600">Leave blank or unchanged to keep the original name.</p>
            </div>
          )}

          {/* ── STEP 3: result ── */}
          {step === 'done' && result && (
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-emerald-400">
                <CheckCircle2 size={18} />
                <span className="font-semibold">Import complete</span>
              </div>

              {result.weights_imported.length > 0 && (
                <div className="bg-slate-800/60 rounded-lg p-3 border border-slate-700/50 space-y-1">
                  <div className="text-xs text-slate-500 uppercase font-bold mb-2">Weights imported</div>
                  {result.weights_imported.map(w => (
                    <div key={w.new_id} className="flex items-center gap-2 text-xs">
                      <CheckCircle2 size={10} className="text-emerald-400 shrink-0" />
                      <span className="text-white font-medium">{w.name}</span>
                      <span className="text-slate-600 font-mono">{w.new_id.slice(0, 10)}</span>
                    </div>
                  ))}
                </div>
              )}

              {result.jobs_imported.length > 0 && (
                <div className="bg-slate-800/60 rounded-lg p-3 border border-slate-700/50 space-y-1">
                  <div className="text-xs text-slate-500 uppercase font-bold mb-2">Jobs imported</div>
                  {result.jobs_imported.map(j => (
                    <div key={j.new_id} className="flex items-center gap-2 text-xs">
                      <CheckCircle2 size={10} className="text-emerald-400 shrink-0" />
                      <span className="text-slate-300 font-mono">{j.new_id.slice(0, 10)}</span>
                    </div>
                  ))}
                </div>
              )}

              {result.errors.length > 0 && (
                <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 text-xs text-red-400 space-y-1">
                  {result.errors.map((e, i) => (
                    <div key={i} className="flex items-start gap-1.5">
                      <AlertTriangle size={10} className="shrink-0 mt-0.5" /> {e}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="flex items-start gap-2 bg-red-500/10 border border-red-500/20 rounded-lg p-3 text-red-400 text-sm">
              <AlertTriangle size={14} className="shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 pb-5 flex justify-end gap-2">
          {step === 'done' ? (
            <button onClick={() => { onClose(); if (onDone) onDone(); }}
              className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-medium transition-colors cursor-pointer">
              Done
            </button>
          ) : (
            <>
              <button onClick={onClose}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-sm transition-colors cursor-pointer border border-slate-700">
                Cancel
              </button>
              {step === 'rename' && (
                <button onClick={handleImport} disabled={loading}
                  className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-2 cursor-pointer">
                  {loading ? <Loader2 size={14} className="animate-spin" /> : <Package size={14} />}
                  {loading ? 'Importing…' : 'Import'}
                </button>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
