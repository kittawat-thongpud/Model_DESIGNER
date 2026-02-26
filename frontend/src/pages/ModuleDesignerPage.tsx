/**
 * Module Designer Page — custom nn.Module block editor.
 *
 * Create and manage custom PyTorch modules that get registered
 * into the Ultralytics namespace for use in model YAML configs.
 */
import { useState, useEffect } from 'react';
import { api } from '../services/api';
import type { ModuleSummary } from '../types';

interface Props {
  onBack: () => void;
}

export default function ModuleDesignerPage({ onBack }: Props) {
  const [modules, setModules] = useState<ModuleSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.listModules().then(setModules).finally(() => setLoading(false));
  }, []);

  return (
    <div className="flex-1 flex flex-col p-6 overflow-auto">
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <button onClick={onBack} className="text-slate-400 hover:text-white transition-colors">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
        </button>
        <h1 className="text-2xl font-bold text-white">Module Designer</h1>
        <span className="text-sm text-slate-500">Custom nn.Module Blocks</span>
      </div>

      {loading ? (
        <div className="flex items-center justify-center flex-1">
          <div className="animate-spin w-8 h-8 border-2 border-purple-500 border-t-transparent rounded-full" />
        </div>
      ) : (
        <div className="grid grid-cols-12 gap-6 flex-1 min-h-0">
          {/* Module List */}
          <div className="col-span-4 bg-slate-900/50 rounded-xl border border-slate-800 p-4 overflow-auto">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">
                Custom Modules ({modules.length})
              </h2>
              <button className="px-3 py-1.5 text-xs font-medium rounded-lg bg-purple-500/10 text-purple-400 border border-purple-500/20 hover:bg-purple-500/20 transition-colors">
                + New Module
              </button>
            </div>
            {modules.length === 0 ? (
              <div className="text-center py-12">
                <div className="w-12 h-12 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center mx-auto mb-3">
                  <svg className="w-6 h-6 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M14.25 9.75L16.5 12l-2.25 2.25m-4.5 0L7.5 12l2.25-2.25M6 20.25h12A2.25 2.25 0 0020.25 18V6A2.25 2.25 0 0018 3.75H6A2.25 2.25 0 003.75 6v12A2.25 2.25 0 006 20.25z" />
                  </svg>
                </div>
                <p className="text-sm text-slate-500">No custom modules yet</p>
                <p className="text-xs text-slate-600 mt-1">Create your first nn.Module block</p>
              </div>
            ) : (
              <div className="space-y-2">
                {modules.map(m => (
                  <div key={m.module_id} className="px-3 py-3 rounded-lg bg-slate-800/50 hover:bg-slate-700/50 cursor-pointer transition-colors">
                    <div className="text-sm text-slate-300 font-mono font-medium">{m.name}</div>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-400">{m.category}</span>
                      <span className="text-[10px] text-slate-600">{m.arg_count} args</span>
                    </div>
                    {m.description && (
                      <p className="text-xs text-slate-500 mt-1 truncate">{m.description}</p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Code Editor Area */}
          <div className="col-span-8 bg-slate-900/50 rounded-xl border border-slate-800 p-6 flex flex-col items-center justify-center">
            <div className="text-center max-w-md">
              <div className="w-16 h-16 rounded-2xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" />
                </svg>
              </div>
              <h2 className="text-xl font-semibold text-white mb-2">Module Code Editor</h2>
              <p className="text-sm text-slate-400 mb-4">
                Write custom <code className="text-purple-400">nn.Module</code> classes that integrate
                directly into the Ultralytics model YAML system.
              </p>
              <p className="text-xs text-slate-500">
                Select a module from the list or create a new one to start editing.
                Modules are automatically registered into the Ultralytics namespace.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
