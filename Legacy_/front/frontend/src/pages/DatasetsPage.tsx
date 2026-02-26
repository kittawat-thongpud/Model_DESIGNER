/**
 * Datasets Page — displays available datasets with their metadata.
 */
import { useEffect, useState } from 'react';
import { api } from '../services/api';
import type { DatasetInfo } from '../types';
import { Database, RefreshCw } from 'lucide-react';

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [loading, setLoading] = useState(true);

  const loadDatasets = async () => {
    try {
      const data = await api.listDatasets();
      setDatasets(data);
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  useEffect(() => {
    loadDatasets();
  }, []);

  return (
    <div className="flex-1 overflow-y-auto p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">Datasets</h1>
            <p className="text-slate-400 text-sm mt-1">Available datasets for training and validation</p>
          </div>
          <button onClick={loadDatasets} className="px-4 py-2 text-sm font-medium text-slate-300 hover:text-white hover:bg-slate-800 rounded-lg transition-colors flex items-center gap-2 cursor-pointer">
            <RefreshCw size={14} /> Refresh
          </button>
        </div>

        {loading ? (
          <div className="text-center text-slate-500 py-20">Loading datasets...</div>
        ) : datasets.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-24 text-center">
            <Database size={48} className="text-slate-700 mb-4" />
            <p className="text-slate-400 text-sm">No datasets available</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {datasets.map((ds) => (
              <div key={ds.name} className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-lg hover:border-slate-600 transition-all">
                <div className="p-5 border-b border-slate-800 flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
                    <Database size={18} />
                  </div>
                  <h3 className="text-white font-semibold text-sm">{ds.display_name}</h3>
                </div>
                <div className="p-4 space-y-3">
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div>
                      <p className="text-slate-500 mb-0.5">Input Shape</p>
                      <p className="text-white font-mono">{ds.input_shape.join(' × ')}</p>
                    </div>
                    <div>
                      <p className="text-slate-500 mb-0.5">Classes</p>
                      <p className="text-white font-bold">{ds.num_classes}</p>
                    </div>
                    <div>
                      <p className="text-slate-500 mb-0.5">Train Size</p>
                      <p className="text-white">{ds.train_size.toLocaleString()}</p>
                    </div>
                    <div>
                      <p className="text-slate-500 mb-0.5">Test Size</p>
                      <p className="text-white">{ds.test_size.toLocaleString()}</p>
                    </div>
                  </div>
                </div>
                <div className="px-4 pb-4">
                  <div className="flex flex-wrap gap-1">
                    {ds.classes.slice(0, 8).map((cls) => (
                      <span key={cls} className="px-1.5 py-0.5 rounded text-[9px] bg-slate-800 text-slate-400 border border-slate-700">{cls}</span>
                    ))}
                    {ds.classes.length > 8 && <span className="px-1.5 py-0.5 rounded text-[9px] bg-slate-800 text-slate-500">+{ds.classes.length - 8}</span>}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
