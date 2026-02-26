/**
 * WeightTransferCard — inspect keys, extract partial weights, and transfer
 * weights between weight files. Extracted from WeightDetailPage.tsx.
 */
import { useState } from 'react';
import { api } from '../services/api';
import { RefreshCw } from 'lucide-react';

interface Props {
  weightId: string;
}

export default function WeightTransferCard({ weightId }: Props) {
  const [keys, setKeys] = useState<{ key: string; node_id: string; shape: number[]; numel: number }[]>([]);
  const [showKeys, setShowKeys] = useState(false);
  const [loadingKeys, setLoadingKeys] = useState(false);

  // Extract state
  const [extractIds, setExtractIds] = useState('');
  const [extracting, setExtracting] = useState(false);
  const [extractResult, setExtractResult] = useState<string | null>(null);

  // Transfer state
  const [sourceId, setSourceId] = useState('');
  const [transferring, setTransferring] = useState(false);
  const [transferResult, setTransferResult] = useState<string | null>(null);

  const handleInspect = async () => {
    setLoadingKeys(true);
    try {
      const data = await api.inspectWeightKeys(weightId);
      setKeys(data);
      setShowKeys(true);
    } catch { /* ignore */ }
    setLoadingKeys(false);
  };

  const handleExtract = async () => {
    const ids = extractIds.split(',').map((s) => s.trim()).filter(Boolean);
    if (!ids.length) return;
    setExtracting(true);
    setExtractResult(null);
    try {
      const res = await api.extractPartialWeight(weightId, ids);
      setExtractResult(`Extracted ${res.keys_extracted} keys → ${res.weight_id.slice(0, 8)}`);
    } catch (e: unknown) {
      setExtractResult(e instanceof Error ? e.message : 'Failed');
    }
    setExtracting(false);
  };

  const handleTransfer = async () => {
    if (!sourceId.trim()) return;
    setTransferring(true);
    setTransferResult(null);
    try {
      const res = await api.transferWeights(weightId, sourceId.trim());
      setTransferResult(`Matched ${res.matched_keys}/${res.total_target_keys} keys (${(res.match_ratio * 100).toFixed(0)}%)`);
    } catch (e: unknown) {
      setTransferResult(e instanceof Error ? e.message : 'Failed');
    }
    setTransferring(false);
  };

  // Group keys by node_id for display
  const nodeGroups = keys.reduce<Record<string, number>>((acc, k) => {
    acc[k.node_id] = (acc[k.node_id] || 0) + k.numel;
    return acc;
  }, {});

  return (
    <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 space-y-4">
      <h3 className="text-white font-semibold flex items-center gap-2">
        <RefreshCw size={18} className="text-amber-400" /> Weight Transfer
      </h3>

      {/* Inspect */}
      <div>
        <button
          onClick={handleInspect}
          disabled={loadingKeys}
          className="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded-lg transition-colors cursor-pointer border border-slate-700"
        >
          {loadingKeys ? 'Loading...' : showKeys ? 'Refresh Keys' : 'Inspect Keys'}
        </button>
        {showKeys && keys.length > 0 && (
          <div className="mt-2 max-h-32 overflow-y-auto text-[10px] font-mono space-y-0.5">
            {Object.entries(nodeGroups).map(([nid, numel]) => (
              <div key={nid} className="flex justify-between text-slate-400 py-0.5 border-b border-slate-800/50">
                <span className="text-slate-300">{nid}</span>
                <span>{(numel / 1000).toFixed(1)}K params</span>
              </div>
            ))}
            <div className="text-slate-500 pt-1">{keys.length} total keys</div>
          </div>
        )}
      </div>

      {/* Extract Partial */}
      <div className="space-y-1.5">
        <div className="text-[11px] font-medium text-slate-400">Extract by Node IDs</div>
        <div className="flex gap-2">
          <input
            value={extractIds}
            onChange={(e) => setExtractIds(e.target.value)}
            placeholder="n2, n3, n4"
            className="flex-1 px-2 py-1.5 text-xs bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-indigo-500 focus:outline-none"
          />
          <button
            onClick={handleExtract}
            disabled={extracting}
            className="text-xs bg-indigo-600 hover:bg-indigo-500 text-white px-3 py-1.5 rounded-lg transition-colors cursor-pointer disabled:opacity-50"
          >
            {extracting ? '...' : 'Extract'}
          </button>
        </div>
        {extractResult && <div className="text-[10px] text-emerald-400">{extractResult}</div>}
      </div>

      {/* Transfer From */}
      <div className="space-y-1.5">
        <div className="text-[11px] font-medium text-slate-400">Transfer From Weight</div>
        <div className="flex gap-2">
          <input
            value={sourceId}
            onChange={(e) => setSourceId(e.target.value)}
            placeholder="source weight ID"
            className="flex-1 px-2 py-1.5 text-xs bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-indigo-500 focus:outline-none"
          />
          <button
            onClick={handleTransfer}
            disabled={transferring}
            className="text-xs bg-amber-600 hover:bg-amber-500 text-white px-3 py-1.5 rounded-lg transition-colors cursor-pointer disabled:opacity-50"
          >
            {transferring ? '...' : 'Transfer'}
          </button>
        </div>
        {transferResult && <div className="text-[10px] text-emerald-400">{transferResult}</div>}
      </div>
    </div>
  );
}
