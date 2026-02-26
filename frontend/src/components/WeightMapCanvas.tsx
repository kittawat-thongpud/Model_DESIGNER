/**
 * WeightMapCanvas — Modal wrapper around WeightMappingPanel.
 *
 * Used by TrainingDesignerPage and CreateJobModal for inline weight mapping.
 */
import { useRef, useCallback } from 'react';
import { X, Check } from 'lucide-react';
import WeightMappingPanel, { type GroupMapping, type MappingPanelHandle } from './WeightMappingPanel';
import type { MappingKey } from '../types';

interface WeightMapCanvasProps {
  targetWeightId: string;
  onApply?: (result: {
    sourceWeightId: string;
    mapping: MappingKey[];
    freezeNodeIds: string[];
  }) => void;
  onClose: () => void;
}

export default function WeightMapCanvas({
  targetWeightId,
  onApply,
  onClose,
}: WeightMapCanvasProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const mappingsRef = useRef<GroupMapping[]>([]);

  const handleMappingsChange = useCallback((m: GroupMapping[]) => {
    mappingsRef.current = m;
  }, []);

  const handleApply = useCallback(() => {
    // Read state from the panel's exposed handle
    const el = panelRef.current?.querySelector('[data-mapping-panel]');
    const handle = el ? (el as unknown as Record<string, MappingPanelHandle>).__mappingState : null;

    const mappings = handle?.getMappings() ?? mappingsRef.current;
    const sourceWeightId = handle?.getSourceWeightId() ?? '';
    const frozenGroups = handle?.getFrozenGroups() ?? new Set<string>();

    const allKeys: MappingKey[] = [];
    const freezeNodeIds: string[] = [];

    for (const m of mappings) {
      for (const k of m.keys) {
        if (k.src_key && k.tgt_key && k.matched) {
          allKeys.push(k);
        }
      }
      if (frozenGroups.has(m.tgtPrefix)) {
        freezeNodeIds.push(m.tgtPrefix);
      }
    }

    onApply?.({ sourceWeightId, mapping: allKeys, freezeNodeIds });
    onClose();
  }, [onApply, onClose]);

  const totalMapped = mappingsRef.current.length;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl w-[90vw] max-w-[1200px] h-[85vh] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 shrink-0">
          <div className="flex items-center gap-3">
            <span className="text-lg">🔄</span>
            <div>
              <h2 className="text-white font-semibold text-sm">Weight Transfer Mapping</h2>
              <p className="text-[10px] text-slate-500">Drag source layers → target layers, or use Auto Map</p>
            </div>
          </div>
          <button onClick={onClose} className="text-slate-500 hover:text-white p-1 cursor-pointer">
            <X size={18} />
          </button>
        </div>

        {/* Mapping Panel */}
        <div ref={panelRef} className="flex-1 min-h-0">
          <WeightMappingPanel
            targetWeightId={targetWeightId}
            showFreezeControls={true}
            showImport={true}
            onMappingsChange={handleMappingsChange}
          />
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 px-6 py-3 border-t border-slate-800 shrink-0 bg-slate-900/50">
          <button
            onClick={onClose}
            className="px-4 py-2 text-xs text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors cursor-pointer"
          >
            Cancel
          </button>
          <button
            onClick={handleApply}
            className="flex items-center gap-1.5 px-4 py-2 text-xs bg-green-600 hover:bg-green-500 text-white rounded-lg transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-default"
          >
            <Check size={14} />
            Apply Mapping
          </button>
        </div>
      </div>
    </div>
  );
}
