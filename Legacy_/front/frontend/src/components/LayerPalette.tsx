/**
 * Layer Palette — drag-to-add sidebar with available layer types.
 * Tailwind + Lucide design matching the modern dark slate UI.
 */
import { useState, useEffect } from 'react';
import type { ModelPackage } from '../types';
import { useDesignerStore } from '../store/designerStore';
import { useNodeCatalogStore } from '../store/nodeCatalogStore';
import { api } from '../services/api';
import PackageManager from './PackageManager';
import { Search } from 'lucide-react';

const CATEGORIES = ['I/O', 'Processing', 'Activation', 'Reshape', 'Regularization'] as const;

export default function LayerPalette() {
  const addNode = useDesignerStore((s) => s.addNode);
  const [packages, setPackages] = useState<ModelPackage[]>([]);
  const [expandedPkg, setExpandedPkg] = useState<string | null>(null);
  const [showManager, setShowManager] = useState(false);
  const [search, setSearch] = useState('');

  useEffect(() => {
    const load = () => api.listPackages().then(setPackages).catch(() => {});
    load();
    const interval = setInterval(load, 5000);
    return () => clearInterval(interval);
  }, []);

  const entries = useNodeCatalogStore((s) => s.entries);

  const handleDragStart = (e: React.DragEvent, type: string) => {
    e.dataTransfer.setData('application/layer-type', type);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragStartPackage = (e: React.DragEvent, pkg: ModelPackage) => {
    e.dataTransfer.setData('application/layer-type', 'Package');
    e.dataTransfer.setData('application/package-id', pkg.id);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleClick = (type: string) => {
    addNode(type, { x: 250 + Math.random() * 200, y: 150 + Math.random() * 200 });
  };

  const getPkgIO = (pkg: ModelPackage) => {
    const nodes = (pkg as any).nodes || [];
    const inputs = nodes.filter((n: any) => n.type === 'Input');
    const outputs = nodes.filter((n: any) => n.type === 'Output');
    return { inputs: inputs.length, outputs: outputs.length };
  };

  const lowerSearch = search.toLowerCase();

  return (
    <div className="flex flex-col h-full">
      {/* Search */}
      <div className="p-3 border-b border-slate-800 shrink-0">
        <div className="relative">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            type="text"
            placeholder="Search layers..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-md pl-8 pr-3 py-1.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all"
          />
        </div>
      </div>

      {/* Categories */}
      <div className="flex-1 overflow-y-auto p-3 space-y-5">
        {CATEGORIES.map((cat) => {
          const layers = entries.filter((d) =>
            d.category === cat && d.type !== 'Package' &&
            (!search || d.label.toLowerCase().includes(lowerSearch) || d.type.toLowerCase().includes(lowerSearch))
          );
          if (layers.length === 0) return null;

          return (
            <div key={cat}>
              <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2">{cat}</h4>
              <div className="space-y-1">
                {layers.map((def) => (
                  <div
                    key={def.type}
                    className="flex items-center gap-3 p-2 rounded-md hover:bg-slate-800 border border-transparent hover:border-slate-700 cursor-grab transition-colors"
                    draggable
                    onDragStart={(e) => handleDragStart(e, def.type)}
                    onClick={() => handleClick(def.type)}
                    title={`Drag or click to add ${def.label}`}
                  >
                    <div className="w-6 h-6 rounded flex items-center justify-center text-sm shrink-0" style={{ background: `${def.color}20`, color: def.color }}>
                      {def.icon || '●'}
                    </div>
                    <span className="text-sm text-slate-300">{def.label}</span>
                  </div>
                ))}
              </div>
            </div>
          );
        })}

        {/* Packages */}
        {packages.length > 0 && (
          <div>
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Packages</h4>
              <button
                className="text-[10px] text-indigo-400 hover:text-indigo-300 font-mono cursor-pointer"
                onClick={() => setShowManager(true)}
              >
                Manage
              </button>
            </div>
            <div className="space-y-1">
              {packages.map((pkg) => {
                const io = getPkgIO(pkg);
                const isExpanded = expandedPkg === pkg.id;
                return (
                  <div key={pkg.id}>
                    <div
                      className="flex items-center gap-3 p-2 rounded-md hover:bg-slate-800 border border-transparent hover:border-slate-700 cursor-grab transition-colors"
                      draggable
                      onDragStart={(e) => handleDragStartPackage(e, pkg)}
                      onClick={() => addNode('Package', { x: 250 + Math.random() * 200, y: 150 + Math.random() * 200 }, { packageId: pkg.id })}
                      title={pkg.description || pkg.name}
                    >
                      <div className="w-3 h-3 rounded-sm shrink-0 bg-slate-500" />
                      <span className="text-sm text-slate-300 flex-1 truncate">{pkg.name}</span>
                      <button
                        className="text-slate-500 hover:text-white text-xs cursor-pointer"
                        onClick={(e) => { e.stopPropagation(); setExpandedPkg(isExpanded ? null : pkg.id); }}
                      >
                        {isExpanded ? '▾' : '▸'}
                      </button>
                    </div>
                    {isExpanded && (
                      <div className="ml-6 mt-1 mb-2 text-xs space-y-1">
                        {pkg.description && <p className="text-slate-500">{pkg.description}</p>}
                        <p>
                          <span className="text-emerald-400">📥 {io.inputs} input{io.inputs !== 1 ? 's' : ''}</span>
                          <span className="text-red-400 ml-2">📤 {io.outputs} output{io.outputs !== 1 ? 's' : ''}</span>
                        </p>
                        {((pkg as any).exposed_params || []).length > 0 && (
                          <p className="text-slate-600 text-[10px]">
                            ⚙️ {((pkg as any).exposed_params || []).map((p: any) => p.name).join(', ')}
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
      <PackageManager isOpen={showManager} onClose={() => setShowManager(false)} />
    </div>
  );
}
