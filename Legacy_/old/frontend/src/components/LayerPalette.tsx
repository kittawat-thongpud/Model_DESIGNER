/**
 * Layer Palette — drag-to-add sidebar with available layer types.
 */
import { useState, useEffect } from 'react';
import { LAYER_DEFINITIONS, type LayerType, type ModelPackage } from '../types';
import { useDesignerStore } from '../store/designerStore';
import { api } from '../services/api';
import PackageManager from './PackageManager';

const CATEGORIES = ['I/O', 'Processing', 'Activation', 'Reshape', 'Regularization'] as const;

export default function LayerPalette() {
  const addNode = useDesignerStore((s) => s.addNode);
  const [packages, setPackages] = useState<ModelPackage[]>([]);
  const [expandedPkg, setExpandedPkg] = useState<string | null>(null);
  const [showManager, setShowManager] = useState(false);

  useEffect(() => {
    const load = () => api.listPackages().then(setPackages).catch(() => {});
    load();
    const interval = setInterval(load, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleDragStart = (e: React.DragEvent, type: LayerType) => {
    e.dataTransfer.setData('application/layer-type', type);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragStartPackage = (e: React.DragEvent, pkg: ModelPackage) => {
    e.dataTransfer.setData('application/layer-type', 'Package');
    e.dataTransfer.setData('application/package-id', pkg.id);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleClick = (type: LayerType) => {
    addNode(type, { x: 250 + Math.random() * 200, y: 150 + Math.random() * 200 });
  };

  // Extract I/O info from package data
  const getPkgIO = (pkg: ModelPackage) => {
    const nodes = (pkg as any).nodes || [];
    const inputs = nodes.filter((n: any) => n.type === 'Input');
    const outputs = nodes.filter((n: any) => n.type === 'Output');
    return { inputs: inputs.length, outputs: outputs.length };
  };

  return (
    <div className="layer-palette">
      <h3 className="palette-title">🧱 Layers</h3>

      {CATEGORIES.map((cat) => {
        const layers = Object.values(LAYER_DEFINITIONS).filter((d) => d.category === cat);
        if (layers.length === 0) return null;

        return (
          <div key={cat} className="palette-category">
            <h4 className="category-label">{cat}</h4>
            {layers.map((def) => (
              <div
                key={def.type}
                className="palette-item"
                style={{ '--item-color': def.color } as React.CSSProperties}
                draggable
                onDragStart={(e) => handleDragStart(e, def.type)}
                onClick={() => handleClick(def.type)}
                title={`Drag or click to add ${def.label}`}
              >
                <span className="palette-icon">{def.icon}</span>
                <span className="palette-label">{def.label}</span>
              </div>
            ))}
          </div>
        );
      })}

      {/* Packages */}
      {packages.length > 0 && (
        <div className="palette-category">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h4 className="category-label">📦 Packages</h4>
            <button 
              className="pkg-manage-btn"
              onClick={() => setShowManager(true)}
              title="Manage Packages"
            >
              Manage
            </button>
          </div>
          {packages.map((pkg) => {
            const io = getPkgIO(pkg);
            const isExpanded = expandedPkg === pkg.id;
            return (
              <div key={pkg.id}>
                <div
                  className="palette-item"
                  style={{ '--item-color': '#607D8B' } as React.CSSProperties}
                  draggable
                  onDragStart={(e) => handleDragStartPackage(e, pkg)}
                  onClick={() => addNode('Package', { x: 250 + Math.random() * 200, y: 150 + Math.random() * 200 }, { packageId: pkg.id })}
                  title={pkg.description || pkg.name}
                >
                  <span className="palette-icon">📦</span>
                  <span className="palette-label">{pkg.name}</span>
                  <button
                    className="pkg-expand-btn"
                    onClick={(e) => { e.stopPropagation(); setExpandedPkg(isExpanded ? null : pkg.id); }}
                    title="View details"
                  >
                    {isExpanded ? '▾' : '▸'}
                  </button>
                </div>
                {isExpanded && (
                  <div className="pkg-details">
                    {pkg.description && (
                      <div className="pkg-detail-row pkg-desc">{pkg.description}</div>
                    )}
                    <div className="pkg-detail-row">
                      <span style={{ color: '#4ade80' }}>📥 {io.inputs} input{io.inputs !== 1 ? 's' : ''}</span>
                      <span style={{ color: '#f87171', marginLeft: '8px' }}>📤 {io.outputs} output{io.outputs !== 1 ? 's' : ''}</span>
                    </div>
                    {((pkg as any).exposed_params || []).length > 0 && (
                      <div className="pkg-detail-row" style={{ fontSize: '10px', opacity: 0.7 }}>
                        ⚙️ {((pkg as any).exposed_params || []).map((p: any) => p.name).join(', ')}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
      <PackageManager isOpen={showManager} onClose={() => setShowManager(false)} />
      
      <style>{`
        .pkg-manage-btn {
            background: none;
            border: none;
            color: var(--accent-primary);
            font-size: 0.75rem;
            cursor: pointer;
            padding: 0 4px;
            opacity: 0.8;
            font-family: var(--font-mono);
        }
        .pkg-manage-btn:hover {
            opacity: 1;
            text-decoration: underline;
        }
      `}</style>
    </div>
  );
}
