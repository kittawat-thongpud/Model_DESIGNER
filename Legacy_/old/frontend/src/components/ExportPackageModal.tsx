
import React, { useState } from 'react';
import { useDesignerStore } from '../store/designerStore';
import { api } from '../services/api';
import type { LayerType, ModelNode, ModelEdge } from '../types';

interface ExportPackageModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ExportPackageModal: React.FC<ExportPackageModalProps> = ({ isOpen, onClose }) => {
  const { nodes, edges, globalVars } = useDesignerStore();
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [selectedGlobals, setSelectedGlobals] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  if (!isOpen) return null;

  // ─── Analyze I/O nodes ─────────────────────────────────────────────────
  const inputNodes = nodes.filter(n => n.data.layerType === 'Input');
  const outputNodes = nodes.filter(n => n.data.layerType === 'Output');

  const describeShape = (n: typeof nodes[0]) => {
    const p = (n.data.params || {}) as Record<string, unknown>;
    if (n.data.layerType === 'Input') {
      return `${p.channels ?? '?'}×${p.height ?? '?'}×${p.width ?? '?'}`;
    }
    if (n.data.layerType === 'Output') {
      return `${p.num_classes ?? p.out_features ?? '?'}`;
    }
    return '?';
  };

  const handleSubmit = async () => {
    if (!name.trim()) {
      setError('Name is required');
      return;
    }
    setLoading(true);
    setError('');
    
    try {
      // Map ReactFlow nodes/edges to ModelGraph format
      const graphNodes: ModelNode[] = nodes.map(n => ({
        id: n.id,
        type: n.data.layerType as LayerType,
        position: n.position,
        params: n.data.params as any,
        enabledByGlobal: n.data.enabledByGlobal as string | undefined
      }));

      const graphEdges: ModelEdge[] = edges.map(e => ({
        source: e.source,
        target: e.target,
        source_handle: e.sourceHandle || undefined,
        target_handle: e.targetHandle || undefined
      }));

      await api.createPackage({
        graph: {
          nodes: graphNodes,
          edges: graphEdges,
          globals: globalVars,
          meta: { 
            name, 
            version: '1.0', 
            created_at: new Date().toISOString(), 
            updated_at: new Date().toISOString(), 
            description 
          }
        },
        name,
        description,
        exposed_globals: selectedGlobals
      });
      // Optionally notify success
      alert(`Package "${name}" created successfully!`);
      onClose();
    } catch (err: any) {
      setError(err.message || 'Failed to export package');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-backdrop">
      <div className="modal-content" style={{ width: '500px' }}>
        <h2 className="modal-title">Export as Package</h2>
        <p className="modal-subtitle">Save current graph as a reusable component.</p>
        
        {error && <div className="error-banner">{error}</div>}

        {/* ─── I/O Summary ─────────────────────────────────── */}
        <div className="tcm-section" style={{ display: 'flex', gap: '16px' }}>
          <div style={{
            flex: 1, padding: '10px', borderRadius: '6px',
            background: 'rgba(76,175,80,0.1)', border: '1px solid rgba(76,175,80,0.3)'
          }}>
            <div style={{ fontSize: '12px', opacity: 0.7, marginBottom: '4px' }}>📥 Inputs</div>
            <div style={{ fontWeight: 600 }}>{inputNodes.length} node{inputNodes.length !== 1 ? 's' : ''}</div>
            {inputNodes.map((n, i) => (
              <div key={i} style={{ fontSize: '11px', color: '#aaa', marginTop: '2px' }}>
                {n.data.label as string}: {describeShape(n)}
              </div>
            ))}
          </div>
          <div style={{
            flex: 1, padding: '10px', borderRadius: '6px',
            background: 'rgba(233,30,99,0.1)', border: '1px solid rgba(233,30,99,0.3)'
          }}>
            <div style={{ fontSize: '12px', opacity: 0.7, marginBottom: '4px' }}>📤 Outputs</div>
            <div style={{ fontWeight: 600 }}>{outputNodes.length} node{outputNodes.length !== 1 ? 's' : ''}</div>
            {outputNodes.map((n, i) => (
              <div key={i} style={{ fontSize: '11px', color: '#aaa', marginTop: '2px' }}>
                {n.data.label as string}: {describeShape(n)}
              </div>
            ))}
          </div>
        </div>
        
        <div className="tcm-section">
          <label className="tcm-label">Package Name</label>
          <input 
            className="tcm-input" 
            value={name} 
            onChange={e => setName(e.target.value)} 
            placeholder="e.g. ResBlock, AttentionLayer" 
            autoFocus
          />
        </div>
        
        <div className="tcm-section">
          <label className="tcm-label">Description</label>
          <textarea 
            className="tcm-input" 
            value={description} 
            onChange={e => setDescription(e.target.value)} 
            rows={2}
          />
        </div>
        
        <div className="tcm-section">
          <label className="tcm-label">Exposed Parameters</label>
          <p className="tcm-hint">Select global variables to expose as package inputs.</p>
          
          <div className="params-list" style={{ 
            maxHeight: '150px', 
            overflowY: 'auto', 
            border: '1px solid #ddd', 
            borderRadius: '4px',
            padding: '8px' 
          }}>
            {globalVars.length === 0 && <p style={{color: '#888', fontStyle: 'italic'}}>No global variables found.</p>}
            
            {globalVars.map(g => (
              <label key={g.id} style={{ display: 'flex', alignItems: 'center', marginBottom: '4px', cursor: 'pointer' }}>
                <input 
                  type="checkbox" 
                  checked={selectedGlobals.includes(g.id)}
                  onChange={e => {
                    if (e.target.checked) setSelectedGlobals([...selectedGlobals, g.id]);
                    else setSelectedGlobals(selectedGlobals.filter(id => id !== g.id));
                  }}
                  style={{ marginRight: '8px' }}
                />
                <span style={{ fontWeight: 500 }}>{g.name}</span>
                <span style={{ marginLeft: 'auto', color: '#666', fontSize: '12px', background: '#eee', padding: '1px 4px', borderRadius: '4px' }}>
                  {g.type}
                </span>
              </label>
            ))}
          </div>
        </div>
        
        <div className="modal-actions">
          <button className="tcm-btn" onClick={onClose} disabled={loading}>
            Cancel
          </button>
          <button className="tcm-btn primary" onClick={handleSubmit} disabled={loading}>
            {loading ? 'Exporting...' : 'Export Package'}
          </button>
        </div>
      </div>
    </div>
  );
};
