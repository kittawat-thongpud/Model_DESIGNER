/**
 * PackageManager.tsx — Modal for managing saved packages.
 */
import { useState, useEffect } from 'react';
import { api } from '../services/api';
import { useDesignerStore } from '../store/designerStore';
import type { ModelPackage, ModelGraph } from '../types';

interface PackageManagerProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function PackageManager({ isOpen, onClose }: PackageManagerProps) {
  const [packages, setPackages] = useState<ModelPackage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const loadGraph = useDesignerStore((s) => s.loadGraph);
  const addLog = useDesignerStore((s) => s.addLog);

  useEffect(() => {
    if (isOpen) loadPackages();
  }, [isOpen]);

  const loadPackages = async () => {
    setLoading(true);
    try {
      const data = await api.listPackages();
      // Sort by created_at desc
      data.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
      setPackages(data);
    } catch (err) {
      setError('Failed to load packages');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id: string, name: string) => {
    if (!confirm(`Are you sure you want to delete package "${name}"?`)) return;
    try {
      await api.deletePackage(id);
      addLog('INFO', 'package', `Deleted package: ${name}`);
      loadPackages();
    } catch (err) {
      addLog('ERROR', 'package', `Failed to delete package: ${err}`);
    }
  };

  const handleEdit = async (pkg: ModelPackage) => {
    if (!confirm(`Load package "${pkg.name}" into the designer?\nAny unsaved changes in the current project will be lost.`)) return;
    
    // Convert package structure to ModelGraph format
    // Packages store 'nodes' and 'globals', need to reconstruct graph
    // Note: stored package 'nodes' are ModelNode[], compatible with loadGraph
    
    try {
        // We need to fetch the full package again to ensure we have everything or just use what we have?
        // listPackages might return summary. Let's rely on list for now or fetch individual if needed.
        // Actually, listPackages returns ModelPackage[] which contains nodes/edges/globals.
        const fullPkg = await api.getPackage(pkg.id);
        
        const graph: ModelGraph = {
            nodes: fullPkg.nodes || [],
            edges: fullPkg.edges || [],
            globals: fullPkg.globals || [],
            meta: {
                name: fullPkg.name,
                description: fullPkg.description || '',
                version: '1.0',
                created_at: fullPkg.created_at,
                updated_at: new Date().toISOString(),
            }
        };
        
        loadGraph(graph);
        onClose();
    } catch (err) {
        addLog('ERROR', 'package', 'Failed to load package for editing');
    }
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay">
      <div className="modal-content" style={{ maxWidth: '800px', width: '90%' }}>
        <div className="modal-header">
          <h3>📦 Manage Packages</h3>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>

        <div className="modal-body">
          {error && <div className="error-banner">{error}</div>}
          
          <div className="package-list">
            {loading ? (
              <div className="loading-spinner">Loading...</div>
            ) : packages.length === 0 ? (
              <p className="empty-state">No packages found.</p>
            ) : (
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Description</th>
                      <th>I/O</th>
                      <th>Created</th>
                      <th style={{ textAlign: 'right' }}>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {packages.map((pkg) => {
                        const inputs = (pkg.nodes || []).filter((n: any) => n.type === 'Input').length;
                        const outputs = (pkg.nodes || []).filter((n: any) => n.type === 'Output').length;
                        return (
                          <tr key={pkg.id}>
                            <td style={{ fontWeight: 500 }}>{pkg.name}</td>
                            <td style={{ color: 'var(--text-secondary)', fontSize: '0.9em' }}>
                              {pkg.description || '-'}
                            </td>
                            <td>
                                <span style={{ color: '#4ade80' }}>{inputs} in</span>,{' '}
                                <span style={{ color: '#f87171' }}>{outputs} out</span>
                            </td>
                            <td style={{ fontSize: '0.85em', color: 'var(--text-muted)' }}>
                              {new Date(pkg.created_at).toLocaleDateString()}
                            </td>
                            <td className="actions-cell">
                              <button 
                                className="action-btn"
                                onClick={() => handleEdit(pkg)}
                                title="Edit in Designer"
                              >
                                ✏️ Edit
                              </button>
                              <button 
                                className="action-btn danger"
                                onClick={() => handleDelete(pkg.id, pkg.name)}
                                title="Delete Package"
                              >
                                🗑️
                              </button>
                            </td>
                          </tr>
                        );
                    })}
                  </tbody>
                </table>
            )}
          </div>
        </div>

        <div className="modal-footer">
          <button className="btn secondary" onClick={onClose}>Close</button>
        </div>
      </div>
      
      <style>{`
        .package-list {
            min-height: 200px;
            max-height: 60vh;
            overflow-y: auto;
        }
        .data-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95em;
        }
        .data-table th, .data-table td {
            text-align: left;
            padding: 10px;
            border-bottom: 1px solid var(--border-default);
        }
        .data-table th {
            color: var(--text-muted);
            font-weight: 500;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .actions-cell {
            text-align: right !important;
            white-space: nowrap;
        }
        .action-btn {
            background: none;
            border: 1px solid var(--border-default);
            color: var(--text-primary);
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
            margin-left: 6px;
            font-size: 0.9em;
            transition: all 0.2s;
        }
        .action-btn:hover {
            background: var(--bg-elevated);
            border-color: var(--text-muted);
        }
        .action-btn.danger:hover {
            background: rgba(248, 113, 113, 0.1);
            border-color: var(--red);
            color: var(--red);
        }
      `}</style>
    </div>
  );
}
