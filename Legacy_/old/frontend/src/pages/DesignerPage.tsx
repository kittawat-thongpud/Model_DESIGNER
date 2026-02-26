/**
 * Designer Page — project list → editor toggle.
 *
 * Default view: grid of saved model projects.
 * Click to open → shows node editor.
 * Right-click → context menu (Open, Edit, Delete).
 */
import { useState, useEffect, useCallback } from 'react';
import { useDesignerStore } from '../store/designerStore';
import { api } from '../services/api';
import type { ModelSummary } from '../types';

import TopBar from '../components/TopBar';
import LayerPalette from '../components/LayerPalette';
import GlobalVarsPanel from '../components/GlobalVarsPanel';
import NodeCanvas from '../components/NodeCanvas';
import PropertiesPanel from '../components/PropertiesPanel';
import LogPanel from '../components/LogPanel';
import CodeModal from '../components/CodeModal';
import InferencePanel from '../components/InferencePanel';

export default function DesignerPage() {
  const [view, setView] = useState<'list' | 'editor'>('list');
  const [projects, setProjects] = useState<ModelSummary[]>([]);
  const [loading, setLoading] = useState(true);

  // Context menu
  const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number; project: ModelSummary } | null>(null);

  // Edit name dialog
  const [editDialog, setEditDialog] = useState<{ project: ModelSummary; newName: string } | null>(null);
  const [leftTab, setLeftTab] = useState<'palette' | 'globals' | 'test'>('palette');

  // Delete confirm
  const [deleteConfirm, setDeleteConfirm] = useState<ModelSummary | null>(null);

  const loadModel = useDesignerStore((s) => s.loadModel);
  const pendingBuild = useDesignerStore((s) => s.pendingBuild);
  const confirmBuildReplace = useDesignerStore((s) => s.confirmBuildReplace);
  const cancelBuildReplace = useDesignerStore((s) => s.cancelBuildReplace);
  const pendingSave = useDesignerStore((s) => s.pendingSave);
  const confirmSaveReplace = useDesignerStore((s) => s.confirmSaveReplace);
  const cancelSaveReplace = useDesignerStore((s) => s.cancelSaveReplace);
  const modelId = useDesignerStore((s) => s.modelId);

  const fetchProjects = useCallback(async () => {
    setLoading(true);
    try {
      const list = await api.listModels();
      setProjects(list);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (view === 'list') fetchProjects();
  }, [view, fetchProjects]);

  // Close context menu on click outside
  useEffect(() => {
    const handler = () => setCtxMenu(null);
    window.addEventListener('click', handler);
    return () => window.removeEventListener('click', handler);
  }, []);

  const handleOpen = async (project: ModelSummary) => {
    await loadModel(project.id);
    setView('editor');
  };

  const handleNewModel = () => {
    // Reset store to blank
    useDesignerStore.setState({
      nodes: [],
      edges: [],
      modelName: 'Untitled',
      modelDescription: '',
      modelId: null,
      selectedNodeId: null,
      generatedCode: null,
      showCode: false,
    });
    setView('editor');
  };

  const handleContextMenu = (e: React.MouseEvent, project: ModelSummary) => {
    e.preventDefault();
    setCtxMenu({ x: e.clientX, y: e.clientY, project });
  };

  const handleDelete = async (project: ModelSummary) => {
    try {
      await api.deleteModel(project.id);
      setProjects((prev) => prev.filter((p) => p.id !== project.id));
    } catch {
      // ignore
    }
    setDeleteConfirm(null);
  };

  const handleEditSave = async () => {
    if (!editDialog) return;
    try {
      // Load model, change name, save
      const graph = await api.loadModel(editDialog.project.id);
      graph.meta.name = editDialog.newName;
      await api.saveModel(graph);
      fetchProjects();
    } catch {
      // ignore
    }
    setEditDialog(null);
  };

  const formatDate = (d: string) => {
    try {
      return new Date(d).toLocaleDateString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric',
        hour: '2-digit', minute: '2-digit',
      });
    } catch {
      return d;
    }
  };

  // ─── Editor view ───────────────────────────────────────────────────────────
  if (view === 'editor') {
    return (
      <div className="designer-page">
        <div className="editor-back-bar">
          <button className="btn btn-ghost" onClick={() => setView('list')}>← Projects</button>
        </div>
        <TopBar />
        <div className="designer-layout">
          <div className="designer-left-sidebar">
            <div className="designer-sidebar-tabs">
              <button 
                className={leftTab === 'palette' ? 'active' : ''} 
                onClick={() => setLeftTab('palette')}
              >
                Layers
              </button>
              <button 
                className={leftTab === 'globals' ? 'active' : ''} 
                onClick={() => setLeftTab('globals')}
              >
                Globals
              </button>
              <button 
                className={leftTab === 'test' ? 'active' : ''} 
                onClick={() => setLeftTab('test')}
              >
                Test
              </button>
            </div>
            <div className="sidebar-tab-content">
              {leftTab === 'palette' && <LayerPalette />}
              {leftTab === 'globals' && <GlobalVarsPanel />}
              {leftTab === 'test' && <InferencePanel modelId={modelId || ''} />}
            </div>
          </div>
          <NodeCanvas />
          <PropertiesPanel />
        </div>
        <LogPanel />
        <CodeModal />

        {/* Build replace confirmation dialog */}
        {pendingBuild && (
          <div className="modal-overlay" onClick={cancelBuildReplace}>
            <div className="confirm-dialog" onClick={(e) => e.stopPropagation()}>
              <h3>⚠️ Build Already Exists</h3>
              <p>A build named <strong>"{pendingBuild.modelName}"</strong> already exists.</p>
              <p>Do you want to replace it?</p>
              <div className="confirm-actions">
                <button className="btn btn-secondary" onClick={cancelBuildReplace}>Cancel</button>
                <button className="btn btn-accent" onClick={confirmBuildReplace}>Replace</button>
              </div>
            </div>
          </div>
        )}

        {/* Save replace confirmation dialog */}
        {pendingSave && (
          <div className="modal-overlay" onClick={cancelSaveReplace}>
            <div className="confirm-dialog" onClick={(e) => e.stopPropagation()}>
              <h3>⚠️ Model Already Exists</h3>
              <p>A model named <strong>"{pendingSave.graph.meta.name}"</strong> already exists.</p>
              <p>Do you want to overwrite it?</p>
              <div className="confirm-actions">
                <button className="btn btn-secondary" onClick={cancelSaveReplace}>Cancel</button>
                <button className="btn btn-accent" onClick={confirmSaveReplace}>Overwrite</button>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  // ─── Project list view ─────────────────────────────────────────────────────
  return (
    <div className="page-container">
      <div className="page-header">
        <h2 className="page-title">🔥 Designer Projects</h2>
        <button className="btn btn-accent" onClick={handleNewModel}>+ New Model</button>
      </div>

      {loading ? (
        <div className="loading-text">Loading projects...</div>
      ) : projects.length === 0 ? (
        <div className="empty-state">
          <span className="empty-icon">📐</span>
          <h3>No projects yet</h3>
          <p>Create a new model to get started.</p>
          <button className="btn btn-accent" onClick={handleNewModel}>+ New Model</button>
        </div>
      ) : (
        <div className="project-grid">
          {projects.map((p) => (
            <div
              key={p.id}
              className="project-card"
              onClick={() => handleOpen(p)}
              onContextMenu={(e) => handleContextMenu(e, p)}
            >
              <div className="project-card-header">
                <span className="project-card-icon">🧠</span>
                <span className="project-card-name">{p.name}</span>
              </div>
              <div className="project-card-meta">
                <span>{p.node_count} layers · {p.edge_count} connections</span>
              </div>
              {p.description && (
                <div className="project-card-desc">{p.description}</div>
              )}
              <div className="project-card-date">{formatDate(p.updated_at)}</div>
            </div>
          ))}
        </div>
      )}

      {/* Context menu */}
      {ctxMenu && (
        <div
          className="context-menu"
          style={{ top: ctxMenu.y, left: ctxMenu.x }}
          onClick={(e) => e.stopPropagation()}
        >
          <button onClick={() => { handleOpen(ctxMenu.project); setCtxMenu(null); }}>
            📂 Open
          </button>
          <button onClick={() => {
            setEditDialog({ project: ctxMenu.project, newName: ctxMenu.project.name });
            setCtxMenu(null);
          }}>
            ✏️ Edit Name
          </button>
          <button className="danger" onClick={() => {
            setDeleteConfirm(ctxMenu.project);
            setCtxMenu(null);
          }}>
            🗑️ Delete
          </button>
        </div>
      )}

      {/* Edit name dialog */}
      {editDialog && (
        <div className="modal-overlay" onClick={() => setEditDialog(null)}>
          <div className="confirm-dialog" onClick={(e) => e.stopPropagation()}>
            <h3>✏️ Edit Model Name</h3>
            <input
              type="text"
              className="dialog-input"
              value={editDialog.newName}
              onChange={(e) => setEditDialog({ ...editDialog, newName: e.target.value })}
              autoFocus
              onKeyDown={(e) => e.key === 'Enter' && handleEditSave()}
            />
            <div className="confirm-actions">
              <button className="btn btn-secondary" onClick={() => setEditDialog(null)}>Cancel</button>
              <button className="btn btn-accent" onClick={handleEditSave}>Save</button>
            </div>
          </div>
        </div>
      )}

      {/* Delete confirm dialog */}
      {deleteConfirm && (
        <div className="modal-overlay" onClick={() => setDeleteConfirm(null)}>
          <div className="confirm-dialog" onClick={(e) => e.stopPropagation()}>
            <h3>🗑️ Delete Project</h3>
            <p>Are you sure you want to delete <strong>"{deleteConfirm.name}"</strong>?</p>
            <p className="warning-text">This action cannot be undone.</p>
            <div className="confirm-actions">
              <button className="btn btn-secondary" onClick={() => setDeleteConfirm(null)}>Cancel</button>
              <button className="btn btn-danger" onClick={() => handleDelete(deleteConfirm)}>Delete</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
