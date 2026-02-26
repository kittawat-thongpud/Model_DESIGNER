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

import { Network } from 'lucide-react';
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
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Back bar + TopBar */}
        <div className="flex items-center border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm shrink-0">
          <button
            className="px-4 py-2 text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors border-r border-slate-800 cursor-pointer"
            onClick={() => setView('list')}
          >
            ← Projects
          </button>
          <div className="flex-1"><TopBar /></div>
        </div>

        {/* Workspace */}
        <div className="flex-1 flex overflow-hidden min-h-0">
          {/* Left sidebar: Layers / Globals / Test tabs */}
          <div className="w-72 border-r border-slate-800 bg-slate-900/50 flex flex-col shrink-0">
            <div className="flex border-b border-slate-800 shrink-0">
              {(['palette', 'globals', 'test'] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setLeftTab(tab)}
                  className={`flex-1 py-2.5 text-xs font-medium uppercase tracking-wider transition-colors cursor-pointer ${
                    leftTab === tab
                      ? 'text-indigo-400 border-b-2 border-indigo-500 bg-indigo-500/5'
                      : 'text-slate-500 hover:text-slate-300'
                  }`}
                >
                  {tab === 'palette' ? 'Layers' : tab === 'globals' ? 'Globals' : 'Test'}
                </button>
              ))}
            </div>
            <div className="flex-1 overflow-y-auto">
              {leftTab === 'palette' && <LayerPalette />}
              {leftTab === 'globals' && <GlobalVarsPanel />}
              {leftTab === 'test' && <InferencePanel modelId={modelId || ''} />}
            </div>
          </div>

          {/* Canvas - fills all remaining space */}
          <div className="flex-1 min-w-0 min-h-0">
            <NodeCanvas />
          </div>

          {/* Properties panel */}
          <div className="w-72 shrink-0">
            <PropertiesPanel />
          </div>
        </div>

        <LogPanel />
        <CodeModal />

        {/* Build replace confirmation */}
        {pendingBuild && (
          <div className="modal-overlay" onClick={cancelBuildReplace}>
            <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 max-w-md shadow-2xl" onClick={(e) => e.stopPropagation()}>
              <h3 className="text-lg font-semibold text-white mb-3">Build Already Exists</h3>
              <p className="text-slate-400 text-sm mb-1">A build named <strong className="text-white">"{pendingBuild.modelName}"</strong> already exists.</p>
              <p className="text-slate-400 text-sm mb-5">Do you want to replace it?</p>
              <div className="flex justify-end gap-3">
                <button className="px-4 py-2 text-sm text-slate-300 hover:bg-slate-700 rounded-lg transition-colors cursor-pointer" onClick={cancelBuildReplace}>Cancel</button>
                <button className="px-4 py-2 text-sm text-white bg-indigo-600 hover:bg-indigo-500 rounded-lg shadow transition-colors cursor-pointer" onClick={confirmBuildReplace}>Replace</button>
              </div>
            </div>
          </div>
        )}

        {/* Save replace confirmation */}
        {pendingSave && (
          <div className="modal-overlay" onClick={cancelSaveReplace}>
            <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 max-w-md shadow-2xl" onClick={(e) => e.stopPropagation()}>
              <h3 className="text-lg font-semibold text-white mb-3">Model Already Exists</h3>
              <p className="text-slate-400 text-sm mb-1">A model named <strong className="text-white">"{pendingSave.graph.meta.name}"</strong> already exists.</p>
              <p className="text-slate-400 text-sm mb-5">Do you want to overwrite it?</p>
              <div className="flex justify-end gap-3">
                <button className="px-4 py-2 text-sm text-slate-300 hover:bg-slate-700 rounded-lg transition-colors cursor-pointer" onClick={cancelSaveReplace}>Cancel</button>
                <button className="px-4 py-2 text-sm text-white bg-indigo-600 hover:bg-indigo-500 rounded-lg shadow transition-colors cursor-pointer" onClick={confirmSaveReplace}>Overwrite</button>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  // ─── Project list view ─────────────────────────────────────────────────────
  return (
    <div className="flex-1 p-8 overflow-y-auto">
      <div className="max-w-6xl mx-auto space-y-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">Designer Projects</h1>
            <p className="text-slate-400 text-sm mt-1">Create and manage your neural network models.</p>
          </div>
          <button
            className="px-4 py-2 bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-400 hover:to-red-400 text-white rounded-lg text-sm font-medium shadow-lg shadow-orange-500/20 transition-all cursor-pointer flex items-center gap-2"
            onClick={handleNewModel}
          >
            + New Model
          </button>
        </div>

        {loading ? (
          <div className="text-center text-slate-500 py-20">Loading projects...</div>
        ) : projects.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-24 text-center">
            <Network size={48} className="text-slate-700 mb-4" />
            <h3 className="text-lg font-semibold text-white mb-2">No projects yet</h3>
            <p className="text-slate-500 mb-6">Create a new model to get started.</p>
            <button
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-medium shadow-lg shadow-indigo-600/20 transition-colors cursor-pointer"
              onClick={handleNewModel}
            >
              + New Model
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {projects.map((p) => (
              <div
                key={p.id}
                className="bg-slate-900 border border-slate-800 rounded-xl p-5 cursor-pointer hover:border-slate-600 hover:shadow-xl transition-all group"
                onClick={() => handleOpen(p)}
                onContextMenu={(e) => handleContextMenu(e, p)}
              >
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-10 h-10 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
                    <Network size={20} />
                  </div>
                  <div className="min-w-0">
                    <h4 className="text-white font-medium truncate">{p.name}</h4>
                    <p className="text-xs text-slate-500">{p.node_count} layers · {p.edge_count} connections</p>
                  </div>
                </div>
                {p.description && (
                  <p className="text-xs text-slate-500 mb-3 line-clamp-2">{p.description}</p>
                )}
                <p className="text-[10px] text-slate-600">{formatDate(p.updated_at)}</p>
              </div>
            ))}
          </div>
        )}

        {/* Context menu */}
        {ctxMenu && (
          <div
            className="canvas-context-menu"
            style={{ top: ctxMenu.y, left: ctxMenu.x }}
            onClick={(e) => e.stopPropagation()}
          >
            <button className="ctx-menu-item" onClick={() => { handleOpen(ctxMenu.project); setCtxMenu(null); }}>
              <span className="ctx-menu-icon">📂</span> Open
            </button>
            <button className="ctx-menu-item" onClick={() => { setEditDialog({ project: ctxMenu.project, newName: ctxMenu.project.name }); setCtxMenu(null); }}>
              <span className="ctx-menu-icon">✏️</span> Edit Name
            </button>
            <button className="ctx-menu-item danger" onClick={() => { setDeleteConfirm(ctxMenu.project); setCtxMenu(null); }}>
              <span className="ctx-menu-icon">🗑️</span> Delete
            </button>
          </div>
        )}

        {/* Edit name dialog */}
        {editDialog && (
          <div className="modal-overlay" onClick={() => setEditDialog(null)}>
            <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 max-w-md shadow-2xl" onClick={(e) => e.stopPropagation()}>
              <h3 className="text-lg font-semibold text-white mb-4">Edit Model Name</h3>
              <input
                type="text"
                className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500 mb-5"
                value={editDialog.newName}
                onChange={(e) => setEditDialog({ ...editDialog, newName: e.target.value })}
                autoFocus
                onKeyDown={(e) => e.key === 'Enter' && handleEditSave()}
              />
              <div className="flex justify-end gap-3">
                <button className="px-4 py-2 text-sm text-slate-300 hover:bg-slate-700 rounded-lg transition-colors cursor-pointer" onClick={() => setEditDialog(null)}>Cancel</button>
                <button className="px-4 py-2 text-sm text-white bg-indigo-600 hover:bg-indigo-500 rounded-lg shadow transition-colors cursor-pointer" onClick={handleEditSave}>Save</button>
              </div>
            </div>
          </div>
        )}

        {/* Delete confirm */}
        {deleteConfirm && (
          <div className="modal-overlay" onClick={() => setDeleteConfirm(null)}>
            <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 max-w-md shadow-2xl" onClick={(e) => e.stopPropagation()}>
              <h3 className="text-lg font-semibold text-white mb-3">Delete Project</h3>
              <p className="text-slate-400 text-sm mb-1">Are you sure you want to delete <strong className="text-white">"{deleteConfirm.name}"</strong>?</p>
              <p className="text-red-400 text-xs mb-5">This action cannot be undone.</p>
              <div className="flex justify-end gap-3">
                <button className="px-4 py-2 text-sm text-slate-300 hover:bg-slate-700 rounded-lg transition-colors cursor-pointer" onClick={() => setDeleteConfirm(null)}>Cancel</button>
                <button className="px-4 py-2 text-sm text-white bg-red-600 hover:bg-red-500 rounded-lg shadow transition-colors cursor-pointer" onClick={() => handleDelete(deleteConfirm)}>Delete</button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
