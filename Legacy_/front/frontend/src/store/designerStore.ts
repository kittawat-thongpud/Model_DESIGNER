/**
 * Zustand store for the Model DESIGNER application.
 * Manages React Flow nodes/edges, selected node, and UI state.
 */
import { create } from 'zustand';
import {
  type Node,
  type Edge,
  type OnNodesChange,
  type OnEdgesChange,
  type OnConnect,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
} from '@xyflow/react';
import type { NodeParams, ModelGraph, TrainStatus, GlobalVariable } from '../types';
import { api } from '../services/api';
import { validateConnection } from '../utils/edgeValidator';
import { useNodeCatalogStore } from './nodeCatalogStore';

// ─── Store types ─────────────────────────────────────────────────────────────

interface DesignerState {
  // Graph
  nodes: Node[];
  edges: Edge[];
  globalVars: GlobalVariable[];
  modelName: string;
  modelDescription: string;
  modelId: string | null;

  // UI
  selectedNodeId: string | null;
  generatedCode: string | null;
  showCode: boolean;
  showLogs: boolean;
  logs: Array<{ timestamp: string; level: string; category: string; message: string }>;
  edgeError: string | null;

  // Training
  trainStatus: TrainStatus | null;
  isTraining: boolean;

  // Build duplicate check
  pendingBuild: { modelId: string; modelName: string; code: string; className: string } | null;
  
  // Save duplicate check
  pendingSave: { graph: ModelGraph; modelId: string } | null;

  // Actions — graph
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onConnect: OnConnect;
  addNode: (type: string, position: { x: number; y: number }, extra?: { packageId?: string }) => void;
  updateNodeParams: (nodeId: string, params: NodeParams) => void;
  deleteNode: (nodeId: string) => void;
  duplicateNode: (nodeId: string) => void;
  addGroup: (label: string, color: string, nodeIds: string[]) => void;
  ungroupNodes: (groupId: string) => void;
  deleteSelected: () => void;
  updateGroup: (groupId: string, updates: { label?: string; color?: string; description?: string }) => void;
  validateEdge: (sourceId: string, targetId: string) => boolean;
  clearEdgeError: () => void;
  selectNode: (nodeId: string | null) => void;
  setModelName: (name: string) => void;
  setModelDescription: (desc: string) => void;

  // Actions — global variables
  addGlobalVar: (gvar: GlobalVariable) => void;
  updateGlobalVar: (id: string, updates: Partial<GlobalVariable>) => void;
  deleteGlobalVar: (id: string) => void;
  setNodeEnabledByGlobal: (nodeId: string, varName: string | null) => void;

  // Actions — API
  saveModel: () => Promise<void>;
  loadModel: (id: string) => Promise<void>;
  loadGraph: (graph: ModelGraph) => void;
  buildModel: () => Promise<void>;
  startTraining: (config: Record<string, unknown>) => Promise<void>;
  stopTraining: () => Promise<void>;
  refreshLogs: () => Promise<void>;
  confirmBuildReplace: () => Promise<void>;
  cancelBuildReplace: () => void;
  confirmSaveReplace: () => Promise<void>;
  cancelSaveReplace: () => void;

  // Actions — UI
  setShowCode: (show: boolean) => void;
  setShowLogs: (show: boolean) => void;
  addLog: (level: string, category: string, message: string) => void;
}

let nodeCounter = 0;

/**
 * Convert a ModelGraph into React Flow nodes and edges.
 * Shared by loadModel and loadGraph to eliminate duplication.
 */
function _graphToReactFlow(graph: ModelGraph): { nodes: Node[]; edges: Edge[] } {
  nodeCounter = 0;
  const catalog = useNodeCatalogStore.getState();
  const nodes: Node[] = graph.nodes.map((n) => {
    const def = catalog.getDef(n.type);
    nodeCounter = Math.max(nodeCounter, parseInt(n.id.replace('node_', '')) || 0);
    return {
      id: n.id,
      type: 'layerNode',
      position: n.position,
      data: {
        layerType: n.type,
        label: def?.label ?? n.type,
        params: n.params,
        color: def?.color ?? '#666',
        icon: def?.icon ?? '❓',
        hasInput: def?.hasInput ?? true,
        hasOutput: def?.hasOutput ?? true,
        enabledByGlobal: n.enabledByGlobal,
      },
    };
  });

  const edges: Edge[] = graph.edges.map((e, i) => ({
    id: `e_${i}`,
    source: e.source,
    target: e.target,
    animated: true,
  }));

  return { nodes, edges };
}

export const useDesignerStore = create<DesignerState>((set, get) => ({
  // ── Initial state ──────────────────────────────────────────────────────────
  nodes: [],
  edges: [],
  globalVars: [],
  modelName: 'Untitled',
  modelDescription: '',
  modelId: null,
  selectedNodeId: null,
  generatedCode: null,
  showCode: false,
  showLogs: false,
  edgeError: null,
  logs: [],
  // Training
  trainStatus: null,
  isTraining: false,

  // Build duplicate check
  pendingBuild: null,
  
  // Save duplicate check
  pendingSave: null,

  // ── Graph changes from React Flow ──────────────────────────────────────────
  onNodesChange: (changes) =>
    set((state) => ({ nodes: applyNodeChanges(changes, state.nodes) })),

  onEdgesChange: (changes) =>
    set((state) => ({ edges: applyEdgeChanges(changes, state.edges) })),

  onConnect: (connection) => {
    const { nodes, edges } = get();
    if (!connection.source || !connection.target) return;
    const result = validateConnection(connection.source, connection.target, nodes, edges);
    if (!result.valid) {
      set({ edgeError: result.reason || 'Invalid connection' });
      get().addLog('WARNING', 'model', `Edge rejected: ${result.reason}`);
      // Auto-clear error after 3s
      setTimeout(() => set({ edgeError: null }), 3000);
      return;
    }
    set((state) => ({
      edges: addEdge({ ...connection, animated: true }, state.edges),
      edgeError: null,
    }));
  },

  validateEdge: (sourceId, targetId) => {
    const { nodes, edges } = get();
    return validateConnection(sourceId, targetId, nodes, edges).valid;
  },

  clearEdgeError: () => set({ edgeError: null }),

  // ── Add a new layer node ───────────────────────────────────────────────────
  addNode: (type, position, extra) => {
    const catalog = useNodeCatalogStore.getState();
    const def = catalog.getDef(type);
    const id = `node_${++nodeCounter}`;

    const defaultParams: NodeParams = {};
    for (const p of (def?.params ?? [])) {
      defaultParams[p.name] = p.default;
    }

    const newNode: Node = {
      id,
      type: 'layerNode',
      position,
      data: {
        layerType: type,
        label: def?.label ?? type,
        params: defaultParams,
        color: def?.color ?? '#666',
        icon: def?.icon ?? '❓',
        hasInput: def?.hasInput ?? true,
        hasOutput: def?.hasOutput ?? true,
        packageId: extra?.packageId,
      },
    };

    set((state) => ({ nodes: [...state.nodes, newNode] }));
    get().addLog('INFO', 'model', `Added ${def?.label ?? type} node`);

    // For Package nodes, fetch metadata to populate I/O info and params
    if (type === 'Package' && extra?.packageId) {
      api.getPackage(extra.packageId).then((pkg) => {
        const inputNodes = (pkg.nodes || []).filter((n: any) => n.type === 'Input');
        const outputNodes = (pkg.nodes || []).filter((n: any) => n.type === 'Output');
        const inputSummary = inputNodes.map((n: any) => {
          const p = n.params || {};
          return `${p.channels ?? '?'}×${p.height ?? '?'}×${p.width ?? '?'}`;
        }).join(', ') || 'none';
        const outputSummary = outputNodes.map((n: any) => {
          const p = n.params || {};
          return `${p.num_classes ?? p.out_features ?? '?'}`;
        }).join(', ') || 'none';

        // Build params from exposed_params
        const pkgParams: NodeParams = {};
        for (const ep of (pkg.exposed_params || [])) {
          pkgParams[ep.name] = ep.default;
        }

        set((state) => ({
          nodes: state.nodes.map((n) =>
            n.id === id
              ? {
                  ...n,
                  data: {
                    ...n.data,
                    label: pkg.name || 'Package',
                    params: pkgParams,
                    _pkgInputs: inputNodes.length,
                    _pkgOutputs: outputNodes.length,
                    _pkgInputShape: inputSummary,
                    _pkgOutputShape: outputSummary,
                    _pkgDescription: pkg.description || '',
                  },
                }
              : n
          ),
        }));
      }).catch(() => { /* package fetch failed, keep defaults */ });
    }
  },

  updateNodeParams: (nodeId, params) =>
    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === nodeId ? { ...n, data: { ...n.data, params: { ...(n.data.params as Record<string, unknown>), ...params } } } : n
      ),
    })),

  deleteNode: (nodeId) =>
    set((state) => ({
      nodes: state.nodes.filter((n) => n.id !== nodeId),
      edges: state.edges.filter((e) => e.source !== nodeId && e.target !== nodeId),
      selectedNodeId: state.selectedNodeId === nodeId ? null : state.selectedNodeId,
    })),

  duplicateNode: (nodeId) => {
    const node = get().nodes.find((n) => n.id === nodeId);
    if (!node) return;
    const id = node.type === 'groupNode' ? `group_${++nodeCounter}` : `node_${++nodeCounter}`;

    // Only copy essential properties — do NOT spread the whole node
    // (avoids copying React Flow internals: measured, internals, width, height, etc.)
    const newNode: Node = {
      id,
      type: node.type,
      position: { x: node.position.x + 40, y: node.position.y + 40 },
      data: { ...node.data, params: { ...(node.data.params as Record<string, unknown> || {}) } },
      selected: false,
      ...(node.parentId ? { parentId: node.parentId, extent: node.extent } : {}),
      ...(node.style ? { style: { ...node.style } } : {}),
    };

    set((state) => ({ nodes: [...state.nodes, newNode] }));
    get().addLog('INFO', 'model', `Duplicated node ${nodeId} → ${id}`);
  },

  addGroup: (label, color, nodeIds) => {
    const { nodes } = get();
    const selected = nodes.filter((n) => nodeIds.includes(n.id));
    if (selected.length === 0) return;

    // Convert positions to absolute (canvas-level) for nodes inside a parent group
    const absPositions = selected.map((n) => {
      if (n.parentId) {
        const parent = nodes.find((p) => p.id === n.parentId);
        if (parent) {
          return { x: n.position.x + parent.position.x, y: n.position.y + parent.position.y };
        }
      }
      return { x: n.position.x, y: n.position.y };
    });

    // Compute bounding box
    const xs = absPositions.map((p) => p.x);
    const ys = absPositions.map((p) => p.y);
    const minX = Math.min(...xs) - 20;
    const minY = Math.min(...ys) - 40;
    const maxX = Math.max(...xs) + 200;
    const maxY = Math.max(...ys) + 120;

    const groupId = `group_${++nodeCounter}`;
    const groupNode: Node = {
      id: groupId,
      type: 'groupNode',
      position: { x: minX, y: minY },
      style: { width: maxX - minX, height: maxY - minY },
      data: { label, color, description: '' },
    };

    // Re-parent selected nodes inside the new group (detach from old parent)
    const absMap = new Map(selected.map((n, i) => [n.id, absPositions[i]]));
    const updatedNodes = nodes.map((n) => {
      if (nodeIds.includes(n.id)) {
        const abs = absMap.get(n.id)!;
        return {
          ...n,
          parentId: groupId,
          position: { x: abs.x - minX, y: abs.y - minY },
          extent: 'parent' as const,
        };
      }
      return n;
    });

    set({ nodes: [groupNode, ...updatedNodes] });
    get().addLog('INFO', 'model', `Grouped ${nodeIds.length} nodes into "${label}"`);
  },

  ungroupNodes: (groupId) => {
    const { nodes } = get();
    const group = nodes.find((n) => n.id === groupId);
    if (!group) return;

    // Move children back to canvas-level coordinates
    const updatedNodes = nodes
      .filter((n) => n.id !== groupId)
      .map((n) => {
        if (n.parentId === groupId) {
          return {
            ...n,
            parentId: undefined,
            extent: undefined,
            position: {
              x: n.position.x + group.position.x,
              y: n.position.y + group.position.y,
            },
          };
        }
        return n;
      });

    set({ nodes: updatedNodes });
    get().addLog('INFO', 'model', `Ungrouped "${group.data.label}"`);
  },

  deleteSelected: () => {
    const { nodes, edges } = get();
    const selectedIds = new Set(nodes.filter((n) => n.selected).map((n) => n.id));
    if (selectedIds.size === 0) return;
    set({
      nodes: nodes.filter((n) => !selectedIds.has(n.id)),
      edges: edges.filter((e) => !selectedIds.has(e.source) && !selectedIds.has(e.target)),
      selectedNodeId: null,
    });
    get().addLog('INFO', 'model', `Deleted ${selectedIds.size} selected nodes`);
  },

  updateGroup: (groupId, updates) =>
    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === groupId ? { ...n, data: { ...n.data, ...updates } } : n
      ),
    })),

  selectNode: (nodeId) => set({ selectedNodeId: nodeId }),

  setModelName: (name) => set({ modelName: name }),
  setModelDescription: (desc) => set({ modelDescription: desc }),

  // ── Global variables ───────────────────────────────────────────────────────
  addGlobalVar: (gvar) =>
    set((s) => ({ globalVars: [...s.globalVars, gvar] })),

  updateGlobalVar: (id, updates) =>
    set((s) => ({
      globalVars: s.globalVars.map((v) => (v.id === id ? { ...v, ...updates } : v)),
    })),

  deleteGlobalVar: (id) =>
    set((s) => ({
      globalVars: s.globalVars.filter((v) => v.id !== id),
      // Also clear any node references to this var
      nodes: s.nodes.map((n) => {
        const gvar = s.globalVars.find((v) => v.id === id);
        if (!gvar) return n;
        const newData = { ...n.data };
        // Clear enabledByGlobal references
        if (newData.enabledByGlobal === gvar.name) {
          newData.enabledByGlobal = undefined;
        }
        // Clear $var_name param references
        if (newData.params) {
          const newParams = { ...newData.params } as Record<string, unknown>;
          for (const [k, v] of Object.entries(newParams)) {
            if (typeof v === 'string' && v === `$${gvar.name}`) {
              delete newParams[k];
            }
          }
          newData.params = newParams;
        }
        return { ...n, data: newData };
      }),
    })),

  setNodeEnabledByGlobal: (nodeId, varName) =>
    set((s) => ({
      nodes: s.nodes.map((n) =>
        n.id === nodeId
          ? { ...n, data: { ...n.data, enabledByGlobal: varName ?? undefined } }
          : n
      ),
    })),

  // ── API actions ────────────────────────────────────────────────────────────
  saveModel: async () => {
    const { nodes, edges, modelName, modelDescription } = get();
    const graph: ModelGraph = {
      meta: {
        name: modelName,
        version: '1.0',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        description: modelDescription,
      },
      nodes: nodes.map((n) => ({
        id: n.id,
        type: n.data.layerType as string,
        position: n.position,
        params: n.data.params as Record<string, number | number[] | string | boolean | undefined>,
        enabledByGlobal: n.data.enabledByGlobal as string | undefined,
      })),
      edges: edges.map((e) => ({
        source: e.source,
        target: e.target,
        source_handle: e.sourceHandle ?? undefined,
        target_handle: e.targetHandle ?? undefined,
      })),
      globals: get().globalVars,
    };

    if (get().modelId) {
      graph.id = get().modelId!;
    }

    try {
      const result = await api.saveModel(graph);
      
      if (result.exists) {
        // Name collision -> Prompt replace
        set({ pendingSave: { graph, modelId: result.model_id } });
        get().addLog('WARNING', 'model', `Model "${modelName}" already exists. Replace?`);
      } else {
        set({ modelId: result.model_id });
        get().addLog('INFO', 'model', `Model saved: ${result.model_id}`);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      get().addLog('ERROR', 'model', `Save failed: ${msg}`);
    }
  },

  confirmSaveReplace: async () => {
    const pending = get().pendingSave;
    if (!pending) return;
    try {
      const result = await api.saveModel(pending.graph, true);
      set({ modelId: result.model_id, pendingSave: null });
      get().addLog('INFO', 'model', `Model overwritten: ${result.model_id}`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      get().addLog('ERROR', 'model', `Save replace failed: ${msg}`);
      set({ pendingSave: null });
    }
  },

  cancelSaveReplace: () => {
    set({ pendingSave: null });
    get().addLog('INFO', 'model', 'Save replace cancelled');
  },

  loadModel: async (id) => {
    try {
      const graph = await api.loadModel(id);
      const { nodes, edges } = _graphToReactFlow(graph);

      set({
        nodes,
        edges,
        globalVars: graph.globals ?? [],
        modelName: graph.meta.name,
        modelDescription: graph.meta.description,
        modelId: id,
        selectedNodeId: null,
      });
      get().addLog('INFO', 'model', `Model loaded: ${graph.meta.name}`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      get().addLog('ERROR', 'model', `Load failed: ${msg}`);
    }
  },

  loadGraph: (graph) => {
    const { nodes, edges } = _graphToReactFlow(graph);

    set({
      nodes,
      edges,
      globalVars: graph.globals ?? [],
      modelName: graph.meta.name,
      modelDescription: graph.meta.description,
      modelId: null,
      selectedNodeId: null,
    });
    get().addLog('INFO', 'model', `Loaded graph: ${graph.meta.name}`);
  },

  buildModel: async () => {
    const { modelId } = get();
    if (!modelId) {
      get().addLog('WARNING', 'model', 'Save the model first before building');
      return;
    }
    try {
      const result = await api.buildModel(modelId);
      if (result.exists) {
        // Duplicate found — ask user to confirm
        set({
          pendingBuild: {
            modelId,
            modelName: result.model_name || '',
            code: result.code,
            className: result.class_name,
          },
        });
        get().addLog('WARNING', 'model', `Build "${result.model_name}" already exists. Replace?`);
      } else {
        set({ generatedCode: result.code, showCode: true, pendingBuild: null });
        get().addLog('INFO', 'model', `Code generated and saved: ${result.class_name}`);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      get().addLog('ERROR', 'model', `Build failed: ${msg}`);
    }
  },

  confirmBuildReplace: async () => {
    const pending = get().pendingBuild;
    if (!pending) return;
    try {
      const result = await api.buildModel(pending.modelId, true);
      set({ generatedCode: result.code, showCode: true, pendingBuild: null });
      get().addLog('INFO', 'model', `Build replaced: ${result.class_name}`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      get().addLog('ERROR', 'model', `Build replace failed: ${msg}`);
      set({ pendingBuild: null });
    }
  },

  cancelBuildReplace: () => {
    set({ pendingBuild: null });
    get().addLog('INFO', 'model', 'Build replace cancelled');
  },

  startTraining: async (config) => {
    const { modelId } = get();
    if (!modelId) {
      get().addLog('WARNING', 'training', 'Save the model first');
      return;
    }
    try {
      const result = await api.startTraining({ model_id: modelId, ...config } as any);
      set({ isTraining: true });
      get().addLog('INFO', 'training', `Training started: job ${result.job_id}`);

      // Poll for status
      const interval = setInterval(async () => {
        try {
          const status = await api.getTrainStatus(result.job_id);
          set({ trainStatus: status });
          if (status.status === 'completed' || status.status === 'failed' || status.status === 'stopped') {
            set({ isTraining: false });
            clearInterval(interval);
            get().addLog('INFO', 'training', `Training ${status.status}: ${status.message}`);
          }
        } catch {
          clearInterval(interval);
          set({ isTraining: false });
        }
      }, 2000);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      get().addLog('ERROR', 'training', `Training failed: ${msg}`);
    }
  },

  stopTraining: async () => {
    const { trainStatus } = get();
    if (!trainStatus) return;
    try {
      await api.stopTraining(trainStatus.job_id);
      get().addLog('INFO', 'training', 'Stop signal sent');
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      get().addLog('ERROR', 'training', `Stop failed: ${msg}`);
    }
  },

  refreshLogs: async () => {
    try {
      const logs = await api.getLogs({ limit: 50 });
      set({
        logs: logs.map((l) => ({
          timestamp: l.timestamp,
          level: l.level,
          category: l.category,
          message: l.message,
        })),
      });
    } catch {
      // ignore
    }
  },

  // ── UI ─────────────────────────────────────────────────────────────────────
  setShowCode: (show) => set({ showCode: show }),
  setShowLogs: (show) => set({ showLogs: show }),

  addLog: (level, category, message) =>
    set((state) => ({
      logs: [{ timestamp: new Date().toISOString(), level, category, message }, ...state.logs].slice(0, 200),
    })),
}));
