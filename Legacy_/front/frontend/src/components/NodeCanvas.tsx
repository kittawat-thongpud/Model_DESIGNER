/**
 * Node Canvas — React Flow wrapper with drag-and-drop support.
 * Computes tensor shapes for all nodes and injects them into node data.
 * Includes edge validation — invalid connections are rejected with visual feedback.
 */
import { useCallback, useRef, useMemo, useState } from 'react';
import {
  ReactFlow,
  Controls,
  MiniMap,
  Background,
  BackgroundVariant,
  SelectionMode,
  type NodeTypes,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import LayerNode from './LayerNode';
import GroupNode from './GroupNode';
import { CanvasContextMenu, type ContextMenuState } from './CanvasContextMenu';
import { useDesignerStore } from '../store/designerStore';
import { computeGraphShapes, formatShape } from '../utils/shapeInference';

const nodeTypes: NodeTypes = {
  layerNode: LayerNode as any,
  groupNode: GroupNode as any,
};

export default function NodeCanvas() {
  const nodes = useDesignerStore((s) => s.nodes);
  const edges = useDesignerStore((s) => s.edges);
  const onNodesChange = useDesignerStore((s) => s.onNodesChange);
  const onEdgesChange = useDesignerStore((s) => s.onEdgesChange);
  const onConnect = useDesignerStore((s) => s.onConnect);
  const addNode = useDesignerStore((s) => s.addNode);
  const selectNode = useDesignerStore((s) => s.selectNode);
  const validateEdge = useDesignerStore((s) => s.validateEdge);
  const edgeError = useDesignerStore((s) => s.edgeError);

  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);

  // Compute shapes for all nodes whenever nodes/edges change
  const shapeMap = useMemo(() => computeGraphShapes(nodes, edges), [nodes, edges]);

  // Inject shape info into each node's data
  const nodesWithShapes = useMemo(() => {
    return nodes.map((node) => {
      const info = shapeMap.get(node.id);
      if (!info) return node;
      return {
        ...node,
        data: {
          ...node.data,
          _outputShape: formatShape(info.output),
          _inputShape: formatShape(info.input),
          _inferred: info.inferred,
        },
      };
    });
  }, [nodes, shapeMap]);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const type = e.dataTransfer.getData('application/layer-type');
      if (!type) return;

      const bounds = reactFlowWrapper.current?.getBoundingClientRect();
      if (!bounds) return;

      const position = {
        x: e.clientX - bounds.left - 80,
        y: e.clientY - bounds.top - 25,
      };

      // Support packageId from drag-and-drop
      const packageId = e.dataTransfer.getData('application/package-id');
      addNode(type, position, packageId ? { packageId } : undefined);
    },
    [addNode]
  );

  const onPaneClick = useCallback(() => {
    selectNode(null);
    setContextMenu(null);
  }, [selectNode]);

  // ─── Right-click context menu ───────────────────────────────────────
  const onNodeContextMenu = useCallback(
    (e: React.MouseEvent, node: { id: string }) => {
      e.preventDefault();
      e.stopPropagation();
      const selected = useDesignerStore.getState().nodes.filter((n) => n.selected);
      // Only treat as multi-select if the right-clicked node is part of the selection
      const clickedIsSelected = selected.some((n) => n.id === node.id);
      setContextMenu({
        x: e.clientX,
        y: e.clientY,
        nodeId: node.id,
        isMultiSelect: clickedIsSelected && selected.length > 1,
      });
    },
    []
  );

  const onPaneContextMenu = useCallback((e: React.MouseEvent | MouseEvent) => {
    e.preventDefault();
    const selected = useDesignerStore.getState().nodes.filter((n) => n.selected);
    if (selected.length > 1) {
      setContextMenu({
        x: (e as React.MouseEvent).clientX,
        y: (e as React.MouseEvent).clientY,
        isMultiSelect: true,
      });
    } else {
      setContextMenu({
        x: (e as React.MouseEvent).clientX,
        y: (e as React.MouseEvent).clientY,
      });
    }
  }, []);

  const isValidConnection = useCallback(
    (connection: { source?: string | null; target?: string | null }) => {
      if (!connection.source || !connection.target) return false;
      return validateEdge(connection.source, connection.target);
    },
    [validateEdge]
  );

  const defaultEdgeOptions = useMemo(
    () => ({
      animated: true,
      style: { stroke: '#6366f1', strokeWidth: 2 },
    }),
    []
  );

  return (
    <div className="flex-1 relative bg-slate-950 overflow-hidden" ref={reactFlowWrapper}>
      <ReactFlow
        nodes={nodesWithShapes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onPaneClick={onPaneClick}
        onNodeContextMenu={onNodeContextMenu}
        onPaneContextMenu={onPaneContextMenu}
        nodeTypes={nodeTypes}
        defaultEdgeOptions={defaultEdgeOptions}
        isValidConnection={isValidConnection}
        selectionMode={SelectionMode.Partial}
        selectionOnDrag
        panOnDrag={[1]}  /* middle-click pan */
        fitView
        proOptions={{ hideAttribution: true }}
        colorMode="dark"
      >
        <Controls
          position="bottom-right"
          style={{ background: '#1e293b', borderRadius: 8, border: '1px solid #334155' }}
        />
        <MiniMap
          position="bottom-left"
          style={{ background: '#1e293b', borderRadius: 8, border: '1px solid #334155' }}
          nodeColor={(node) => (node.data?.color as string) || '#666'}
          maskColor="rgba(0,0,0,0.6)"
        />
        <Background variant={BackgroundVariant.Dots} gap={24} size={1} color="#1e293b" />
      </ReactFlow>

      {/* Edge validation error toast */}
      {edgeError && (
        <div className="edge-error-toast">
          <span className="edge-error-icon">⚠️</span>
          <span>{edgeError}</span>
        </div>
      )}

      {/* Context menu */}
      {contextMenu && (
        <CanvasContextMenu
          menu={contextMenu}
          onClose={() => setContextMenu(null)}
        />
      )}
    </div>
  );
}
