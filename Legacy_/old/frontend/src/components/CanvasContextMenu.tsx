/**
 * Canvas Context Menu — right-click context menu for canvas operations.
 * Supports node-level, selection-level, and canvas-level actions.
 */
import React from 'react';
import { useDesignerStore } from '../store/designerStore';

export interface ContextMenuState {
  x: number;
  y: number;
  nodeId?: string;         // if right-clicked on a specific node
  isMultiSelect?: boolean; // if multiple nodes are selected
}

interface Props {
  menu: ContextMenuState;
  onClose: () => void;
}

export const CanvasContextMenu: React.FC<Props> = ({ menu, onClose }) => {
  const deleteNode = useDesignerStore((s) => s.deleteNode);
  const duplicateNode = useDesignerStore((s) => s.duplicateNode);
  const addGroup = useDesignerStore((s) => s.addGroup);
  const deleteSelected = useDesignerStore((s) => s.deleteSelected);
  const ungroupNodes = useDesignerStore((s) => s.ungroupNodes);
  const nodes = useDesignerStore((s) => s.nodes);

  const selectedNodes = nodes.filter((n) => n.selected);
  const isGroup = menu.nodeId
    ? nodes.find((n) => n.id === menu.nodeId)?.type === 'groupNode'
    : false;

  const handleAction = (action: () => void) => {
    action();
    onClose();
  };

  const menuItems: Array<{ label: string; icon: string; action: () => void; danger?: boolean }> = [];

  if (menu.nodeId && !menu.isMultiSelect) {
    // Single node context
    menuItems.push(
      { label: 'Duplicate', icon: '📋', action: () => duplicateNode(menu.nodeId!) },
    );
    if (isGroup) {
      menuItems.push(
        { label: 'Ungroup', icon: '📂', action: () => ungroupNodes(menu.nodeId!) },
      );
    }
    menuItems.push(
      { label: 'Delete', icon: '🗑️', action: () => deleteNode(menu.nodeId!), danger: true },
    );
  } else if (menu.isMultiSelect || selectedNodes.length > 1) {
    // Multi-selection context
    const selectedIds = selectedNodes.map((n) => n.id);
    menuItems.push(
      {
        label: `Group ${selectedNodes.length} nodes`,
        icon: '📦',
        action: () => addGroup('Group', '#6366f1', selectedIds),
      },
      { label: 'Delete selected', icon: '🗑️', action: () => deleteSelected(), danger: true },
    );
  } else {
    // Empty canvas context
    menuItems.push(
      { label: 'Select All', icon: '⬜', action: () => {
        // Select all nodes via store
        useDesignerStore.getState().onNodesChange(
          nodes.map((n) => ({ type: 'select' as const, id: n.id, selected: true }))
        );
      }},
    );
  }

  return (
    <div
      className="canvas-context-menu"
      style={{ left: menu.x, top: menu.y }}
      onMouseLeave={onClose}
    >
      {menuItems.map((item, i) => (
        <button
          key={i}
          className={`ctx-menu-item ${item.danger ? 'danger' : ''}`}
          onClick={() => handleAction(item.action)}
        >
          <span className="ctx-menu-icon">{item.icon}</span>
          <span>{item.label}</span>
        </button>
      ))}
    </div>
  );
};
