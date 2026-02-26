/**
 * Group Node — visual container for grouping multiple nodes.
 * Renders as a colored, semi-transparent rectangle with a label header.
 * Supports selection, and properties editing (label, color).
 */
import { memo } from 'react';
import { type NodeProps, NodeResizer } from '@xyflow/react';
import { useDesignerStore } from '../store/designerStore';

function GroupNode({ id, data, selected }: NodeProps) {
  const selectNode = useDesignerStore((s) => s.selectNode);

  const {
    label = 'Group',
    color = '#6366f1',
    description = '',
  } = data as {
    label?: string;
    color?: string;
    description?: string;
  };

  return (
    <div
      className={`group-node ${selected ? 'selected' : ''}`}
      style={{
        '--group-color': color,
        minWidth: '200px',
        minHeight: '150px',
        width: '100%',
        height: '100%',
      } as React.CSSProperties}
      onClick={(e) => {
        e.stopPropagation();
        selectNode(id);
      }}
    >
      <NodeResizer
        isVisible={selected}
        minWidth={150}
        minHeight={100}
        lineClassName="group-resize-line"
        handleClassName="group-resize-handle"
      />
      <div className="group-header">
        <span
          className="group-color-dot"
          style={{ background: color }}
        />
        <span className="group-label">{label}</span>
      </div>
      {description && (
        <div className="group-description">{description}</div>
      )}
    </div>
  );
}

export default memo(GroupNode);
