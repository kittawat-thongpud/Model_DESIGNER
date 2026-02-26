/**
 * Custom React Flow node for rendering a neural network layer.
 * Each node shows its type icon, label, parameters, and computed output shape.
 * 
 * Supports global variable display:
 * - Param values starting with $ are shown in accent color
 * - Nodes with enabledByGlobal set to a false bool are dimmed
 */
import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { useDesignerStore } from '../store/designerStore';

function LayerNode({ id, data, selected }: NodeProps) {
  const selectNode = useDesignerStore((s) => s.selectNode);
  const globalVars = useDesignerStore((s) => s.globalVars);
  const {
    label, color, icon, hasInput, hasOutput, params, layerType,
    _outputShape, _inferred, enabledByGlobal,
    _pkgInputs, _pkgOutputs, _pkgInputShape, _pkgOutputShape, _pkgDescription,
  } = data as {
    layerType: string;
    label: string;
    color: string;
    icon: string;
    hasInput: boolean;
    hasOutput: boolean;
    params: Record<string, unknown>;
    _outputShape?: string;
    _inferred?: Record<string, number>;
    enabledByGlobal?: string;
    _pkgInputs?: number;
    _pkgOutputs?: number;
    _pkgInputShape?: string;
    _pkgOutputShape?: string;
    _pkgDescription?: string;
  };

  // Check if this node is conditionally disabled
  const isConditionallyDisabled = (() => {
    if (!enabledByGlobal) return false;
    const gvar = globalVars.find((v) => v.name === enabledByGlobal);
    return gvar ? gvar.value === false : false;
  })();

  // Merge inferred values with user params for display
  const displayParams = { ...params };
  if (_inferred) {
    for (const [key, val] of Object.entries(_inferred)) {
      displayParams[key] = val;
    }
  }

  const visibleParams = Object.entries(displayParams || {}).filter(
    ([key]) => !key.startsWith('_')
  );

  return (
    <div
      className={`layer-node ${selected ? 'selected' : ''} ${isConditionallyDisabled ? 'conditionally-disabled' : ''}`}
      style={{ '--node-color': color } as React.CSSProperties}
      onClick={() => selectNode(id)}
    >
      {hasInput && (
        <Handle type="target" position={Position.Left} className="handle handle-input" />
      )}

      <div className="node-header">
        <span className="node-icon">{icon}</span>
        <span className="node-label">{label}</span>
        {enabledByGlobal && (
          <span className={`node-conditional-badge ${isConditionallyDisabled ? 'off' : 'on'}`} title={`Enabled by $${enabledByGlobal}`}>
            ⚡
          </span>
        )}
      </div>

      <div className="node-body">
        {visibleParams.map(([key, value]) => {
          const isInferred = _inferred && key in _inferred;
          const isGlobalRef = typeof value === 'string' && value.startsWith('$');
          return (
            <div key={key} className={`node-param ${isInferred ? 'inferred' : ''}`}>
              <span className="param-key">{key}</span>
              <span className={`param-value ${isGlobalRef ? 'global-ref' : ''}`}>
                {String(value)}
              </span>
            </div>
          );
        })}
        {visibleParams.length === 0 && layerType !== 'Package' && (
          <div className="node-param">
            <span className="param-key" style={{ opacity: 0.5 }}>no params</span>
          </div>
        )}

        {/* Package I/O badges */}
        {layerType === 'Package' && (_pkgInputs != null || _pkgOutputs != null) && (
          <>
            <div className="node-param" style={{ borderTop: '1px solid rgba(255,255,255,0.06)', marginTop: '2px', paddingTop: '4px' }}>
              <span className="param-key" style={{ color: '#4ade80' }}>📥 in</span>
              <span className="param-value">{_pkgInputs ?? 0} ({_pkgInputShape || '?'})</span>
            </div>
            <div className="node-param">
              <span className="param-key" style={{ color: '#f87171' }}>📤 out</span>
              <span className="param-value">{_pkgOutputs ?? 0} ({_pkgOutputShape || '?'})</span>
            </div>
            {_pkgDescription && (
              <div className="node-param">
                <span className="param-key" style={{ opacity: 0.5, fontSize: '9px', fontStyle: 'italic' }}>{_pkgDescription}</span>
              </div>
            )}
          </>
        )}

        {layerType === 'Package' && _pkgInputs == null && visibleParams.length === 0 && (
          <div className="node-param">
            <span className="param-key" style={{ opacity: 0.5 }}>loading...</span>
          </div>
        )}
      </div>

      {/* Output shape badge */}
      {_outputShape && _outputShape !== '—' && (
        <div className="node-shape-badge">
          <span className="shape-label">out</span>
          <span className="shape-value">{_outputShape}</span>
        </div>
      )}

      {hasOutput && (
        <Handle type="source" position={Position.Right} className="handle handle-output" />
      )}
    </div>
  );
}

export default memo(LayerNode);
