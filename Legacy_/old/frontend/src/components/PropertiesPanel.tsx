/**
 * Properties Panel — edit params of the selected node.
 * For Input nodes, shows a dataset selector to auto-fill shape from dataset metadata.
 * For Output nodes, shows a dataset selector to auto-fill num_classes.
 * For other nodes, shows the standard parameter editor.
 * 
 * Supports global variable binding: any param can reference a $var_name from the
 * global variables store by clicking the 🔗 button next to the input field.
 */
import { useState, useEffect } from 'react';
import { useDesignerStore } from '../store/designerStore';
import { LAYER_DEFINITIONS, type LayerType, type DatasetInfo, type ParamDefinition } from '../types';
import { api } from '../services/api';

export default function PropertiesPanel() {
  const selectedNodeId = useDesignerStore((s) => s.selectedNodeId);
  const nodes = useDesignerStore((s) => s.nodes);
  const updateNodeParams = useDesignerStore((s) => s.updateNodeParams);
  const deleteNode = useDesignerStore((s) => s.deleteNode);
  const globalVars = useDesignerStore((s) => s.globalVars);
  const setNodeEnabledByGlobal = useDesignerStore((s) => s.setNodeEnabledByGlobal);

  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [inputMode, setInputMode] = useState<'manual' | 'dataset'>('manual');
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [packageParams, setPackageParams] = useState<ParamDefinition[]>([]);

  const selectedNode = nodes.find((n) => n.id === selectedNodeId);
  const isIONode =
    selectedNode &&
    (selectedNode.data.layerType === 'Input' || selectedNode.data.layerType === 'Output');

  // Fetch datasets when an Input or Output node is selected
  useEffect(() => {
    if (isIONode) {
      api.listDatasets().then(setDatasets).catch(() => {});

      const params = (selectedNode.data.params || {}) as Record<string, number | string>;
      if (params._datasetSource) {
        setInputMode('dataset');
        setSelectedDataset(params._datasetSource as string);
      } else {
        setInputMode('manual');
        setSelectedDataset('');
      }
    } else if (selectedNode?.data.layerType === 'Package' && selectedNode.data.packageId) {
       // Fetch package params
       api.getPackage(selectedNode.data.packageId as string).then(pkg => {
          // Convert PackageParameter to ParamDefinition
          const defs: ParamDefinition[] = pkg.exposed_params.map(p => ({
              name: p.name,
              type: p.type === 'int' || p.type === 'float' ? 'number' : 'string',
              default: p.default as any,
              label: p.name, // Use name as label
          }));
          setPackageParams(defs);
       }).catch(() => setPackageParams([]));
    }
  }, [selectedNodeId]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!selectedNode) {
    return (
      <div className="properties-panel empty">
        <div className="empty-state">
          <span className="empty-icon">🔧</span>
          <p>Select a node to edit its properties</p>
        </div>
      </div>
    );
  }
  // ─── Group node editing ──────────────────────────────────────────────────────
  if (selectedNode.type === 'groupNode') {
    const groupData = selectedNode.data as { label?: string; color?: string; description?: string };
    const updateGroup = useDesignerStore.getState().updateGroup;
    return (
      <div className="properties-panel">
        <div className="props-header">
          <span className="props-icon">📦</span>
          <span className="props-title">Group Properties</span>
        </div>
        <div className="props-body">
          <div className="prop-field">
            <label className="prop-label">Label</label>
            <input
              className="prop-input"
              value={groupData.label || ''}
              onChange={(e) => updateGroup(selectedNode.id, { label: e.target.value })}
            />
          </div>
          <div className="prop-field">
            <label className="prop-label">Color</label>
            <input
              type="color"
              className="prop-input"
              value={groupData.color || '#6366f1'}
              onChange={(e) => updateGroup(selectedNode.id, { color: e.target.value })}
              style={{ height: '36px', padding: '2px' }}
            />
          </div>
          <div className="prop-field">
            <label className="prop-label">Description</label>
            <textarea
              className="prop-input"
              value={groupData.description || ''}
              onChange={(e) => updateGroup(selectedNode.id, { description: e.target.value })}
              rows={2}
              style={{ resize: 'vertical' }}
            />
          </div>
        </div>
      </div>
    );
  }

  const layerType = selectedNode.data.layerType as LayerType;
  const def = LAYER_DEFINITIONS[layerType];
  const params = (selectedNode.data.params || {}) as Record<string, number | string>;
  
  // Use dynamic params for Package nodes
  const displayParams = layerType === 'Package' ? packageParams : (def?.params || []);

  const handleChange = (paramName: string, value: string) => {
    const numVal = Number(value);
    updateNodeParams(selectedNode.id, { [paramName]: isNaN(numVal) ? value : numVal });
  };

  // ─── Global variable binding helpers ─────────────────────────────────────────

  const isGlobalRef = (paramName: string): boolean => {
    const val = params[paramName];
    return typeof val === 'string' && val.startsWith('$');
  };

  const getCompatibleGlobals = (paramDef: ParamDefinition) => {
    // Allow ALL variable types to link — filter by best-match first, fallback to all
    return globalVars.filter((gvar) => {
      if (paramDef.type === 'number') return gvar.type === 'float' || gvar.type === 'int';
      if (paramDef.type === 'string') return gvar.type === 'str' || gvar.type === 'selector';
      if (paramDef.type === 'boolean') return gvar.type === 'bool';
      if (paramDef.type === 'array') return false; // arrays can't bind to a single global
      // Fallback: allow any global
      return true;
    });
  };

  const toggleGlobalBinding = (paramDef: ParamDefinition) => {
    if (isGlobalRef(paramDef.name)) {
      // Unbind: restore default value
      updateNodeParams(selectedNode.id, { [paramDef.name]: paramDef.default });
    } else {
      // Bind: pick first compatible global, or show empty
      const compatible = getCompatibleGlobals(paramDef);
      if (compatible.length > 0) {
        updateNodeParams(selectedNode.id, { [paramDef.name]: `$${compatible[0].name}` });
      }
    }
  };

  // ─── Param field with optional global binding ──────────────────────────────

  const renderParamField = (paramDef: ParamDefinition, readOnly = false) => {
    const isBound = isGlobalRef(paramDef.name);
    const compatible = getCompatibleGlobals(paramDef);
    const hasGlobals = compatible.length > 0;

    return (
      <div key={paramDef.name} className="prop-field">
        <label className="prop-label">
          {paramDef.label}
          {hasGlobals && (
            <button
              className={`prop-bind-btn ${isBound ? 'bound' : ''}`}
              onClick={() => toggleGlobalBinding(paramDef)}
              title={isBound ? 'Unbind global variable' : 'Bind to global variable'}
            >
              {isBound ? '🔗' : '🔗'}
            </button>
          )}
        </label>

        {isBound ? (
          <select
            className="prop-input prop-input-global"
            value={String(params[paramDef.name])}
            onChange={(e) =>
              updateNodeParams(selectedNode.id, { [paramDef.name]: e.target.value })
            }
          >
            {compatible.map((gvar) => (
              <option key={gvar.id} value={`$${gvar.name}`}>
                ${gvar.name} ({gvar.type} = {String(gvar.value)})
              </option>
            ))}
          </select>
        ) : paramDef.type === 'boolean' ? (
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={Boolean(params[paramDef.name] ?? paramDef.default)}
              onChange={(e) =>
                updateNodeParams(selectedNode.id, { [paramDef.name]: e.target.checked })
              }
            />
            <span style={{ fontSize: '12px', opacity: 0.7 }}>
              {params[paramDef.name] ? 'Enabled' : 'Disabled'}
            </span>
          </label>
        ) : (
          <input
            type={paramDef.type === 'number' ? 'number' : 'text'}
            className="prop-input"
            value={params[paramDef.name] ?? paramDef.default}
            min={paramDef.min}
            max={paramDef.max}
            readOnly={readOnly}
            onChange={(e) => handleChange(paramDef.name, e.target.value)}
          />
        )}
      </div>
    );
  };

  // ─── enabledByGlobal selector ──────────────────────────────────────────────

  const boolGlobals = globalVars.filter((v) => v.type === 'bool');
  const currentEnabledBy = selectedNode.data.enabledByGlobal as string | undefined;

  const enabledBySection = boolGlobals.length > 0 && layerType !== 'Input' && layerType !== 'Output' ? (
    <div className="props-conditional-section">
      <label className="prop-label">
        ⚡ Enabled by
        <span className="prop-label-hint">conditional layer</span>
      </label>
      <select
        className="prop-input prop-input-conditional"
        value={currentEnabledBy ?? ''}
        onChange={(e) =>
          setNodeEnabledByGlobal(selectedNode.id, e.target.value || null)
        }
      >
        <option value="">Always enabled</option>
        {boolGlobals.map((gvar) => (
          <option key={gvar.id} value={gvar.name}>
            ${gvar.name} ({gvar.value ? 'true' : 'false'})
          </option>
        ))}
      </select>
    </div>
  ) : null;

  // ─── Dataset select handler (shared by Input & Output) ─────────────────────

  const handleDatasetSelect = (datasetName: string) => {
    setSelectedDataset(datasetName);
    const ds = datasets.find((d) => d.name === datasetName);
    if (!ds) return;

    if (layerType === 'Input' && ds.input_shape.length >= 3) {
      updateNodeParams(selectedNode.id, {
        channels: ds.input_shape[0],
        height: ds.input_shape[1],
        width: ds.input_shape[2],
        _datasetSource: datasetName,
        task_type: ds.task_type || 'classification',
      });
    } else if (layerType === 'Output') {
      updateNodeParams(selectedNode.id, {
        num_classes: ds.num_classes,
        out_features: ds.num_classes,
        _datasetSource: datasetName,
        task_type: ds.task_type || 'classification',
      });
    }
  };

  const handleModeSwitch = (mode: 'manual' | 'dataset') => {
    setInputMode(mode);
    if (mode === 'manual') {
      setSelectedDataset('');
      updateNodeParams(selectedNode.id, { _datasetSource: '' });
    }
  };

  // ─── Shared header ─────────────────────────────────────────────────────────

  const header = (
    <div className="props-header" style={{ borderColor: def?.color || '#666' }}>
      <span className="props-icon">{def?.icon || '❓'}</span>
      <div>
        <h3 className="props-title">{def?.label || layerType}</h3>
        <span className="props-id">{selectedNode.id}</span>
      </div>
    </div>
  );

  const footer = (
    <div className="props-footer">
      <button className="btn btn-danger" onClick={() => deleteNode(selectedNode.id)}>
        🗑️ Delete Node
      </button>
    </div>
  );

  // ─── Input / Output node special UI ────────────────────────────────────────

  if (layerType === 'Input' || layerType === 'Output') {
    const currentDataset = datasets.find((d) => d.name === selectedDataset);

    return (
      <div className="properties-panel">
        {header}

        <div className="props-body">
          {/* Mode toggle */}
          <div className="input-mode-toggle">
            <button
              className={`mode-btn ${inputMode === 'dataset' ? 'active' : ''}`}
              onClick={() => handleModeSwitch('dataset')}
            >
              📦 From Dataset
            </button>
            <button
              className={`mode-btn ${inputMode === 'manual' ? 'active' : ''}`}
              onClick={() => handleModeSwitch('manual')}
            >
              ✏️ Manual
            </button>
          </div>

          {/* Dataset mode */}
          {inputMode === 'dataset' && (
            <div className="dataset-selector">
              <label className="prop-label">Dataset</label>
              <select
                className="prop-select"
                value={selectedDataset}
                onChange={(e) => handleDatasetSelect(e.target.value)}
              >
                <option value="">Select a dataset…</option>
                {datasets.map((ds) => (
                  <option key={ds.name} value={ds.name}>
                    {ds.display_name}
                    {layerType === 'Input'
                      ? ` (${ds.input_shape.join('×')})`
                      : ` (${ds.num_classes} classes)`}
                  </option>
                ))}
              </select>

              {currentDataset && (
                <div className="dataset-info-card">
                  {layerType === 'Input' && (
                    <div className="dataset-info-row">
                      <span className="dataset-info-label">Shape</span>
                      <span className="dataset-info-value">
                        {currentDataset.input_shape.join(' × ')}
                      </span>
                      <span className="dataset-info-value">
                        {currentDataset.input_shape.join(' × ')}
                      </span>
                    </div>
                  )}
                  <div className="dataset-info-row">
                    <span className="dataset-info-label">Task</span>
                    <span className={`tcm-badge-task ${currentDataset.task_type || 'classification'}`}>
                      {(currentDataset.task_type || 'classification').toUpperCase()}
                    </span>
                  </div>
                  <div className="dataset-info-row">
                    <span className="dataset-info-label">Classes</span>
                    <span className="dataset-info-value">{currentDataset.num_classes}</span>
                  </div>
                  <div className="dataset-info-row">
                    <span className="dataset-info-label">Train</span>
                    <span className="dataset-info-value">
                      {currentDataset.train_size.toLocaleString()}
                    </span>
                  </div>
                  <div className="dataset-info-row">
                    <span className="dataset-info-label">Test</span>
                    <span className="dataset-info-value">
                      {currentDataset.test_size.toLocaleString()}
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Params (read-only in dataset mode) */}
          <div className="input-shape-section">
            <h4 className="shape-section-title">
              {layerType === 'Input' ? 'Input Shape' : 'Output Config'}{' '}
              {inputMode === 'dataset' && <span className="auto-badge">auto</span>}
            </h4>

            {layerType === 'Output' && inputMode === 'manual' && (
              <div style={{ marginBottom: '1rem' }}>
                <label className="prop-label">
                  Task Type <span className="prop-label-hint">manual override</span>
                </label>
                <select
                  className="prop-select"
                  value={params.task_type || 'classification'}
                  onChange={(e) =>
                    updateNodeParams(selectedNode.id, { task_type: e.target.value })
                  }
                >
                  <option value="classification">Classification</option>
                  <option value="detection">Detection</option>
                </select>
              </div>
            )}

            {def?.params.map((paramDef) => renderParamField(paramDef, inputMode === 'dataset'))}
          </div>
        </div>

        {footer}
      </div>
    );
  }

  // ─── Default properties panel for other layers ──────────────────────────────
  return (
    <div className="properties-panel">
      {header}

      <div className="props-body">
        {displayParams.map((paramDef) => renderParamField(paramDef))}

        {(!displayParams || displayParams.length === 0) && (
          <p className="no-params">This layer has no editable parameters.</p>
        )}

        {enabledBySection}
      </div>

      {footer}
    </div>
  );
}
