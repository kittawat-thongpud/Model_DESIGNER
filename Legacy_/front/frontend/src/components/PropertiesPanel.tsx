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
import { useNodeCatalogStore } from '../store/nodeCatalogStore';
import type { DatasetInfo, ParamDefinition } from '../types';
import { api } from '../services/api';
import { Settings } from 'lucide-react';

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
      <div className="w-72 border-l border-slate-800 bg-slate-900/50 flex flex-col items-center justify-center text-slate-500 shrink-0">
        <Settings size={32} className="mb-3 opacity-30" />
        <p className="text-sm">Select a node to edit</p>
      </div>
    );
  }
  // ─── Group node editing ──────────────────────────────────────────────────────
  if (selectedNode.type === 'groupNode') {
    const groupData = selectedNode.data as { label?: string; color?: string; description?: string };
    const updateGroup = useDesignerStore.getState().updateGroup;
    return (
      <div className="w-72 border-l border-slate-800 bg-slate-900/50 flex flex-col shrink-0">
        <div className="p-4 border-b border-slate-800 flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">📦</div>
          <h3 className="text-white font-semibold text-sm">Group Properties</h3>
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1.5">Label</label>
            <input className="w-full bg-slate-950 border border-slate-800 rounded-md px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500" value={groupData.label || ''} onChange={(e) => updateGroup(selectedNode.id, { label: e.target.value })} />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1.5">Color</label>
            <input type="color" className="w-full h-9 bg-slate-950 border border-slate-800 rounded-md p-0.5 cursor-pointer" value={groupData.color || '#6366f1'} onChange={(e) => updateGroup(selectedNode.id, { color: e.target.value })} />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1.5">Description</label>
            <textarea className="w-full bg-slate-950 border border-slate-800 rounded-md px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500 resize-y" value={groupData.description || ''} onChange={(e) => updateGroup(selectedNode.id, { description: e.target.value })} rows={2} />
          </div>
        </div>
      </div>
    );
  }

  const layerType = selectedNode.data.layerType as string;
  const def = useNodeCatalogStore.getState().getDef(layerType);
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

  const paramHints: Record<string, string> = {
    channels: 'Number of input channels (e.g. 1 for grayscale, 3 for RGB)',
    height: 'Input image height in pixels',
    width: 'Input image width in pixels',
    in_channels: 'Must match the output channels of the previous layer',
    out_channels: 'Number of filters / output feature maps',
    kernel_size: 'Size of the convolution kernel (e.g. 3 = 3×3)',
    stride: 'Step size of the kernel sliding over input',
    padding: 'Zero-padding added to input edges',
    in_features: 'Must match flattened output size of previous layer',
    out_features: 'Number of output neurons',
    num_features: 'Must match channel count from previous layer',
    num_classes: 'Number of classification categories',
    p: 'Probability of dropping a neuron (0 to 1)',
    dim: 'Dimension along which to apply the operation',
    scale_factor: 'Multiplier for spatial dimensions',
    mode: 'Interpolation mode: nearest or bilinear',
  };

  const inputCls = "w-full bg-slate-950 border border-slate-800 rounded-md px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500 transition-colors";

  const renderParamField = (paramDef: ParamDefinition, readOnly = false) => {
    const isBound = isGlobalRef(paramDef.name);
    const compatible = getCompatibleGlobals(paramDef);
    const hasGlobals = compatible.length > 0;
    const hint = paramHints[paramDef.name];

    return (
      <div key={paramDef.name}>
        <label className="flex items-center gap-1.5 text-xs font-medium text-slate-400 mb-1.5" title={hint}>
          {paramDef.label}
          {hint && (
            <span className="w-3.5 h-3.5 rounded-full bg-slate-800 text-slate-500 text-[9px] inline-flex items-center justify-center cursor-help" title={hint}>?</span>
          )}
          {hasGlobals && (
            <button
              className={`ml-auto text-xs cursor-pointer ${isBound ? 'text-indigo-400' : 'text-slate-600 hover:text-slate-400'}`}
              onClick={() => toggleGlobalBinding(paramDef)}
              title={isBound ? 'Unbind global variable' : 'Bind to global variable'}
            >
              🔗
            </button>
          )}
        </label>

        {isBound ? (
          <select
            className={`${inputCls} appearance-none`}
            value={String(params[paramDef.name])}
            onChange={(e) => updateNodeParams(selectedNode.id, { [paramDef.name]: e.target.value })}
          >
            {compatible.map((gvar) => (
              <option key={gvar.id} value={`$${gvar.name}`}>
                ${gvar.name} ({gvar.type} = {String(gvar.value)})
              </option>
            ))}
          </select>
        ) : paramDef.type === 'boolean' ? (
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={Boolean(params[paramDef.name] ?? paramDef.default)}
              onChange={(e) => updateNodeParams(selectedNode.id, { [paramDef.name]: e.target.checked })}
              className="accent-indigo-500"
            />
            <span className="text-xs text-slate-500">{params[paramDef.name] ? 'Enabled' : 'Disabled'}</span>
          </label>
        ) : (
          <input
            type={paramDef.type === 'number' ? 'number' : 'text'}
            className={inputCls}
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
    <div className="mt-5 pt-4 border-t border-slate-800">
      <label className="flex items-center gap-1.5 text-xs font-medium text-slate-400 mb-1.5">
        ⚡ Enabled by
        <span className="text-slate-600 text-[10px]">conditional</span>
      </label>
      <select
        className={`${inputCls} appearance-none`}
        value={currentEnabledBy ?? ''}
        onChange={(e) => setNodeEnabledByGlobal(selectedNode.id, e.target.value || null)}
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
    <div className="p-4 border-b border-slate-800 flex items-center gap-3 shrink-0">
      <div
        className="w-10 h-10 rounded-lg flex items-center justify-center text-lg border"
        style={{ background: `${def?.color || '#666'}15`, borderColor: `${def?.color || '#666'}30`, color: def?.color || '#666' }}
      >
        {def?.icon || '❓'}
      </div>
      <div className="min-w-0">
        <h4 className="text-white font-medium text-sm">{def?.label || layerType}</h4>
        <p className="text-[10px] text-slate-500 font-mono truncate">{selectedNode.id}</p>
      </div>
    </div>
  );

  const footer = (
    <div className="p-4 border-t border-slate-800 shrink-0">
      <button
        className="w-full py-2 bg-red-500/10 text-red-400 hover:bg-red-500/20 rounded-md text-sm font-medium transition-colors border border-red-500/20 cursor-pointer"
        onClick={() => deleteNode(selectedNode.id)}
      >
        Delete Node
      </button>
    </div>
  );

  // ─── Input / Output node special UI ────────────────────────────────────────

  if (layerType === 'Input' || layerType === 'Output') {
    const currentDataset = datasets.find((d) => d.name === selectedDataset);

    return (
      <div className="w-72 border-l border-slate-800 bg-slate-900/50 flex flex-col shrink-0">
        {header}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Mode toggle */}
          <div className="flex rounded-lg overflow-hidden border border-slate-800">
            <button
              className={`flex-1 py-1.5 text-xs font-medium transition-colors cursor-pointer ${inputMode === 'dataset' ? 'bg-indigo-600 text-white' : 'bg-slate-950 text-slate-400 hover:text-white'}`}
              onClick={() => handleModeSwitch('dataset')}
            >
              📦 Dataset
            </button>
            <button
              className={`flex-1 py-1.5 text-xs font-medium transition-colors cursor-pointer ${inputMode === 'manual' ? 'bg-indigo-600 text-white' : 'bg-slate-950 text-slate-400 hover:text-white'}`}
              onClick={() => handleModeSwitch('manual')}
            >
              ✏️ Manual
            </button>
          </div>

          {/* Dataset mode */}
          {inputMode === 'dataset' && (
            <div className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1.5">Dataset</label>
                <select className={`${inputCls} appearance-none`} value={selectedDataset} onChange={(e) => handleDatasetSelect(e.target.value)}>
                  <option value="">Select a dataset…</option>
                  {datasets.map((ds) => (
                    <option key={ds.name} value={ds.name}>
                      {ds.display_name}
                      {layerType === 'Input' ? ` (${ds.input_shape.join('×')})` : ` (${ds.num_classes} classes)`}
                    </option>
                  ))}
                </select>
              </div>

              {currentDataset && (
                <div className="bg-slate-950 rounded-lg border border-slate-800 p-3 text-xs space-y-2">
                  {layerType === 'Input' && (
                    <div className="flex justify-between"><span className="text-slate-500">Shape</span><span className="text-white font-mono">{currentDataset.input_shape.join(' × ')}</span></div>
                  )}
                  <div className="flex justify-between"><span className="text-slate-500">Task</span><span className="text-indigo-400 uppercase text-[10px] font-medium">{currentDataset.task_type || 'classification'}</span></div>
                  <div className="flex justify-between"><span className="text-slate-500">Classes</span><span className="text-white">{currentDataset.num_classes}</span></div>
                  <div className="flex justify-between"><span className="text-slate-500">Train</span><span className="text-white">{currentDataset.train_size.toLocaleString()}</span></div>
                  <div className="flex justify-between"><span className="text-slate-500">Test</span><span className="text-white">{currentDataset.test_size.toLocaleString()}</span></div>
                </div>
              )}
            </div>
          )}

          {/* Params */}
          <div className="space-y-4">
            <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-2">
              {layerType === 'Input' ? 'Input Shape' : 'Output Config'}
              {inputMode === 'dataset' && <span className="px-1.5 py-0.5 text-[9px] bg-emerald-500/10 text-emerald-400 rounded border border-emerald-500/20">auto</span>}
            </h4>

            {layerType === 'Output' && inputMode === 'manual' && (
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1.5">Task Type</label>
                <select className={`${inputCls} appearance-none`} value={params.task_type || 'classification'} onChange={(e) => updateNodeParams(selectedNode.id, { task_type: e.target.value })}>
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
    <div className="w-72 border-l border-slate-800 bg-slate-900/50 flex flex-col shrink-0">
      {header}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {displayParams.map((paramDef) => renderParamField(paramDef))}

        {(!displayParams || displayParams.length === 0) && (
          <p className="text-xs text-slate-500 text-center py-4">No editable parameters.</p>
        )}

        {enabledBySection}
      </div>
      {footer}
    </div>
  );
}
