/**
 * Create Train Job Modal — Comprehensive training configuration modal.
 * 
 * Extracted from TrainDesignerPage to provide a modal interface for creating
 * training jobs directly from the Training Jobs page.
 */
import { useState, useEffect } from 'react';
import { api } from '../services/api';
import type { ModelSummary, DatasetInfo, TrainConfig } from '../types';
import { Play, Settings, Loader2, Search, Layers, X } from 'lucide-react';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onJobCreated: (jobId: string) => void;
}

const DEFAULT_CONFIG: TrainConfig = {
  data: '',
  imgsz: 640,
  batch: 16,
  workers: 8,
  epochs: 100,
  patience: 100,
  device: '',
  seed: 0,
  deterministic: true,
  amp: true,
  close_mosaic: 10,
  optimizer: 'auto',
  lr0: 0.01,
  lrf: 0.01,
  momentum: 0.937,
  weight_decay: 0.0005,
  warmup_epochs: 3.0,
  warmup_momentum: 0.8,
  warmup_bias_lr: 0.1,
  cos_lr: false,
  nbs: 64,
  pretrained: '',
  yolo_model: '',
  use_yolo_pretrained: true,
  freeze: 0,
  resume: false,
  box: 7.5,
  cls: 0.5,
  dfl: 1.5,
  pose: 12.0,
  kobj: 1.0,
  overlap_mask: true,
  mask_ratio: 4,
  conf: null,
  iou: 0.7,
  max_det: 300,
  agnostic_nms: false,
  rect: false,
  single_cls: false,
  hsv_h: 0.015,
  hsv_s: 0.7,
  hsv_v: 0.4,
  degrees: 0.0,
  translate: 0.1,
  scale: 0.5,
  shear: 0.0,
  perspective: 0.0,
  flipud: 0.0,
  fliplr: 0.5,
  bgr: 0.0,
  mosaic: 1.0,
  mixup: 0.0,
  copy_paste: 0.0,
  erasing: 0.4,
  crop_fraction: 1.0,
  auto_augment: '',
  save_period: -1,
  val: true,
  plots: true,
  record_gradients: false,
  gradient_interval: 1,
  record_weights: false,
  weight_interval: 1,
  sample_per_class: 0,
};

type Tab = 'general' | 'optimizer' | 'loss' | 'augmentation' | 'validation';

// Official YOLO model as virtual model card (scale selected via existing scale selector)
const OFFICIAL_YOLO_MODELS: ModelSummary[] = [
  {
    model_id: 'yolo:yolov8',
    name: 'YOLOv8 (Official)',
    task: 'detect',
    layer_count: 225,
    params: 3157200,  // Base params for 'n' scale
    flops: 8.9,       // Base FLOPs for 'n' scale
    input_shape: [3, 640, 640],
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  },
];

export default function CreateTrainJobModal({ isOpen, onClose, onJobCreated }: Props) {
  const [customModels, setCustomModels] = useState<ModelSummary[]>([]);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [weights, setWeights] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);

  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [modelScale, setModelScale] = useState<string>('n');
  const [config, setConfig] = useState<TrainConfig>(DEFAULT_CONFIG);
  const [activeTab, setActiveTab] = useState<Tab>('general');
  const [searchQuery, setSearchQuery] = useState('');
  const [partitionSplitConfig, setPartitionSplitConfig] = useState<Record<string, {train: boolean; val: boolean; test: boolean}>>({});
  const [datasetPartitions, setDatasetPartitions] = useState<Record<string, any>>({});
  
  // Merge official YOLO models with custom models
  const models = [...OFFICIAL_YOLO_MODELS, ...customModels];

  useEffect(() => {
    if (isOpen) {
      setLoading(true);
      Promise.all([api.listModels(), api.listDatasets(), api.listWeights()])
        .then(([m, d, w]) => {
          setCustomModels(m ?? []);
          setDatasets(d ?? []);
          setWeights(w ?? []);
        })
        .finally(() => setLoading(false));
    }
  }, [isOpen]);

  // Calculate scale-specific metrics locally
  const getScaledMetrics = (baseParams?: number, baseFlops?: number, scale: string = 'n') => {
    if (!baseParams) return { params: undefined, flops: undefined };
    
    // Approximate scale multipliers (based on YOLOv8 scaling)
    const scaleMultipliers: Record<string, { depth: number; width: number }> = {
      'n': { depth: 0.33, width: 0.25 },
      's': { depth: 0.33, width: 0.50 },
      'm': { depth: 0.67, width: 0.75 },
      'l': { depth: 1.00, width: 1.00 },
      'x': { depth: 1.00, width: 1.25 }
    };
    
    const mult = scaleMultipliers[scale] || scaleMultipliers['n'];
    // Params scale roughly with width^2 * depth
    const paramScale = Math.pow(mult.width, 2) * mult.depth;
    // FLOPs scale roughly with width^2 * depth
    const flopScale = Math.pow(mult.width, 2) * mult.depth;
    
    return {
      params: Math.round(baseParams * paramScale),
      flops: baseFlops ? baseFlops * flopScale : undefined
    };
  };

  // Fetch partition data when dataset is selected
  useEffect(() => {
    if (config.data && !datasetPartitions[config.data]) {
      api.getDatasetPartitions(config.data)
        .then(partitionData => {
          setDatasetPartitions(prev => ({
            ...prev,
            [config.data]: partitionData
          }));
        })
        .catch(err => {
          console.error(`Failed to fetch partitions for ${config.data}:`, err);
        });
    }
  }, [config.data]);

  const handleStartTraining = async () => {
    if (!selectedModelId) return;
    if (!config.data) {
      alert('Please select a dataset');
      return;
    }
    const selectedPartitions = Object.entries(partitionSplitConfig).filter(([_, splits]) => 
      splits.train || splits.val || splits.test
    );
    if (selectedPartitions.length === 0) {
      alert('Please select at least one partition with at least one split');
      return;
    }

    setSubmitting(true);
    try {
      const partitionConfigs = Object.entries(partitionSplitConfig)
        .filter(([_, splits]) => splits.train || splits.val || splits.test)
        .map(([partition_id, splits]) => ({
          partition_id,
          train: splits.train,
          val: splits.val,
          test: splits.test
        }));
      
      // Detect if official YOLO model is selected
      const finalConfig = { ...config };
      if (selectedModelId.startsWith('yolo:')) {
        // Use YOLOv8 with selected scale (e.g., 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')
        finalConfig.yolo_model = `yolov8${modelScale}`;
        // use_yolo_pretrained is already set by user via radio buttons in UI
      }
      
      const res = await api.startTraining({
        model_id: selectedModelId,
        model_scale: modelScale,
        config: finalConfig,
        partitions: partitionConfigs,
      });
      onJobCreated(res.job_id);
      onClose();
      // Reset state
      setSelectedModelId(null);
      setModelScale('n');
      setConfig(DEFAULT_CONFIG);
      setActiveTab('general');
      setSearchQuery('');
      setPartitionSplitConfig({});
    } catch (err) {
      alert(`Failed to start training: ${err}`);
    } finally {
      setSubmitting(false);
    }
  };

  const updateConfig = (key: keyof TrainConfig, value: any) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  const selectedModel = models.find(m => m.model_id === selectedModelId);
  const filteredModels = models.filter(m => 
    m.name.toLowerCase().includes(searchQuery.toLowerCase()) || 
    m.task.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Normalize task names for compatibility check (detect <-> detection)
  const normalizeTask = (task: string) => {
    if (task === 'detect') return 'detection';
    if (task === 'detection') return 'detection';
    if (task === 'classify') return 'classification';
    if (task === 'classification') return 'classification';
    return task;
  };

  const isTaskCompatible = (datasetTask: string, modelTask: string) => {
    return normalizeTask(datasetTask) === normalizeTask(modelTask);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div 
        className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-full max-w-7xl h-[90vh] mx-4 flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-800 flex-shrink-0">
          <div>
            <h2 className="text-2xl font-bold text-white">Create Training Job</h2>
            <p className="text-sm text-slate-500 mt-1">Configure and launch Ultralytics training</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Modal Content */}
        {loading ? (
          <div className="flex items-center justify-center flex-1">
            <Loader2 className="w-8 h-8 text-emerald-500 animate-spin" />
          </div>
        ) : (
          <div className="flex gap-6 flex-1 min-h-0 p-6">
            
            {/* Left Sidebar: Model Selection */}
            <div className="w-80 flex flex-col bg-slate-900/50 rounded-xl border border-slate-800 overflow-hidden flex-shrink-0">
              <div className="p-4 border-b border-slate-800">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={14} />
                  <input 
                    type="text" 
                    placeholder="Search models..." 
                    value={searchQuery}
                    onChange={e => setSearchQuery(e.target.value)}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg pl-9 pr-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500"
                  />
                </div>
              </div>
              
              <div className="flex-1 overflow-y-auto p-2 space-y-1">
                {filteredModels.length === 0 ? (
                  <div className="text-center py-8 text-slate-500 text-sm">No models found</div>
                ) : (
                  filteredModels.map(m => (
                    <button
                      key={m.model_id}
                      onClick={() => setSelectedModelId(m.model_id)}
                      className={`w-full text-left p-3 rounded-lg border transition-all ${
                        selectedModelId === m.model_id
                          ? 'bg-emerald-500/10 border-emerald-500/50 shadow-[0_0_15px_rgba(16,185,129,0.1)]'
                          : 'bg-transparent border-transparent hover:bg-slate-800/50 hover:border-slate-700'
                      }`}
                    >
                      <div className="flex justify-between items-start mb-1">
                        <span className={`font-medium ${selectedModelId === m.model_id ? 'text-emerald-400' : 'text-slate-200'}`}>
                          {m.name}
                        </span>
                        <div className="flex items-center gap-1">
                          {m.model_id.startsWith('yolo:') && (
                            <span className="text-[10px] bg-amber-500/20 text-amber-400 px-1.5 py-0.5 rounded border border-amber-500/30 font-medium">
                              OFFICIAL
                            </span>
                          )}
                          <span className="text-[10px] bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded border border-slate-700">
                            {m.task}
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center gap-3 text-xs text-slate-500">
                        <span className="flex items-center gap-1"><Layers size={10} /> {m.layer_count} layers</span>
                        {!m.model_id.startsWith('yolo:') && <span>{new Date(m.updated_at).toLocaleDateString()}</span>}
                        {m.model_id.startsWith('yolo:') && <span className="text-amber-400">Pretrained on COCO</span>}
                      </div>
                      {!m.model_id.startsWith('yolo:') && (
                        <div className="mt-1 text-[10px] text-slate-600">
                          Task: <span className="text-slate-500">{m.task}</span>
                        </div>
                      )}
                    </button>
                  ))
                )}
              </div>
            </div>

            {/* Main Area: Configuration */}
            <div className="flex-1 bg-slate-900/50 rounded-xl border border-slate-800 flex flex-col overflow-hidden">
              {selectedModelId ? (
                <>
                  {/* Config Header */}
                  <div className="p-6 border-b border-slate-800 flex justify-between items-start">
                    <div className="flex-1">
                      <h3 className="text-xl font-bold text-white flex items-center gap-2">
                        <Settings size={20} className="text-emerald-400" />
                        Configuration: {selectedModel?.name}
                        {selectedModelId.startsWith('yolo:') && (
                          <span className="text-[11px] bg-amber-500/20 text-amber-400 px-2 py-1 rounded border border-amber-500/30 font-medium">
                            OFFICIAL YOLO
                          </span>
                        )}
                      </h3>
                      <div className="flex items-center gap-4 mt-2 text-sm flex-wrap">
                        {selectedModelId.startsWith('yolo:') && (
                          <span className="text-amber-400 text-xs">
                            ⚡ Pretrained on COCO dataset — ready for transfer learning
                          </span>
                        )}
                        <span className="text-slate-500">
                          Task: <span className="text-slate-300">{selectedModel?.task}</span>
                        </span>
                        {!selectedModelId.startsWith('yolo:') && (
                          <span className="text-slate-500">
                            ID: <span className="font-mono text-xs text-slate-400">{selectedModelId.substring(0, 8)}</span>
                          </span>
                        )}
                        <span className="text-slate-500">
                          Scale: <span className="text-emerald-400 font-mono text-xs font-bold">{modelScale.toUpperCase()}</span>
                          {selectedModelId.startsWith('yolo:') && (
                            <span className="text-amber-400 ml-1">(YOLOv8{modelScale})</span>
                          )}
                        </span>
                        {selectedModel?.input_shape && (
                          <span className="text-slate-500">
                            Input: <span className="text-emerald-400 font-mono text-xs">[{selectedModel.input_shape.join(', ')}]</span>
                          </span>
                        )}
                        {selectedModel?.params && (() => {
                          const scaled = getScaledMetrics(selectedModel.params, selectedModel.flops, modelScale);
                          return (
                            <>
                              <span className="text-slate-500">
                                Params: <span className="text-indigo-400 font-mono text-xs">{((scaled.params || 0) / 1e6).toFixed(2)}M</span>
                              </span>
                              {scaled.flops && (
                                <span className="text-slate-500">
                                  FLOPs: <span className="text-amber-400 font-mono text-xs">{scaled.flops.toFixed(1)} G</span>
                                </span>
                              )}
                            </>
                          );
                        })()}
                      </div>
                    </div>
                    <button
                      onClick={handleStartTraining}
                      disabled={submitting || !config.data}
                      className="px-6 py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-all flex items-center gap-2 shadow-lg shadow-emerald-900/20"
                    >
                      {submitting ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} fill="currentColor" />}
                      Start Training
                    </button>
                  </div>

                  {/* Tabs */}
                  <div className="flex border-b border-slate-800 px-6">
                    {(['general', 'optimizer', 'loss', 'augmentation', 'validation'] as Tab[]).map(tab => (
                      <button
                        key={tab}
                        onClick={() => setActiveTab(tab)}
                        className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                          activeTab === tab
                            ? 'border-emerald-500 text-emerald-400'
                            : 'border-transparent text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        {tab.charAt(0).toUpperCase() + tab.slice(1)}
                      </button>
                    ))}
                  </div>

                  {/* Form Content */}
                  <div className="flex-1 overflow-y-auto p-6">
                    
                    {/* GENERAL TAB */}
                    {activeTab === 'general' && (
                      <div className="grid grid-cols-2 gap-x-8 gap-y-6 max-w-4xl">
                        <div className="col-span-2">
                          <label className="block text-xs font-medium text-emerald-400 mb-1.5">Dataset (Required)</label>
                          <select 
                            value={config.data}
                            onChange={e => updateConfig('data', e.target.value)}
                            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2.5 text-sm text-white focus:border-emerald-500 focus:outline-none font-mono"
                          >
                            <option value="">Select a dataset...</option>
                            {datasets
                              .filter(d => d.available && selectedModel && isTaskCompatible(d.task_type, selectedModel.task))
                              .map(d => (
                                <option key={d.name} value={d.name}>
                                  {d.display_name} — {d.task_type} — [{d.input_shape.join(', ')}] — {d.num_classes} classes
                                </option>
                              ))
                            }
                            {datasets.filter(d => d.available && selectedModel && !isTaskCompatible(d.task_type, selectedModel.task)).length > 0 && (
                              <optgroup label="⚠️ Incompatible Task Type">
                                {datasets
                                  .filter(d => d.available && selectedModel && !isTaskCompatible(d.task_type, selectedModel.task))
                                  .map(d => (
                                    <option key={d.name} value={d.name} disabled>
                                      {d.display_name} — {d.task_type} — [{d.input_shape.join(', ')}] — {d.num_classes} classes
                                    </option>
                                  ))
                                }
                              </optgroup>
                            )}
                          </select>
                        </div>

                        <div className="col-span-2">
                          <label className="block text-xs font-medium text-slate-400 mb-1.5">Model Scale</label>
                          <div className="flex gap-2">
                            {['n', 's', 'm', 'l', 'x'].map(scale => (
                              <button
                                key={scale}
                                onClick={() => setModelScale(scale)}
                                className={`flex-1 px-4 py-2.5 rounded-lg border transition-all text-sm font-medium ${
                                  modelScale === scale
                                    ? 'bg-emerald-500/10 border-emerald-500/50 text-emerald-400 shadow-[0_0_10px_rgba(16,185,129,0.1)]'
                                    : 'bg-slate-800 border-slate-700 text-slate-400 hover:border-slate-600 hover:text-slate-300'
                                }`}
                              >
                                {scale.toUpperCase()}
                              </button>
                            ))}
                          </div>
                          <p className="text-[10px] text-slate-500 mt-1.5">
                            Scale affects model size: n (nano) → s (small) → m (medium) → l (large) → x (xlarge)
                          </p>
                        </div>

                        {/* Training Mode for Official YOLO */}
                        {selectedModelId?.startsWith('yolo:') && (
                          <div className="col-span-2">
                            <label className="block text-xs font-medium text-amber-400 mb-1.5">Training Mode</label>
                            <div className="flex gap-3">
                              <label className={`flex-1 flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-all border-2 ${
                                config.use_yolo_pretrained === true
                                  ? 'bg-emerald-500/10 border-emerald-500/50 shadow-[0_0_10px_rgba(16,185,129,0.1)]'
                                  : 'bg-slate-800/50 border-slate-700 hover:border-slate-600'
                              }`}>
                                <input
                                  type="radio"
                                  checked={config.use_yolo_pretrained === true}
                                  onChange={() => updateConfig('use_yolo_pretrained', true)}
                                  className="w-4 h-4 text-emerald-500 focus:ring-emerald-500"
                                />
                                <div className="flex-1">
                                  <div className="text-sm font-medium text-white">Pretrained (Transfer Learning)</div>
                                  <div className="text-xs text-slate-400 mt-0.5">Use COCO pretrained weights — faster convergence, better accuracy</div>
                                </div>
                              </label>
                              <label className={`flex-1 flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-all border-2 ${
                                config.use_yolo_pretrained === false
                                  ? 'bg-emerald-500/10 border-emerald-500/50 shadow-[0_0_10px_rgba(16,185,129,0.1)]'
                                  : 'bg-slate-800/50 border-slate-700 hover:border-slate-600'
                              }`}>
                                <input
                                  type="radio"
                                  checked={config.use_yolo_pretrained === false}
                                  onChange={() => updateConfig('use_yolo_pretrained', false)}
                                  className="w-4 h-4 text-emerald-500 focus:ring-emerald-500"
                                />
                                <div className="flex-1">
                                  <div className="text-sm font-medium text-white">From Scratch</div>
                                  <div className="text-xs text-slate-400 mt-0.5">Train without pretrained weights — slower but fully custom</div>
                                </div>
                              </label>
                            </div>
                          </div>
                        )}

                        <div className="col-span-2">
                          <div className="flex items-start gap-2">
                            <div className="flex-1 text-xs text-slate-500">
                              <p className="mb-1">
                                <span className="text-emerald-400">✓ Compatible:</span> Datasets matching model task ({selectedModel?.task})
                              </p>
                              {selectedModel && datasets.filter(d => d.available && isTaskCompatible(d.task_type, selectedModel.task)).length === 0 && (
                                <p className="text-amber-400">
                                  ⚠️ No compatible datasets found. Model expects <span className="font-mono">{selectedModel.task}</span> task.
                                </p>
                              )}
                            </div>
                            {config.data && datasets.find(d => d.name === config.data) && (
                              <div className="px-3 py-2 bg-slate-800/50 border border-slate-700 rounded-lg text-xs">
                                <div className="text-slate-400 mb-1">Selected Dataset Info:</div>
                                <div className="space-y-0.5 text-slate-300">
                                  <div>Input: <span className="font-mono text-emerald-400">[{datasets.find(d => d.name === config.data)?.input_shape.join(', ')}]</span></div>
                                  <div>Classes: <span className="text-emerald-400">{datasets.find(d => d.name === config.data)?.num_classes}</span></div>
                                  <div>Task: <span className="text-emerald-400">{datasets.find(d => d.name === config.data)?.task_type}</span></div>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>

                        {/* Partition Selection */}
                        {config.data && (() => {
                          const partitionData = datasetPartitions[config.data];
                          if (!partitionData) {
                            return (
                              <div className="col-span-2">
                                <div className="text-sm text-slate-400 flex items-center gap-2">
                                  <Loader2 size={14} className="animate-spin" />
                                  Loading partition data...
                                </div>
                              </div>
                            );
                          }
                          
                          // Filter partitions that have at least one non-zero split
                          const availablePartitions = partitionData.partitions.filter((p: any) => 
                            p.train_count > 0 || p.val_count > 0 || p.test_count > 0
                          );
                          
                          const allPartitionIds = availablePartitions.map((p: any) => p.id);
                          
                          return (
                            <div className="col-span-2">
                              <label className="block text-xs font-medium text-emerald-400 mb-2">
                                Select Dataset Partitions
                              </label>
                              <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                                <div className="flex items-center justify-between mb-3">
                                  <div>
                                    <span className="text-sm text-slate-300 block">Choose which partitions to use for training:</span>
                                    <span className="text-xs text-slate-500 mt-1 block">
                                      {availablePartitions.length} partition{availablePartitions.length > 1 ? 's' : ''} available • 
                                      Method: {partitionData.method}
                                    </span>
                                  </div>
                                  <button
                                    onClick={() => {
                                      const allSelected = availablePartitions.every((p: any) => {
                                        const splits = partitionSplitConfig[p.id];
                                        return splits && splits.train && splits.val && splits.test;
                                      });
                                      
                                      if (allSelected) {
                                        setPartitionSplitConfig({});
                                      } else {
                                        const newConfig: Record<string, {train: boolean; val: boolean; test: boolean}> = {};
                                        availablePartitions.forEach((p: any) => {
                                          newConfig[p.id] = {
                                            train: p.train_count > 0,
                                            val: p.val_count > 0,
                                            test: p.test_count > 0
                                          };
                                        });
                                        setPartitionSplitConfig(newConfig);
                                      }
                                    }}
                                    className="text-xs text-emerald-400 hover:text-emerald-300 font-medium transition-colors"
                                  >
                                    Select All Splits
                                  </button>
                                </div>
                                
                                <div className="space-y-3">
                                  {availablePartitions.map((partition: any) => {
                                    const splits = partitionSplitConfig[partition.id] || { train: false, val: false, test: false };
                                    const isAnySelected = splits.train || splits.val || splits.test;
                                    
                                    return (
                                      <div
                                        key={partition.id}
                                        className={`p-4 rounded-lg border-2 transition-all ${
                                          isAnySelected
                                            ? 'bg-emerald-500/10 border-emerald-500/50 shadow-[0_0_10px_rgba(16,185,129,0.1)]'
                                            : 'bg-slate-800/30 border-slate-700'
                                        }`}
                                      >
                                        <div className="flex items-center justify-between mb-3">
                                          <div className={`text-sm font-medium ${isAnySelected ? 'text-emerald-400' : 'text-slate-300'}`}>
                                            {partition.name}
                                          </div>
                                          <div className="text-xs text-slate-500">
                                            {partition.percent}% of dataset
                                          </div>
                                        </div>
                                        
                                        <div className="grid grid-cols-3 gap-2">
                                          {partition.train_count > 0 && (
                                            <label className="flex items-center gap-2 p-2 bg-slate-900/50 rounded cursor-pointer hover:bg-slate-900/70 transition-colors">
                                              <input
                                                type="checkbox"
                                                checked={splits.train}
                                                onChange={(e) => {
                                                  setPartitionSplitConfig(prev => ({
                                                    ...prev,
                                                    [partition.id]: { ...splits, train: e.target.checked }
                                                  }));
                                                }}
                                                className="w-3 h-3 rounded border-slate-600 text-emerald-500 focus:ring-emerald-500 focus:ring-offset-slate-900"
                                              />
                                              <div className="flex-1">
                                                <div className="text-[10px] text-slate-500">Train</div>
                                                <div className={`text-xs font-medium ${splits.train ? 'text-emerald-400' : 'text-slate-400'}`}>
                                                  {partition.train_count.toLocaleString()}
                                                </div>
                                              </div>
                                            </label>
                                          )}
                                          {partition.val_count > 0 && (
                                            <label className="flex items-center gap-2 p-2 bg-slate-900/50 rounded cursor-pointer hover:bg-slate-900/70 transition-colors">
                                              <input
                                                type="checkbox"
                                                checked={splits.val}
                                                onChange={(e) => {
                                                  setPartitionSplitConfig(prev => ({
                                                    ...prev,
                                                    [partition.id]: { ...splits, val: e.target.checked }
                                                  }));
                                                }}
                                                className="w-3 h-3 rounded border-slate-600 text-emerald-500 focus:ring-emerald-500 focus:ring-offset-slate-900"
                                              />
                                              <div className="flex-1">
                                                <div className="text-[10px] text-slate-500">Val</div>
                                                <div className={`text-xs font-medium ${splits.val ? 'text-emerald-400' : 'text-slate-400'}`}>
                                                  {partition.val_count.toLocaleString()}
                                                </div>
                                              </div>
                                            </label>
                                          )}
                                          {partition.test_count > 0 && (
                                            <label className="flex items-center gap-2 p-2 bg-slate-900/50 rounded cursor-pointer hover:bg-slate-900/70 transition-colors">
                                              <input
                                                type="checkbox"
                                                checked={splits.test}
                                                onChange={(e) => {
                                                  setPartitionSplitConfig(prev => ({
                                                    ...prev,
                                                    [partition.id]: { ...splits, test: e.target.checked }
                                                  }));
                                                }}
                                                className="w-3 h-3 rounded border-slate-600 text-emerald-500 focus:ring-emerald-500 focus:ring-offset-slate-900"
                                              />
                                              <div className="flex-1">
                                                <div className="text-[10px] text-slate-500">Test</div>
                                                <div className={`text-xs font-medium ${splits.test ? 'text-emerald-400' : 'text-slate-400'}`}>
                                                  {partition.test_count.toLocaleString()}
                                                </div>
                                              </div>
                                            </label>
                                          )}
                                        </div>
                                      </div>
                                    );
                                  })}
                                </div>
                                
                                <div className="mt-4 p-3 bg-slate-900/50 border border-slate-700/50 rounded-lg">
                                  <div className="flex items-start gap-2 text-xs">
                                    <span className="text-emerald-400 mt-0.5">💡</span>
                                    <div className="flex-1 text-slate-400">
                                      <span className="text-emerald-400 font-medium">Granular Split Selection:</span>
                                      <ul className="mt-1 space-y-1 list-disc list-inside">
                                        <li>Select specific splits (train/val/test) for each partition</li>
                                        <li>Example: Partition A [train + test], Partition B [train + val + test]</li>
                                        <li>Unselected splits will not be used</li>
                                        <li>Full flexibility to mix and match as needed</li>
                                      </ul>
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </div>
                          );
                        })()}

                        <div className="space-y-4">
                          <h4 className="text-sm font-semibold text-white border-b border-slate-800 pb-2">Training Params</h4>
                          <NumberInput label="Epochs" value={config.epochs} onChange={v => updateConfig('epochs', v)} />
                          <NumberInput label="Batch Size" value={config.batch} onChange={v => updateConfig('batch', v)} />
                          <NumberInput label="Image Size" value={config.imgsz} onChange={v => updateConfig('imgsz', v)} />
                          <NumberInput label="Workers" value={config.workers} onChange={v => updateConfig('workers', v)} />
                          <TextInput label="Device" value={config.device} onChange={v => updateConfig('device', v)} placeholder="e.g. 0,1 or cpu" />
                        </div>

                        <div className="space-y-4">
                          <h4 className="text-sm font-semibold text-white border-b border-slate-800 pb-2">Advanced</h4>
                          <div>
                            <label className="block text-xs font-medium text-slate-400 mb-1.5">Pretrained Weights</label>
                            <select
                              value={config.pretrained}
                              onChange={e => updateConfig('pretrained', e.target.value)}
                              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2.5 text-sm text-white focus:border-emerald-500 focus:outline-none"
                            >
                              <option value="">None (Train from scratch)</option>
                              {weights.map(w => (
                                <option key={w.weight_id} value={w.weight_id}>
                                  {w.name || w.weight_id} — {w.model_name} — {(w.file_size_bytes / 1024 / 1024).toFixed(1)} MB
                                </option>
                              ))}
                            </select>
                          </div>
                          <NumberInput label="Freeze Layers" value={Number(config.freeze) || 0} onChange={v => updateConfig('freeze', v)} />
                          <NumberInput label="Patience" value={config.patience} onChange={v => updateConfig('patience', v)} />
                          <NumberInput label="Seed" value={config.seed} onChange={v => updateConfig('seed', v)} />
                          <NumberInput label="Save Period (epochs)" value={config.save_period} onChange={v => updateConfig('save_period', v)} />
                          <div className="pt-2 space-y-2">
                            <Toggle label="Resume Training" checked={Boolean(config.resume)} onChange={v => updateConfig('resume', v)} />
                            <Toggle label="Deterministic" checked={config.deterministic} onChange={v => updateConfig('deterministic', v)} />
                            <Toggle label="AMP (Mixed Precision)" checked={config.amp} onChange={v => updateConfig('amp', v)} />
                            <Toggle label="Save Plots" checked={config.plots} onChange={v => updateConfig('plots', v)} />
                            <Toggle label="Validate during training" checked={config.val} onChange={v => updateConfig('val', v)} />
                          </div>
                        </div>

                        <div className="space-y-4">
                          <h4 className="text-sm font-semibold text-white border-b border-slate-800 pb-2">Checkpoints & Monitoring</h4>
                          <div className="space-y-3">
                            <div className="p-3 bg-slate-800/50 rounded-lg space-y-2">
                              <NumberInput 
                                label="Close Mosaic (last N epochs)" 
                                value={config.close_mosaic} 
                                onChange={v => updateConfig('close_mosaic', v)} 
                              />
                              <p className="text-xs text-slate-500">Disable mosaic augmentation in final epochs</p>
                            </div>

                            <div className="p-3 bg-slate-800/50 rounded-lg space-y-2">
                              <Toggle 
                                label="Record Gradients" 
                                checked={Boolean(config.record_gradients)} 
                                onChange={v => updateConfig('record_gradients', v)} 
                              />
                              {config.record_gradients ? (
                                <NumberInput 
                                  label="Gradient Interval (epochs)" 
                                  value={Number(config.gradient_interval) || 1} 
                                  onChange={v => updateConfig('gradient_interval', v)} 
                                />
                              ) : null}
                              <p className="text-xs text-slate-500">Record gradient statistics for analysis</p>
                            </div>
                            
                            <div className="p-3 bg-slate-800/50 rounded-lg space-y-2">
                              <Toggle 
                                label="Record Weights" 
                                checked={Boolean(config.record_weights)} 
                                onChange={v => updateConfig('record_weights', v)} 
                              />
                              {config.record_weights ? (
                                <NumberInput 
                                  label="Weight Interval (epochs)" 
                                  value={Number(config.weight_interval) || 1} 
                                  onChange={v => updateConfig('weight_interval', v)} 
                                />
                              ) : null}
                              <p className="text-xs text-slate-500">Record weight statistics for analysis</p>
                            </div>

                            <div className="p-3 bg-slate-800/50 rounded-lg space-y-2">
                              <NumberInput 
                                label="Validation Samples per Class" 
                                value={config.sample_per_class} 
                                onChange={v => updateConfig('sample_per_class', v)} 
                              />
                              <p className="text-xs text-slate-500">Number of validation images to save for each class (0 = disabled)</p>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* OPTIMIZER TAB */}
                    {activeTab === 'optimizer' && (
                      <div className="grid grid-cols-2 gap-x-8 gap-y-6 max-w-4xl">
                        <div className="col-span-2">
                           <label className="block text-xs font-medium text-slate-400 mb-1.5">Optimizer</label>
                           <select 
                              value={config.optimizer}
                              onChange={e => updateConfig('optimizer', e.target.value)}
                              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2.5 text-sm text-white focus:border-emerald-500 focus:outline-none"
                            >
                              {['auto', 'SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp'].map(opt => (
                                <option key={opt} value={opt}>{opt}</option>
                              ))}
                            </select>
                        </div>
                        
                        <div className="space-y-4">
                          <h4 className="text-sm font-semibold text-white border-b border-slate-800 pb-2">Hyperparameters</h4>
                          <NumberInput label="Initial LR (lr0)" value={config.lr0} onChange={v => updateConfig('lr0', v)} step={0.001} />
                          <NumberInput label="Final LR Factor (lrf)" value={config.lrf} onChange={v => updateConfig('lrf', v)} step={0.001} />
                          <NumberInput label="Momentum" value={config.momentum} onChange={v => updateConfig('momentum', v)} step={0.001} />
                          <NumberInput label="Weight Decay" value={config.weight_decay} onChange={v => updateConfig('weight_decay', v)} step={0.0001} />
                          <Toggle label="Cosine LR Scheduler" checked={config.cos_lr} onChange={v => updateConfig('cos_lr', v)} />
                        </div>

                        <div className="space-y-4">
                          <h4 className="text-sm font-semibold text-white border-b border-slate-800 pb-2">Warmup</h4>
                          <NumberInput label="Warmup Epochs" value={config.warmup_epochs} onChange={v => updateConfig('warmup_epochs', v)} />
                          <NumberInput label="Warmup Momentum" value={config.warmup_momentum} onChange={v => updateConfig('warmup_momentum', v)} step={0.01} />
                          <NumberInput label="Warmup Bias LR" value={config.warmup_bias_lr} onChange={v => updateConfig('warmup_bias_lr', v)} step={0.01} />
                          <NumberInput label="Nominal Batch Size (nbs)" value={Number(config.nbs) || 64} onChange={v => updateConfig('nbs', v)} />
                        </div>
                      </div>
                    )}

                    {/* LOSS TAB */}
                    {activeTab === 'loss' && (
                      <div className="grid grid-cols-2 gap-x-8 max-w-4xl">
                        <div className="space-y-4">
                          <h4 className="text-sm font-semibold text-white border-b border-slate-800 pb-2">Detection Loss</h4>
                          <NumberInput label="Box Loss Gain" value={config.box} onChange={v => updateConfig('box', v)} step={0.1} />
                          <NumberInput label="Cls Loss Gain" value={config.cls} onChange={v => updateConfig('cls', v)} step={0.1} />
                          <NumberInput label="DFL Loss Gain" value={config.dfl} onChange={v => updateConfig('dfl', v)} step={0.1} />
                        </div>

                        <div className="space-y-4">
                          <h4 className="text-sm font-semibold text-white border-b border-slate-800 pb-2">Pose / Seg / Legacy</h4>
                          <NumberInput label="Pose Gain" value={Number(config.pose) || 12.0} onChange={v => updateConfig('pose', v)} step={0.1} />
                          <NumberInput label="Keypoint Obj Gain (kobj)" value={config.kobj} onChange={v => updateConfig('kobj', v)} step={0.1} />
                          <NumberInput label="Mask Ratio" value={config.mask_ratio} onChange={v => updateConfig('mask_ratio', v)} />
                          <Toggle label="Overlap Mask" checked={config.overlap_mask} onChange={v => updateConfig('overlap_mask', v)} />
                        </div>
                      </div>
                    )}

                    {/* AUGMENTATION TAB */}
                    {activeTab === 'augmentation' && (
                      <div className="grid grid-cols-2 gap-x-8 gap-y-6 max-w-4xl">
                        <div className="space-y-4">
                          <h4 className="text-sm font-semibold text-white border-b border-slate-800 pb-2">Color & Intensity</h4>
                          <SliderInput label="HSV Hue" value={config.hsv_h} onChange={v => updateConfig('hsv_h', v)} min={0} max={1} step={0.001} />
                          <SliderInput label="HSV Saturation" value={config.hsv_s} onChange={v => updateConfig('hsv_s', v)} min={0} max={1} step={0.01} />
                          <SliderInput label="HSV Value" value={config.hsv_v} onChange={v => updateConfig('hsv_v', v)} min={0} max={1} step={0.01} />
                        </div>

                        <div className="space-y-4">
                          <h4 className="text-sm font-semibold text-white border-b border-slate-800 pb-2">Geometric</h4>
                          <SliderInput label="Degrees (Rotation)" value={config.degrees} onChange={v => updateConfig('degrees', v)} min={-180} max={180} />
                          <SliderInput label="Translate" value={config.translate} onChange={v => updateConfig('translate', v)} min={0} max={1} />
                          <SliderInput label="Scale" value={config.scale} onChange={v => updateConfig('scale', v)} min={0} max={1} />
                          <SliderInput label="Shear" value={config.shear} onChange={v => updateConfig('shear', v)} min={0} max={180} />
                          <SliderInput label="Perspective" value={config.perspective} onChange={v => updateConfig('perspective', v)} min={0} max={0.001} step={0.0001} />
                          <SliderInput label="Flip Up-Down" value={config.flipud} onChange={v => updateConfig('flipud', v)} min={0} max={1} />
                          <SliderInput label="Flip Left-Right" value={config.fliplr} onChange={v => updateConfig('fliplr', v)} min={0} max={1} />
                        </div>

                        <div className="space-y-4 col-span-2">
                           <h4 className="text-sm font-semibold text-white border-b border-slate-800 pb-2">Composition</h4>
                           <div className="grid grid-cols-2 gap-8">
                              <div className="space-y-4">
                                <SliderInput label="Mosaic" value={config.mosaic} onChange={v => updateConfig('mosaic', v)} min={0} max={1} />
                                <SliderInput label="Mixup" value={config.mixup} onChange={v => updateConfig('mixup', v)} min={0} max={1} />
                                <SliderInput label="Copy Paste" value={config.copy_paste} onChange={v => updateConfig('copy_paste', v)} min={0} max={1} />
                                <SliderInput label="Erasing" value={config.erasing} onChange={v => updateConfig('erasing', v)} min={0} max={1} />
                                <SliderInput label="Crop Fraction" value={config.crop_fraction} onChange={v => updateConfig('crop_fraction', v)} min={0} max={1} />
                              </div>
                              <div className="space-y-4">
                                <SliderInput label="BGR" value={config.bgr} onChange={v => updateConfig('bgr', v)} min={0} max={1} />
                                <NumberInput label="Close Mosaic (last N epochs)" value={config.close_mosaic} onChange={v => updateConfig('close_mosaic', v)} />
                                <div className="pt-2">
                                  <label className="block text-xs font-medium text-slate-400 mb-1">Auto Augment</label>
                                  <select
                                    value={String(config.auto_augment || '')}
                                    onChange={e => updateConfig('auto_augment', e.target.value)}
                                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none"
                                  >
                                    <option value="">None</option>
                                    <option value="randaugment">RandAugment</option>
                                    <option value="autoaugment">AutoAugment</option>
                                    <option value="augmix">AugMix</option>
                                  </select>
                                </div>
                              </div>
                           </div>
                        </div>
                      </div>
                    )}

                    {/* VALIDATION TAB */}
                    {activeTab === 'validation' && (
                      <div className="grid grid-cols-2 gap-x-8 max-w-4xl">
                        <div className="space-y-4">
                          <h4 className="text-sm font-semibold text-white border-b border-slate-800 pb-2">Inference / NMS</h4>
                          <div>
                            <label className="block text-xs font-medium text-slate-400 mb-1">Confidence Threshold</label>
                            <input
                              type="number"
                              value={config.conf ?? ''}
                              placeholder="Auto"
                              onChange={e => updateConfig('conf', e.target.value === '' ? null : parseFloat(e.target.value))}
                              step={0.01} min={0} max={1}
                              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none"
                            />
                          </div>
                          <NumberInput label="IoU Threshold (NMS)" value={Number(config.iou) || 0.7} onChange={v => updateConfig('iou', v)} step={0.01} min={0} max={1} />
                          <NumberInput label="Max Detections" value={Number(config.max_det) || 300} onChange={v => updateConfig('max_det', v)} />
                          <div className="pt-2 space-y-2">
                            <Toggle label="Agnostic NMS" checked={Boolean(config.agnostic_nms)} onChange={v => updateConfig('agnostic_nms', v)} />
                          </div>
                        </div>

                        <div className="space-y-4">
                          <h4 className="text-sm font-semibold text-white border-b border-slate-800 pb-2">Training Mode</h4>
                          <div className="space-y-2">
                            <Toggle label="Rectangular Training (rect)" checked={Boolean(config.rect)} onChange={v => updateConfig('rect', v)} />
                            <Toggle label="Single Class Mode" checked={Boolean(config.single_cls)} onChange={v => updateConfig('single_cls', v)} />
                          </div>
                          <p className="text-xs text-slate-500 mt-2">rect: uses variable stride per batch for efficiency. single_cls: treats all classes as one.</p>
                        </div>
                      </div>
                    )}
                    
                  </div>
                </>
              ) : (
                <div className="flex-1 flex flex-col items-center justify-center text-slate-500">
                  <div className="w-16 h-16 rounded-2xl bg-slate-800/50 border border-slate-700 flex items-center justify-center mb-4">
                     <Settings size={32} className="text-slate-600" />
                  </div>
                  <h3 className="text-lg font-medium text-slate-300">No Model Selected</h3>
                  <p className="text-sm max-w-xs text-center mt-2">Select a model from the sidebar to configure and start a training job.</p>
                </div>
              )}
            </div>

          </div>
        )}
      </div>
    </div>
  );
}

// ─── Reusable Inputs ─────────────────────────────────────────────────────────

function NumberInput({ label, value, onChange, step = 1, min, max }: { label: string; value: number; onChange: (v: number) => void; step?: number; min?: number; max?: number }) {
  return (
    <div>
      <label className="block text-xs font-medium text-slate-400 mb-1">{label}</label>
      <input 
        type="number" 
        value={value} 
        onChange={e => onChange(parseFloat(e.target.value))}
        step={step}
        min={min}
        max={max}
        className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none"
      />
    </div>
  );
}

function TextInput({ label, value, onChange, placeholder }: { label: string; value: string; onChange: (v: string) => void; placeholder?: string }) {
  return (
    <div>
      <label className="block text-xs font-medium text-slate-400 mb-1">{label}</label>
      <input 
        type="text" 
        value={value} 
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none"
      />
    </div>
  );
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-center gap-2 cursor-pointer group">
      <div className={`w-9 h-5 rounded-full relative transition-colors ${checked ? 'bg-emerald-600' : 'bg-slate-700 group-hover:bg-slate-600'}`}>
        <div className={`absolute top-1 w-3 h-3 rounded-full bg-white transition-transform ${checked ? 'left-5' : 'left-1'}`} />
      </div>
      <span className="text-sm text-slate-300 group-hover:text-white transition-colors">{label}</span>
      <input type="checkbox" className="hidden" checked={checked} onChange={e => onChange(e.target.checked)} />
    </label>
  );
}

function SliderInput({ label, value, onChange, min, max, step = 0.01 }: { label: string; value: number; onChange: (v: number) => void; min: number; max: number; step?: number }) {
  return (
    <div>
      <div className="flex justify-between mb-1">
        <label className="text-xs font-medium text-slate-400">{label}</label>
        <span className="text-xs font-mono text-emerald-400">{value.toFixed(step < 0.1 ? 3 : 1)}</span>
      </div>
      <input 
        type="range" 
        value={value} 
        onChange={e => onChange(parseFloat(e.target.value))}
        min={min}
        max={max}
        step={step}
        className="w-full accent-emerald-500 h-1.5 bg-slate-700 rounded-full appearance-none cursor-pointer"
      />
    </div>
  );
}
