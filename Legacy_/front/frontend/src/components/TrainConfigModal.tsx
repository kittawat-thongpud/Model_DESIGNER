/**
 * TrainConfigModal — Full training configuration with collapsible sections.
 * Shows all 38 PyTorch-native parameters organized by category.
 * Augmentation section only shows for image datasets.
 */
import { useState, useEffect, useMemo } from 'react';

import type { DatasetInfo } from '../types';
import { useDesignerStore } from '../store/designerStore';
import { api } from '../services/api';
import { getTopologicalSort } from '../utils/graphUtils';

// ── Default config (matches backend JobConfig defaults) ──────────────────────
const DEFAULT_CONFIG = {
  // Basic Training
  dataset: 'mnist',
  epochs: 5,
  batch_size: 64,
  imgsz: 0,
  device: 'auto',
  workers: 2,
  patience: 0,
  val: true,
  seed: 0,
  deterministic: false,

  // Optimizer
  optimizer: 'Adam' as 'Adam' | 'AdamW' | 'SGD',
  lr0: 0.001,
  lrf: 0.01,
  momentum: 0.9,
  weight_decay: 0.0005,
  warmup_epochs: 0,
  warmup_momentum: 0.8,
  warmup_bias_lr: 0.1,
  cos_lr: false,

  // Model
  pretrained: '',
  freeze: 0,
  amp: false,

  // Augmentation
  hsv_h: 0.015,
  hsv_s: 0.7,
  hsv_v: 0.4,
  degrees: 0.0,
  translate: 0.1,
  scale: 0.5,
  shear: 0.0,
  flipud: 0.0,
  fliplr: 0.5,
  erasing: 0.0,
  auto_augment: '',
  crop_fraction: 1.0,

  // Loss
  cls_weight: 1.0,

  // Weight Recording
  weight_record_enabled: false,
  weight_record_frequency: 5,
  weight_record_layers: [] as string[],

  // System
  save_period: 0,
  nbs: 64,
};

// Image datasets that support augmentation
const IMAGE_DATASETS = ['mnist', 'cifar10', 'fashion_mnist'];

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onStart: (config: Record<string, unknown>) => void;
}

export default function TrainConfigModal({ isOpen, onClose, onStart }: Props) {
  const globalVars = useDesignerStore((state) => state.globalVars);
  const nodes = useDesignerStore((state) => state.nodes);
  const edges = useDesignerStore((state) => state.edges);
  const [config, setConfig] = useState({ ...DEFAULT_CONFIG });
  const [globalOverrides, setGlobalOverrides] = useState<Record<string, any>>({});
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  
  // Weights selection logic
  // Now we typically allow all layers (except Input/Group/Notes) to be recorded (outputs/weights)
  const weightableLayers = useMemo(() => {
    // Determine topological order
    const sortedIds = getTopologicalSort(nodes as any[], edges);

    // Filter relevant nodes
    const relevantNodes = nodes.filter(n => n.type === 'layerNode' && n.data.layerType !== 'Input');

    // Create a map for sorting
    const orderMap = new Map(sortedIds.map((id, index) => [id, index]));

    // Sort relevant nodes by their topological index
    relevantNodes.sort((a, b) => {
      const idxA = orderMap.has(a.id) ? orderMap.get(a.id)! : Infinity;
      const idxB = orderMap.has(b.id) ? orderMap.get(b.id)! : Infinity;
      return idxA - idxB;
    });

    return relevantNodes.map(n => ({ id: n.id, label: n.data.label || n.id }));
  }, [nodes, edges]);

  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    globals: true,
    basic: true,
    optimizer: false,
    model: false,
    augmentation: false,
    loss: false,
    weights: false,
    system: false,
  });

  // Initialize overrides with current values
  useEffect(() => {
    if (isOpen) {
      const initial: Record<string, any> = {};
      globalVars.forEach(g => {
        initial[g.name] = g.value;
      });
      setGlobalOverrides(initial);
    }
  }, [isOpen, globalVars]);

  // Fetch datasets on open
  useEffect(() => {
    if (isOpen) {
      api.listDatasets().then(setDatasets).catch(() => {});
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const isImageDataset = IMAGE_DATASETS.includes(config.dataset.toLowerCase());
  const currentDataset = datasets.find(d => d.name === config.dataset);

  const toggleSection = (section: string) => {
    setExpandedSections((prev) => ({ ...prev, [section]: !prev[section] }));
  };

  const set = <K extends keyof typeof config>(key: K, value: (typeof config)[K]) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  const handleStart = () => {
    // Strip defaults to keep payload clean (backend has same defaults)
    const payload: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(config)) {
      if (value !== (DEFAULT_CONFIG as any)[key]) {
        payload[key] = value;
      }
    }
    // Always send required fields
    payload.dataset = config.dataset;
    payload.epochs = config.epochs;
    payload.batch_size = config.batch_size;
    payload.lr0 = config.lr0;
    
    // Add global overrides
    payload.global_overrides = globalOverrides;

    // Weight recording layers are already in config.weight_record_layers
    payload.weight_record_layers = config.weight_record_layers;
    
    onStart(payload);
  };

  // ── Helpers ──

  const NumberField = ({ label, value, onChange, min, max, step, help }: {
    label: string; value: number; onChange: (v: number) => void;
    min?: number; max?: number; step?: number; help?: string;
  }) => (
    <div className="tcm-field">
      <label className="tcm-label">
        {label}
        {help && <span className="tcm-help" title={help}>ⓘ</span>}
      </label>
      <input type="number" className="tcm-input" value={value}
        min={min} max={max} step={step}
        onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  );

  const ToggleField = ({ label, value, onChange, help }: {
    label: string; value: boolean; onChange: (v: boolean) => void; help?: string;
  }) => (
    <div className="tcm-field tcm-field-toggle">
      <label className="tcm-label">
        {label}
        {help && <span className="tcm-help" title={help}>ⓘ</span>}
      </label>
      <button className={`tcm-toggle ${value ? 'on' : 'off'}`}
        onClick={() => onChange(!value)}>
        {value ? 'ON' : 'OFF'}
      </button>
    </div>
  );

  const SelectField = ({ label, value, options, onChange, help }: {
    label: string; value: string; options: { value: string; label: string }[];
    onChange: (v: string) => void; help?: string;
  }) => (
    <div className="tcm-field">
      <label className="tcm-label">
        {label}
        {help && <span className="tcm-help" title={help}>ⓘ</span>}
      </label>
      <select className="tcm-select" value={value}
        onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  );

  const SectionHeader = ({ id, icon, title, count }: {
    id: string; icon: string; title: string; count: number;
  }) => (
    <button className={`tcm-section-header ${expandedSections[id] ? 'expanded' : ''}`}
      onClick={() => toggleSection(id)}>
      <span className="tcm-section-icon">{icon}</span>
      <span className="tcm-section-title">{title}</span>
      <span className="tcm-section-count">{count}</span>
      <span className="tcm-section-chevron">{expandedSections[id] ? '▾' : '▸'}</span>
    </button>
  );

  // ── Not implementable items (shown greyed out with reason) ──
  const NotAvailable = ({ label, reason }: { label: string; reason: string }) => (
    <div className="tcm-field tcm-disabled" title={reason}>
      <label className="tcm-label">{label} <span className="tcm-na-badge">N/A</span></label>
      <span className="tcm-na-reason">{reason}</span>
    </div>
  );

  return (
    <div className="tcm-overlay" onClick={onClose}>
      <div className="tcm-modal" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="tcm-header">
          <h2>🚀 Training Configuration</h2>
          <button className="tcm-close" onClick={onClose}>✕</button>
        </div>

        {/* Body */}
        <div className="tcm-body">

          {/* ─── 0. Global Variables ──────────────────────────────────── */}
          {globalVars.length > 0 && (
            <>
              <SectionHeader id="globals" icon="🌐" title="Global Variables" count={globalVars.length} />
              {expandedSections.globals && (
                <div className="tcm-section">
                  {globalVars.map((gvar) => (
                    <div key={gvar.id} className="tcm-field">
                      <label className="tcm-label">
                        ${gvar.name}
                        <span className="tcm-badge-type">{gvar.type}</span>
                      </label>
                      
                      {/* Boolean Toggle */}
                      {gvar.type === 'bool' && (
                        <button 
                          className={`tcm-toggle ${globalOverrides[gvar.name] ? 'on' : 'off'}`}
                          onClick={() => setGlobalOverrides(prev => ({ ...prev, [gvar.name]: !prev[gvar.name] }))}
                        >
                          {globalOverrides[gvar.name] ? 'TRUE' : 'FALSE'}
                        </button>
                      )}

                      {/* Number Input */}
                      {(gvar.type === 'int' || gvar.type === 'float') && (
                        <input 
                          type="number" 
                          className="tcm-input" 
                          value={globalOverrides[gvar.name] ?? ''}
                          step={gvar.type === 'float' ? 0.0001 : 1}
                          onChange={(e) => setGlobalOverrides(prev => ({ 
                            ...prev, 
                            [gvar.name]: gvar.type === 'int' ? parseInt(e.target.value) : parseFloat(e.target.value) 
                          }))}
                        />
                      )}

                      {/* String/Selector Input */}
                      {(gvar.type === 'str' || gvar.type === 'selector') && (
                         gvar.options && gvar.options.length > 0 ? (
                           <select 
                             className="tcm-select"
                             value={globalOverrides[gvar.name] ?? ''}
                             onChange={(e) => setGlobalOverrides(prev => ({ ...prev, [gvar.name]: e.target.value }))}
                           >
                             {gvar.options.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                           </select>
                         ) : (
                           <input 
                             type="text" 
                             className="tcm-input" 
                             value={globalOverrides[gvar.name] ?? ''}
                             onChange={(e) => setGlobalOverrides(prev => ({ ...prev, [gvar.name]: e.target.value }))}
                           />
                         )
                      )}
                    </div>
                  ))}
                </div>
              )}
            </>
          )}

          {/* ─── 1. Basic Training ────────────────────────────────────── */}
          <SectionHeader id="basic" icon="⚡" title="Basic Training" count={9} />
          {expandedSections.basic && (
            <div className="tcm-section">
              <SelectField label="Dataset" value={config.dataset}
                options={datasets.map((d) => ({ value: d.name, label: d.display_name }))}
                onChange={(v) => set('dataset', v)}
                help="Training dataset" />
              
              {currentDataset && (
                <div className="tcm-field tcm-dataset-details">
                   <div style={{ marginLeft: '120px', display: 'flex', gap: '8px', fontSize: '12px' }}>
                      <span className={`tcm-badge-task ${currentDataset.task_type || 'classification'}`}>
                        {(currentDataset.task_type || 'classification').toUpperCase()}
                      </span>
                      <span className="tcm-badge-classes" style={{ color: '#666', padding: '2px 0' }}>
                         {currentDataset.num_classes} Classes
                      </span>
                   </div>
                </div>
              )}

              <NumberField label="Epochs" value={config.epochs}
                onChange={(v) => set('epochs', v)} min={1} max={1000}
                help="Total training iterations over the dataset" />

              <NumberField label="Batch Size" value={config.batch_size}
                onChange={(v) => set('batch_size', v)} min={1} max={1024}
                help="Samples per batch (DataLoader batch_size)" />

              <NumberField label="Image Size" value={config.imgsz}
                onChange={(v) => set('imgsz', v)} min={0} max={2048}
                help="Resize images to this size. 0 = use native size" />

              <SelectField label="Device" value={config.device}
                options={[
                  { value: 'auto', label: 'Auto (GPU if available)' },
                  { value: 'cpu', label: 'CPU' },
                  { value: 'cuda', label: 'GPU (CUDA)' },
                ]}
                onChange={(v) => set('device', v)}
                help="model.to(device)" />

              <NumberField label="Workers" value={config.workers}
                onChange={(v) => set('workers', v)} min={0} max={16}
                help="DataLoader num_workers for data loading" />

              <NumberField label="Patience" value={config.patience}
                onChange={(v) => set('patience', v)} min={0} max={100}
                help="Early stopping: stop if val_loss doesn't improve for N epochs. 0 = disabled" />

              <ToggleField label="Validation" value={config.val}
                onChange={(v) => set('val', v)}
                help="Run validation after each epoch" />

              <NumberField label="Seed" value={config.seed}
                onChange={(v) => set('seed', v)} min={0}
                help="Random seed. 0 = random. torch.manual_seed()" />

              <ToggleField label="Deterministic" value={config.deterministic}
                onChange={(v) => set('deterministic', v)}
                help="torch.use_deterministic_algorithms(True)" />
            </div>
          )}

          {/* ─── 2. Optimizer ─────────────────────────────────────────── */}
          <SectionHeader id="optimizer" icon="🔧" title="Optimizer" count={9} />
          {expandedSections.optimizer && (
            <div className="tcm-section">
              <SelectField label="Optimizer" value={config.optimizer}
                options={[
                  { value: 'Adam', label: 'Adam' },
                  { value: 'AdamW', label: 'AdamW' },
                  { value: 'SGD', label: 'SGD' },
                ]}
                onChange={(v) => set('optimizer', v as any)}
                help="torch.optim optimizer class" />

              <NumberField label="Learning Rate (lr0)" value={config.lr0}
                onChange={(v) => set('lr0', v)} min={0.000001} max={1} step={0.0001}
                help="Initial learning rate" />

              <NumberField label="LR Final Factor (lrf)" value={config.lrf}
                onChange={(v) => set('lrf', v)} min={0.001} max={1} step={0.01}
                help="Final LR = lr0 × lrf (used in scheduler)" />

              <NumberField label="Momentum" value={config.momentum}
                onChange={(v) => set('momentum', v)} min={0} max={1} step={0.01}
                help="SGD momentum / Adam beta1" />

              <NumberField label="Weight Decay" value={config.weight_decay}
                onChange={(v) => set('weight_decay', v)} min={0} max={0.1} step={0.0001}
                help="L2 regularization" />

              <NumberField label="Warmup Epochs" value={config.warmup_epochs}
                onChange={(v) => set('warmup_epochs', v)} min={0} max={20}
                help="Linear warmup for N epochs" />

              <NumberField label="Warmup Momentum" value={config.warmup_momentum}
                onChange={(v) => set('warmup_momentum', v)} min={0} max={1} step={0.01}
                help="Starting momentum during warmup" />

              <NumberField label="Warmup Bias LR" value={config.warmup_bias_lr}
                onChange={(v) => set('warmup_bias_lr', v)} min={0} max={1} step={0.01}
                help="Starting LR for bias params during warmup" />

              <ToggleField label="Cosine LR Scheduler" value={config.cos_lr}
                onChange={(v) => set('cos_lr', v)}
                help="CosineAnnealingLR vs StepLR" />
            </div>
          )}

          {/* ─── 3. Model Structure ───────────────────────────────────── */}
          <SectionHeader id="model" icon="🏗️" title="Model Structure" count={5} />
          {expandedSections.model && (
            <div className="tcm-section">
              <div className="tcm-field">
                <label className="tcm-label">
                  Model
                  <span className="tcm-help" title="Loaded from canvas graph">ⓘ</span>
                </label>
                <span className="tcm-readonly">From canvas graph</span>
              </div>

              <div className="tcm-field">
                <label className="tcm-label">
                  Pretrained Weights
                  <span className="tcm-help" title="Load saved weights as starting point">ⓘ</span>
                </label>
                <input type="text" className="tcm-input" value={config.pretrained}
                  onChange={(e) => set('pretrained', e.target.value)}
                  placeholder="Weight ID (leave empty for random init)" />
              </div>

              <NumberField label="Freeze Layers" value={config.freeze}
                onChange={(v) => set('freeze', v)} min={0} max={50}
                help="Freeze first N layers (requires_grad=False)" />

              <div className="tcm-field">
                <label className="tcm-label">
                  Dropout
                  <span className="tcm-help" title="Already configured in model via canvas Dropout nodes">ⓘ</span>
                </label>
                <span className="tcm-readonly">Configured via canvas nodes</span>
              </div>

              <ToggleField label="Mixed Precision (AMP)" value={config.amp}
                onChange={(v) => set('amp', v)}
                help="torch.cuda.amp — FP16 training for faster GPU training" />
            </div>
          )}

          {/* ─── 4. Augmentation ──────────────────────────────────────── */}
          <SectionHeader id="augmentation" icon="🎨"
            title={`Augmentation ${!isImageDataset ? '(image only)' : ''}`} count={16} />
          {expandedSections.augmentation && (
            <div className={`tcm-section ${!isImageDataset ? 'tcm-section-disabled' : ''}`}>
              {!isImageDataset && (
                <div className="tcm-section-notice">
                  ⚠️ Augmentation is only available for image datasets (MNIST, CIFAR-10, Fashion-MNIST)
                </div>
              )}

              <h4 className="tcm-subheading">Color</h4>
              <NumberField label="Hue (hsv_h)" value={config.hsv_h}
                onChange={(v) => set('hsv_h', v)} min={0} max={0.5} step={0.005}
                help="ColorJitter hue augmentation" />
              <NumberField label="Saturation (hsv_s)" value={config.hsv_s}
                onChange={(v) => set('hsv_s', v)} min={0} max={1} step={0.1}
                help="ColorJitter saturation" />
              <NumberField label="Brightness (hsv_v)" value={config.hsv_v}
                onChange={(v) => set('hsv_v', v)} min={0} max={1} step={0.1}
                help="ColorJitter brightness" />

              <h4 className="tcm-subheading">Geometric</h4>
              <NumberField label="Rotation (degrees)" value={config.degrees}
                onChange={(v) => set('degrees', v)} min={0} max={180}
                help="RandomRotation ±degrees" />
              <NumberField label="Translate" value={config.translate}
                onChange={(v) => set('translate', v)} min={0} max={0.9} step={0.05}
                help="RandomAffine translate fraction" />
              <NumberField label="Scale" value={config.scale}
                onChange={(v) => set('scale', v)} min={0} max={0.9} step={0.05}
                help="RandomAffine scale ±fraction" />
              <NumberField label="Shear" value={config.shear}
                onChange={(v) => set('shear', v)} min={0} max={90} step={1}
                help="RandomAffine shear ±degrees" />
              <NumberField label="Crop Fraction" value={config.crop_fraction}
                onChange={(v) => set('crop_fraction', v)} min={0.1} max={1} step={0.05}
                help="RandomResizedCrop scale. 1.0 = no crop" />

              <h4 className="tcm-subheading">Flip</h4>
              <NumberField label="Horizontal Flip (fliplr)" value={config.fliplr}
                onChange={(v) => set('fliplr', v)} min={0} max={1} step={0.1}
                help="RandomHorizontalFlip probability" />
              <NumberField label="Vertical Flip (flipud)" value={config.flipud}
                onChange={(v) => set('flipud', v)} min={0} max={1} step={0.1}
                help="RandomVerticalFlip probability" />

              <h4 className="tcm-subheading">Advanced</h4>
              <NumberField label="Random Erasing" value={config.erasing}
                onChange={(v) => set('erasing', v)} min={0} max={1} step={0.1}
                help="RandomErasing probability" />

              <SelectField label="Auto Augment" value={config.auto_augment}
                options={[
                  { value: '', label: 'None' },
                  { value: 'randaugment', label: 'RandAugment' },
                  { value: 'autoaugment', label: 'AutoAugment' },
                  { value: 'trivialaugmentwide', label: 'TrivialAugmentWide' },
                ]}
                onChange={(v) => set('auto_augment', v)}
                help="torchvision auto augmentation policy" />

              <h4 className="tcm-subheading">Not Available (YOLO-specific)</h4>
              <NotAvailable label="Mosaic" reason="Requires custom collate + bbox handling" />
              <NotAvailable label="MixUp" reason="Batch-level mixing + label interpolation" />
              <NotAvailable label="CutMix" reason="Custom batch transform + label mixing" />
              <NotAvailable label="Copy-Paste" reason="Object-level: needs segmentation masks" />
            </div>
          )}

          {/* ─── 5. Loss ──────────────────────────────────────────────── */}
          <SectionHeader id="loss" icon="📉" title="Loss Function" count={3} />
          {expandedSections.loss && (
            <div className="tcm-section">
              <NumberField label="Classification Weight (cls)" value={config.cls_weight}
                onChange={(v) => set('cls_weight', v)} min={0.1} max={10} step={0.1}
                help="CrossEntropyLoss weight multiplier" />

              <NotAvailable label="Box Loss (box)" reason="Detection-specific — CIoU/DIoU loss for bounding boxes" />
              <NotAvailable label="DFL Loss (dfl)" reason="Distribution Focal Loss — YOLO detection-specific" />
            </div>
          )}

          {/* ─── 6. Weight Recording ──────────────────────────────────── */}
          <SectionHeader id="weights" icon="🔬" title="Weight Recording" count={3} />
          {expandedSections.weights && (
            <div className="tcm-section">
              <ToggleField label="Enable Recording" value={config.weight_record_enabled}
                onChange={(v) => set('weight_record_enabled', v)}
                help="Record weight distributions during training for playback" />

              {config.weight_record_enabled && (
                <>
                  <NumberField label="Frequency (Epochs)" value={config.weight_record_frequency}
                    onChange={(v) => set('weight_record_frequency', v)} min={1} max={50}
                    help="Record every N epochs" />
                  
                  <div className="tcm-field">
                    <div className="tcm-label-row" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <label className="tcm-label" style={{ marginBottom: 0 }}>
                        Layers to Record
                        <span className="tcm-help" title="Select specific layers to record weights/activations for. If none selected, all are recorded by default in some modes, but explicit selection is better.">ⓘ</span>
                      </label>
                      {weightableLayers.length > 0 && (
                        <label className="tcm-checkbox-item" style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                          <input
                            type="checkbox"
                            checked={weightableLayers.every(l => config.weight_record_layers.includes(l.id))}
                            onChange={(e) => {
                              if (e.target.checked) {
                                // Select All
                                setConfig(prev => ({
                                  ...prev,
                                  weight_record_layers: weightableLayers.map(l => l.id)
                                }));
                              } else {
                                // Deselect All
                                setConfig(prev => ({
                                  ...prev,
                                  weight_record_layers: []
                                }));
                              }
                            }}
                          />
                          Select All
                        </label>
                      )}
                    </div>
                    <div className="tcm-layer-checkbox-grid">
                      {weightableLayers.length === 0 ? (
                        <span className="tcm-no-layers">No weightable layers found in model</span>
                      ) : (
                        weightableLayers.map(layer => (
                          <label key={layer.id} className="tcm-checkbox-item">
                            <input 
                              type="checkbox" 
                              checked={config.weight_record_layers.includes(layer.id)}
                              onChange={(e) => {
                                const current = config.weight_record_layers;
                                if (e.target.checked) {
                                  setConfig(prev => ({ ...prev, weight_record_layers: [...current, layer.id] }));
                                } else {
                                  setConfig(prev => ({ ...prev, weight_record_layers: current.filter(id => id !== layer.id) }));
                                }
                              }}
                            />
                            <span className="tcm-checkbox-label">
                              {layer.label} ({layer.id})
                            </span>
                          </label>
                        ))
                      )}
                    </div>
                  </div>
                </>
              )}
            </div>
          )}

          {/* ─── 7. System ────────────────────────────────────────────── */}
          <SectionHeader id="system" icon="⚙️" title="System" count={10} />
          {expandedSections.system && (
            <div className="tcm-section">
              <NumberField label="Save Period" value={config.save_period}
                onChange={(v) => set('save_period', v)} min={0} max={100}
                help="Save checkpoint every N epochs. 0 = only at end" />

              <div className="tcm-field">
                <label className="tcm-label">Save Final Weights</label>
                <span className="tcm-readonly">Always ON (torch.save)</span>
              </div>

              <NotAvailable label="Cache" reason="Custom Dataset class — torchvision already caches after download" />
              <NotAvailable label="Rect Training" reason="YOLO-specific custom batch collation" />
              <NotAvailable label="Multi-Scale" reason="Requires modifying tensor size mid-batch" />
              <NotAvailable label="Close Mosaic" reason="Depends on Mosaic (not implemented)" />
              <NotAvailable label="Overlap Mask" reason="Segmentation-specific" />
              <NotAvailable label="Mask Ratio" reason="Segmentation-specific" />
              <NotAvailable label="Single Class" reason="Label remapping — niche use case" />
              <NumberField label="Nominal Batch Size (NBS)" value={config.nbs}
                onChange={(v) => set('nbs', v)} min={1} max={1024}
                help="Gradient accumulation: effective batch = nbs. Loss scaled by nbs/batch_size steps" />
            </div>
          )}

        </div>

        {/* Footer */}
        <div className="tcm-footer">
          <button className="btn btn-ghost" onClick={onClose}>Cancel</button>
          <button className="btn btn-accent" onClick={handleStart}>
            🚀 Start Training
          </button>
        </div>
      </div>
    </div>
  );
}
