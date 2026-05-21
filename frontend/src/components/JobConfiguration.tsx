/**
 * JobConfiguration — dynamic training config display.
 *
 * Fields are grouped into Training / Optimizer / Augmentation / Inference
 * via FIELD_GROUP_MAP. A conditional "Model Config" column surfaces
 * arch-specific fields (training_mode, cs2ga_lr_*) from three sources:
 *   1. Direct config keys
 *   2. Underscore-prefixed copies (_training_mode, _cs2ga_lr_*)
 *   3. Fallback from config._training_profile.lr_group_overrides
 *
 * Any config key not in FIELD_GROUP_MAP or SKIP_FIELDS is appended to
 * the Training column so new fields are never silently dropped.
 */
import React, { useState } from 'react';
import {
  Settings, ChevronDown, ChevronUp,
  Cpu, Sliders, Image as ImageIcon, Server, Layers,
} from 'lucide-react';
import type { TrainConfig } from '../types';

// ── Field classification ─────────────────────────────────────────────────────

type GroupName = 'Training' | 'Optimizer' | 'Augmentation' | 'Inference';

const FIELD_GROUP_MAP: Record<string, GroupName> = {
  // Training
  epochs: 'Training', patience: 'Training', batch: 'Training',
  imgsz: 'Training', workers: 'Training', cache: 'Training',
  device: 'Training', seed: 'Training', deterministic: 'Training',
  amp: 'Training', ema: 'Training', pin_memory: 'Training',
  close_mosaic: 'Training', freeze: 'Training', resume: 'Training',
  save_period: 'Training', val: 'Training', plots: 'Training',
  rect: 'Training', single_cls: 'Training', overlap_mask: 'Training',
  mask_ratio: 'Training', kobj: 'Training', sample_per_class: 'Training',
  pretrained: 'Training', use_yolo_pretrained: 'Training',
  // Optimizer & Loss
  optimizer: 'Optimizer', lr0: 'Optimizer', lrf: 'Optimizer',
  momentum: 'Optimizer', weight_decay: 'Optimizer',
  warmup_epochs: 'Optimizer', warmup_momentum: 'Optimizer',
  warmup_bias_lr: 'Optimizer', cos_lr: 'Optimizer', nbs: 'Optimizer',
  box: 'Optimizer', cls: 'Optimizer', dfl: 'Optimizer', pose: 'Optimizer',
  // Augmentation
  mosaic: 'Augmentation', mixup: 'Augmentation', copy_paste: 'Augmentation',
  erasing: 'Augmentation', crop_fraction: 'Augmentation',
  fliplr: 'Augmentation', flipud: 'Augmentation',
  degrees: 'Augmentation', translate: 'Augmentation', scale: 'Augmentation',
  shear: 'Augmentation', perspective: 'Augmentation',
  hsv_h: 'Augmentation', hsv_s: 'Augmentation', hsv_v: 'Augmentation',
  bgr: 'Augmentation', auto_augment: 'Augmentation',
  // Inference
  conf: 'Inference', iou: 'Inference', max_det: 'Inference',
  agnostic_nms: 'Inference',
};

/** Preferred display order within each group (unlisted extras appended at end) */
const GROUP_FIELD_ORDER: Record<GroupName, string[]> = {
  Training: [
    'epochs', 'patience', 'batch', 'imgsz', 'workers', 'cache', 'device',
    'amp', 'ema', 'pin_memory', 'seed', 'deterministic',
    'close_mosaic', 'freeze', 'resume', 'save_period',
    'val', 'plots', 'rect', 'single_cls',
    'overlap_mask', 'mask_ratio', 'kobj', 'sample_per_class',
    'pretrained', 'use_yolo_pretrained',
  ],
  Optimizer: [
    'optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay',
    'warmup_epochs', 'warmup_momentum', 'warmup_bias_lr', 'cos_lr', 'nbs',
    'box', 'cls', 'dfl', 'pose',
  ],
  Augmentation: [
    'mosaic', 'mixup', 'copy_paste', 'erasing', 'crop_fraction',
    'fliplr', 'flipud', 'degrees', 'translate', 'scale',
    'shear', 'perspective', 'hsv_h', 'hsv_s', 'hsv_v', 'bgr', 'auto_augment',
  ],
  Inference: ['conf', 'iou', 'max_det', 'agnostic_nms'],
};

/** Human-readable label overrides (fallback: snake_case → Title Case) */
const LABEL_MAP: Record<string, string> = {
  lr0: 'LR₀', lrf: 'LR Final', cos_lr: 'Cos LR', nbs: 'NBS',
  amp: 'AMP', ema: 'EMA', iou: 'IoU', bgr: 'BGR',
  hsv_h: 'HSV-H', hsv_s: 'HSV-S', hsv_v: 'HSV-V',
  warmup_epochs: 'Warmup Epochs', warmup_momentum: 'Warmup Mom',
  warmup_bias_lr: 'Warmup Bias LR', weight_decay: 'Wt Decay',
  close_mosaic: 'Close Mosaic', save_period: 'Save Period',
  pin_memory: 'Pin Mem', single_cls: 'Single Cls',
  agnostic_nms: 'Agnostic NMS', copy_paste: 'Copy-Paste',
  crop_fraction: 'Crop Frac', auto_augment: 'Auto-Aug',
  max_det: 'Max Det', overlap_mask: 'Overlap Mask',
  mask_ratio: 'Mask Ratio', use_yolo_pretrained: 'YOLO Pretrain',
  sample_per_class: 'Samp/Class',
  // Model (arch) fields
  training_mode:     'Training Mode',
  cs2ga_lr_sparse:   'Projection LR',
  cs2ga_lr_gamma:    'LayerScale LR',
  cs2ga_lr_norm:     'Norm LR',
  cs2ga_lr_backbone: 'Backbone LR',
};

const HIGHLIGHT_FIELDS = new Set([
  'lr0', 'lrf', 'box', 'cls', 'dfl', 'training_mode', 'cs2ga_lr_backbone',
]);

/** Fields to completely skip in regular columns (internal, metadata, or shown elsewhere) */
const SKIP_FIELDS = new Set([
  'data', 'class_names', 'dataset_name', 'enable_deep_metrics', 'nan_retries',
  'model_arch', 'yolo_model', '_training_profile',
  // Ultralytics internal
  'cfg', 'project', 'name', 'exist_ok',
]);

const ARCH_FIELD_KEYS = [
  'training_mode', 'cs2ga_lr_sparse', 'cs2ga_lr_gamma',
  'cs2ga_lr_norm', 'cs2ga_lr_backbone',
] as const;
type ArchFieldKey = typeof ARCH_FIELD_KEYS[number];

const TRAINING_MODE_LABELS: Record<string, string> = {
  full:           'Full Training',
  attention_only: 'Attention Only',
  joint_finetune: 'Joint Fine-Tune',
};

// ── Helpers ──────────────────────────────────────────────────────────────────

function toLabel(key: string): string {
  return LABEL_MAP[key] ?? key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function formatCache(value: unknown): string {
  if (value === true)  return 'Disk';
  if (value === false) return 'None';
  if (typeof value !== 'string') return '-';
  const v = value.trim().toLowerCase();
  return v === 'auto' ? 'Auto' : v === 'ram' ? 'RAM' : v === 'disk' ? 'Disk'
       : (v === 'none' || v === 'off') ? 'None' : value;
}

function fmtValue(key: string, value: unknown): string {
  if (value === undefined || value === null) return 'Auto';
  if (key === 'cache')          return formatCache(value);
  if (key === 'training_mode')  return TRAINING_MODE_LABELS[String(value)] ?? String(value);
  if (key.startsWith('cs2ga_lr_')) return `${Number(value).toFixed(2)} ×`;
  if (key === 'device')         return String(value) || 'Auto';
  if (key === 'pretrained') {
    const s = String(value);
    // weight_id (short hash) or empty
    return s.length > 20 ? s.slice(0, 18) + '…' : (s || 'None');
  }
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'number')  return Number.isInteger(value) ? String(value)
    : parseFloat(value.toPrecision(5)).toString();
  if (Array.isArray(value))       return value.join(', ');
  const s = String(value);
  return s.length > 28 ? s.slice(0, 26) + '…' : s;
}

// ── Types ────────────────────────────────────────────────────────────────────

interface JobConfigurationProps {
  config: TrainConfig;
  datasetName?: string | null;
  partitions?: Array<{
    partition_id: string; train: boolean; val: boolean; test: boolean;
    dataset_name?: string; partition_name?: string;
  }>;
  modelScale?: string;
}

// ── Component ────────────────────────────────────────────────────────────────

const JobConfiguration: React.FC<JobConfigurationProps> = ({
  config, datasetName, modelScale,
}) => {
  const [showConfig, setShowConfig] = useState(false);

  // ── Resolve arch model fields (3-level fallback) ─────────────────────────
  const _profile    = config['_training_profile'] as Record<string, any> | undefined;
  const _lrOverride = _profile?.lr_group_overrides as Record<string, number> | undefined;

  const resolvedArch: Record<ArchFieldKey, unknown> = {
    training_mode:     config['training_mode']     ?? config['_training_mode']     ?? _profile?.name,
    cs2ga_lr_sparse:   config['cs2ga_lr_sparse']   ?? config['_cs2ga_lr_sparse']   ?? _lrOverride?.['sgb_sparse'],
    cs2ga_lr_gamma:    config['cs2ga_lr_gamma']     ?? config['_cs2ga_lr_gamma']    ?? _lrOverride?.['sgb_gamma'],
    cs2ga_lr_norm:     config['cs2ga_lr_norm']      ?? config['_cs2ga_lr_norm']     ?? _lrOverride?.['sgb_norm_group'],
    cs2ga_lr_backbone: config['cs2ga_lr_backbone']  ?? config['_cs2ga_lr_backbone'] ?? _lrOverride?.['base'],
  };
  const archFields   = ARCH_FIELD_KEYS.filter(k => resolvedArch[k] != null);
  const hasArchFields = archFields.length > 0;
  const modelArch    = config['model_arch'] as string | undefined;

  // ── Build dynamic groups from actual config keys ─────────────────────────
  const grouped: Record<GroupName, string[]> = {
    Training: [], Optimizer: [], Augmentation: [], Inference: [],
  };
  const extraKeys: string[] = [];  // keys not in FIELD_GROUP_MAP

  for (const key of Object.keys(config)) {
    if (SKIP_FIELDS.has(key))         continue;  // metadata / internal
    if (key.startsWith('_'))          continue;  // underscore-prefixed internal
    if (key in resolvedArch)          continue;  // arch fields → Model column
    const v = config[key];
    if (v === undefined || v === null || v === '') continue;  // empty

    const group = FIELD_GROUP_MAP[key];
    if (group) {
      grouped[group].push(key);
    } else {
      extraKeys.push(key);  // unknown → appended to Training
    }
  }

  // Sort each group by preferred order; unknown keys within a group go last
  const GROUPS: GroupName[] = ['Training', 'Optimizer', 'Augmentation', 'Inference'];
  for (const g of GROUPS) {
    const order = GROUP_FIELD_ORDER[g];
    grouped[g].sort((a, b) => {
      const ia = order.indexOf(a);
      const ib = order.indexOf(b);
      if (ia === -1 && ib === -1) return a.localeCompare(b);
      if (ia === -1) return 1;
      if (ib === -1) return -1;
      return ia - ib;
    });
  }
  // Append truly unknown keys to Training
  grouped['Training'].push(...extraKeys.sort());

  const gridCols = hasArchFields ? 'lg:grid-cols-5' : 'lg:grid-cols-4';

  return (
    <div className="bg-[#0f1117] border border-slate-800 rounded-xl overflow-hidden shadow-sm mt-6">
      <div
        className="px-6 py-4 flex justify-between items-center cursor-pointer bg-slate-900/50 hover:bg-slate-900 transition-colors"
        onClick={() => setShowConfig(!showConfig)}
      >
        <h3 className="text-white font-semibold flex items-center gap-2">
          <Settings size={18} className="text-slate-400" /> Training Configuration
        </h3>
        {showConfig
          ? <ChevronUp size={18} className="text-slate-500" />
          : <ChevronDown size={18} className="text-slate-500" />}
      </div>

      {showConfig && (
        <div className={`p-6 grid grid-cols-1 md:grid-cols-2 ${gridCols} gap-8 text-sm border-t border-slate-800 bg-[#0f1117]`}>

          {/* ── Training ────────────────────────────────────────────── */}
          <ConfigColumn title="Training" icon={<Cpu size={14} />}>
            {grouped.Training.map(key => (
              <ConfigItem
                key={key}
                label={toLabel(key)}
                value={fmtValue(key, config[key])}
                highlight={HIGHLIGHT_FIELDS.has(key)}
              />
            ))}
          </ConfigColumn>

          {/* ── Optimizer & Loss ────────────────────────────────────── */}
          <ConfigColumn title="Optimizer & Loss" icon={<Sliders size={14} />}>
            {grouped.Optimizer.map(key => (
              <ConfigItem
                key={key}
                label={toLabel(key)}
                value={fmtValue(key, config[key])}
                highlight={HIGHLIGHT_FIELDS.has(key)}
              />
            ))}
          </ConfigColumn>

          {/* ── Augmentation ────────────────────────────────────────── */}
          <ConfigColumn title="Augmentation" icon={<ImageIcon size={14} />}>
            {grouped.Augmentation.map(key => (
              <ConfigItem
                key={key}
                label={toLabel(key)}
                value={fmtValue(key, config[key])}
                highlight={HIGHLIGHT_FIELDS.has(key)}
              />
            ))}
          </ConfigColumn>

          {/* ── Inference & System ──────────────────────────────────── */}
          <ConfigColumn title="Inference & System" icon={<Server size={14} />}>
            {grouped.Inference.map(key => (
              <ConfigItem
                key={key}
                label={toLabel(key)}
                value={fmtValue(key, config[key])}
                highlight={HIGHLIGHT_FIELDS.has(key)}
              />
            ))}
            {typeof modelScale === 'string' && modelScale && (
              <ConfigItem label="Model Scale" value={modelScale.toUpperCase()} highlight />
            )}
            {/* Model Info */}
            {(config.layer_count || typeof config.model_params === 'number') && (
              <div className="mt-4 pt-2 border-t border-slate-800">
                <span className="text-xs text-slate-500 block mb-2">Model Info</span>
                <div className="space-y-1">
                  {config.layer_count && (
                    <ConfigItem label="Layers" value={String(config.layer_count)} />
                  )}
                  {typeof config.model_params === 'number' && (
                    <ConfigItem label="Params" value={`${(config.model_params / 1e6).toFixed(2)}M`} highlight />
                  )}
                  {typeof config.model_flops === 'number' && (
                    <ConfigItem label="FLOPs" value={`${config.model_flops.toFixed(1)} G`} highlight />
                  )}
                </div>
              </div>
            )}
            {/* Dataset */}
            <div className="mt-4 pt-2 border-t border-slate-800">
              <span className="text-xs text-slate-500 block mb-1">Dataset</span>
              <span className="bg-slate-900 px-2 py-1 rounded text-xs text-slate-400 block border border-slate-800">
                {config.dataset_name || datasetName || 'unknown'}
              </span>
            </div>
          </ConfigColumn>

          {/* ── Model Config (arch-specific, conditional) ───────────── */}
          {hasArchFields && (
            <ConfigColumn title="Model Config" icon={<Layers size={14} />}>
              {modelArch && (
                <div className="mb-3 pb-2 border-b border-slate-800">
                  <span className="text-xs text-slate-500 block mb-1">Arch Plugin</span>
                  <span className="font-mono text-xs text-indigo-400 break-all">{modelArch}</span>
                </div>
              )}
              {archFields.map(key => (
                <ConfigItem
                  key={key}
                  label={toLabel(key)}
                  value={fmtValue(key, resolvedArch[key])}
                  highlight={HIGHLIGHT_FIELDS.has(key)}
                />
              ))}
            </ConfigColumn>
          )}

        </div>
      )}
    </div>
  );
};

// ── Sub-components ────────────────────────────────────────────────────────────

const ConfigColumn: React.FC<{
  title: string; icon: React.ReactNode; children: React.ReactNode;
}> = ({ title, icon, children }) => (
  <div>
    <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2">
      {icon} {title}
    </h4>
    <div className="space-y-2">{children}</div>
  </div>
);

const ConfigItem: React.FC<{
  label: string; value: string | number; highlight?: boolean;
}> = ({ label, value, highlight }) => (
  <div className="flex justify-between border-b border-slate-800/50 pb-1 last:border-0 hover:bg-slate-800/30 px-1 rounded transition-colors">
    <span className={highlight ? 'text-indigo-400' : 'text-slate-400'}>{label}</span>
    <span className="text-slate-300 font-mono text-right max-w-[55%] truncate">{value}</span>
  </div>
);

export default JobConfiguration;
