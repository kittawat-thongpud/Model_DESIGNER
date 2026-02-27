import React, { useState } from 'react';
import { 
  Settings, 
  ChevronDown, 
  ChevronUp, 
  Cpu, 
  Sliders, 
  Image as ImageIcon, 
  Server 
} from 'lucide-react';
import { TrainConfig } from '../types';
import { fmtDataset } from '../utils/format';

interface JobConfigurationProps {
  config: TrainConfig;
  datasetName?: string | null;
  partitions?: Array<{partition_id: string; train: boolean; val: boolean; test: boolean}>;
  modelScale?: string;
}

const JobConfiguration: React.FC<JobConfigurationProps> = ({ config, datasetName, partitions, modelScale }) => {
  const [showConfig, setShowConfig] = useState(false);

  // Helper to safely access config values
  const getValue = (key: string, defaultValue: any = '-') => {
    return config[key] !== undefined ? config[key] : defaultValue;
  };

  return (
    <div className="bg-[#0f1117] border border-slate-800 rounded-xl overflow-hidden shadow-sm mt-6">
      <div 
        className="px-6 py-4 flex justify-between items-center cursor-pointer bg-slate-900/50 hover:bg-slate-900 transition-colors"
        onClick={() => setShowConfig(!showConfig)}
      >
         <h3 className="text-white font-semibold flex items-center gap-2">
            <Settings size={18} className="text-slate-400" /> Training Configuration
         </h3>
         {showConfig ? <ChevronUp size={18} className="text-slate-500" /> : <ChevronDown size={18} className="text-slate-500" />}
      </div>
      
      {showConfig && (
        <div className="p-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 text-sm border-t border-slate-800 bg-[#0f1117]">
           
           {/* Column 1: Training Basics */}
           <ConfigColumn title="Training" icon={<Cpu size={14} />}>
              <ConfigItem label="Epochs" value={getValue('epochs')} />
              <ConfigItem label="Batch Size" value={getValue('batch')} />
              <ConfigItem label="Image Size" value={getValue('imgsz')} />
              <ConfigItem label="Workers" value={getValue('workers')} />
              <ConfigItem label="Device" value={getValue('device') || 'Auto'} />
              <ConfigItem label="Patience" value={getValue('patience')} />
              <ConfigItem label="Freeze" value={getValue('freeze')} />
              <ConfigItem label="Close Mosaic" value={getValue('close_mosaic')} />
              <ConfigItem label="Save Period" value={getValue('save_period')} />
              <ConfigItem label="Samples/Class" value={getValue('sample_per_class')} />
              <ConfigItem label="Resume" value={String(getValue('resume'))} />
              <ConfigItem label="AMP" value={String(getValue('amp'))} />
              <ConfigItem label="Rect" value={String(getValue('rect'))} />
              <ConfigItem label="Single Cls" value={String(getValue('single_cls'))} />
           </ConfigColumn>

           {/* Column 2: Optimizer + Loss */}
           <ConfigColumn title="Optimizer & Loss" icon={<Sliders size={14} />}>
              <ConfigItem label="Optimizer" value={getValue('optimizer')} />
              <ConfigItem label="lr0" value={getValue('lr0')} highlight />
              <ConfigItem label="lrf" value={getValue('lrf')} />
              <ConfigItem label="Momentum" value={getValue('momentum')} />
              <ConfigItem label="Weight Decay" value={getValue('weight_decay')} />
              <ConfigItem label="Warmup Epochs" value={getValue('warmup_epochs')} />
              <ConfigItem label="Warmup Mom" value={getValue('warmup_momentum')} />
              <ConfigItem label="Warmup Bias LR" value={getValue('warmup_bias_lr')} />
              <ConfigItem label="Cos LR" value={String(getValue('cos_lr'))} />
              <ConfigItem label="NBS" value={getValue('nbs')} />
              <ConfigItem label="Box Gain" value={getValue('box')} highlight />
              <ConfigItem label="Cls Gain" value={getValue('cls')} highlight />
              <ConfigItem label="DFL Gain" value={getValue('dfl')} highlight />
              <ConfigItem label="Pose Gain" value={getValue('pose')} />
              <ConfigItem label="kobj" value={getValue('kobj')} />
              <ConfigItem label="Mask Ratio" value={getValue('mask_ratio')} />
              <ConfigItem label="Overlap Mask" value={String(getValue('overlap_mask'))} />
           </ConfigColumn>

           {/* Column 3: Augmentation */}
           <ConfigColumn title="Augmentation" icon={<ImageIcon size={14} />}>
              <ConfigItem label="Mosaic" value={getValue('mosaic')} />
              <ConfigItem label="Mixup" value={getValue('mixup')} />
              <ConfigItem label="Copy Paste" value={getValue('copy_paste')} />
              <ConfigItem label="Erasing" value={getValue('erasing')} />
              <ConfigItem label="Crop Frac" value={getValue('crop_fraction')} />
              <ConfigItem label="Flip LR" value={getValue('fliplr')} />
              <ConfigItem label="Flip UD" value={getValue('flipud')} />
              <ConfigItem label="Degrees" value={getValue('degrees')} />
              <ConfigItem label="Translate" value={getValue('translate')} />
              <ConfigItem label="Scale" value={getValue('scale')} />
              <ConfigItem label="Shear" value={getValue('shear')} />
              <ConfigItem label="Perspective" value={getValue('perspective')} />
              <ConfigItem label="HSV-H" value={getValue('hsv_h')} />
              <ConfigItem label="HSV-S" value={getValue('hsv_s')} />
              <ConfigItem label="HSV-V" value={getValue('hsv_v')} />
              <ConfigItem label="BGR" value={getValue('bgr')} />
              {getValue('auto_augment') !== '-' && getValue('auto_augment') && <ConfigItem label="Auto Aug" value={getValue('auto_augment')} />}
           </ConfigColumn>

           {/* Column 4: Inference + System */}
           <ConfigColumn title="Inference & System" icon={<Server size={14} />}>
              <ConfigItem label="Conf" value={getValue('conf') ?? 'Auto'} />
              <ConfigItem label="IoU" value={getValue('iou')} />
              <ConfigItem label="Max Det" value={getValue('max_det')} />
              <ConfigItem label="Agnostic NMS" value={String(getValue('agnostic_nms'))} />
              <ConfigItem label="Seed" value={getValue('seed')} />
              {modelScale && <ConfigItem label="Model Scale" value={modelScale.toUpperCase()} highlight />}
              <div className="mt-4 pt-2 border-t border-slate-800">
                <span className="text-xs text-slate-500 block mb-2">Model Info</span>
                <div className="space-y-1">
                  <ConfigItem label="Layers" value={String(config.layer_count || '-')} />
                  {typeof config.model_params === 'number' && (
                    <ConfigItem label="Params" value={`${(config.model_params / 1e6).toFixed(2)}M`} highlight />
                  )}
                  {typeof config.model_flops === 'number' && (
                    <ConfigItem label="FLOPs" value={`${config.model_flops.toFixed(1)} G`} highlight />
                  )}
                </div>
              </div>
              <div className="mt-4 pt-2 border-t border-slate-800">
                <span className="text-xs text-slate-500 block mb-1">Dataset</span>
                <span className="bg-slate-900 px-2 py-1 rounded text-xs text-slate-400 block border border-slate-800">
                  {fmtDataset(getValue('data') as string, datasetName)}
                </span>
              </div>
              {partitions && partitions.length > 0 && (
                <div className="mt-4 pt-2 border-t border-slate-800">
                  <span className="text-xs text-slate-500 block mb-2">Partitions ({partitions.length})</span>
                  <div className="space-y-1">
                    {partitions.map((p, idx) => (
                      <div key={idx} className="bg-slate-900 px-2 py-1.5 rounded text-xs border border-slate-800">
                        <div className="text-emerald-400 font-medium mb-0.5">{p.partition_id}</div>
                        <div className="text-slate-500 text-[10px]">
                          {[p.train && 'train', p.val && 'val', p.test && 'test'].filter(Boolean).join(' + ')}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
           </ConfigColumn>

        </div>
      )}
    </div>
  );
};

const ConfigColumn: React.FC<{ title: string; icon: React.ReactNode; children: React.ReactNode }> = ({ title, icon, children }) => (
  <div>
    <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2">
      {icon} {title}
    </h4>
    <div className="space-y-2">
      {children}
    </div>
  </div>
);

const ConfigItem: React.FC<{ label: string; value: string | number; highlight?: boolean }> = ({ label, value, highlight }) => (
  <div className="flex justify-between border-b border-slate-800/50 pb-1 last:border-0 hover:bg-slate-800/30 px-1 rounded transition-colors">
    <span className={highlight ? "text-indigo-400" : "text-slate-400"}>{label}</span>
    <span className="text-slate-300 font-mono">{value}</span>
  </div>
);

export default JobConfiguration;
