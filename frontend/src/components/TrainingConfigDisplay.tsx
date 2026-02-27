import { Settings, Database, Zap, Sliders, Save, Activity } from 'lucide-react';

interface TrainingConfigDisplayProps {
  config: Record<string, unknown>;
}

export default function TrainingConfigDisplay({ config }: TrainingConfigDisplayProps) {
  const sections = [
    {
      title: 'Data & Augmentation',
      icon: Database,
      fields: [
        { key: 'data', label: 'Dataset' },
        { key: 'imgsz', label: 'Image Size' },
        { key: 'batch', label: 'Batch Size' },
        { key: 'workers', label: 'Workers' },
        { key: 'hsv_h', label: 'HSV Hue' },
        { key: 'hsv_s', label: 'HSV Saturation' },
        { key: 'hsv_v', label: 'HSV Value' },
        { key: 'degrees', label: 'Rotation' },
        { key: 'translate', label: 'Translation' },
        { key: 'scale', label: 'Scale' },
        { key: 'shear', label: 'Shear' },
        { key: 'perspective', label: 'Perspective' },
        { key: 'flipud', label: 'Flip UD' },
        { key: 'fliplr', label: 'Flip LR' },
        { key: 'mosaic', label: 'Mosaic' },
        { key: 'mixup', label: 'Mixup' },
        { key: 'copy_paste', label: 'Copy Paste' },
      ],
    },
    {
      title: 'Training',
      icon: Activity,
      fields: [
        { key: 'epochs', label: 'Epochs' },
        { key: 'patience', label: 'Patience' },
        { key: 'device', label: 'Device' },
        { key: 'seed', label: 'Seed' },
        { key: 'deterministic', label: 'Deterministic' },
        { key: 'amp', label: 'AMP' },
        { key: 'ema', label: 'EMA' },
        { key: 'close_mosaic', label: 'Close Mosaic' },
      ],
    },
    {
      title: 'Optimizer & Learning Rate',
      icon: Zap,
      fields: [
        { key: 'optimizer', label: 'Optimizer' },
        { key: 'lr0', label: 'Initial LR' },
        { key: 'lrf', label: 'Final LR Factor' },
        { key: 'momentum', label: 'Momentum' },
        { key: 'weight_decay', label: 'Weight Decay' },
        { key: 'warmup_epochs', label: 'Warmup Epochs' },
        { key: 'warmup_momentum', label: 'Warmup Momentum' },
        { key: 'cos_lr', label: 'Cosine LR' },
      ],
    },
    {
      title: 'Loss Weights',
      icon: Sliders,
      fields: [
        { key: 'box', label: 'Box Loss' },
        { key: 'cls', label: 'Class Loss' },
        { key: 'dfl', label: 'DFL Loss' },
      ],
    },
    {
      title: 'Model & Checkpoints',
      icon: Save,
      fields: [
        { key: 'pretrained', label: 'Pretrained' },
        { key: 'freeze', label: 'Freeze Layers' },
        { key: 'save_period', label: 'Save Period' },
        { key: 'val', label: 'Validation' },
        { key: 'plots', label: 'Plots' },
      ],
    },
    {
      title: 'Monitoring',
      icon: Settings,
      fields: [
        { key: 'record_gradients', label: 'Record Gradients' },
        { key: 'gradient_interval', label: 'Gradient Interval' },
        { key: 'record_weights', label: 'Record Weights' },
        { key: 'weight_interval', label: 'Weight Interval' },
      ],
    },
  ];

  const formatValue = (value: unknown): string => {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value === 'boolean') return value ? 'Yes' : 'No';
    if (typeof value === 'number') return value.toString();
    if (typeof value === 'string') return value || 'N/A';
    if (Array.isArray(value)) return value.join(', ');
    return String(value);
  };

  return (
    <div className="space-y-6">
      {sections.map((section) => {
        const Icon = section.icon;
        const hasValues = section.fields.some((f) => config[f.key] !== undefined);
        
        if (!hasValues) return null;

        return (
          <div key={section.title} className="bg-gray-800 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3 pb-2 border-b border-gray-700">
              <Icon className="w-4 h-4 text-blue-400" />
              <h3 className="font-semibold text-white">{section.title}</h3>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
              {section.fields.map((field) => {
                const value = config[field.key];
                if (value === undefined) return null;

                return (
                  <div key={field.key} className="bg-gray-900 rounded p-2">
                    <div className="text-xs text-gray-400 mb-1">{field.label}</div>
                    <div className="text-sm text-white font-mono truncate" title={formatValue(value)}>
                      {formatValue(value)}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
