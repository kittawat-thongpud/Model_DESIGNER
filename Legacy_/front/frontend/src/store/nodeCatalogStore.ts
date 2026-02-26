/**
 * Node Catalog Store — single source of truth for all node type definitions.
 *
 * Fetches the full catalog from the backend /api/registry/nodes on app init.
 * Every component (LayerPalette, PropertiesPanel, edgeValidator, shapeInference,
 * designerStore) consumes from this store instead of static LAYER_DEFINITIONS.
 *
 * Falls back to a minimal embedded snapshot if the backend is unreachable.
 */
import { create } from 'zustand';
import { api } from '../services/api';
import type { NodeRegistryEntry, AutoParamDef, ShapeRule, ParamDefinition } from '../types';


// ─── Minimal fallback definitions (used when backend is unreachable) ────────

const FALLBACK_ENTRIES: NodeRegistryEntry[] = [
  {
    type: 'Input', label: 'Input', category: 'I/O', color: '#4CAF50', icon: '📥',
    params: [
      { name: 'channels', type: 'number', default: 1, label: 'Channels', min: 1 },
      { name: 'height', type: 'number', default: 28, label: 'Height', min: 1 },
      { name: 'width', type: 'number', default: 28, label: 'Width', min: 1 },
    ],
    inputs: [], outputs: [{ name: 'out', pin_type: 'Tensor2D', label: 'Output' }],
    hasInput: false, hasOutput: true, allowMultipleInputs: false,
    autoParams: [], shapeRule: { rule_type: 'none_to_2d' },
  },
  {
    type: 'Output', label: 'Output', category: 'I/O', color: '#f44336', icon: '📤',
    params: [
      { name: 'num_classes', type: 'number', default: 10, label: 'Num Classes', min: 1 },
      { name: 'out_features', type: 'number', default: 10, label: 'Out Features', min: 1 },
    ],
    inputs: [{ name: 'in', pin_type: 'Any', label: 'Input' }], outputs: [],
    hasInput: true, hasOutput: false, allowMultipleInputs: false,
    autoParams: [], shapeRule: { rule_type: 'terminal' },
  },
  {
    type: 'Conv2d', label: 'Conv2D', category: 'Processing', color: '#2196F3', icon: '🔲',
    params: [
      { name: 'in_channels', type: 'number', default: 1, label: 'In Channels', min: 1 },
      { name: 'out_channels', type: 'number', default: 32, label: 'Out Channels', min: 1 },
      { name: 'kernel_size', type: 'number', default: 3, label: 'Kernel Size', min: 1 },
      { name: 'stride', type: 'number', default: 1, label: 'Stride', min: 1 },
      { name: 'padding', type: 'number', default: 1, label: 'Padding', min: 0 },
    ],
    inputs: [{ name: 'in', pin_type: 'Tensor2D', label: 'Input' }],
    outputs: [{ name: 'out', pin_type: 'Tensor2D', label: 'Output' }],
    hasInput: true, hasOutput: true, allowMultipleInputs: false,
    autoParams: [{ name: 'in_channels', from_field: 'channels' }],
    shapeRule: { rule_type: 'conv2d', out_channels_param: 'out_channels', kernel_param: 'kernel_size', stride_param: 'stride', padding_param: 'padding' },
  },
  {
    type: 'MaxPool2d', label: 'MaxPool2D', category: 'Processing', color: '#9C27B0', icon: '🔻',
    params: [
      { name: 'kernel_size', type: 'number', default: 2, label: 'Kernel Size', min: 1 },
      { name: 'stride', type: 'number', default: 2, label: 'Stride', min: 1 },
    ],
    inputs: [{ name: 'in', pin_type: 'Tensor2D', label: 'Input' }],
    outputs: [{ name: 'out', pin_type: 'Tensor2D', label: 'Output' }],
    hasInput: true, hasOutput: true, allowMultipleInputs: false,
    autoParams: [],
    shapeRule: { rule_type: 'pool2d', kernel_param: 'kernel_size', stride_param: 'stride' },
  },
  {
    type: 'Linear', label: 'Linear (FC)', category: 'Processing', color: '#E91E63', icon: '🔗',
    params: [
      { name: 'in_features', type: 'number', default: 512, label: 'In Features', min: 1 },
      { name: 'out_features', type: 'number', default: 10, label: 'Out Features', min: 1 },
    ],
    inputs: [{ name: 'in', pin_type: 'Tensor1D', label: 'Input' }],
    outputs: [{ name: 'out', pin_type: 'Tensor1D', label: 'Output' }],
    hasInput: true, hasOutput: true, allowMultipleInputs: false,
    autoParams: [{ name: 'in_features', from_field: 'features' }],
    shapeRule: { rule_type: 'linear', out_features_param: 'out_features' },
  },
  {
    type: 'Flatten', label: 'Flatten', category: 'Reshape', color: '#607D8B', icon: '📏',
    params: [],
    inputs: [{ name: 'in', pin_type: 'Tensor2D', label: 'Input' }],
    outputs: [{ name: 'out', pin_type: 'Tensor1D', label: 'Output' }],
    hasInput: true, hasOutput: true, allowMultipleInputs: false,
    autoParams: [], shapeRule: { rule_type: 'flatten' },
  },
  {
    type: 'ReLU', label: 'ReLU', category: 'Activation', color: '#FF9800', icon: '⚡',
    params: [],
    inputs: [{ name: 'in', pin_type: 'Any', label: 'Input' }],
    outputs: [{ name: 'out', pin_type: 'Any', label: 'Output' }],
    hasInput: true, hasOutput: true, allowMultipleInputs: false,
    autoParams: [], shapeRule: { rule_type: 'passthrough' },
  },
  {
    type: 'BatchNorm2d', label: 'BatchNorm2D', category: 'Regularization', color: '#00BCD4', icon: '📊',
    params: [{ name: 'num_features', type: 'number', default: 32, label: 'Num Features', min: 1 }],
    inputs: [{ name: 'in', pin_type: 'Tensor2D', label: 'Input' }],
    outputs: [{ name: 'out', pin_type: 'Tensor2D', label: 'Output' }],
    hasInput: true, hasOutput: true, allowMultipleInputs: false,
    autoParams: [{ name: 'num_features', from_field: 'channels' }],
    shapeRule: { rule_type: 'passthrough' },
  },
  {
    type: 'Dropout', label: 'Dropout', category: 'Regularization', color: '#795548', icon: '🎲',
    params: [{ name: 'p', type: 'number', default: 0.5, label: 'Probability', min: 0, max: 1 }],
    inputs: [{ name: 'in', pin_type: 'Any', label: 'Input' }],
    outputs: [{ name: 'out', pin_type: 'Any', label: 'Output' }],
    hasInput: true, hasOutput: true, allowMultipleInputs: false,
    autoParams: [], shapeRule: { rule_type: 'passthrough' },
  },
  {
    type: 'Softmax', label: 'Softmax', category: 'Activation', color: '#FF5722', icon: '📈',
    params: [{ name: 'dim', type: 'number', default: 1, label: 'Dimension', min: 0 }],
    inputs: [{ name: 'in', pin_type: 'Any', label: 'Input' }],
    outputs: [{ name: 'out', pin_type: 'Any', label: 'Output' }],
    hasInput: true, hasOutput: true, allowMultipleInputs: false,
    autoParams: [], shapeRule: { rule_type: 'passthrough' },
  },
  {
    type: 'Upsample', label: 'Upsample', category: 'Reshape', color: '#8BC34A', icon: '⬆️',
    params: [
      { name: 'scale_factor', type: 'number', default: 2.0, label: 'Scale Factor', min: 1 },
      { name: 'mode', type: 'string', default: 'nearest', label: 'Mode (nearest/bilinear)' },
    ],
    inputs: [{ name: 'in', pin_type: 'Tensor2D', label: 'Input' }],
    outputs: [{ name: 'out', pin_type: 'Tensor2D', label: 'Output' }],
    hasInput: true, hasOutput: true, allowMultipleInputs: false,
    autoParams: [], shapeRule: { rule_type: 'upsample', scale_param: 'scale_factor' },
  },
  {
    type: 'Concatenate', label: 'Concatenate', category: 'Processing', color: '#FFC107', icon: '🔗',
    params: [{ name: 'dim', type: 'number', default: 1, label: 'Dimension', min: 0 }],
    inputs: [{ name: 'in', pin_type: 'Tensor2D', label: 'Input' }],
    outputs: [{ name: 'out', pin_type: 'Tensor2D', label: 'Output' }],
    hasInput: true, hasOutput: true, allowMultipleInputs: true,
    autoParams: [], shapeRule: { rule_type: 'passthrough' },
  },
  {
    type: 'Package', label: 'Package', category: 'Package', color: '#607D8B', icon: '📦',
    params: [],
    inputs: [{ name: 'in', pin_type: 'Any', label: 'Input' }],
    outputs: [{ name: 'out', pin_type: 'Any', label: 'Output' }],
    hasInput: true, hasOutput: true, allowMultipleInputs: false,
    autoParams: [], shapeRule: { rule_type: 'passthrough' },
  },
];


// ─── Helper: pin type to shape category ─────────────────────────────────────

export type ShapeCategory = '2d' | '1d' | 'any' | 'none';

export function pinTypeToShape(pinType: string): ShapeCategory {
  switch (pinType) {
    case 'Tensor2D': return '2d';
    case 'Tensor1D': return '1d';
    case 'Image':    return '2d';
    case 'BBoxList': return '1d';
    case 'Scalar':   return '1d';
    case 'Any':      return 'any';
    case 'None':     return 'none';
    default:         return 'any';
  }
}


// ─── Store Interface ────────────────────────────────────────────────────────

interface NodeCatalogState {
  entries: NodeRegistryEntry[];
  byType: Record<string, NodeRegistryEntry>;
  categories: string[];
  isLoaded: boolean;
  error: string | null;

  // Actions
  load: () => Promise<void>;

  // Lookups (pure functions that read state)
  getDef: (type: string) => NodeRegistryEntry | undefined;
  getParams: (type: string) => ParamDefinition[];
  getOutputPin: (type: string) => string;    // returns pin_type string
  getInputPin: (type: string) => string;     // returns pin_type string
  getOutputShape: (type: string) => ShapeCategory;
  getInputShape: (type: string) => ShapeCategory;
  getAllowMultipleInputs: (type: string) => boolean;
  getAutoParams: (type: string) => AutoParamDef[];
  getShapeRule: (type: string) => ShapeRule | null;
  pinsCompatible: (srcPinType: string, tgtPinType: string) => boolean;
}


// ─── Build derived state from entries ───────────────────────────────────────

function buildDerived(entries: NodeRegistryEntry[]) {
  const byType: Record<string, NodeRegistryEntry> = {};
  const catSet = new Set<string>();
  for (const e of entries) {
    byType[e.type] = e;
    catSet.add(e.category);
  }
  return { byType, categories: Array.from(catSet) };
}


// ─── Store ──────────────────────────────────────────────────────────────────

export const useNodeCatalogStore = create<NodeCatalogState>((set, get) => {
  // Initialize with fallback so the app works before load() completes
  const initial = buildDerived(FALLBACK_ENTRIES);

  return {
    entries: FALLBACK_ENTRIES,
    byType: initial.byType,
    categories: initial.categories,
    isLoaded: false,
    error: null,

    load: async () => {
      try {
        const data = await api.getNodeRegistry();
        // Ensure autoParams is always present (backend may omit empty arrays)
        const entries = data.map((e) => ({
          ...e,
          autoParams: e.autoParams ?? [],
        }));
        const derived = buildDerived(entries);
        set({
          entries,
          byType: derived.byType,
          categories: derived.categories,
          isLoaded: true,
          error: null,
        });
      } catch (err) {
        console.warn('[nodeCatalog] Backend unreachable, using fallback definitions');
        set({ isLoaded: true, error: String(err) });
      }
    },

    getDef: (type) => get().byType[type],

    getParams: (type) => get().byType[type]?.params ?? [],

    getOutputPin: (type) => {
      const entry = get().byType[type];
      return entry?.outputs?.[0]?.pin_type ?? 'None';
    },

    getInputPin: (type) => {
      const entry = get().byType[type];
      return entry?.inputs?.[0]?.pin_type ?? 'None';
    },

    getOutputShape: (type) => pinTypeToShape(get().getOutputPin(type)),

    getInputShape: (type) => pinTypeToShape(get().getInputPin(type)),

    getAllowMultipleInputs: (type) => get().byType[type]?.allowMultipleInputs ?? false,

    getAutoParams: (type) => get().byType[type]?.autoParams ?? [],

    getShapeRule: (type) => get().byType[type]?.shapeRule ?? null,

    pinsCompatible: (srcPinType, tgtPinType) => {
      if (srcPinType === 'None' || tgtPinType === 'None') return false;
      if (srcPinType === 'Any' || tgtPinType === 'Any') return true;
      return srcPinType === tgtPinType;
    },
  };
});
