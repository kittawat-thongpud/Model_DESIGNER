/**
 * Edge Validator — checks whether a connection between two layer nodes
 * is valid based on tensor dimension compatibility.
 *
 * Each layer operates on a specific tensor shape category:
 *   - "2d"  → (batch, C, H, W)   e.g. Conv2d, MaxPool2d, BatchNorm2d
 *   - "1d"  → (batch, features)   e.g. Linear
 *   - "any" → passes through      e.g. ReLU, Dropout, Softmax
 *   - "reshape" → transforms 2d→1d (Flatten)
 */
import { LAYER_DEFINITIONS, type LayerType } from '../types';
import type { Node } from '@xyflow/react';

// ─── Shape categories ────────────────────────────────────────────────────────

export type ShapeCategory = '2d' | '1d' | 'any' | 'none';

/** What shape category a layer OUTPUTS */
const OUTPUT_SHAPE: Record<LayerType, ShapeCategory> = {
  Input:       '2d',
  Conv2d:      '2d',
  MaxPool2d:   '2d',
  BatchNorm2d: '2d',
  Flatten:     '1d',
  Linear:      '1d',
  ReLU:        'any',   // passes through
  Dropout:     'any',
  Softmax:     'any',
  Output:      'none',  // terminal
  Package:     'any',   // opaque sub-graph
  Upsample:    '2d',
  Concatenate: '2d',
};

/** What shape category a layer ACCEPTS as input */
const INPUT_SHAPE: Record<LayerType, ShapeCategory> = {
  Input:       'none',  // no input
  Conv2d:      '2d',
  MaxPool2d:   '2d',
  BatchNorm2d: '2d',
  Flatten:     '2d',
  Linear:      '1d',
  ReLU:        'any',
  Dropout:     'any',
  Softmax:     'any',
  Output:      'any',
  Package:     'any',   // opaque sub-graph
  Upsample:    '2d',
  Concatenate: '2d',
};

// ─── Compatibility check ─────────────────────────────────────────────────────

export interface EdgeValidationResult {
  valid: boolean;
  reason?: string;
}

/**
 * Check if connecting sourceType → targetType is valid.
 */
export function checkEdgeCompatibility(
  sourceType: LayerType,
  targetType: LayerType,
): EdgeValidationResult {
  const srcOutput = OUTPUT_SHAPE[sourceType];
  const tgtInput  = INPUT_SHAPE[targetType];

  // Source has no output handle (e.g. Output node)
  if (srcOutput === 'none') {
    return { valid: false, reason: `${sourceType} has no output` };
  }

  // Target has no input handle (e.g. Input node)
  if (tgtInput === 'none') {
    return { valid: false, reason: `${targetType} cannot receive connections` };
  }

  // If either side is "any", always compatible
  if (srcOutput === 'any' || tgtInput === 'any') {
    return { valid: true };
  }

  // Both must match: 2d→2d or 1d→1d
  if (srcOutput === tgtInput) {
    return { valid: true };
  }

  // Mismatch
  const srcLabel = LAYER_DEFINITIONS[sourceType]?.label || sourceType;
  const tgtLabel = LAYER_DEFINITIONS[targetType]?.label || targetType;
  return {
    valid: false,
    reason: `${srcLabel} outputs ${srcOutput.toUpperCase()} but ${tgtLabel} expects ${tgtInput.toUpperCase()}`,
  };
}

/**
 * Resolve the "effective" output shape of a node by tracing back through
 * pass-through layers (any→any). Used when a pass-through node is the source.
 *
 * For basic checks we just look at the direct source/target types,
 * but this helper can be used for more advanced validation later.
 */
export function resolveEffectiveOutput(
  sourceNodeId: string,
  nodes: Node[],
  edges: { source: string; target: string }[],
): ShapeCategory {
  const node = nodes.find((n) => n.id === sourceNodeId);
  if (!node) return 'any';

  const layerType = node.data.layerType as LayerType;
  const shape = OUTPUT_SHAPE[layerType];

  // If this node has a definite shape, return it
  if (shape !== 'any') return shape;

  // Otherwise trace backwards through the graph
  const incomingEdge = edges.find((e) => e.target === sourceNodeId);
  if (!incomingEdge) return 'any'; // unconnected pass-through
  return resolveEffectiveOutput(incomingEdge.source, nodes, edges);
}

/**
 * Resolve the "effective" input shape by tracing forward through pass-through layers.
 */
export function resolveEffectiveInput(
  targetNodeId: string,
  nodes: Node[],
  edges: { source: string; target: string }[],
): ShapeCategory {
  const node = nodes.find((n) => n.id === targetNodeId);
  if (!node) return 'any';

  const layerType = node.data.layerType as LayerType;
  const shape = INPUT_SHAPE[layerType];

  if (shape !== 'any') return shape;

  // Trace forward
  const outgoingEdge = edges.find((e) => e.source === targetNodeId);
  if (!outgoingEdge) return 'any';
  return resolveEffectiveInput(outgoingEdge.target, nodes, edges);
}

/**
 * Full connection validation: checks type compatibility + prevents duplicate edges
 * and self-loops.
 */
export function validateConnection(
  sourceId: string,
  targetId: string,
  nodes: Node[],
  edges: { source: string; target: string }[],
): EdgeValidationResult {
  // Self-loop
  if (sourceId === targetId) {
    return { valid: false, reason: 'Cannot connect a node to itself' };
  }

  // Duplicate edge
  if (edges.some((e) => e.source === sourceId && e.target === targetId)) {
    return { valid: false, reason: 'Connection already exists' };
  }

  // Target already has an incoming connection (unless acts as multi-input)
  const targetDef = nodes.find((n) => n.id === targetId);
  const targetLayerType = targetDef?.data.layerType as LayerType;
  const allowMulti = LAYER_DEFINITIONS[targetLayerType]?.allowMultipleInputs ?? false;

  if (!allowMulti && edges.some((e) => e.target === targetId)) {
    return { valid: false, reason: 'This node already has an input connection' };
  }

  const sourceNode = nodes.find((n) => n.id === sourceId);
  const targetNode = nodes.find((n) => n.id === targetId);
  if (!sourceNode || !targetNode) {
    return { valid: false, reason: 'Invalid node' };
  }

  const sourceType = sourceNode.data.layerType as LayerType;
  const targetType = targetNode.data.layerType as LayerType;

  // Basic type compatibility
  const typeCheck = checkEdgeCompatibility(sourceType, targetType);
  if (!typeCheck.valid) return typeCheck;

  // For pass-through layers, resolve the effective shape
  const effectiveOutput = resolveEffectiveOutput(sourceId, nodes, edges);
  const effectiveInput  = resolveEffectiveInput(targetId, nodes, edges);

  if (effectiveOutput !== 'any' && effectiveInput !== 'any' && effectiveOutput !== effectiveInput) {
    return {
      valid: false,
      reason: `Shape mismatch: chain produces ${effectiveOutput.toUpperCase()} but downstream expects ${effectiveInput.toUpperCase()}`,
    };
  }

  return { valid: true };
}
