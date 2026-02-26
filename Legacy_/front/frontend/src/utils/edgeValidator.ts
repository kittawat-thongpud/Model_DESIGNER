/**
 * Edge Validator — checks whether a connection between two layer nodes
 * is valid based on tensor dimension compatibility.
 *
 * Uses the nodeCatalogStore as the single source of truth for pin types,
 * shape categories, and multi-input rules. No static fallback maps.
 */
import type { Node } from '@xyflow/react';
import { useNodeCatalogStore, type ShapeCategory } from '../store/nodeCatalogStore';

export type { ShapeCategory } from '../store/nodeCatalogStore';

// ─── Catalog accessors (non-hook, for use outside React components) ─────────

function getOutputShape(layerType: string): ShapeCategory {
  return useNodeCatalogStore.getState().getOutputShape(layerType);
}

function getInputShape(layerType: string): ShapeCategory {
  return useNodeCatalogStore.getState().getInputShape(layerType);
}

function getAllowMultipleInputs(layerType: string): boolean {
  return useNodeCatalogStore.getState().getAllowMultipleInputs(layerType);
}

function getLabel(layerType: string): string {
  return useNodeCatalogStore.getState().getDef(layerType)?.label ?? layerType;
}

// ─── Compatibility check ─────────────────────────────────────────────────────

export interface EdgeValidationResult {
  valid: boolean;
  reason?: string;
}

/**
 * Check if connecting sourceType → targetType is valid.
 */
export function checkEdgeCompatibility(
  sourceType: string,
  targetType: string,
): EdgeValidationResult {
  const srcOutput = getOutputShape(sourceType);
  const tgtInput  = getInputShape(targetType);

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
  return {
    valid: false,
    reason: `${getLabel(sourceType)} outputs ${srcOutput.toUpperCase()} but ${getLabel(targetType)} expects ${tgtInput.toUpperCase()}`,
  };
}

/**
 * Resolve the "effective" output shape of a node by tracing back through
 * pass-through layers (any→any). Used when a pass-through node is the source.
 */
export function resolveEffectiveOutput(
  sourceNodeId: string,
  nodes: Node[],
  edges: { source: string; target: string }[],
): ShapeCategory {
  const node = nodes.find((n) => n.id === sourceNodeId);
  if (!node) return 'any';

  const layerType = node.data.layerType as string;
  const shape = getOutputShape(layerType);

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

  const layerType = node.data.layerType as string;
  const shape = getInputShape(layerType);

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
  const targetLayerType = targetDef?.data.layerType as string;
  const allowMulti = getAllowMultipleInputs(targetLayerType);

  if (!allowMulti && edges.some((e) => e.target === targetId)) {
    return { valid: false, reason: 'This node already has an input connection' };
  }

  const sourceNode = nodes.find((n) => n.id === sourceId);
  const targetNode = nodes.find((n) => n.id === targetId);
  if (!sourceNode || !targetNode) {
    return { valid: false, reason: 'Invalid node' };
  }

  const sourceType = sourceNode.data.layerType as string;
  const targetType = targetNode.data.layerType as string;

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
