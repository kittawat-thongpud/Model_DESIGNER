/**
 * Shape Inference — compute the tensor shape at every node in the graph.
 *
 * Data-driven from the nodeCatalogStore: each node's shapeRule and autoParams
 * are read from the catalog, so adding new node types on the backend
 * automatically enables shape inference with zero frontend code changes.
 */
import type { Node, Edge } from '@xyflow/react';
import type { ShapeRule, AutoParamDef } from '../types';
import { useNodeCatalogStore } from '../store/nodeCatalogStore';

export interface TensorShape {
  /** '2d' = (C, H, W), '1d' = (features,), 'none' = terminal */
  kind: '2d' | '1d' | 'none';
  channels?: number;
  height?: number;
  width?: number;
  features?: number;
}

export interface NodeShapeInfo {
  input: TensorShape;
  output: TensorShape;
  /** Inferred param overrides (e.g. in_channels for Conv2d) */
  inferred: Record<string, number>;
}

// ─── Shape rule interpreters ────────────────────────────────────────────────

function p(params: Record<string, number | string>, name: string | undefined, fallback: number): number {
  if (!name) return fallback;
  const v = Number(params[name]);
  return isNaN(v) ? fallback : v;
}

function applyAutoParams(
  autoParams: AutoParamDef[],
  inputShape: TensorShape,
): Record<string, number> {
  const inferred: Record<string, number> = {};
  for (const ap of autoParams) {
    if (ap.from_field === 'channels' && inputShape.kind === '2d' && inputShape.channels != null) {
      inferred[ap.name] = inputShape.channels;
    } else if (ap.from_field === 'features' && inputShape.kind === '1d' && inputShape.features != null) {
      inferred[ap.name] = inputShape.features;
    } else if (ap.from_field === 'height' && inputShape.kind === '2d' && inputShape.height != null) {
      inferred[ap.name] = inputShape.height;
    } else if (ap.from_field === 'width' && inputShape.kind === '2d' && inputShape.width != null) {
      inferred[ap.name] = inputShape.width;
    }
  }
  return inferred;
}

function applyShapeRule(
  rule: ShapeRule | null | undefined,
  inputShape: TensorShape,
  params: Record<string, number | string>,
): TensorShape {
  if (!rule) return { ...inputShape }; // no rule → passthrough

  switch (rule.rule_type) {
    case 'none_to_2d': {
      const c = p(params, 'channels', 1);
      const h = p(params, 'height', 28);
      const w = p(params, 'width', 28);
      return { kind: '2d', channels: c, height: h, width: w };
    }

    case 'conv2d': {
      const inH = inputShape.kind === '2d' ? inputShape.height! : 28;
      const inW = inputShape.kind === '2d' ? inputShape.width! : 28;
      const outC = p(params, rule.out_channels_param, 32);
      const ks = p(params, rule.kernel_param, 3);
      const stride = p(params, rule.stride_param, 1);
      const padding = p(params, rule.padding_param, 0);
      const outH = Math.floor((inH + 2 * padding - ks) / stride) + 1;
      const outW = Math.floor((inW + 2 * padding - ks) / stride) + 1;
      return { kind: '2d', channels: outC, height: outH, width: outW };
    }

    case 'pool2d': {
      const inC = inputShape.kind === '2d' ? inputShape.channels! : 1;
      const inH = inputShape.kind === '2d' ? inputShape.height! : 28;
      const inW = inputShape.kind === '2d' ? inputShape.width! : 28;
      const ks = p(params, rule.kernel_param, 2);
      const stride = p(params, rule.stride_param, ks);
      const outH = Math.floor((inH - ks) / stride) + 1;
      const outW = Math.floor((inW - ks) / stride) + 1;
      return { kind: '2d', channels: inC, height: outH, width: outW };
    }

    case 'flatten': {
      if (inputShape.kind === '2d') {
        const f = inputShape.channels! * inputShape.height! * inputShape.width!;
        return { kind: '1d', features: f };
      }
      return { kind: '1d', features: 0 };
    }

    case 'linear': {
      const outF = p(params, rule.out_features_param, 10);
      return { kind: '1d', features: outF };
    }

    case 'upsample': {
      if (inputShape.kind === '2d') {
        const scale = p(params, rule.scale_param, 2);
        return {
          kind: '2d',
          channels: inputShape.channels!,
          height: Math.floor(inputShape.height! * scale),
          width: Math.floor(inputShape.width! * scale),
        };
      }
      return { ...inputShape };
    }

    case 'passthrough':
      return { ...inputShape };

    case 'terminal':
      return { kind: 'none' };

    default:
      return { ...inputShape }; // unknown rule → passthrough
  }
}

// ─── Main shape computation ─────────────────────────────────────────────────

/**
 * Compute tensor shapes for every node in the graph.
 * Returns a Map from nodeId → NodeShapeInfo.
 */
export function computeGraphShapes(
  nodes: Node[],
  edges: Edge[],
): Map<string, NodeShapeInfo> {
  const catalog = useNodeCatalogStore.getState();
  const result = new Map<string, NodeShapeInfo>();
  const nodeMap = new Map<string, Node>();
  nodes.forEach((n) => nodeMap.set(n.id, n));

  // Build adjacency
  const inEdge = new Map<string, string>(); // target → source
  for (const e of edges) {
    inEdge.set(e.target, e.source);
  }

  // Topological sort
  const inDeg = new Map<string, number>();
  const adj = new Map<string, string[]>();
  nodes.forEach((n) => {
    inDeg.set(n.id, 0);
    adj.set(n.id, []);
  });
  for (const e of edges) {
    adj.get(e.source)?.push(e.target);
    inDeg.set(e.target, (inDeg.get(e.target) || 0) + 1);
  }
  const queue: string[] = [];
  inDeg.forEach((deg, id) => { if (deg === 0) queue.push(id); });
  const sorted: string[] = [];
  while (queue.length > 0) {
    const nid = queue.shift()!;
    sorted.push(nid);
    for (const nb of adj.get(nid) || []) {
      const d = (inDeg.get(nb) || 1) - 1;
      inDeg.set(nb, d);
      if (d === 0) queue.push(nb);
    }
  }

  // Shape propagation
  const shapeAt = new Map<string, TensorShape>();

  for (const nid of sorted) {
    const node = nodeMap.get(nid);
    if (!node) continue;

    const layerType = node.data.layerType as string;
    const params = (node.data.params || {}) as Record<string, number | string>;

    // Get input shape from predecessor
    const predId = inEdge.get(nid);
    const inputShape: TensorShape = predId && shapeAt.has(predId)
      ? { ...shapeAt.get(predId)! }
      : { kind: 'none' };

    // Auto-infer params from upstream shape
    const autoParams = catalog.getAutoParams(layerType);
    const inferred = applyAutoParams(autoParams, inputShape);

    // Apply shape rule to compute output shape
    const shapeRule = catalog.getShapeRule(layerType);
    const outputShape = applyShapeRule(shapeRule, inputShape, params);

    shapeAt.set(nid, outputShape);
    result.set(nid, { input: inputShape, output: outputShape, inferred });
  }

  return result;
}

/**
 * Format a TensorShape as a display string like "[32, 14, 14]" or "[128]".
 */
export function formatShape(shape: TensorShape): string {
  if (shape.kind === '2d') {
    return `[${shape.channels}, ${shape.height}, ${shape.width}]`;
  }
  if (shape.kind === '1d') {
    return `[${shape.features}]`;
  }
  return '—';
}
