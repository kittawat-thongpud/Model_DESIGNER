/**
 * Shape Inference — compute the tensor shape at every node in the graph.
 *
 * Traces from the Input node through topological order, computing
 * (C, H, W) or (features,) at each step. Provides both the output
 * shape and the inferred input params for each node.
 */
import type { Node, Edge } from '@xyflow/react';
import type { LayerType } from '../types';

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

/**
 * Compute tensor shapes for every node in the graph.
 * Returns a Map from nodeId → NodeShapeInfo.
 */
export function computeGraphShapes(
  nodes: Node[],
  edges: Edge[],
): Map<string, NodeShapeInfo> {
  const result = new Map<string, NodeShapeInfo>();
  const nodeMap = new Map<string, Node>();
  nodes.forEach((n) => nodeMap.set(n.id, n));

  // Build adjacency
  const inEdge = new Map<string, string>(); // target → source
  const outEdge = new Map<string, string>(); // source → target

  for (const e of edges) {
    inEdge.set(e.target, e.source);
    outEdge.set(e.source, e.target);
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

    const layerType = node.data.layerType as LayerType;
    const params = (node.data.params || {}) as Record<string, number | string>;

    // Get input shape from predecessor
    const predId = inEdge.get(nid);
    const inputShape: TensorShape = predId && shapeAt.has(predId)
      ? { ...shapeAt.get(predId)! }
      : { kind: 'none' };

    let outputShape: TensorShape = { kind: 'none' };
    const inferred: Record<string, number> = {};

    switch (layerType) {
      case 'Input': {
        const c = Number(params.channels) || 1;
        const h = Number(params.height) || 28;
        const w = Number(params.width) || 28;
        outputShape = { kind: '2d', channels: c, height: h, width: w };
        break;
      }

      case 'Conv2d': {
        const inC = inputShape.kind === '2d' ? inputShape.channels! : 1;
        const inH = inputShape.kind === '2d' ? inputShape.height! : 28;
        const inW = inputShape.kind === '2d' ? inputShape.width! : 28;
        const outC = Number(params.out_channels) || 32;
        const ks = Number(params.kernel_size) || 3;
        const stride = Number(params.stride) || 1;
        const padding = Number(params.padding) || 0;
        const outH = Math.floor((inH + 2 * padding - ks) / stride) + 1;
        const outW = Math.floor((inW + 2 * padding - ks) / stride) + 1;
        inferred.in_channels = inC;
        outputShape = { kind: '2d', channels: outC, height: outH, width: outW };
        break;
      }

      case 'MaxPool2d': {
        const inC = inputShape.kind === '2d' ? inputShape.channels! : 1;
        const inH = inputShape.kind === '2d' ? inputShape.height! : 28;
        const inW = inputShape.kind === '2d' ? inputShape.width! : 28;
        const ks = Number(params.kernel_size) || 2;
        const stride = Number(params.stride) || ks;
        const outH = Math.floor((inH - ks) / stride) + 1;
        const outW = Math.floor((inW - ks) / stride) + 1;
        outputShape = { kind: '2d', channels: inC, height: outH, width: outW };
        break;
      }

      case 'BatchNorm2d': {
        const inC = inputShape.kind === '2d' ? inputShape.channels! : 1;
        inferred.num_features = inC;
        outputShape = { ...inputShape };
        break;
      }

      case 'Flatten': {
        if (inputShape.kind === '2d') {
          const f = inputShape.channels! * inputShape.height! * inputShape.width!;
          outputShape = { kind: '1d', features: f };
        } else {
          outputShape = { kind: '1d', features: 0 };
        }
        break;
      }

      case 'Linear': {
        const outF = Number(params.out_features) || 10;
        if (inputShape.kind === '1d') {
          inferred.in_features = inputShape.features!;
        }
        outputShape = { kind: '1d', features: outF };
        break;
      }

      case 'ReLU':
      case 'Dropout':
      case 'Softmax': {
        outputShape = { ...inputShape };
        break;
      }

      case 'Output': {
        outputShape = { kind: 'none' };
        break;
      }
    }

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
