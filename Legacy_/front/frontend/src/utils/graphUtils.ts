// Simplified Topological Sort
// We use any for Node/Edge to avoid dependency on reactflow if not strictly needed or missing

/**
 * Perform a topological sort on the graph nodes to determine execution order.
 * Returns an array of node IDs in topological order.
 */
export const getTopologicalSort = (nodes: any[], edges: any[]): string[] => {
  const adjacencyList = new Map<string, string[]>();
  const inDegree = new Map<string, number>();

  // Initialize graph
  nodes.forEach(node => {
    adjacencyList.set(node.id, []);
    inDegree.set(node.id, 0);
  });

  // Build graph
  edges.forEach(edge => {
    if (adjacencyList.has(edge.source) && adjacencyList.has(edge.target)) {
      adjacencyList.get(edge.source)?.push(edge.target);
      inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1);
    }
  });

  // Kahn's algorithm
  const queue: string[] = [];
  inDegree.forEach((degree, id) => {
    if (degree === 0) {
      queue.push(id);
    }
  });

  // Sort queue by node position (y-coordinate) to have deterministic order for parallel branches
  // This helps keep 'Input' usually at the top if multiple starts exist
  queue.sort((a, b) => {
    const nodeA = nodes.find(n => n.id === a);
    const nodeB = nodes.find(n => n.id === b);
    return (nodeA?.position.y || 0) - (nodeB?.position.y || 0);
  });

  const result: string[] = [];
  
  while (queue.length > 0) {
    const u = queue.shift()!;
    result.push(u);

    const neighbors = adjacencyList.get(u) || [];
    
    // Sort neighbors by position too for deterministic traversal
    neighbors.sort((a, b) => {
        const nodeA = nodes.find(n => n.id === a);
        const nodeB = nodes.find(n => n.id === b);
        return (nodeA?.position.y || 0) - (nodeB?.position.y || 0);
    });

    for (const v of neighbors) {
      inDegree.set(v, (inDegree.get(v) || 0) - 1);
      if (inDegree.get(v) === 0) {
        queue.push(v);
      }
    }
  }

  // If graph has cycles, result might stick. 
  // For training config, we just want a "best effort" sort.
  // Append any remaining nodes that weren't visited (cycles)
  if (result.length !== nodes.length) {
    const visited = new Set(result);
    const remaining = nodes.filter(n => !visited.has(n.id)).map(n => n.id);
    result.push(...remaining);
  }

  return result;
};
