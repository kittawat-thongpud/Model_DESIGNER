"""
Convert Ultralytics YAML model definition to Model Designer graph format.

Handles:
- Backbone layers → graph nodes
- Head layers → graph nodes
- Sequential connections (e.g., Classify head)
- Multi-input connections (e.g., Concat, Detect)
"""
from __future__ import annotations
import yaml
from typing import Any


def yaml_to_graph(yaml_content: str | dict) -> dict:
    """
    Convert YAML model definition to graph format.
    
    Args:
        yaml_content: YAML string or dict with backbone/head structure
        
    Returns:
        dict with:
            - yaml_def: normalized YAML definition
            - _graph: graph metadata with nodes and edges
    """
    if isinstance(yaml_content, str):
        yaml_def = yaml.safe_load(yaml_content)
    else:
        yaml_def = yaml_content
    
    if not yaml_def:
        raise ValueError("Empty YAML content")
    
    # Extract layers
    backbone = yaml_def.get("backbone", [])
    head = yaml_def.get("head", [])
    
    # Normalize layers to dict format
    backbone_layers = _normalize_layers(backbone)
    head_layers = _normalize_layers(head)
    
    # Build graph
    nodes = []
    edges = []
    node_id = 0
    
    # Track layer index to node_id mapping
    layer_to_node = {}  # layer_index -> node_id
    
    # Process backbone
    for i, layer in enumerate(backbone_layers):
        node = _create_node(node_id, layer, i, "backbone")
        nodes.append(node)
        layer_to_node[i] = node_id
        
        # Create edges based on 'from' field
        from_indices = _resolve_from(layer["from"], i)
        for from_idx in from_indices:
            if from_idx in layer_to_node:
                edges.append({
                    "id": f"e{len(edges)}",
                    "source": layer_to_node[from_idx],
                    "target": node_id,
                })
        
        node_id += 1
    
    # Process head
    bb_len = len(backbone_layers)
    for i, layer in enumerate(head_layers):
        layer_idx = bb_len + i
        node = _create_node(node_id, layer, layer_idx, "head")
        nodes.append(node)
        layer_to_node[layer_idx] = node_id
        
        # Create edges based on 'from' field
        from_indices = _resolve_from(layer["from"], layer_idx)
        for from_idx in from_indices:
            if from_idx in layer_to_node:
                edges.append({
                    "id": f"e{len(edges)}",
                    "source": layer_to_node[from_idx],
                    "target": node_id,
                })
        
        node_id += 1
    
    # Build result
    result = {
        "nc": yaml_def.get("nc", 80),
        "scales": yaml_def.get("scales", {}),
        "kpt_shape": yaml_def.get("kpt_shape"),
        "backbone": backbone_layers,
        "head": head_layers,
        "_graph": {
            "nodes": nodes,
            "edges": edges,
        }
    }
    
    return result


def _normalize_layers(layers: list) -> list[dict]:
    """Convert layers from YAML array format to dict format."""
    result = []
    for layer in layers:
        if isinstance(layer, list):
            # [from, repeats, module, args]
            result.append({
                "from": layer[0] if len(layer) > 0 else -1,
                "repeats": layer[1] if len(layer) > 1 else 1,
                "module": str(layer[2]) if len(layer) > 2 else "Conv",
                "args": layer[3] if len(layer) > 3 and isinstance(layer[3], list) else [],
            })
        elif isinstance(layer, dict):
            # Already dict format
            result.append({
                "from": layer.get("from", layer.get("from_", -1)),
                "repeats": layer.get("repeats", 1),
                "module": layer.get("module", "Conv"),
                "args": layer.get("args", []),
            })
    return result


def _resolve_from(from_val: int | list, current_idx: int) -> list[int]:
    """
    Resolve 'from' field to absolute layer indices.
    
    Args:
        from_val: -1, or index, or list of indices
        current_idx: current layer index
        
    Returns:
        list of absolute layer indices
    """
    if isinstance(from_val, list):
        result = []
        for f in from_val:
            if f == -1:
                result.append(current_idx - 1)
            elif f < 0:
                result.append(current_idx + f)
            else:
                result.append(f)
        return result
    else:
        if from_val == -1:
            return [current_idx - 1] if current_idx > 0 else []
        elif from_val < 0:
            return [current_idx + from_val]
        else:
            return [from_val]


def _create_node(node_id: int, layer: dict, layer_idx: int, section: str) -> dict:
    """
    Create a graph node from a layer definition.
    
    Args:
        node_id: unique node ID
        layer: layer dict with from, repeats, module, args
        layer_idx: layer index in the model
        section: "backbone" or "head"
        
    Returns:
        node dict with id, type, position, data
    """
    module = layer["module"]
    args = layer["args"]
    repeats = layer["repeats"]
    
    # Auto-position nodes in a grid
    # Backbone: column 0-2, Head: column 3-5
    col = 0 if section == "backbone" else 3
    row = layer_idx if section == "backbone" else (layer_idx - 10)  # offset head rows
    
    x = col * 300 + 100
    y = row * 120 + 100
    
    # Format label
    label = f"{module}"
    if args:
        args_str = str(args)[:30]  # truncate long args
        label += f"\n{args_str}"
    if repeats > 1:
        label += f"\nx{repeats}"
    
    return {
        "id": str(node_id),
        "type": "moduleNode",
        "position": {"x": x, "y": y},
        "data": {
            "label": label,
            "module": module,
            "args": args,
            "repeats": repeats,
            "layerIndex": layer_idx,
            "section": section,
        }
    }


def import_yaml_file(file_path: str) -> dict:
    """
    Import a YAML file and convert to graph format.
    
    Args:
        file_path: path to YAML file
        
    Returns:
        dict with yaml_def and _graph
    """
    with open(file_path, 'r') as f:
        content = f.read()
    return yaml_to_graph(content)
