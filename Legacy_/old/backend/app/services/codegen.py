"""
Code generation service — converts a model graph JSON into a PyTorch nn.Module class.
Uses topological sort on edges to determine forward() execution order.
Refactored to support branching and merging (Concatenate).
"""
from __future__ import annotations
from ..schemas.model_schema import ModelGraph, NodeSchema


# ─── Layer type → PyTorch code templates ──────────────────────────────────────

LAYER_TEMPLATES: dict[str, str] = {
    "Conv2d": "nn.Conv2d(in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})",
    "ReLU": "nn.ReLU()",
    "MaxPool2d": "nn.MaxPool2d(kernel_size={kernel_size}, stride={stride})",
    "Flatten": "nn.Flatten()",
    "Linear": "nn.Linear(in_features={in_features}, out_features={out_features})",
    "BatchNorm2d": "nn.BatchNorm2d(num_features={num_features})",
    "Dropout": "nn.Dropout(p={p})",
    "Softmax": "nn.Softmax(dim={dim})",
    "Upsample": "nn.Upsample(scale_factor={scale_factor}, mode='{mode}')",
    # Concatenate is functional, handled separately
}

# Default params for layer types that might have missing optional fields
DEFAULT_PARAMS: dict[str, dict] = {
    "Conv2d": {"stride": 1, "padding": 0},
    "MaxPool2d": {"stride": None},
    "Dropout": {"p": 0.5},
    "Softmax": {"dim": 1},
    "Upsample": {"scale_factor": 2.0, "mode": "nearest"},
    "Concatenate": {"dim": 1},
}


def topological_sort(nodes: list[NodeSchema], edges: list) -> list[str]:
    """Return node IDs in topological order based on edges."""
    id_set = {n.id for n in nodes}
    in_degree: dict[str, int] = {n.id: 0 for n in nodes}
    adj: dict[str, list[str]] = {n.id: [] for n in nodes}

    for e in edges:
        if e.source in id_set and e.target in id_set:
            adj[e.source].append(e.target)
            in_degree[e.target] += 1

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    result: list[str] = []

    while queue:
        nid = queue.pop(0)
        result.append(nid)
        for neighbor in adj[nid]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result


def _sanitize_class_name(name: str) -> str:
    """Convert model name to a valid Python class name."""
    cleaned = "".join(c if c.isalnum() else "_" for c in name)
    if cleaned and cleaned[0].isdigit():
        cleaned = "Model_" + cleaned
    return cleaned or "GeneratedModel"


def _get_input_node(graph: ModelGraph) -> NodeSchema | None:
    """Find the Input node in the graph."""
    for n in graph.nodes:
        if n.type == "Input":
            return n
    return None


def _compute_shapes(
    graph: ModelGraph,
    sorted_ids: list[str],
    input_shape: tuple[int, int, int]
) -> dict[str, dict]:
    """
    Walk through layers in topological order and compute the tensor shape at each node output.
    Returns a dict mapping node_id -> inferred_params (for that layer).
    
    We also track output_shapes map: node_id -> (c, h, w) or (features,)
    """
    node_map = {n.id: n for n in graph.nodes}
    
    # Adjacency for inputs (reverse edges)
    adj_in: dict[str, list[str]] = {n.id: [] for n in graph.nodes}
    for e in graph.edges:
        if e.target in adj_in:
            adj_in[e.target].append(e.source)

    # Track output shapes: node_id -> dict(c, h, w, flat_features, is_flat)
    # Using a structured dict to handle both 2D and 1D states
    # ShapeState: { c: int, h: int, w: int, flat: bool }
    shapes: dict[str, dict] = {}
    
    # Initialize Input node(s)
    c_in, h_in, w_in = input_shape
    for nid in sorted_ids:
        node = node_map[nid]
        if node.type == "Input":
            # Input params override defaults if present
            c = int(node.params.get("channels", c_in))
            h = int(node.params.get("height", h_in))
            w = int(node.params.get("width", w_in))
            shapes[nid] = {"c": c, "h": h, "w": w, "flat": False}
    
    inferred_params_map: dict[str, dict] = {}

    for nid in sorted_ids:
        node = node_map[nid]
        if node.type == "Input":
            continue
            
        # Get input shape from previous node(s)
        sources = adj_in.get(nid, [])
        if not sources:
            # Orphan node, skip or assume default? 
            # We'll assume default (1, 28, 28) if no input, or skip
            shapes[nid] = {"c": 1, "h": 28, "w": 28, "flat": False}
            continue

        # For most layers, we take the shape of the first input
        # For Concatenate, we merge
        # For now, simplistic approach: take first source
        parent_shape = shapes.get(sources[0], {"c": 1, "h": 28, "w": 28, "flat": False})
        c, h, w, is_flat = parent_shape["c"], parent_shape["h"], parent_shape["w"], parent_shape["flat"]
        
        user_params = {**node.params}
        user_params.pop("_datasetSource", None)
        inferred: dict = {}
        
        layer_type = node.type
        
        if layer_type == "Conv2d":
            out_channels = int(user_params.get("out_channels", 32))
            kernel_size = int(user_params.get("kernel_size", 3))
            stride = int(user_params.get("stride", 1))
            padding = int(user_params.get("padding", 0))

            inferred = {
                "in_channels": c,
                "out_channels": out_channels,
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
            }
            h = (h + 2 * padding - kernel_size) // stride + 1
            w = (w + 2 * padding - kernel_size) // stride + 1
            c = out_channels
            is_flat = False

        elif layer_type == "MaxPool2d":
            kernel_size = int(user_params.get("kernel_size", 2))
            stride_val = user_params.get("stride", kernel_size)
            stride = int(stride_val) if stride_val is not None else kernel_size

            inferred = {"kernel_size": kernel_size, "stride": stride}
            h = (h - kernel_size) // stride + 1
            w = (w - kernel_size) // stride + 1
            is_flat = False

        elif layer_type == "BatchNorm2d":
            inferred = {"num_features": c}
            
        elif layer_type == "Flatten":
            is_flat = True
            inferred = {}
            # c becomes flat features count, h/w irrelevant
            c = c * h * w
            h, w = 1, 1

        elif layer_type == "Linear":
            out_features = int(user_params.get("out_features", 10))
            
            if is_flat:
                # Auto-infer input features from the flatten shape
                in_f = c
            else:
                # Not flat? Respect user param or default to c (last dim)
                in_f = int(user_params.get("in_features", 0))
                if in_f == 0:
                    in_f = c
            
            inferred = {"in_features": in_f, "out_features": out_features}
            c = out_features
            h, w = 1, 1
            is_flat = True

        elif layer_type == "Dropout":
            p = user_params.get("p", 0.5)
            inferred = {"p": p}

        elif layer_type == "Softmax":
            dim = user_params.get("dim", 1)
            inferred = {"dim": dim}

        elif layer_type == "ReLU":
            inferred = {}
            
        elif layer_type == "Upsample":
            scale = float(user_params.get("scale_factor", 2.0))
            mode = str(user_params.get("mode", "nearest"))
            inferred = {"scale_factor": scale, "mode": mode}
            h = int(h * scale)
            w = int(w * scale)
            is_flat = False
            
        elif layer_type == "Concatenate":
            dim = int(user_params.get("dim", 1))
            inferred = {"dim": dim}
            # Calculate output shape based on dim
            # If dim=1 (Channels), sum channels. H/W must match.
            total_c = 0
            # Check all sources
            for sid in sources:
                s_shape = shapes.get(sid)
                if s_shape:
                    total_c += s_shape["c"]
                    # Assume h, w match for valid graph
            
            c = total_c
            # Keep h, w from first parent
        
        shapes[nid] = {"c": c, "h": h, "w": w, "flat": is_flat}
        inferred_params_map[nid] = inferred

    return inferred_params_map


def generate_code(graph: ModelGraph) -> tuple[str, str]:
    """
    Generate PyTorch nn.Module code from a model graph.
    Supports branching and merging via variable name tracking.
    """
    class_name = _sanitize_class_name(graph.meta.name)
    node_map = {n.id: n for n in graph.nodes}
    sorted_ids = topological_sort(graph.nodes, graph.edges)
    
    # ── Build globals lookup ──────────────────────────────────────────────────
    globals_map = {g.name: g.value for g in graph.globals}
    globals_type_map = {g.name: g.type for g in graph.globals}

    # ── Find used globals ─────────────────────────────────────────────────────
    used_globals: set[str] = set()
    param_global_refs: dict[str, dict[str, str]] = {}
    
    for node in graph.nodes:
        refs: dict[str, str] = {}
        for key, val in node.params.items():
            if isinstance(val, str) and val.startswith("$"):
                var_name = val[1:]
                if var_name in globals_map:
                    used_globals.add(var_name)
                    refs[key] = var_name
        param_global_refs[node.id] = refs
        if node.enabled_by_global and node.enabled_by_global in globals_map:
            used_globals.add(node.enabled_by_global)

    # ── Resolve globals for shape inference ───────────────────────────────────
    # We clone nodes to verify we don't mutate input graph destructively for caller?
    # Actually graph is passed by value (Pydantic model)? No, ref.
    # But checking input constraints, we should be fine modifying params locally if needed.
    # For now, we just rely on the fact that _compute_shapes reads params.
    # We should inject global values into params for _compute_shapes to work correctly with dynamic values?
    # Code gen keeps $var, but shape inference needs numbers.
    
    # Apply global values to params for shape calculation
    for node in graph.nodes:
        for key, val in node.params.items():
            if isinstance(val, str) and val.startswith("$"):
                var_name = val[1:]
                if var_name in globals_map:
                    node.params[key] = globals_map[var_name]

    # ── Input Shape ───────────────────────────────────────────────────────────
    input_node = _get_input_node(graph)
    if input_node:
        in_c = int(input_node.params.get("channels", 1))
        in_h = int(input_node.params.get("height", 28))
        in_w = int(input_node.params.get("width", 28))
    else:
        in_c, in_h, in_w = 1, 28, 28

    # ── Infer Shapes ──────────────────────────────────────────────────────────
    inferred_params = _compute_shapes(graph, sorted_ids, (in_c, in_h, in_w))
    
    # ── Build Connected Set ───────────────────────────────────────────────────
    # Adjacencies
    adj_out: dict[str, list[str]] = {n.id: [] for n in graph.nodes}
    adj_in: dict[str, list[str]] = {n.id: [] for n in graph.nodes}
    for e in graph.edges:
        if e.source in node_map and e.target in node_map:
            adj_out[e.source].append(e.target)
            adj_in[e.target].append(e.source)
            
    # Filter reachable from Input
    connected: set[str] = set()
    if input_node:
        q = [input_node.id]
        connected.add(input_node.id)
        while q:
            curr = q.pop(0)
            for neighbor in adj_out[curr]:
                if neighbor not in connected:
                    connected.add(neighbor)
                    q.append(neighbor)
    
    # Filter active layers (exclude Input, include only connected)
    active_layers = [nid for nid in sorted_ids if nid in connected and node_map[nid].type != "Input"]

    # ── Build __init__ ────────────────────────────────────────────────────────
    init_lines = []
    
    # Store globals
    for gname in sorted(used_globals):
        init_lines.append(f"        self.{gname} = {gname}")

    # Create distinct layers
    layer_var_map: dict[str, str] = {} # node_id -> self.layer_name
    
    for i, nid in enumerate(active_layers):
        node = node_map[nid]
        if node.type == "Concatenate":
            # Functional, no init needed
            continue
            
        # Use node ID as variable name to allow mapping back to frontend
        # node_ids are like "node_1", which are valid python identifiers
        layer_name = nid
        layer_var_map[nid] = layer_name
        
        template = LAYER_TEMPLATES.get(node.type)
        if not template:
            init_lines.append(f"        # Unsupported: {node.type}")
            continue
            
        # Merge params
        defaults = DEFAULT_PARAMS.get(node.type, {})
        inferred = inferred_params.get(nid, {})
        params = {**defaults, **inferred}
        
        # Override with globals
        refs = param_global_refs.get(nid, {})
        for k, v in refs.items():
            params[k] = v
            
        try:
            code_str = template.format(**params)
        except KeyError as e:
            code_str = f"None # Error: missing param {e}"
            
        if node.enabled_by_global:
             init_lines.append(f"        self.{layer_name} = {code_str} if {node.enabled_by_global} else None")
        else:
             init_lines.append(f"        self.{layer_name} = {code_str}")

    # ── Init Args ─────────────────────────────────────────────────────────────
    init_kwargs = []
    for gname in sorted(used_globals):
        val = globals_map[gname]
        default = f'"{val}"' if isinstance(val, str) else str(val)
        if isinstance(val, bool): default = str(val)
        init_kwargs.append(f"{gname}={default}")
    
    init_sig = ", ".join(init_kwargs)
    if init_sig: init_sig = ", " + init_sig

    # ── Build forward ─────────────────────────────────────────────────────────
    forward_lines = []
    
    # Map node_id -> output variable name
    # Input node output is 'x'
    var_map: dict[str, str] = {}
    if input_node:
        var_map[input_node.id] = "x"
    
    for nid in active_layers:
        node = node_map[nid]
        sources = adj_in.get(nid, [])
        
        # Determine input variable(s)
        if not sources:
            # Should not happen if connected check passed
            input_var = "x" 
        elif len(sources) == 1:
            input_var = var_map.get(sources[0], "x")
        else:
            # Multiple inputs, used for Concatenate
            input_vars = [var_map.get(s, "x") for s in sources]
        
        output_var = f"x_{nid}"
        var_map[nid] = output_var
        
        if node.type == "Concatenate":
            dim = node.params.get("dim", 1)
            # Globals?
            if "dim" in param_global_refs.get(nid, {}):
                dim = f"self.{param_global_refs[nid]['dim']}"
                
            input_list = ", ".join(input_vars) # type: ignore
            forward_lines.append(f"        {output_var} = torch.cat([{input_list}], dim={dim})")
            
        elif node.type == "Output":
             # Output node is identity but sets final return
             # Usually we just return the input to Output node
             # But if graph continues? 
             # For now, just identity.
             forward_lines.append(f"        {output_var} = {input_var}")
        else:
            layer_name = layer_var_map.get(nid)
            if node.enabled_by_global:
                forward_lines.append(f"        if self.{layer_name} is not None:")
                forward_lines.append(f"            {output_var} = self.{layer_name}({input_var})")
                forward_lines.append(f"        else:")
                forward_lines.append(f"            {output_var} = {input_var}")
            else:
                forward_lines.append(f"        {output_var} = self.{layer_name}({input_var})")

    # Final return
    # Find Output node(s)
    outputs = [n for n in graph.nodes if n.type == "Output"]
    if outputs:
        # Return output of last Output node
        # Sort by topology?
        # Just take the last one in sorted_ids
        for nid in reversed(sorted_ids):
            if node_map[nid].type == "Output":
                final_var = var_map.get(nid, "x")
                forward_lines.append(f"        return {final_var}")
                break
    else:
        # If no output node, return last computed var
        if active_layers:
            last_nid = active_layers[-1]
            forward_lines.append(f"        return {var_map[last_nid]}")
        else:
            forward_lines.append(f"        return x")

    # ── Globals Comment ───────────────────────────────────────────────────────
    globals_comment = ""
    if used_globals:
        glines = []
        for g in sorted(used_globals):
            glines.append(f"#   {g} ({globals_type_map[g]}) = {globals_map[g]}")
        globals_comment = "# ─── Global Config ────────────────────────────────────────────────────────────\n" + "\n".join(glines) + "\n\n"

    code = f'''import torch
import torch.nn as nn

{globals_comment}class {class_name}(nn.Module):
    """Auto-generated by Model DESIGNER."""

    def __init__(self{init_sig}):
        super().__init__()
{chr(10).join(init_lines)}

    def forward(self, x):
{chr(10).join(forward_lines)}
'''
    return class_name, code
