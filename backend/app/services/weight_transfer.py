"""
Partial weight extraction & transfer — enables transfer learning workflows.

State-dict keys follow the pattern: ``{node_id}.{submodule}.{param}``
(e.g. ``n2.weight``, ``n4.cv1.conv.weight``).  We can filter/match by
node-ID prefix to extract or inject subsets.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import torch

from ..config import WEIGHTS_DIR
from . import weight_storage


# ── Ultralytics checkpoint unwrapping / saving ───────────────────────────────

def _save_state_dict(sd: dict, pt_path: Path) -> None:
    """Save a flat state_dict back to disk, preserving checkpoint format if the
    original file was an Ultralytics checkpoint (has a 'model' key).

    If the original was a plain state_dict, save as-is.
    If it was an Ultralytics checkpoint, load the original, update the model's
    state_dict in-place, and save back the full checkpoint so YOLO() can load it.
    """
    if not pt_path.exists():
        torch.save(sd, pt_path)
        return

    try:
        original = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception:
        torch.save(sd, pt_path)
        return

    if isinstance(original, dict) and "model" in original:
        model_obj = original["model"]
        import torch.nn as nn
        if isinstance(model_obj, nn.Module):
            # Add back 'model.' prefix that was stripped by _load_state_dict_safe
            prefixed = {f"model.{k}": v for k, v in sd.items()}
            model_obj.load_state_dict(prefixed, strict=False)
            torch.save(original, pt_path)
            return
        elif isinstance(model_obj, dict):
            prefixed = {f"model.{k}": v for k, v in sd.items()}
            original["model"] = prefixed
            torch.save(original, pt_path)
            return

    # Plain state_dict fallback
    torch.save(sd, pt_path)


def _load_state_dict_safe(pt_path: Path) -> dict:
    """Load a .pt file and return a flat state_dict.

    Handles both:
      - Plain state_dict files (legacy Model DESIGNER format)
      - Ultralytics checkpoint files (nested dict with 'model' key)
    """
    # Ensure backend/ is in sys.path so custom packages (e.g. hsg_det) are importable
    import sys as _sys
    _backend_dir = str(Path(__file__).resolve().parents[2])
    if _backend_dir not in _sys.path:
        _sys.path.insert(0, _backend_dir)
    # Register all arch plugin custom modules before loading
    try:
        from ..plugins.loader import all_arch_plugins
        for _ap in all_arch_plugins():
            try:
                _ap.register_modules()
            except Exception:
                pass
    except Exception:
        pass

    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    # Already a flat state_dict (all values are tensors)
    if isinstance(data, dict) and data and all(isinstance(v, torch.Tensor) for v in list(data.values())[:5]):
        return data

    # Ultralytics checkpoint: {'model': nn.Module or OrderedDict, ...}
    if isinstance(data, dict) and "model" in data:
        model_obj = data["model"]
        # If it's an nn.Module, extract state_dict
        if hasattr(model_obj, "state_dict"):
            sd = model_obj.state_dict()
        # If it's already a dict (state_dict)
        elif isinstance(model_obj, dict):
            sd = model_obj
        else:
            sd = {}
        # Strip Ultralytics 'model.' prefix so keys are '0.conv.weight' not 'model.0.conv.weight'
        if sd and all(k.startswith("model.") for k in list(sd.keys())[:10]):
            sd = {k[len("model."):]: v for k, v in sd.items()}
        return sd

    # Fallback: try weights_only=True for strict state_dict loading
    try:
        return torch.load(pt_path, map_location="cpu", weights_only=True)
    except Exception:
        return data if isinstance(data, dict) else {}


# ── Extraction ───────────────────────────────────────────────────────────────

def extract_partial(
    source_weight_id: str,
    node_ids: list[str],
) -> tuple[str, int]:
    """Extract a subset of a state_dict filtered to specific node IDs.

    Parameters
    ----------
    source_weight_id : str
        Weight to read from.
    node_ids : list[str]
        Node IDs whose parameters should be kept (prefix match).

    Returns
    -------
    (new_weight_id, key_count) — ID of the new partial .pt file and
    how many keys were extracted.
    """
    src_path = weight_storage.weight_pt_path(source_weight_id)
    if not src_path.exists():
        raise FileNotFoundError(f"Weight file not found: {source_weight_id}")

    sd: dict = _load_state_dict_safe(src_path)
    prefixes = tuple(f"{nid}." for nid in node_ids)

    partial_sd = {k: v for k, v in sd.items() if k.startswith(prefixes)}
    if not partial_sd:
        raise ValueError(f"No matching keys for node IDs: {node_ids}")

    new_id = uuid.uuid4().hex[:12]
    out_dir = WEIGHTS_DIR / new_id
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(partial_sd, out_dir / "weight.pt")

    # Save minimal metadata
    src_meta = weight_storage.load_weight_meta(source_weight_id) or {}
    import json, datetime
    meta = {
        "weight_id": new_id,
        "model_id": src_meta.get("model_id", ""),
        "model_name": src_meta.get("model_name", ""),
        "source_weight_id": source_weight_id,
        "extracted_node_ids": node_ids,
        "key_count": len(partial_sd),
        "partial": True,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return new_id, len(partial_sd)


# ── Transfer ─────────────────────────────────────────────────────────────────

def transfer_weights(
    source_weight_id: str,
    target_weight_id: str,
    node_id_map: dict[str, str] | None = None,
) -> tuple[int, int, list[str]]:
    """Transfer matching parameters from source into target weight file.

    Parameters
    ----------
    source_weight_id : str
        Weight to copy parameters FROM.
    target_weight_id : str
        Weight to copy parameters INTO (modified in-place).
    node_id_map : dict | None
        Optional rename mapping ``{source_node_id: target_node_id}``.
        If None, keys are matched by exact name.

    Returns
    -------
    (matched, total_target, matched_keys) — count of matched params,
    total params in target, and list of matched key names.
    """
    src_path = weight_storage.weight_pt_path(source_weight_id)
    tgt_path = weight_storage.weight_pt_path(target_weight_id)
    if not src_path.exists():
        raise FileNotFoundError(f"Source weight not found: {source_weight_id}")
    if not tgt_path.exists():
        raise FileNotFoundError(f"Target weight not found: {target_weight_id}")

    src_sd: dict = _load_state_dict_safe(src_path)
    tgt_sd: dict = _load_state_dict_safe(tgt_path)

    # Optionally rename source keys
    if node_id_map:
        renamed_sd: dict = {}
        for k, v in src_sd.items():
            new_key = k
            for old_prefix, new_prefix in node_id_map.items():
                if k.startswith(f"{old_prefix}."):
                    new_key = f"{new_prefix}.{k[len(old_prefix) + 1:]}"
                    break
            renamed_sd[new_key] = v
        src_sd = renamed_sd

    # Match: only copy keys that exist in target AND have same shape
    matched_keys: list[str] = []
    for k in tgt_sd:
        if k in src_sd and src_sd[k].shape == tgt_sd[k].shape:
            tgt_sd[k] = src_sd[k]
            matched_keys.append(k)

    # Save updated target (preserving original checkpoint format)
    _save_state_dict(tgt_sd, tgt_path)

    return len(matched_keys), len(tgt_sd), matched_keys


# ── Inspection ───────────────────────────────────────────────────────────────

def inspect_weight_keys(weight_id: str) -> list[dict]:
    """Return a list of all state_dict keys with shapes and node grouping."""
    pt_path = weight_storage.weight_pt_path(weight_id)
    if not pt_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_id}")

    sd: dict = _load_state_dict_safe(pt_path)
    result = []
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        node_id = k.split(".")[0] if "." in k else k
        result.append({
            "key": k,
            "node_id": node_id,
            "shape": list(v.shape),
            "dtype": str(v.dtype),
            "numel": v.numel(),
        })
    return result


# ── Auto-Mapping ──────────────────────────────────────────────────────────────

def _group_by_prefix(sd: dict) -> dict[str, dict[str, torch.Tensor]]:
    """Group state_dict keys by their first dotted segment (node/layer prefix)."""
    from collections import OrderedDict
    groups: dict[str, dict[str, torch.Tensor]] = OrderedDict()
    for k, v in sd.items():
        prefix = k.split(".")[0] if "." in k else k
        groups.setdefault(prefix, {})[k] = v
    return groups


def _suffix_set(keys: dict[str, torch.Tensor]) -> set[str]:
    """Extract suffixes (everything after the first dot) from a group."""
    result = set()
    for k in keys:
        if "." in k:
            result.add(k.split(".", 1)[1])
        else:
            result.add(k)
    return result


def _shape_signature(keys: dict[str, torch.Tensor]) -> list[tuple[str, list[int]]]:
    """Return sorted list of (suffix, shape) pairs for matching."""
    pairs = []
    for k, v in keys.items():
        suffix = k.split(".", 1)[1] if "." in k else k
        pairs.append((suffix, list(v.shape)))
    pairs.sort(key=lambda x: x[0])
    return pairs


def _leaf_suffix(suffix: str) -> str:
    """Extract the leaf portion of a suffix for structural matching.

    Examples::

        m.0.cv1.bn.bias       → cv1.bn.bias
        bottlenecks.0.cv1.bn.bias → cv1.bn.bias
        conv.weight           → conv.weight
        weight                → weight

    Strategy: strip leading numeric-index segments and their container name.
    e.g. ``m.0.`` or ``bottlenecks.0.`` → gone, keep the rest.
    """
    parts = suffix.split(".")
    # Skip leading segments that look like container + index patterns
    # e.g. ["m", "0", "cv1", "bn", "bias"] → start from "cv1"
    # e.g. ["bottlenecks", "0", "cv1", "bn", "bias"] → start from "cv1"
    i = 0
    while i < len(parts) - 1:
        # If current is a name and next is a digit, skip both
        if i + 1 < len(parts) and parts[i + 1].isdigit():
            i += 2
        elif parts[i].isdigit():
            i += 1
        else:
            break
    return ".".join(parts[i:]) if i < len(parts) else suffix


def _match_keys_smart(
    src_keys: dict[str, torch.Tensor],
    tgt_keys: dict[str, torch.Tensor],
    src_prefix: str,
    tgt_prefix: str,
) -> list[dict]:
    """Match source keys to target keys using multi-level strategy.

    Levels:
    1. Exact suffix match (current behavior)
    2. Leaf-suffix match (strip container/index prefixes)
    3. Index-based match by shape (positional fallback)
    """
    # Build suffix maps
    src_suf_map: dict[str, tuple[str, torch.Tensor]] = {}
    for sk, sv in src_keys.items():
        suf = sk.split(".", 1)[1] if "." in sk else sk
        src_suf_map[suf] = (sk, sv)

    tgt_suf_map: dict[str, tuple[str, torch.Tensor]] = {}
    for tk, tv in tgt_keys.items():
        suf = tk.split(".", 1)[1] if "." in tk else tk
        tgt_suf_map[suf] = (tk, tv)

    key_mappings: list[dict] = []
    matched_src: set[str] = set()
    matched_tgt: set[str] = set()

    # ── Level 1: Exact suffix match ──
    for tgt_suf in sorted(tgt_suf_map.keys()):
        if tgt_suf in src_suf_map:
            tk, tv = tgt_suf_map[tgt_suf]
            sk, sv = src_suf_map[tgt_suf]
            key_mappings.append({
                "src_key": sk, "tgt_key": tk,
                "src_shape": list(sv.shape), "tgt_shape": list(tv.shape),
                "matched": list(sv.shape) == list(tv.shape),
            })
            matched_src.add(tgt_suf)
            matched_tgt.add(tgt_suf)

    # ── Level 2: Leaf-suffix match (strip container/index prefixes) ──
    remaining_src = {s: v for s, v in src_suf_map.items() if s not in matched_src}
    remaining_tgt = {s: v for s, v in tgt_suf_map.items() if s not in matched_tgt}

    if remaining_src and remaining_tgt:
        # Build leaf-suffix maps
        src_by_leaf: dict[str, list[str]] = {}
        for suf in remaining_src:
            leaf = _leaf_suffix(suf)
            src_by_leaf.setdefault(leaf, []).append(suf)

        tgt_by_leaf: dict[str, list[str]] = {}
        for suf in remaining_tgt:
            leaf = _leaf_suffix(suf)
            tgt_by_leaf.setdefault(leaf, []).append(suf)

        for leaf in sorted(tgt_by_leaf.keys()):
            if leaf not in src_by_leaf:
                continue
            src_candidates = [s for s in src_by_leaf[leaf] if s not in matched_src]
            tgt_candidates = [t for t in tgt_by_leaf[leaf] if t not in matched_tgt]
            # Match 1:1 in order
            for s_suf, t_suf in zip(src_candidates, tgt_candidates):
                sk, sv = src_suf_map[s_suf]
                tk, tv = tgt_suf_map[t_suf]
                key_mappings.append({
                    "src_key": sk, "tgt_key": tk,
                    "src_shape": list(sv.shape), "tgt_shape": list(tv.shape),
                    "matched": list(sv.shape) == list(tv.shape),
                })
                matched_src.add(s_suf)
                matched_tgt.add(t_suf)

    # ── Level 3: Index-based match by shape ──
    remaining_src = [(s, src_suf_map[s]) for s in sorted(src_suf_map) if s not in matched_src]
    remaining_tgt = [(s, tgt_suf_map[s]) for s in sorted(tgt_suf_map) if s not in matched_tgt]

    if remaining_src and remaining_tgt:
        used_src_idx: set[int] = set()
        for t_idx, (t_suf, (tk, tv)) in enumerate(remaining_tgt):
            for s_idx, (s_suf, (sk, sv)) in enumerate(remaining_src):
                if s_idx in used_src_idx:
                    continue
                if list(sv.shape) == list(tv.shape):
                    key_mappings.append({
                        "src_key": sk, "tgt_key": tk,
                        "src_shape": list(sv.shape), "tgt_shape": list(tv.shape),
                        "matched": True,
                    })
                    used_src_idx.add(s_idx)
                    matched_src.add(s_suf)
                    matched_tgt.add(t_suf)
                    break

    # ── Add remaining unmatched ──
    for t_suf in sorted(tgt_suf_map):
        if t_suf not in matched_tgt:
            tk, tv = tgt_suf_map[t_suf]
            key_mappings.append({
                "src_key": None, "tgt_key": tk,
                "src_shape": None, "tgt_shape": list(tv.shape),
                "matched": False,
            })

    for s_suf in sorted(src_suf_map):
        if s_suf not in matched_src:
            sk, sv = src_suf_map[s_suf]
            key_mappings.append({
                "src_key": sk, "tgt_key": None,
                "src_shape": list(sv.shape), "tgt_shape": None,
                "matched": False,
            })

    # Sort: matched first, then unmatched
    key_mappings.sort(key=lambda m: (m["tgt_key"] or "", m["src_key"] or ""))
    return key_mappings


def auto_map(
    source_weight_id: str,
    target_weight_id: str,
) -> list[dict]:
    """Generate an auto-mapping preview between source and target weights.

    Strategy:
    1. Group both state_dicts by prefix (layer/node).
    2. For each target group, find the best source group by:
       a. Suffix pattern overlap (e.g. both have .conv.weight, .bn.weight)
       b. Shape compatibility (all matched suffixes have same tensor shapes)
       c. Sequential order preference (1st unmatched source ↔ 1st unmatched target)

    Returns
    -------
    list[dict], each entry:
        {
            "src_prefix": str,
            "tgt_prefix": str,
            "status": "matched" | "shape_mismatch" | "unmatched",
            "keys": [
                {"src_key": str, "tgt_key": str, "shape": list, "matched": bool},
                ...
            ]
        }
    Also includes unmatched source/target groups with status "unmatched".
    """
    src_path = weight_storage.weight_pt_path(source_weight_id)
    tgt_path = weight_storage.weight_pt_path(target_weight_id)
    if not src_path.exists():
        raise FileNotFoundError(f"Source weight not found: {source_weight_id}")
    if not tgt_path.exists():
        raise FileNotFoundError(f"Target weight not found: {target_weight_id}")

    src_sd: dict = _load_state_dict_safe(src_path)
    tgt_sd: dict = _load_state_dict_safe(tgt_path)

    return auto_map_state_dicts(src_sd, tgt_sd)


def auto_map_state_dicts(
    src_sd: dict[str, torch.Tensor],
    tgt_sd: dict[str, torch.Tensor],
) -> list[dict]:
    """Core auto-mapping logic operating on raw state_dicts."""
    src_groups = _group_by_prefix(src_sd)
    tgt_groups = _group_by_prefix(tgt_sd)

    used_src: set[str] = set()
    result: list[dict] = []

    # For each target group, find best matching source group
    for tgt_prefix, tgt_keys in tgt_groups.items():
        tgt_sigs = _shape_signature(tgt_keys)
        tgt_suffixes = _suffix_set(tgt_keys)

        best_src: str | None = None
        best_score = -1
        best_all_shapes_match = False

        for src_prefix, src_keys in src_groups.items():
            if src_prefix in used_src:
                continue

            src_suffixes = _suffix_set(src_keys)
            overlap = tgt_suffixes & src_suffixes
            if not overlap:
                continue

            # Score = number of matching suffixes with same shape
            shapes_match = 0
            shapes_total = 0
            for suffix in overlap:
                shapes_total += 1
                # Find tensors with this suffix
                src_tensor = None
                tgt_tensor = None
                for sk, sv in src_keys.items():
                    s = sk.split(".", 1)[1] if "." in sk else sk
                    if s == suffix:
                        src_tensor = sv
                        break
                for tk, tv in tgt_keys.items():
                    s = tk.split(".", 1)[1] if "." in tk else tk
                    if s == suffix:
                        tgt_tensor = tv
                        break
                if src_tensor is not None and tgt_tensor is not None:
                    if list(src_tensor.shape) == list(tgt_tensor.shape):
                        shapes_match += 1

            score = shapes_match
            all_match = (shapes_match == shapes_total == len(tgt_suffixes))

            if score > best_score:
                best_score = score
                best_src = src_prefix
                best_all_shapes_match = all_match

        if best_src is not None and best_score > 0:
            used_src.add(best_src)
            src_keys = src_groups[best_src]

            # Build key-level mapping using smart multi-level matcher
            key_mappings = _match_keys_smart(src_keys, tgt_keys, best_src, tgt_prefix)

            all_matched = all(m["matched"] for m in key_mappings if m["src_key"] and m["tgt_key"])
            has_unmatched = any(m["src_key"] is None or m["tgt_key"] is None for m in key_mappings)
            status = "matched" if all_matched and not has_unmatched else "shape_mismatch"
            result.append({
                "src_prefix": best_src,
                "tgt_prefix": tgt_prefix,
                "status": status,
                "keys": key_mappings,
            })
        else:
            # No source found for this target group
            result.append({
                "src_prefix": None,
                "tgt_prefix": tgt_prefix,
                "status": "unmatched",
                "keys": [
                    {
                        "src_key": None,
                        "tgt_key": tk,
                        "src_shape": None,
                        "tgt_shape": list(tv.shape),
                        "matched": False,
                    }
                    for tk, tv in tgt_keys.items()
                ],
            })

    # Add unmatched source groups
    for src_prefix in src_groups:
        if src_prefix not in used_src:
            src_keys = src_groups[src_prefix]
            result.append({
                "src_prefix": src_prefix,
                "tgt_prefix": None,
                "status": "unmatched",
                "keys": [
                    {
                        "src_key": sk,
                        "tgt_key": None,
                        "src_shape": list(sv.shape),
                        "tgt_shape": None,
                        "matched": False,
                    }
                    for sk, sv in src_keys.items()
                ],
            })

    return result


def apply_mapping(
    source_weight_id: str,
    target_weight_id: str,
    mapping: list[dict],
) -> dict:
    """Apply a user-confirmed key mapping from source to target weight.

    Parameters
    ----------
    source_weight_id : str
        Weight to copy parameters FROM.
    target_weight_id : str
        Weight to copy parameters INTO (modified in-place).
    mapping : list[dict]
        Each entry: ``{"src_key": str, "tgt_key": str}``.
        Only entries where both keys are non-null are applied.

    Returns
    -------
    dict with: applied (int), skipped (int), total_target (int),
               applied_keys (list[str]), applied_node_ids (list[str])
    """
    src_path = weight_storage.weight_pt_path(source_weight_id)
    tgt_path = weight_storage.weight_pt_path(target_weight_id)
    if not src_path.exists():
        raise FileNotFoundError(f"Source weight not found: {source_weight_id}")
    if not tgt_path.exists():
        raise FileNotFoundError(f"Target weight not found: {target_weight_id}")

    src_sd: dict = _load_state_dict_safe(src_path)
    tgt_sd: dict = _load_state_dict_safe(tgt_path)

    applied_keys: list[str] = []
    applied_node_ids: set[str] = set()
    skipped = 0

    for entry in mapping:
        src_key = entry.get("src_key")
        tgt_key = entry.get("tgt_key")
        if not src_key or not tgt_key:
            continue
        if src_key not in src_sd:
            skipped += 1
            continue
        if tgt_key not in tgt_sd:
            skipped += 1
            continue
        if list(src_sd[src_key].shape) != list(tgt_sd[tgt_key].shape):
            skipped += 1
            continue

        tgt_sd[tgt_key] = src_sd[src_key]
        applied_keys.append(tgt_key)
        node_id = tgt_key.split(".")[0] if "." in tgt_key else tgt_key
        applied_node_ids.add(node_id)

    # Save updated target (preserving original checkpoint format)
    _save_state_dict(tgt_sd, tgt_path)

    return {
        "applied": len(applied_keys),
        "skipped": skipped,
        "total_target": len(tgt_sd),
        "applied_keys": applied_keys,
        "applied_node_ids": sorted(applied_node_ids),
    }
