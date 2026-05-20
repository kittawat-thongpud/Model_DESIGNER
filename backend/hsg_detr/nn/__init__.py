"""
HSG-DETR neural network module registration.

Mirrors the Mamba-YOLO pattern: injects ``SGTokenBlock`` (and
future blocks) into ``ultralytics.nn.tasks.parse_model`` via source-level
patching so ``base_modules`` recognises them and auto-injects ``[c1, c2]``.

Idempotent — safe to call ``register()`` multiple times.
"""
from __future__ import annotations

import linecache

from .sparse_global_token import (
    SGTokenBlock,
    SGStem,
    SGDown,
    RTDETRDecoderSGB,
)
from .sparse_global_token_v2 import (
    SGTokenBlockV2,
    RTDETRDecoderV2,
)
from .sparse_global_cross import CrossScaleSGA
from .cross_scale_fpn import CrossScaleFPN

# Also include hsg_det's SparseGlobalTokenBlock so that when hsg_detr
# rebuilds parse_model via source-patch, SparseGlobalTokenBlock stays
# in base_modules and gets c1 injected correctly.
try:
    from hsg_det.nn.sparse_global import (
        SparseGlobalBlock,
        SparseGlobalBlockGated,
        SparseGlobalTokenBlock,
    )
    _HSG_DET_MODULES: dict[str, type] = {
        "SparseGlobalBlock": SparseGlobalBlock,
        "SparseGlobalBlockGated": SparseGlobalBlockGated,
        "SparseGlobalTokenBlock": SparseGlobalTokenBlock,
    }
except ImportError:
    _HSG_DET_MODULES = {}

_MODULES: dict[str, type] = {
    "SGTokenBlock": SGTokenBlock,
    "SGStem": SGStem,
    "SGDown": SGDown,
    "RTDETRDecoderSGB": RTDETRDecoderSGB,
    "SGTokenBlockV2": SGTokenBlockV2,
    "RTDETRDecoderV2": RTDETRDecoderV2,
    "CrossScaleSGA": CrossScaleSGA,
    "CrossScaleFPN": CrossScaleFPN,
    **_HSG_DET_MODULES,
}

_registered = False


def _inject_globals(modules: dict[str, type]) -> None:
    """Make class names resolvable by parse_model's ``globals()[m]`` lookup."""
    try:
        import ultralytics.nn.tasks as _tasks
        for name, cls in modules.items():
            setattr(_tasks, name, cls)
    except ImportError:
        pass
    try:
        import ultralytics.nn.modules as _ult_nn
        for name, cls in modules.items():
            setattr(_ult_nn, name, cls)
    except ImportError:
        pass


def _patch_parse_model(modules: dict[str, type]) -> bool:
    """
    Rebuild ``ultralytics.nn.tasks.parse_model`` with HSG-DETR modules added
    to ``base_modules`` (so ``[c1, c2, …]`` is auto-injected).

    Returns ``True`` on success, ``False`` if the installed Ultralytics
    version has an incompatible ``parse_model`` shape.
    """
    import inspect
    import textwrap

    import ultralytics.nn.tasks as _tasks

    fn = _tasks.parse_model
    if getattr(fn, "_hsg_detr_patched", False):
        return True

    try:
        src = inspect.getsource(fn)
        src = textwrap.dedent(src)
    except Exception:
        return False

    # These modules must NOT be in base_modules — they need special
    # channel-injection paths handled by their own elif cases below.
    _special_modules = {"RTDETRDecoderSGB", "RTDETRDecoderV2", "CrossScaleSGA", "CrossScaleFPN"}
    base_names = [n for n in modules.keys() if n not in _special_modules]

    # ── 1. Insert into base_modules frozenset ────────────────────────────
    base_marker = "\n        }\n    )\n    repeat_modules"
    base_insert = "".join(f"\n            {n}," for n in base_names)
    if base_marker not in src:
        return False
    src = src.replace(base_marker, base_insert + base_marker, 1)

    # ── 2. Patch RTDETRDecoder special-case to include both variants ──────
    rtdetr_check = "        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1"
    rtdetr_patch = "        elif m is RTDETRDecoder or m is RTDETRDecoderSGB or m is RTDETRDecoderV2:  # special case, channels arg must be passed in index 1"
    if rtdetr_check in src:
        src = src.replace(rtdetr_check, rtdetr_patch, 1)
        print("[HSG-DETR] Patched RTDETRDecoder special case for RTDETRDecoderSGB + RTDETRDecoderV2")
    else:
        print("[HSG-DETR] WARNING: RTDETRDecoder check NOT FOUND in parse_model source!")

    # ── 3. Add CrossScaleSGA special-case (injects c1 list, stores c2 as list) ──
    # Insert after the RTDETRDecoder args.insert line so c2=list is set before ch.append
    rtdetr_insert_line = "            args.insert(1, [ch[x] for x in f])"
    cs2ga_case = (
        "\n        elif m is CrossScaleSGA:"
        "\n            _cs_ch = [ch[x] for x in f]"
        "\n            args.insert(0, _cs_ch)"
        "\n            args.insert(1, _cs_ch)"
        "\n            c2 = _cs_ch"
        # CrossScaleFPN special case:
        # f = [large_layer, small_layer]
        # c_kv = ch[f[0]] (large-stride, K/V source)
        # c_q  = ch[f[1]] (small-stride, Q source → output channels)
        "\n        elif m is CrossScaleFPN:"
        "\n            _csfpn_c_kv = ch[f[0]]"
        "\n            _csfpn_c_q  = ch[f[1]]"
        "\n            args.insert(0, _csfpn_c_kv)"
        "\n            args.insert(1, _csfpn_c_q)"
        "\n            c2 = _csfpn_c_q"
    )
    if rtdetr_insert_line in src:
        src = src.replace(rtdetr_insert_line, rtdetr_insert_line + cs2ga_case, 1)
        print("[HSG-DETR] Added CrossScaleSGA + CrossScaleFPN special cases to parse_model")
    else:
        print("[HSG-DETR] WARNING: RTDETRDecoder args.insert line NOT FOUND — CrossScaleSGA/CrossScaleFPN not patched!")

    # ── 4. Patch Detect args to flatten list-channels from CrossScaleSGA ──
    # Main Detect group uses args.extend([reg_max, end2end, [ch[x]...]])
    detect_args_old = "            args.extend([reg_max, end2end, [ch[x] for x in f]])"
    detect_args_new = (
        "            _dch = []\n"
        "            for _x in (f if hasattr(f, '__iter__') else [f]):\n"
        "                _c = ch[_x]\n"
        "                _dch.extend(_c if isinstance(_c, (list, tuple)) else [_c])\n"
        "            args.extend([reg_max, end2end, _dch])"
    )
    if detect_args_old in src:
        src = src.replace(detect_args_old, detect_args_new, 1)
        print("[HSG-DETR] Patched Detect args to handle CrossScaleSGA list channels")
    else:
        print("[HSG-DETR] WARNING: Detect args.extend line NOT FOUND — list channel handling not patched!")

    # v10Detect uses args.append([ch[x]...])
    v10_block_old = "        elif m is v10Detect:\n            args.append([ch[x] for x in f])"
    v10_block_new = (
        "        elif m is v10Detect:\n"
        "            _dch = []\n"
        "            for _x in (f if hasattr(f, '__iter__') else [f]):\n"
        "                _c = ch[_x]\n"
        "                _dch.extend(_c if isinstance(_c, (list, tuple)) else [_c])\n"
        "            args.append(_dch)"
    )
    if v10_block_old in src:
        src = src.replace(v10_block_old, v10_block_new, 1)
        print("[HSG-DETR] Patched v10Detect args to handle CrossScaleSGA list channels")
    else:
        print("[HSG-DETR] WARNING: v10Detect block NOT FOUND — skipping v10 patch")

    # ── 3. Compile and install the patched function ───────────────────────
    new_globals = dict(fn.__globals__)
    for name, cls in modules.items():
        new_globals[name] = cls

    try:
        filename = "<hsg_detr_parse_model_patch>"
        linecache.cache[filename] = (
            len(src),
            None,
            [line + "\n" for line in src.splitlines()],
            filename,
        )
        code = compile(src, filename, "exec")
        ns: dict = {}
        exec(code, new_globals, ns)  # noqa: S102
        if "parse_model" not in ns:
            return False
        patched = ns["parse_model"]
        patched._hsg_detr_patched = True  # type: ignore[attr-defined]
        _tasks.parse_model = patched
        return True
    except Exception as exc:
        import warnings
        warnings.warn(
            f"HSG-DETR: could not patch parse_model: {exc}", stacklevel=2
        )
        return False


def register() -> None:
    """Register HSG-DETR modules with Ultralytics.  Idempotent."""
    global _registered
    if _registered:
        return
    _inject_globals(_MODULES)
    _patch_parse_model(_MODULES)
    _registered = True
