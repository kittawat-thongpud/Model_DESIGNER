"""
HSG-DETR neural network module registration.

Mirrors the Mamba-YOLO pattern: injects ``SGTokenBlock`` (and
future blocks) into ``ultralytics.nn.tasks.parse_model`` via source-level
patching so ``base_modules`` recognises them and auto-injects ``[c1, c2]``.

Idempotent — safe to call ``register()`` multiple times.
"""
from __future__ import annotations

from .sparse_global_token import (
    SGTokenBlock,
    SGStem,
    SGDown,
    RTDETRDecoderSGB,
)

_MODULES: dict[str, type] = {
    "SGTokenBlock": SGTokenBlock,
    "SGStem": SGStem,
    "SGDown": SGDown,
    "RTDETRDecoderSGB": RTDETRDecoderSGB,
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

    # RTDETRDecoderSGB must NOT be in base_modules — it needs the special
    # RTDETRDecoder channel-injection path (args.insert(1, [ch[x] for x in f]))
    base_names = [n for n in modules.keys() if n != "RTDETRDecoderSGB"]

    # ── 1. Insert into base_modules frozenset ────────────────────────────
    base_marker = "\n        }\n    )\n    repeat_modules"
    base_insert = "".join(f"\n            {n}," for n in base_names)
    if base_marker not in src:
        return False
    src = src.replace(base_marker, base_insert + base_marker, 1)

    # ── 2. Patch RTDETRDecoder special-case to include RTDETRDecoderSGB ───
    rtdetr_check = "        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1"
    rtdetr_patch = "        elif m is RTDETRDecoder or m is RTDETRDecoderSGB:  # special case, channels arg must be passed in index 1"
    if rtdetr_check in src:
        src = src.replace(rtdetr_check, rtdetr_patch, 1)
        print("[HSG-DETR] Patched RTDETRDecoder special case for RTDETRDecoderSGB")
    else:
        print("[HSG-DETR] WARNING: RTDETRDecoder check NOT FOUND in parse_model source!")

    # ── 3. Compile and install the patched function ───────────────────────
    new_globals = dict(fn.__globals__)
    for name, cls in modules.items():
        new_globals[name] = cls

    try:
        code = compile(src, "<hsg_detr_parse_model_patch>", "exec")
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
