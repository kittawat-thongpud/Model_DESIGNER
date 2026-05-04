"""
MambaYOLO nn package.

Loads SimpleStem, VisionClueMerge, VSSBlock, XSSBlock directly from the
official HZAI-ZJNU/Mamba-YOLO repository (cloned by the installer) and
registers them with Ultralytics so they are resolved by parse_model.

The standard ultralytics parse_model function hardcodes which modules go
into its local `base_modules` / `repeat_modules` frozensets.  Because
those are local variables we cannot modify from outside, we rebuild
parse_model via source-code patching (inspect.getsource + compile/exec)
to add our four modules exactly as the upstream Mamba-YOLO fork does.
"""
from __future__ import annotations

import importlib.util
import linecache
import sys
import types
from pathlib import Path

# ── Repo location (mirrors installer._repo_dir) ───────────────────────────────

def _repo_dir() -> Path:
    try:
        from app.config import DATA_DIR
        return DATA_DIR / "vendor" / "Mamba-YOLO"
    except ImportError:
        return Path(__file__).resolve().parents[2] / "data" / "vendor" / "Mamba-YOLO"


# ── Load modules from cloned repo ─────────────────────────────────────────────

_VENDOR_PKG = "_mamba_yolo_vendor"


def _load_mamba_modules() -> dict:
    """
    Import SimpleStem, VisionClueMerge, VSSBlock, XSSBlock from the
    cloned HZAI-ZJNU/Mamba-YOLO repo using a private namespace package
    so we never shadow the installed ultralytics package.

    Returns a dict {name: class}.
    Raises ImportError if the repo is not yet cloned.
    """
    modules_dir = _repo_dir() / "ultralytics" / "nn" / "modules"
    if not modules_dir.exists():
        raise ImportError(
            f"Mamba-YOLO repo not found at {_repo_dir()}. "
            "Please run the Mamba-YOLO installer first."
        )

    # ── Create a private top-level package backed by modules_dir ─────────────
    if _VENDOR_PKG not in sys.modules:
        pkg = types.ModuleType(_VENDOR_PKG)
        pkg.__path__ = [str(modules_dir)]   # type: ignore[attr-defined]
        pkg.__package__ = _VENDOR_PKG
        pkg.__spec__ = importlib.util.spec_from_loader(_VENDOR_PKG, loader=None)
        sys.modules[_VENDOR_PKG] = pkg

    # ── Evict any stale / partially-loaded vendor modules ────────────────────
    # If a previous attempt crashed mid-exec (e.g. selective_scan missing),
    # the module is already in sys.modules but lacks required attributes.
    # Remove it so the load is retried cleanly.
    _common_key = f"{_VENDOR_PKG}.common_utils_mbyolo"
    if _common_key in sys.modules and not hasattr(sys.modules[_common_key], "SelectiveScanCore"):
        sys.modules.pop(_common_key)

    _yolo_key = f"{_VENDOR_PKG}.mamba_yolo"
    if _yolo_key in sys.modules and not hasattr(sys.modules[_yolo_key], "SimpleStem"):
        sys.modules.pop(_yolo_key)

    # ── Stub any missing selective_scan CUDA extension modules ───────────────
    # common_utils_mbyolo.py has an unconditional `import selective_scan_cuda`
    # in the except-branch of its second try-block (line 27).  If the CUDA
    # extension was never built, that bare import raises ImportError before
    # SelectiveScanCore is defined (line 100), causing the NameError seen in
    # mamba_yolo.py's default argument.  We pre-populate sys.modules with
    # empty stubs so the import succeeds; real modules override them if built.
    _SS_CUDA_NAMES = (
        "selective_scan_cuda_core",
        "selective_scan_cuda",
        "selective_scan_cuda_oflex",
        "selective_scan_cuda_ndstate",
        "selective_scan_cuda_nrow",
    )
    for _ssname in _SS_CUDA_NAMES:
        if _ssname not in sys.modules:
            try:
                importlib.import_module(_ssname)
            except ImportError:
                sys.modules[_ssname] = types.ModuleType(_ssname)

    # ── Load common_utils_mbyolo (provides LayerNorm2d, cross_selective_scan …) ──
    common_name = f"{_VENDOR_PKG}.common_utils_mbyolo"
    if common_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            common_name,
            modules_dir / "common_utils_mbyolo.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        mod.__package__ = _VENDOR_PKG
        sys.modules[common_name] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

    # Ultralytics runs a CPU dummy forward during model construction to infer
    # strides. Mamba's SelectiveScanCore is CUDA-only and crashes on that probe.
    # Patch a CPU-safe fast path for probe-time tensors (shape-preserving).
    _common_mod = sys.modules[common_name]
    _ssc = getattr(_common_mod, "SelectiveScanCore", None)
    if _ssc is not None and not getattr(_ssc, "_md_cpu_probe_patched", False):
        _orig_forward = _ssc.forward

        @staticmethod
        def _forward_cpu_safe(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
            if not getattr(u, "is_cuda", False):
                return u
            return _orig_forward(ctx, u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, oflex)

        _ssc.forward = _forward_cpu_safe
        _ssc._md_cpu_probe_patched = True

    # ── Load mamba_yolo (defines the four public classes) ────────────────────
    yolo_name = f"{_VENDOR_PKG}.mamba_yolo"
    if yolo_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            yolo_name,
            modules_dir / "mamba_yolo.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        mod.__package__ = _VENDOR_PKG
        # The vendor file does `from .common_utils_mbyolo import *`, but some
        # upstream versions define `__all__` without SelectiveScanCore.  That
        # leaves default args like `SelectiveScan=SelectiveScanCore` unresolved
        # during module execution.  Seed the namespace from common_utils first
        # so those definitions are always available.
        for _name, _value in _common_mod.__dict__.items():
            if not _name.startswith("__"):
                mod.__dict__.setdefault(_name, _value)
        sys.modules[yolo_name] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        # ^ triggers `from .common_utils_mbyolo import *` which resolves
        #   to the already-loaded sys.modules entry above ✓

    yolo_mod = sys.modules[yolo_name]
    return {
        "SimpleStem":       yolo_mod.SimpleStem,
        "VisionClueMerge":  yolo_mod.VisionClueMerge,
        "VSSBlock":         yolo_mod.VSSBlock,
        "XSSBlock":         yolo_mod.XSSBlock,
    }


# ── Monkey-patch parse_model ──────────────────────────────────────────────────

def _patch_parse_model(modules: dict) -> bool:
    """
    Rebuild ultralytics.nn.tasks.parse_model with Mamba modules added to:
      • base_modules  — so parse_model injects [c1, c2, *extra_args]
      • repeat_modules — for XSSBlock (n becomes an explicit arg, n→1)

    This mirrors exactly what HZAI-ZJNU/Mamba-YOLO's forked tasks.py does.
    Returns True on success, False if patching was not possible.
    """
    import inspect
    import textwrap
    import ultralytics.nn.tasks as _tasks

    fn = _tasks.parse_model
    if getattr(fn, "_mamba_patched", False):
        return True  # already patched

    try:
        src = inspect.getsource(fn)
        src = textwrap.dedent(src)
    except Exception:
        return False

    base_names = list(modules.keys())

    # ── 1. Insert into base_modules frozenset ─────────────────────────────────
    # Pattern (8-space '}', 4-space ')' then '    repeat_modules'):
    base_marker = "\n        }\n    )\n    repeat_modules"
    base_insert = "".join(f"\n            {n}," for n in base_names)
    if base_marker not in src:
        return False
    src = src.replace(base_marker, base_insert + base_marker, 1)

    # ── 2. Insert XSSBlock into repeat_modules frozenset ─────────────────────
    # Pattern (8-space '}', 4-space ')' then '    for i,'):
    repeat_marker = "\n        }\n    )\n    for i,"
    repeat_insert = "\n            XSSBlock,"
    if repeat_marker in src:
        src = src.replace(repeat_marker, repeat_insert + repeat_marker, 1)

    # ── 3. Compile and install the patched function ───────────────────────────
    new_globals = dict(fn.__globals__)
    for name, cls in modules.items():
        new_globals[name] = cls

    try:
        filename = "<mamba_parse_model_patch>"
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
        patched._mamba_patched = True
        _tasks.parse_model = patched
        return True
    except Exception as exc:
        import warnings
        warnings.warn(f"Mamba-YOLO: could not patch parse_model: {exc}", stacklevel=2)
        return False


# ── Inject names into ultralytics globals ────────────────────────────────────

def _inject_globals(modules: dict) -> None:
    """Make class names resolvable by parse_model's globals()[m] lookup."""
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


def _parse_model_has_modules(modules: dict) -> bool:
    """Return True when the live parser can resolve this plugin's modules."""
    try:
        import ultralytics.nn.tasks as _tasks
    except ImportError:
        return False
    parser_globals = getattr(_tasks.parse_model, "__globals__", {})
    return all(parser_globals.get(name) is cls for name, cls in modules.items())


# ── Public entry point ────────────────────────────────────────────────────────

_registered = False


def register() -> None:
    """
    Load Mamba-YOLO modules from the cloned repo and register them with
    Ultralytics.  Idempotent — safe to call multiple times.

    Raises ImportError if the repo has not been cloned yet.
    """
    global _registered
    modules = _load_mamba_modules()
    if _registered and _parse_model_has_modules(modules):
        return
    _inject_globals(modules)
    _patch_parse_model(modules)
    _registered = True
