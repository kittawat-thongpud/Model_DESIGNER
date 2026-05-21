"""
Plugin management endpoints.

Routes
------
GET  /api/plugins/archs
GET  /api/plugins/mamba_yolo/install-status
POST /api/plugins/mamba_yolo/install
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/plugins")


# ── Arch plugins list ────────────────────────────────────────────────────────

@router.get("/archs")
def list_arch_plugins():
    """Return model architecture families (each family = one model entry with selectable scales)."""
    from ..plugins.loader import arch_families, discover_plugins
    discover_plugins()
    return arch_families()


@router.get("/archs/{plugin_name}/config-fields")
def get_arch_config_fields(plugin_name: str):
    """Return arch-specific training config field schema for dynamic UI rendering.

    ``plugin_name`` accepts a concrete plugin name (``yolo26_cs2ga_n``) or
    a family key (``yolo26_cs2ga``).  Returns an empty list for plugins that
    have not registered custom fields — the UI omits the "Model" tab in that case.
    """
    from ..plugins.loader import resolve_arch_plugin, discover_plugins
    discover_plugins()
    plugin = resolve_arch_plugin(plugin_name)
    if plugin is None:
        from fastapi import HTTPException
        raise HTTPException(404, f"Arch plugin not found: {plugin_name!r}")
    fields = plugin.get_config_fields()
    return [f.to_dict() for f in fields]


@router.get("/archs/{plugin_name}/profiles")
def get_arch_training_profiles(plugin_name: str):
    """Return the named training profiles registered by an arch plugin.

    ``plugin_name`` is the concrete plugin name (e.g. ``yolo26_cs2ga_n``) or
    the family key (e.g. ``yolo26_cs2ga``).  If the plugin has no profiles an
    empty list is returned — the caller should fall back to standard training.
    """
    from ..plugins.loader import resolve_arch_plugin, discover_plugins
    discover_plugins()
    plugin = resolve_arch_plugin(plugin_name)
    if plugin is None:
        from fastapi import HTTPException
        raise HTTPException(404, f"Arch plugin not found: {plugin_name!r}")
    profiles = plugin.get_training_profiles()
    return [p.to_dict() for p in profiles]


# ── MambaYOLO selective_scan ──────────────────────────────────────────────────

@router.get("/mamba_yolo/install-status")
def mamba_yolo_install_status():
    """Return the current status of the selective_scan CUDA extension installer."""
    try:
        from mamba_yolo import installer
        status = installer.get_status()
        log_tail = installer.get_log_tail(n=200)
        return JSONResponse({
            "ok": True,
            "status": status["status"],
            "started_at": status["started_at"],
            "finished_at": status["finished_at"],
            "error": status["error"],
            "log_tail": log_tail,
        })
    except ImportError as exc:
        return JSONResponse(
            {"ok": False, "status": "unavailable", "error": str(exc)},
            status_code=503,
        )


@router.post("/mamba_yolo/install")
def mamba_yolo_install(rebuild: bool = False):
    """Trigger selective_scan install/rebuild in background."""
    try:
        from mamba_yolo import installer
        state = installer.ensure_installed(trigger=True, force=rebuild)
        return JSONResponse({
            "ok": True,
            "status": state,
            "rebuild": rebuild,
            "message": {
                "installing": "Installation started in background (~5–10 min). "
                              "Poll /api/plugins/mamba_yolo/install-status for progress.",
                "installed": "selective_scan is already installed.",
                "idle": "Installation not yet triggered — try again.",
                "failed": "Previous install failed. Check install-status for details.",
            }.get(state, state),
        })
    except ImportError as exc:
        return JSONResponse(
            {"ok": False, "status": "unavailable", "error": str(exc)},
            status_code=503,
        )
