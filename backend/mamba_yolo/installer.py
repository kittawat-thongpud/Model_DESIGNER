"""
Mamba-YOLO Installer.

Clones the full HZAI-ZJNU/Mamba-YOLO repository to a permanent vendor
location, installs the selective_scan CUDA extension from source, and
installs required Python dependencies (einops, timm).

State machine:
  idle  →  installing  →  installed
                       →  failed

Persists to DATA_DIR:
  mamba_yolo_install.json   — state + metadata
  mamba_yolo_install.log    — full install stdout/stderr (line-buffered)

SSE: each log line is also broadcast via logging_service → SYSTEM_LOG_CHANNEL.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Literal

_State = Literal["idle", "installing", "installed", "failed"]

# ── Paths ──────────────────────────────────────────────────────────────────────

def _data_dir() -> Path:
    try:
        from app.config import DATA_DIR  # installed as top-level app package
        return DATA_DIR
    except ImportError:
        return Path(__file__).resolve().parents[1] / "data"


def _vendor_dir() -> Path:
    return _data_dir() / "vendor"


def _repo_dir() -> Path:
    return _vendor_dir() / "Mamba-YOLO"


def _state_path() -> Path:
    return _data_dir() / "mamba_yolo_install.json"


def _log_path() -> Path:
    return _data_dir() / "mamba_yolo_install.log"


# ── Internal state ─────────────────────────────────────────────────────────────

_lock = threading.Lock()

_state: _State = "idle"
_started_at: str | None = None
_finished_at: str | None = None
_error: str | None = None


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── Persistence helpers ────────────────────────────────────────────────────────

def _write_state(state: _State, error: str | None = None) -> None:
    global _state, _finished_at, _error
    _state = state
    if state in ("installed", "failed"):
        _finished_at = _now_iso()
    _error = error
    record = {
        "status": state,
        "started_at": _started_at,
        "finished_at": _finished_at,
        "error": error,
    }
    try:
        _state_path().write_text(json.dumps(record, indent=2))
    except Exception:
        pass


def _load_persisted_state() -> None:
    """Restore state from disk on startup (survives backend restarts)."""
    global _state, _started_at, _finished_at, _error
    p = _state_path()
    if not p.exists():
        return
    try:
        data = json.loads(p.read_text())
        persisted = data.get("status", "idle")
        # If server restarted during install, mark as failed
        if persisted == "installing":
            persisted = "failed"
            data["status"] = "failed"
            data["error"] = "Server restarted during installation — rerun install."
            p.write_text(json.dumps(data, indent=2))
        _state = persisted  # type: ignore[assignment]
        _started_at = data.get("started_at")
        _finished_at = data.get("finished_at")
        _error = data.get("error")
    except Exception:
        pass


def _log_line(line: str) -> None:
    """Append a line to the persistent log file and broadcast via SSE."""
    try:
        with open(_log_path(), "a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")
    except Exception:
        pass
    try:
        from app import logging_service
        logging_service.log(
            "system", "INFO", line.rstrip(),
            component="mamba_yolo_install",
        )
    except Exception:
        pass


# ── Check if already available ─────────────────────────────────────────────────

def _modules_loadable() -> bool:
    """Return True if the cloned repo's module files are present."""
    modules_dir = _repo_dir() / "ultralytics" / "nn" / "modules"
    return (
        (modules_dir / "mamba_yolo.py").exists()
        and (modules_dir / "common_utils_mbyolo.py").exists()
    )


def _prepare_cuda_env() -> tuple[dict[str, str], str | None]:
    """Return subprocess env with best-effort CUDA_HOME detection."""
    env = dict(os.environ)
    cuda_home = env.get("CUDA_HOME") or env.get("CUDA_PATH")

    candidates: list[Path] = []
    if cuda_home:
        candidates.append(Path(cuda_home))

    nvcc = shutil.which("nvcc")
    if nvcc:
        # <cuda>/bin/nvcc -> parent of parent is CUDA root
        candidates.append(Path(nvcc).resolve().parent.parent)

    candidates.extend(
        [
            Path("/usr/local/cuda"),
            Path("/usr/local/cuda-12.8"),
            Path("/usr/local/cuda-12.6"),
            Path("/usr/local/cuda-12.4"),
            Path("/opt/cuda"),
        ]
    )

    resolved: str | None = None
    for c in candidates:
        if (c / "bin" / "nvcc").exists():
            resolved = str(c)
            break

    if resolved:
        env["CUDA_HOME"] = resolved
        env.setdefault("CUDA_PATH", resolved)
        env["PATH"] = f"{resolved}/bin:{env.get('PATH', '')}"
        lib64 = f"{resolved}/lib64"
        env["LD_LIBRARY_PATH"] = (
            f"{lib64}:{env.get('LD_LIBRARY_PATH', '')}"
            if env.get("LD_LIBRARY_PATH")
            else lib64
        )

    return env, resolved


# ── Background install ─────────────────────────────────────────────────────────

_MAMBA_YOLO_REPO = "https://github.com/HZAI-ZJNU/Mamba-YOLO.git"


def _do_install() -> None:
    """Clone full Mamba-YOLO repo and install dependencies in background."""
    import sys
    import importlib

    _log_line("=" * 60)
    _log_line(f"[mamba_yolo] Install started at {_now_iso()}")
    _log_line(f"[mamba_yolo] Repo will be cloned to: {_repo_dir()}")

    vendor = _vendor_dir()
    repo = _repo_dir()

    # ── Step 1: Clone (or update) the full repo ────────────────────────────
    if repo.exists() and (repo / ".git").exists():
        _log_line("[mamba_yolo] Existing clone found — pulling latest...")
        ok = _run_logged(["git", "pull", "--depth=1"], cwd=str(repo))
        if not ok:
            _log_line("[mamba_yolo] WARNING: git pull failed; using existing clone.")
    else:
        if repo.exists():
            shutil.rmtree(repo, ignore_errors=True)
        vendor.mkdir(parents=True, exist_ok=True)

        cmd_clone = ["git", "clone", "--depth=1", _MAMBA_YOLO_REPO, str(repo)]
        _log_line(f"[mamba_yolo] $ {' '.join(cmd_clone)}")
        ok = _run_logged(cmd_clone, cwd=str(vendor))
        if not ok:
            _write_state("failed", "git clone Mamba-YOLO failed — check network/git.")
            _log_line("[mamba_yolo] FAILED: git clone step failed.")
            return

    _log_line(f"[mamba_yolo] Repo ready at {repo}")

    # ── Step 2: Install Python dependencies ───────────────────────────────
    deps = ["einops", "timm"]
    cmd_deps = [sys.executable, "-m", "pip", "install"] + deps
    _log_line(f"[mamba_yolo] $ {' '.join(cmd_deps)}")
    ok = _run_logged(cmd_deps)
    if not ok:
        _write_state("failed", "Failed to install Python dependencies (einops, timm).")
        _log_line("[mamba_yolo] FAILED: could not install einops/timm.")
        return

    # ── Step 3: Install selective_scan CUDA extension ──────────────────────
    scan_dir = repo / "selective_scan"
    if scan_dir.exists():
        scan_env, cuda_home = _prepare_cuda_env()
        if cuda_home:
            _log_line(f"[mamba_yolo] Using CUDA_HOME={cuda_home}")
        else:
            _log_line(
                "[mamba_yolo] WARNING: CUDA_HOME not detected. "
                "Set CUDA_HOME to your CUDA toolkit root (e.g. /usr/local/cuda)."
            )

        cmd_scan = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            str(scan_dir),
        ]
        _log_line(f"[mamba_yolo] $ {' '.join(cmd_scan)}")
        ok = _run_logged(cmd_scan, env=scan_env)
        if not ok:
            msg = "Failed to build selective_scan CUDA extension."
            if not cuda_home:
                msg += " CUDA_HOME is not set or CUDA toolkit was not found on this machine."
            _write_state("failed", msg)
            _log_line("[mamba_yolo] FAILED: selective_scan build failed.")
            return

        try:
            importlib.import_module("selective_scan_cuda_core")
            _log_line("[mamba_yolo] selective_scan_cuda_core import check passed.")
        except Exception as exc:
            _write_state("failed", f"selective_scan built but import failed: {exc}")
            _log_line(f"[mamba_yolo] FAILED: selective_scan import check failed: {exc}")
            return
    else:
        _write_state("failed", f"selective_scan dir not found at {scan_dir}")
        _log_line(f"[mamba_yolo] FAILED: selective_scan dir not found at {scan_dir}")
        return

    # ── Step 4: Verify ────────────────────────────────────────────────────
    if _modules_loadable():
        _write_state("installed")
        _log_line("[mamba_yolo] SUCCESS: Mamba-YOLO modules are ready.")
    else:
        _write_state("failed", "Module files not found after clone.")
        _log_line("[mamba_yolo] FAILED: module files missing after clone.")
    _log_line("=" * 60)


def _run_logged(cmd: list[str], cwd: str | None = None, env: dict[str, str] | None = None) -> bool:
    """Run a subprocess, streaming stdout+stderr to _log_line. Returns True on success."""
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd,
            env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True,
        )
        if proc.stdout:
            for line in proc.stdout:
                _log_line(line)
        proc.wait()
        return proc.returncode == 0
    except Exception as exc:
        _log_line(f"[mamba_yolo] subprocess error: {exc}")
        return False


# ── Public API ─────────────────────────────────────────────────────────────────

def get_repo_dir() -> Path:
    """Return the permanent vendor repo path."""
    return _repo_dir()


def get_status() -> dict:
    """Return the current installer state dict."""
    with _lock:
        return {
            "status": _state,
            "started_at": _started_at,
            "finished_at": _finished_at,
            "error": _error,
        }


def get_log_tail(n: int = 100) -> list[str]:
    """Return the last n lines from the persistent install log."""
    p = _log_path()
    if not p.exists():
        return []
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-n:]
    except Exception:
        return []


def ensure_installed(trigger: bool = True, force: bool = False) -> _State:
    """
    Ensure the Mamba-YOLO repo is cloned and dependencies are installed.

    If currently installing → return current state.
    If installed and force=False → return installed.
    If installed and force=True → re-run install in background.
    If idle/failed and trigger=True → start background install.
    Returns the current state string.
    """
    global _state, _started_at

    with _lock:
        if _state == "installed" and not force:
            return "installed"
        if _state == "installing":
            return "installing"

        # Check if it was installed externally (e.g. manual clone)
        if _modules_loadable() and not force:
            _write_state("installed")
            return "installed"

        if not trigger:
            return _state

        # Start background install
        _state = "installing"
        _started_at = _now_iso()
        _write_state("installing")
        try:
            _log_path().write_text("", encoding="utf-8")
        except Exception:
            pass

    thread = threading.Thread(target=_do_install, daemon=True, name="mamba_yolo_install")
    thread.start()
    return "installing"


# ── Initialise on import ───────────────────────────────────────────────────────
_load_persisted_state()

# If persisted state is "installed" but the files are gone, reset to idle.
if _state == "installed" and not _modules_loadable():
    _state = "idle"
    _write_state("idle")
