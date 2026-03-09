"""
Platform Capability Abstraction — centralizes all platform-specific checks
and auto-fallback decisions with detailed persisted logging.

Design principles:
  - Capability checks instead of OS-name branching where possible
  - All fallback decisions are logged with explicit reason codes
  - Windows GPU + DDP supported but gated on runtime capability probe
  - Never crash startup; degrade gracefully and explain why
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any

# ── Reason codes for fallback decisions ──────────────────────────────────────
# Use these codes in log entries so issues are grep-able in production.

RC_SYMLINK_UNSUPPORTED   = "PLATFORM_SYMLINK_UNSUPPORTED"
RC_REMOTE_FS_DETECTED    = "PLATFORM_REMOTE_FS_DETECTED"
RC_DDP_SPAWN_UNAVAILABLE = "PLATFORM_DDP_SPAWN_UNAVAILABLE"
RC_DDP_WIN_FALLBACK      = "PLATFORM_DDP_WIN_SINGLE_PROCESS_FALLBACK"
RC_CACHE_DOWNGRADE       = "PLATFORM_CACHE_MODE_DOWNGRADED"
RC_PROC_MOUNTS_UNAVAIL   = "PLATFORM_PROC_MOUNTS_UNAVAILABLE"
RC_WINDOWS_PATHS         = "PLATFORM_WINDOWS_PATH_ADJUSTMENT"


# ── Public helpers ────────────────────────────────────────────────────────────

def get_platform_info() -> dict[str, Any]:
    """Return a summary of the current platform for health/readiness endpoints."""
    import torch
    gpu_count = 0
    gpu_names: list[str] = []
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    except Exception:
        pass

    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "python": sys.version.split()[0],
        "arch": platform.machine(),
        "cpu_count": os.cpu_count(),
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "symlink_support": _probe_symlink_support(),
        "proc_mounts_available": Path("/proc/mounts").exists(),
    }


def _probe_symlink_support() -> bool:
    """Test whether symlinks can be created in the temp directory."""
    import tempfile
    tmp = Path(tempfile.gettempdir())
    src = tmp / f"_md_symlink_probe_{os.getpid()}"
    dst = tmp / f"_md_symlink_probe_target_{os.getpid()}"
    try:
        src.write_text("probe")
        dst.symlink_to(src)
        dst.unlink()
        src.unlink()
        return True
    except Exception:
        for p in (src, dst):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        return False


def get_fs_type(path: str | Path) -> str | None:
    """
    Best-effort filesystem type lookup for a path.

    Strategy (in order):
    1. /proc/mounts (Linux/WSL)
    2. Windows: GetVolumeInformation via ctypes
    3. Returns None if undetermined
    """
    try:
        p = Path(path).resolve()
    except Exception:
        p = Path(path)

    # ── Strategy 1: /proc/mounts ──────────────────────────────────────────────
    try:
        mounts_text = Path("/proc/mounts").read_text()
        best_match: str | None = None
        best_len = -1
        for line in mounts_text.splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            mount_point = parts[1]
            fstype = parts[2]
            try:
                if str(p).startswith(mount_point) and len(mount_point) > best_len:
                    best_match = fstype
                    best_len = len(mount_point)
            except Exception:
                continue
        if best_match:
            return best_match
    except OSError:
        # /proc/mounts not available (Windows, macOS, strict container)
        pass

    # ── Strategy 2: Windows GetVolumeInformation ──────────────────────────────
    if platform.system() == "Windows":
        try:
            import ctypes
            drive = str(p.drive) + "\\"
            fs_buf = ctypes.create_unicode_buffer(256)
            ctypes.windll.kernel32.GetVolumeInformationW(
                drive, None, 0, None, None, None, fs_buf, ctypes.sizeof(fs_buf)
            )
            if fs_buf.value:
                return fs_buf.value.upper()
        except Exception:
            pass

    return None


def is_remote_fs(path: str | Path) -> bool:
    """
    Detect whether a path lives on a remote/network filesystem.

    Covers: NFS, CIFS/SMB, FUSE, sshfs, overlay, RunPod network volumes.
    Auto-detects known remote path prefixes (/workspace, /runpod-volume) as fallback.
    """
    _REMOTE_FSTYPES = frozenset({
        "nfs", "nfs4", "cifs", "smbfs", "sshfs", "fuse", "fuseblk",
        "overlay", "overlayfs", "tmpfs", "ramfs",
    })

    fstype = get_fs_type(path)
    if fstype and fstype.lower() in _REMOTE_FSTYPES:
        return True

    # Fallback: known remote-like path prefixes (RunPod, typical NFS mounts)
    try:
        p_str = str(Path(path).resolve())
        for prefix in ("/workspace", "/runpod-volume", "/mnt/nfs", "/mnt/smb"):
            if p_str.startswith(prefix):
                return True
    except Exception:
        pass

    return False


class DDPCapability:
    """Result of a DDP capability probe."""
    __slots__ = ("supported", "start_method", "fallback_reason", "fallback_code")

    def __init__(
        self,
        supported: bool,
        start_method: str,
        fallback_reason: str = "",
        fallback_code: str = "",
    ) -> None:
        self.supported = supported
        self.start_method = start_method
        self.fallback_reason = fallback_reason
        self.fallback_code = fallback_code

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "start_method": self.start_method,
            "fallback_reason": self.fallback_reason,
            "fallback_code": self.fallback_code,
        }


def probe_ddp_capability(device_str: str) -> DDPCapability:
    """
    Probe whether DDP multi-GPU training is viable on the current runtime.

    Auto-fallback rules:
    - Windows: DDP is supported only if 'spawn' start method is available
      and CUDA is accessible. Falls back to single-process with explanation.
    - Linux/macOS: DDP is normally viable; 'spawn' is set for safety.
    - If only 1 GPU or device is 'cpu': DDP not applicable.

    Returns a DDPCapability with supported=True if DDP can proceed, or
    supported=False with fallback_reason and fallback_code explaining the downgrade.
    """
    import multiprocessing as _mp

    _os = platform.system()

    # Parse GPU count from device string e.g. "0,1,2"
    gpu_ids = [d.strip() for d in device_str.split(",") if d.strip().isdigit()]
    if len(gpu_ids) <= 1:
        # Not a multi-GPU config — DDP not needed
        return DDPCapability(supported=False, start_method="none",
                             fallback_reason="Single device specified; DDP not applicable.",
                             fallback_code="DDP_SINGLE_DEVICE")

    # Check CUDA availability
    try:
        import torch
        if not torch.cuda.is_available():
            return DDPCapability(
                supported=False,
                start_method="none",
                fallback_reason="CUDA not available on this host; cannot run DDP.",
                fallback_code="DDP_NO_CUDA",
            )
        available_gpus = torch.cuda.device_count()
        if available_gpus < len(gpu_ids):
            return DDPCapability(
                supported=False,
                start_method="none",
                fallback_reason=(
                    f"Requested {len(gpu_ids)} GPUs but only {available_gpus} detected. "
                    "Falling back to single-process training on device 0."
                ),
                fallback_code="DDP_INSUFFICIENT_GPUS",
            )
    except Exception as _e:
        return DDPCapability(
            supported=False,
            start_method="none",
            fallback_reason=f"Could not probe CUDA: {_e}. DDP disabled.",
            fallback_code="DDP_CUDA_PROBE_FAILED",
        )

    # Probe spawn availability
    can_spawn = False
    try:
        current = _mp.get_start_method(allow_none=True)
        if current == "spawn":
            can_spawn = True
        else:
            _mp.set_start_method("spawn", force=True)
            can_spawn = True
    except RuntimeError:
        can_spawn = False

    if _os == "Windows" and not can_spawn:
        return DDPCapability(
            supported=False,
            start_method="none",
            fallback_reason=(
                "Windows DDP requires 'spawn' start method which is not available "
                "in this process context. Falling back to single-GPU training."
            ),
            fallback_code=RC_DDP_WIN_FALLBACK,
        )

    return DDPCapability(supported=True, start_method="spawn")


def select_ddp_device_with_fallback(
    device_str: str,
    job_id: str | None = None,
) -> tuple[str, DDPCapability]:
    """
    Resolve final device string with DDP auto-fallback and persisted logging.

    Returns (final_device_str, capability).
    If DDP is not viable, final_device_str is downgraded to 'gpu:0' or the
    first device in the list, and the fallback is logged to the job log.
    """
    cap = probe_ddp_capability(device_str)

    if not cap.supported and cap.fallback_reason:
        _log_fallback(
            job_id=job_id,
            reason=cap.fallback_reason,
            code=cap.fallback_code,
            context={"original_device": device_str},
        )
        # Return single device fallback
        first = device_str.split(",")[0].strip()
        return first, cap

    return device_str, cap


def select_cache_mode_with_fallback(
    requested: str | bool | None,
    ds_root: str | Path | None,
    job_id: str | None = None,
    num_gpus: int = 1,
) -> tuple[str | bool, str]:
    """
    Resolve final cache mode with auto-fallback and persisted logging.

    Fallback chain: ram → disk → off (never silently pass unsafe config)
    Returns (final_cache_mode, reason_code).
    """
    import shutil

    if requested is None or requested is False or str(requested).lower() in ("false", "off", ""):
        return False, ""

    req_str = str(requested).lower().strip()

    # Remote FS: never use RAM cache (each DDP process gets own copy → OOM)
    if req_str == "ram" and ds_root and is_remote_fs(ds_root):
        _log_fallback(
            job_id=job_id,
            reason="RAM cache disabled on remote/NFS filesystem. Downgrading to disk cache.",
            code=RC_CACHE_DOWNGRADE,
            context={"requested": requested, "ds_root": str(ds_root)},
        )
        return True, RC_CACHE_DOWNGRADE  # True = disk cache in Ultralytics

    # Multi-GPU DDP: RAM cache multiplies memory per rank → risky
    if req_str == "ram" and num_gpus > 1:
        try:
            import psutil
            free_ram = psutil.virtual_memory().available
            if ds_root and Path(ds_root).exists():
                ds_size = sum(
                    f.stat().st_size for f in Path(ds_root).rglob("*") if f.is_file()
                ) * 30  # decompressed estimate
                needed = ds_size * num_gpus
                if needed > free_ram * 0.8:
                    _log_fallback(
                        job_id=job_id,
                        reason=(
                            f"RAM cache with {num_gpus} GPUs would need ~{needed/1e9:.1f} GB "
                            f"but only {free_ram/1e9:.1f} GB free. Downgrading to disk cache."
                        ),
                        code=RC_CACHE_DOWNGRADE,
                        context={"requested": requested, "num_gpus": num_gpus},
                    )
                    return True, RC_CACHE_DOWNGRADE
        except Exception:
            pass

    # Disk cache: check available space
    if req_str in ("disk", "true") or requested is True:
        try:
            check_path = Path(ds_root) if ds_root else Path.cwd()
            free = shutil.disk_usage(str(check_path)).free
            if free < 2 * 1024 ** 3:  # < 2 GB free
                _log_fallback(
                    job_id=job_id,
                    reason=f"Insufficient disk space for disk cache ({free/1e9:.1f} GB free). Disabling cache.",
                    code=RC_CACHE_DOWNGRADE,
                    context={"requested": requested, "free_gb": round(free / 1e9, 1)},
                )
                return False, RC_CACHE_DOWNGRADE
        except Exception:
            pass

    return requested, ""


# ── Internal helpers ──────────────────────────────────────────────────────────

def _log_fallback(
    reason: str,
    code: str,
    job_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    """
    Persist a fallback decision to the system log and optionally the job log.
    Always includes reason_code so production issues are grep-able.
    """
    payload: dict[str, Any] = {
        "reason_code": code,
        "reason": reason,
        **(context or {}),
    }
    try:
        from .. import logging_service as _ls
        _ls.log("system", "WARNING", f"Platform auto-fallback [{code}]: {reason}", payload)
    except Exception:
        pass

    if job_id:
        try:
            from . import job_storage as _js
            _js.append_job_log(
                job_id,
                "WARNING",
                f"[{code}] {reason}",
                payload,
            )
        except Exception:
            pass
