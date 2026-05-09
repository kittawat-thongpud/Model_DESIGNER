"""RunPod GPU keepalive service.

Keeps the GPU runtime warm while the app is idle without holding a long-lived
CUDA context in the FastAPI process. Each pulse is a short subprocess that
allocates a one-element CUDA tensor, synchronizes, and exits.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .. import logging_service as logger
from ..config import JOBS_DIR


_KEEPALIVE_SCRIPT = """
import torch
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    x = torch.empty((1,), device="cuda")
    x.fill_(1.0)
    torch.cuda.synchronize()
    print(torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA is not available")
"""


class GPUKeepAlive:
    def __init__(self, interval_s: int = 60):
        self.interval_s = max(10, int(interval_s))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="GPUKeepAlive",
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        logger.log("system", "INFO", f"GPU keepalive started (interval={self.interval_s}s)", component="gpu_keepalive")
        while not self._stop.is_set():
            try:
                if not _has_active_training_worker():
                    _pulse_gpu()
            except Exception as exc:
                logger.log("system", "DEBUG", f"GPU keepalive pulse skipped: {exc}", component="gpu_keepalive")
            self._stop.wait(timeout=self.interval_s)
        logger.log("system", "INFO", "GPU keepalive stopped", component="gpu_keepalive")


def _normalize_runpod_gpu_env() -> None:
    nvidia_visible = os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
    if nvidia_visible in ("", "void", "none"):
        if _nvidia_smi_has_gpu():
            os.environ["NVIDIA_VISIBLE_DEVICES"] = "all"
            os.environ.setdefault("NVIDIA_DRIVER_CAPABILITIES", "compute,utility")
            logger.log(
                "system",
                "INFO",
                "Normalized NVIDIA_VISIBLE_DEVICES=all for GPU runtime visibility",
                component="gpu_keepalive",
            )
    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")


def _nvidia_smi_has_gpu() -> bool:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return proc.returncode == 0 and "GPU" in proc.stdout
    except Exception:
        return False


def _has_active_training_worker() -> bool:
    try:
        pid_files = list(Path(JOBS_DIR).glob("*/worker_process.pid"))
    except Exception:
        return False
    for pid_file in pid_files:
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip().splitlines()[0])
        except Exception:
            continue
        if _pid_is_alive(pid):
            return True
    return False


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _pulse_gpu() -> None:
    env = os.environ.copy()
    env.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    proc = subprocess.run(
        [sys.executable, "-c", _KEEPALIVE_SCRIPT],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "CUDA pulse failed").strip())


_keepalive: GPUKeepAlive | None = None


def should_enable_keepalive() -> bool:
    setting = os.environ.get("MODEL_DESIGNER_GPU_KEEPALIVE", os.environ.get("RUNPOD_GPU_KEEPALIVE", "auto")).strip().lower()
    if setting in ("1", "true", "yes", "on"):
        return True
    if setting in ("0", "false", "no", "off"):
        return False
    # Auto mode: enable only when the pod/container appears to have a GPU.
    return _nvidia_smi_has_gpu()


def start_keepalive(interval_s: int | None = None) -> GPUKeepAlive | None:
    global _keepalive
    _normalize_runpod_gpu_env()
    if not should_enable_keepalive():
        logger.log("system", "INFO", "GPU keepalive disabled", component="gpu_keepalive")
        return None
    if _keepalive is None:
        if interval_s is None:
            interval_s = int(os.environ.get("MODEL_DESIGNER_GPU_KEEPALIVE_INTERVAL", os.environ.get("RUNPOD_GPU_KEEPALIVE_INTERVAL", "60")))
        _keepalive = GPUKeepAlive(interval_s=interval_s)
    _keepalive.start()
    return _keepalive


def stop_keepalive(timeout: float = 5.0) -> None:
    global _keepalive
    if _keepalive is not None:
        _keepalive.stop(timeout=timeout)

