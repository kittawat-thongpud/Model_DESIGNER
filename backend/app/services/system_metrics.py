"""
System Metrics Service - GPU, CPU, RAM monitoring for training jobs.
"""
import psutil
from typing import Dict, Any, List

def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics (CPU, RAM, all NVIDIA GPUs if available)."""
    metrics: Dict[str, Any] = {}

    # CPU
    try:
        metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        metrics['cpu_count'] = psutil.cpu_count(logical=True)
    except Exception:
        pass

    # RAM
    try:
        mem = psutil.virtual_memory()
        metrics['ram_used'] = mem.used
        metrics['ram_total'] = mem.total
        metrics['ram_percent'] = mem.percent
    except Exception:
        pass

    # GPU — enumerate all NVIDIA devices, not just index 0
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            gpus: List[Dict[str, Any]] = []
            for idx in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpus.append({
                        "index": idx,
                        "name": pynvml.nvmlDeviceGetName(handle),
                        "memory_used": mem_info.used,
                        "memory_total": mem_info.total,
                        "memory_percent": round((mem_info.used / mem_info.total) * 100, 1),
                        "utilization": util.gpu,
                    })
                except Exception:
                    pass
            if gpus:
                metrics['gpus'] = gpus
                # Keep top-level keys for first GPU for backward compat with existing callers
                metrics['gpu_memory_used'] = gpus[0]['memory_used']
                metrics['gpu_memory_total'] = gpus[0]['memory_total']
                metrics['gpu_memory_percent'] = gpus[0]['memory_percent']
                metrics['gpu_utilization'] = gpus[0]['utilization']
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        # GPU metrics not available (no NVIDIA GPU or pynvml not installed)
        pass

    return metrics


def get_job_metrics(job_id: str) -> Dict[str, Any]:
    """Get system metrics for a specific job (currently returns global metrics)."""
    # TODO: track per-job resource usage (GPU handle pinned to job pid) in a future slice
    return get_system_metrics()
