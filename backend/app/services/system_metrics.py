"""
System Metrics Service - GPU, CPU, RAM monitoring for training jobs.
"""
import psutil
from typing import Dict, Any

def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics (CPU, RAM, GPU if available)."""
    metrics: Dict[str, Any] = {}
    
    # CPU
    try:
        metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
    except:
        pass
    
    # RAM
    try:
        mem = psutil.virtual_memory()
        metrics['ram_used'] = mem.used
        metrics['ram_total'] = mem.total
        metrics['ram_percent'] = mem.percent
    except:
        pass
    
    # GPU (try pynvml for NVIDIA GPUs)
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
        
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        metrics['gpu_memory_used'] = mem_info.used
        metrics['gpu_memory_total'] = mem_info.total
        metrics['gpu_memory_percent'] = (mem_info.used / mem_info.total) * 100
        
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        metrics['gpu_utilization'] = util.gpu
        
        pynvml.nvmlShutdown()
    except:
        # GPU metrics not available (no NVIDIA GPU or pynvml not installed)
        pass
    
    return metrics


def get_job_metrics(job_id: str) -> Dict[str, Any]:
    """Get system metrics for a specific job (currently returns global metrics)."""
    # For now, return global system metrics
    # In the future, could track per-job resource usage
    return get_system_metrics()
