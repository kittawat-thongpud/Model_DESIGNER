"""
Worker Monitor Service - Automatic zombie worker cleanup.

Runs a background thread that periodically checks for and cleans up zombie workers.
"""
from __future__ import annotations
import threading
import time
from typing import Callable

from . import ultra_trainer


class WorkerMonitor:
    """Background monitor for worker health and zombie cleanup."""
    
    def __init__(self, check_interval: int = 60):
        """Initialize worker monitor.
        
        Args:
            check_interval: Seconds between health checks (default: 60)
        """
        self.check_interval = check_interval
        self._monitor_thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._callbacks: list[Callable[[dict], None]] = []
    
    def start(self) -> None:
        """Start the background monitor thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return  # Already running
        
        self._stop_flag.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="WorkerMonitor"
        )
        self._monitor_thread.start()
    
    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background monitor thread.
        
        Args:
            timeout: Max seconds to wait for thread to stop
        """
        self._stop_flag.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=timeout)
    
    def add_callback(self, callback: Callable[[dict], None]) -> None:
        """Add a callback to be called when zombies are detected.
        
        Args:
            callback: Function that receives cleanup result dict
        """
        self._callbacks.append(callback)
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop - runs in background thread."""
        while not self._stop_flag.is_set():
            try:
                # Check worker health
                health = ultra_trainer.get_worker_health()
                
                # If there are dead workers, clean them up
                if health["dead_workers"] > 0:
                    cleaned = ultra_trainer.cleanup_zombie_workers()
                    
                    if cleaned:
                        # Notify callbacks
                        result = {
                            "timestamp": time.time(),
                            "health": health,
                            "cleaned": cleaned
                        }
                        for callback in self._callbacks:
                            try:
                                callback(result)
                            except Exception as e:
                                print(f"Worker monitor callback error: {e}")
            
            except Exception as e:
                print(f"Worker monitor error: {e}")
            
            # Wait for next check (with early exit on stop)
            self._stop_flag.wait(timeout=self.check_interval)


# Global monitor instance
_monitor: WorkerMonitor | None = None


def start_monitor(check_interval: int = 60) -> WorkerMonitor:
    """Start the global worker monitor.
    
    Args:
        check_interval: Seconds between health checks (default: 60)
    
    Returns:
        WorkerMonitor instance
    """
    global _monitor
    if _monitor is None:
        _monitor = WorkerMonitor(check_interval=check_interval)
    _monitor.start()
    return _monitor


def stop_monitor(timeout: float = 5.0) -> None:
    """Stop the global worker monitor.
    
    Args:
        timeout: Max seconds to wait for thread to stop
    """
    global _monitor
    if _monitor:
        _monitor.stop(timeout=timeout)


def get_monitor() -> WorkerMonitor | None:
    """Get the global worker monitor instance."""
    return _monitor
