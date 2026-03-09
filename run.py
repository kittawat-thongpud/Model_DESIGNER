import subprocess
import time
import os
import signal
import sys
import platform
from pathlib import Path

# Config
PROJECT_ROOT = Path(__file__).parent.resolve()
BACKEND_DIR  = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
PID_FILE     = PROJECT_ROOT / ".model_designer.pid"   # tracks our process

IS_WINDOWS = platform.system() == "Windows"


# ── PID helpers ───────────────────────────────────────────────────────────────

def _write_pid(pid: int) -> None:
    PID_FILE.write_text(f"{pid}\n")


def _read_pid() -> int | None:
    """Return pid stored in PID file, or None."""
    try:
        content = PID_FILE.read_text().strip()
        # Handle old format (pgid:pid) if it exists
        if ":" in content:
            return int(content.split(":")[1])
        return int(content)
    except Exception:
        return None


def _pid_is_alive(pid: int) -> bool:
    try:
        # os.kill(pid, 0) works on both Unix and Windows (Python 3.2+)
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _kill_group(pid: int, sig_type: str = "TERM") -> None:
    """Cross-platform process tree killer."""
    try:
        if IS_WINDOWS:
            if sig_type == "TERM":
                # Send CTRL_BREAK_EVENT to the process group
                os.kill(pid, signal.CTRL_BREAK_EVENT)
            else:
                # Force kill process tree on Windows
                subprocess.run(
                    ['taskkill', '/F', '/T', '/PID', str(pid)], 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL
                )
        else:
            pgid = os.getpgid(pid)
            sig = signal.SIGTERM if sig_type == "TERM" else signal.SIGKILL
            os.killpg(pgid, sig)
    except ProcessLookupError:
        pass  # already gone
    except Exception as e:
        print(f"   Warning: kill_group({pid}, {sig_type}): {e}")


def clean_boot() -> None:
    """Kill any stale worker from a previous run before starting fresh."""
    pid = _read_pid()
    if pid is None:
        return
    if not _pid_is_alive(pid):
        PID_FILE.unlink(missing_ok=True)
        return
    
    print(f"⚠️  Found stale worker (pid={pid}) — cleaning up...")
    _kill_group(pid, "TERM")
    
    # Wait up to 5 s for graceful exit
    for _ in range(10):
        time.sleep(0.5)
        if not _pid_is_alive(pid):
            break
    else:
        print("   SIGTERM timeout — forcing SIGKILL...")
        _kill_group(pid, "KILL")
        time.sleep(0.5)
        
    PID_FILE.unlink(missing_ok=True)
    print("   Stale worker cleaned.")


def clean_shutdown(backend_proc: subprocess.Popen) -> None:
    """Gracefully terminate the process group, escalate to SIGKILL if needed."""
    if backend_proc.poll() is not None:
        return  # already dead

    print("   Sending stop signal to process group...")
    _kill_group(backend_proc.pid, "TERM")

    # Give uvicorn up to 8 s to finish in-flight requests
    for _ in range(16):
        time.sleep(0.5)
        if backend_proc.poll() is not None:
            break
    else:
        print("   Timeout — forcing kill...")
        _kill_group(backend_proc.pid, "KILL")
        backend_proc.wait()

    PID_FILE.unlink(missing_ok=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("🚀 Starting Model DESIGNER System...")

    # ── Clean boot: kill stale workers ───────────────────────────────────────
    clean_boot()

    # ── Select python interpreter ─────────────────────────────────────────────
    print("🔹 Launching Backend (FastAPI)...")
    python_exec = os.environ.get("MODEL_DESIGNER_PYTHON") or sys.executable
    print(f"   Python: {python_exec}")

    # ── Build frontend if missing ─────────────────────────────────────────────
    dist_dir = FRONTEND_DIR / "dist"
    npm_cmd = "npm.cmd" if IS_WINDOWS else "npm"
    
    if not dist_dir.exists():
        print("🔹 Building Frontend (production)...")
        if not (FRONTEND_DIR / "node_modules").exists():
            print("   Running 'npm install'...")
            subprocess.run([npm_cmd, "install"], cwd=FRONTEND_DIR, check=True)
        subprocess.run([npm_cmd, "run", "build"], cwd=FRONTEND_DIR, check=True)
        print("   Frontend built.")

    # ── Start uvicorn in its own process group ────────────────────────────────
    backend_cmd = [
        python_exec, "-m", "uvicorn", "app.main:app",
        "--host", "0.0.0.0", "--port", "8000",
        "--workers", "1",
    ]

    # OS-specific process group creation flags
    popen_kwargs = {}
    if IS_WINDOWS:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["preexec_fn"] = os.setsid

    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=BACKEND_DIR,
        stdout=sys.stdout,
        stderr=sys.stderr,
        **popen_kwargs
    )

    # ── Write PID file so next boot can clean up if we crash ─────────────────
    _write_pid(backend_proc.pid)

    print("\n✅ All services running!")
    print("   App: http://localhost:8000  (frontend served from dist/)")
    print("   API: http://localhost:8000/api/")
    print(f"   PID file: {PID_FILE}")
    print("   (Press Ctrl+C to stop)")

    try:
        while True:
            time.sleep(1)
            if backend_proc.poll() is not None:
                print("\n❌ Backend stopped unexpectedly.")
                PID_FILE.unlink(missing_ok=True)
                break
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping services...")
    finally:
        clean_shutdown(backend_proc)
        print("👋 Shutdown complete.")


if __name__ == "__main__":
    run()