import subprocess
import time
import os
import signal
import sys
from pathlib import Path

# Config
PROJECT_ROOT = Path(__file__).parent.resolve()
BACKEND_DIR  = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
PID_FILE     = PROJECT_ROOT / ".model_designer.pid"   # tracks our process group


# ── PID helpers ───────────────────────────────────────────────────────────────

def _write_pid(pgid: int, pid: int) -> None:
    PID_FILE.write_text(f"{pgid}:{pid}\n")


def _read_pid() -> tuple[int, int] | None:
    """Return (pgid, pid) stored in PID file, or None."""
    try:
        parts = PID_FILE.read_text().strip().split(":")
        return int(parts[0]), int(parts[1])
    except Exception:
        return None


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _kill_group(pgid: int, sig: int = signal.SIGTERM) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass  # already gone
    except Exception as e:
        print(f"   Warning: killpg({pgid}, {sig}): {e}")


def clean_boot() -> None:
    """Kill any stale worker from a previous run before starting fresh."""
    stale = _read_pid()
    if stale is None:
        return
    pgid, pid = stale
    if not _pid_is_alive(pid):
        PID_FILE.unlink(missing_ok=True)
        return
    print(f"⚠️  Found stale worker (pid={pid}, pgid={pgid}) — cleaning up...")
    _kill_group(pgid, signal.SIGTERM)
    # Wait up to 5 s for graceful exit
    for _ in range(10):
        time.sleep(0.5)
        if not _pid_is_alive(pid):
            break
    else:
        print("   SIGTERM timeout — forcing SIGKILL...")
        _kill_group(pgid, signal.SIGKILL)
        time.sleep(0.5)
    PID_FILE.unlink(missing_ok=True)
    print("   Stale worker cleaned.")


def clean_shutdown(backend_proc: subprocess.Popen) -> None:
    """Gracefully terminate the process group, escalate to SIGKILL if needed."""
    if backend_proc.poll() is not None:
        return  # already dead

    pgid = os.getpgid(backend_proc.pid)
    print("   Sending SIGTERM to process group...")
    _kill_group(pgid, signal.SIGTERM)

    # Give uvicorn up to 8 s to finish in-flight requests
    for _ in range(16):
        time.sleep(0.5)
        if backend_proc.poll() is not None:
            break
    else:
        print("   SIGTERM timeout — forcing SIGKILL...")
        _kill_group(pgid, signal.SIGKILL)
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
    if not dist_dir.exists():
        print("🔹 Building Frontend (production)...")
        if not (FRONTEND_DIR / "node_modules").exists():
            print("   Running 'npm install'...")
            subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, check=True)
        subprocess.run(["npm", "run", "build"], cwd=FRONTEND_DIR, check=True)
        print("   Frontend built.")

    # ── Start uvicorn in its own process group ────────────────────────────────
    backend_cmd = [
        python_exec, "-m", "uvicorn", "app.main:app",
        "--host", "0.0.0.0", "--port", "8000",
        "--workers", "1",
        # NOTE: large upload limit is set via Starlette middleware in app/main.py
    ]

    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=BACKEND_DIR,
        stdout=sys.stdout,
        stderr=sys.stderr,
        preexec_fn=os.setsid,   # new process group so we can killpg
    )

    # ── Write PID file so next boot can clean up if we crash ─────────────────
    pgid = os.getpgid(backend_proc.pid)
    _write_pid(pgid, backend_proc.pid)

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
