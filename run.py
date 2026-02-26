import subprocess
import time
import os
import signal
import sys
from pathlib import Path

# Config
PROJECT_ROOT = Path(__file__).parent.resolve()
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

def run():
    print("🚀 Starting Model DESIGNER System...")
    
    # 1. Start Backend
    print("🔹 Launching Backend (FastAPI)...")
    
    # Detect virtualenv in backend directory
    venv_python = BACKEND_DIR / "venv" / "bin" / "python"
    
    if venv_python.exists():
        print(f"   Using venv: {venv_python}")
        python_exec = str(venv_python)
    else:
        print("   Using system python (ensure dependencies are installed)")
        python_exec = sys.executable

    # Build frontend dist if not present or stale
    dist_dir = FRONTEND_DIR / "dist"
    if not dist_dir.exists():
        print("🔹 Building Frontend (production)...")
        if not (FRONTEND_DIR / "node_modules").exists():
            print("   Running 'npm install'...")
            subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, check=True)
        subprocess.run(["npm", "run", "build"], cwd=FRONTEND_DIR, check=True)
        print("   Frontend built.")

    # Backend command: uvicorn production (no --reload, 1 worker — training uses threads)
    backend_cmd = [
        python_exec, "-m", "uvicorn", "app.main:app",
        "--host", "0.0.0.0", "--port", "8000",
        "--workers", "1",
    ]

    # Use os.setsid to create a process group for clean killing
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=BACKEND_DIR,
        stdout=sys.stdout,
        stderr=sys.stderr,
        preexec_fn=os.setsid
    )

    print("\n✅ All services running!")
    print("   App: http://localhost:8000  (frontend served from dist/)")
    print("   API: http://localhost:8000/api/")
    print("   (Press Ctrl+C to stop)")

    try:
        while True:
            time.sleep(1)
            if backend_proc.poll() is not None:
                print("\n❌ Backend stopped unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping services...")
    finally:
        if backend_proc.poll() is None:
            try:
                os.killpg(os.getpgid(backend_proc.pid), signal.SIGTERM)
                print("   Killed Backend.")
            except Exception as e:
                print(f"   Error killing backend: {e}")

        backend_proc.wait()
        print("👋 Shutdown complete.")

if __name__ == "__main__":
    run()
