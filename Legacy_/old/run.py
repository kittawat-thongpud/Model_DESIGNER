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

    # Backend command: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    backend_cmd = [
        python_exec, "-m", "uvicorn", "app.main:app", 
        "--reload", "--host", "0.0.0.0", "--port", "8000"
    ]
    
    # Use os.setsid to create a process group for clean killing
    backend_proc = subprocess.Popen(
        backend_cmd, 
        cwd=BACKEND_DIR,
        stdout=sys.stdout, 
        stderr=sys.stderr,
        preexec_fn=os.setsid
    )

    # 2. Start Frontend
    print("🔹 Launching Frontend (Vite)...")
    
    # Frontend command: npm run dev -- --host 0.0.0.0
    frontend_cmd = ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
    
    # Check if node_modules exists
    if not (FRONTEND_DIR / "node_modules").exists():
        print("⚠️  'node_modules' not found in frontend. Running 'npm install'...")
        subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, check=True)

    frontend_proc = subprocess.Popen(
        frontend_cmd,
        cwd=FRONTEND_DIR,
        stdout=sys.stdout,
        stderr=sys.stderr,
        preexec_fn=os.setsid
    )

    print("\n✅ All services running!")
    print("   Backend API: http://localhost:8000")
    print("   Frontend UI: http://localhost:5173")
    print("   (Press Ctrl+C to stop all services)")

    try:
        while True:
            time.sleep(1)
            # Check if any process exited unexpectedly
            if backend_proc.poll() is not None:
                print("\n❌ Backend service stopped unexpectedly.")
                break
            if frontend_proc.poll() is not None:
                print("\n❌ Frontend service stopped unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping services...")
    finally:
        # Kill backend process group
        if backend_proc.poll() is None:
            try:
                os.killpg(os.getpgid(backend_proc.pid), signal.SIGTERM)
                print("   Killed Backend.")
            except Exception as e:
                print(f"   Error killing backend: {e}")

        # Kill frontend process group
        if frontend_proc.poll() is None:
            try:
                os.killpg(os.getpgid(frontend_proc.pid), signal.SIGTERM)
                print("   Killed Frontend.")
            except Exception as e:
                print(f"   Error killing frontend: {e}")
        
        backend_proc.wait()
        frontend_proc.wait()
        print("👋 Governance shutdown complete.")

if __name__ == "__main__":
    run()
