"""
Unified Runner for DataVista AI: Starts both Backend & Frontend in one command.
Usage:
    python app.py
"""
import os
import sys
import subprocess
import time
import signal

def run_both():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(root_dir, "frontend")

    print("\n" + "=" * 60)
    print("  * DataVista AI - Launching Both Servers Concurrently")
    print("=" * 60)
    print("  - Backend API (FastAPI):  http://localhost:8000")
    print("  - Frontend App (Vite):    http://localhost:5173")
    print("=" * 60 + "\n")

    # Start FastAPI Backend
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=root_dir
    )

    # Start Vite Frontend
    is_windows = sys.platform.startswith("win")
    npm_cmd = "npm.cmd" if is_windows else "npm"
    
    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=frontend_dir
    )

    print("Both servers are now running! Press Ctrl+C in this terminal to stop both.\n")

    def signal_handler(sig, frame):
        print("\nStopping PowerMind BI servers...")
        try:
            backend_proc.terminate()
            frontend_proc.terminate()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal_handler)

    try:
        while True:
            time.sleep(1)
            # Check if any process died
            if backend_proc.poll() is not None:
                print("Backend server stopped.")
                frontend_proc.terminate()
                break
            if frontend_proc.poll() is not None:
                print("Frontend server stopped.")
                backend_proc.terminate()
                break
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    run_both()
