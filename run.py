#!/usr/bin/env python
"""
QNA-Auth Quick Start Script

This script handles all setup and startup tasks:
- Creates virtual environment (if needed)
- Installs dependencies
- Initializes database
- Starts backend and frontend servers
- Collects device fingerprints

Usage:
    python run.py [command]

Commands:
    setup       - First-time setup (venv, deps, db)
    start       - Start backend server
    dev         - Start in development mode (backend + frontend)
    frontend    - Start frontend only
    collect     - Collect device fingerprint (interactive)
    train       - Run model training
    evaluate    - Run model evaluation
    test        - Run tests
    clean       - Clean generated files
"""

import subprocess
import sys
import os
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / "venv"
DATA_DIR = ROOT / "data"


def run_command(cmd: list, cwd: Path = ROOT, check: bool = True) -> int:
    """Run a command and return exit code."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if check and result.returncode != 0:
        print(f"Command failed with code {result.returncode}")
    return result.returncode


def get_python():
    """Get the Python executable path."""
    if sys.platform == "win32":
        python = VENV_DIR / "Scripts" / "python.exe"
    else:
        python = VENV_DIR / "bin" / "python"
    
    if python.exists():
        return str(python)
    return sys.executable


def get_pip():
    """Get the pip executable path."""
    if sys.platform == "win32":
        pip = VENV_DIR / "Scripts" / "pip.exe"
    else:
        pip = VENV_DIR / "bin" / "pip"
    
    if pip.exists():
        return str(pip)
    return f"{get_python()} -m pip"


def setup():
    """First-time setup."""
    print("=" * 60)
    print("QNA-Auth Setup")
    print("=" * 60)
    
    # Create virtual environment
    if not VENV_DIR.exists():
        print("\n1. Creating virtual environment...")
        run_command([sys.executable, "-m", "venv", str(VENV_DIR)])
    else:
        print("\n1. Virtual environment already exists")
    
    # Install dependencies
    print("\n2. Installing dependencies...")
    pip = get_pip()
    if isinstance(pip, str) and " " in pip:
        run_command(pip.split() + ["install", "-r", "requirements.txt"])
    else:
        run_command([pip, "install", "-r", "requirements.txt"])
    
    # Create directories
    print("\n3. Creating directories...")
    directories = [
        DATA_DIR,
        ROOT / "model" / "checkpoints",
        ROOT / "model" / "evaluation",
        ROOT / "auth" / "device_embeddings",
        ROOT / "dataset" / "samples",
        ROOT / "dataset" / "processed",
    ]
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)
        print(f"   Created: {d}")
    
    # Initialize database
    print("\n4. Initializing database...")
    python = get_python()
    run_command([python, "scripts/init_db.py"])
    
    # Create .env if not exists
    if not (ROOT / ".env").exists():
        print("\n5. Creating .env file...")
        import shutil
        shutil.copy(ROOT / ".env.example", ROOT / ".env")
        print("   Created .env from .env.example")
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Edit .env file with your configuration")
    print("  2. Run: python run.py start")
    print("  3. Visit: http://localhost:8000/docs")


def start_backend():
    """Start the backend server."""
    print("Starting backend server...")
    python = get_python()
    run_command([python, "server/app.py"])


def start_frontend():
    """Start the frontend dev server."""
    print("Starting frontend server...")
    os.chdir(ROOT / "frontend")
    run_command(["npm", "run", "dev"])


def start_dev():
    """Start both backend and frontend in development mode."""
    import threading
    
    print("Starting development servers...")
    
    # Start backend in background
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    print("Backend starting on http://localhost:8000")
    print("Frontend starting on http://localhost:3000")
    print("\nPress Ctrl+C to stop\n")
    
    # Start frontend (blocking)
    start_frontend()


def run_training():
    """Run model training."""
    print("Running model training...")
    python = get_python()
    run_command([python, "scripts/train_and_evaluate.py"])


def run_evaluation():
    """Run model evaluation."""
    print("Running model evaluation...")
    python = get_python()
    run_command([python, "scripts/run_evaluation.py"])


def run_tests():
    """Run test suite."""
    print("Running tests...")
    python = get_python()
    run_command([python, "-m", "pytest", "-v"])


def clean():
    """Clean generated files."""
    print("Cleaning generated files...")
    import shutil
    
    to_remove = [
        ROOT / "data" / "qna_auth.db",
        ROOT / "__pycache__",
        ROOT / ".pytest_cache",
    ]
    
    # Find all __pycache__ directories
    for pycache in ROOT.rglob("__pycache__"):
        to_remove.append(pycache)
    
    for path in to_remove:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"  Removed: {path}")
    
    print("Clean complete!")


def collect_fingerprint():
    """Collect device fingerprint interactively."""
    print("=" * 60)
    print("🔐 Device Fingerprint Collection")
    print("=" * 60)
    
    # Get device name from user
    device_name = input("\nEnter a name for this device (e.g., 'My Laptop'): ").strip()
    if not device_name:
        device_name = "Unnamed Device"
    
    # Ask for number of samples
    samples_input = input("Number of samples per source [50]: ").strip()
    samples = int(samples_input) if samples_input.isdigit() else 50
    
    # Ask for sources
    print("\nAvailable sources: qrng, camera, microphone, system")
    sources_input = input("Sources to use [qrng,camera,microphone,system]: ").strip()
    sources = sources_input if sources_input else "qrng,camera,microphone,system"
    
    python = get_python()
    run_command([
        python, "scripts/collect_device_fingerprint.py",
        "--device-name", device_name,
        "--samples", str(samples),
        "--sources", sources
    ])


def print_help():
    """Print help message."""
    print(__doc__)


def main():
    commands = {
        "setup": setup,
        "start": start_backend,
        "dev": start_dev,
        "frontend": start_frontend,
        "collect": collect_fingerprint,
        "train": run_training,
        "evaluate": run_evaluation,
        "test": run_tests,
        "clean": clean,
        "help": print_help,
        "-h": print_help,
        "--help": print_help,
    }
    
    if len(sys.argv) < 2:
        print_help()
        return 1
    
    cmd = sys.argv[1].lower()
    
    if cmd not in commands:
        print(f"Unknown command: {cmd}")
        print_help()
        return 1
    
    try:
        commands[cmd]()
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
