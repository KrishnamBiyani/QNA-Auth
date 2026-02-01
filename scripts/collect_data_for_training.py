#!/usr/bin/env python3
"""
QNA-Auth: Standalone data collection script for training.

Run this script by itself to collect noise samples (QRNG, camera, microphone).
No project files needed.

Requirements (install before running):
  pip install numpy requests opencv-python sounddevice

  - numpy          (required)
  - requests       (required for QRNG)
  - opencv-python  (required for camera)
  - sounddevice    (optional, for microphone)

Note: The ANU QRNG API is limited to 1 request per minute. QRNG collection will
wait ~65 seconds between samples (e.g. 50 samples ≈ 50+ minutes). Use fewer
samples (e.g. --num-samples 5) or camera/microphone for faster collection.

  python collect_data_for_training.py

Creates a folder in the current directory with raw samples and manifest.json.
Zip that folder and send it to the project owner for training.

Usage:
  python collect_data_for_training.py
  python collect_data_for_training.py --name "Alice Laptop" --sources qrng,camera --num-samples 50
  python collect_data_for_training.py --name "Bob" --num-samples 30 --zip

Output (in current directory):
  qna_auth_collection_<name>_<timestamp>/
    manifest.json
    qrng/000.npy, 001.npy, ...
    camera/000.npy, ...   (if collected)
    microphone/000.npy, ... (if collected)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# -----------------------------------------------------------------------------
# Dependencies: numpy required; requests for QRNG; cv2 for camera; sounddevice for mic
# -----------------------------------------------------------------------------
try:
    import numpy as np
except ImportError:
    print("This script requires numpy. Run: pip install numpy")
    sys.exit(1)

try:
    import requests
except ImportError:
    requests = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import sounddevice as sd
except ImportError:
    sd = None


def _sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s).strip("._") or "device"


# -----------------------------------------------------------------------------
# QRNG: ANU API (no key required). Rate limit: 1 request per minute.
# -----------------------------------------------------------------------------
ANU_QRNG_URL = "https://qrng.anu.edu.au/API/jsonI.php"
ANU_QRNG_DELAY_SEC = 65  # ANU allows 1 request/minute; wait between calls


def _collect_qrng_standalone(num_samples: int, sample_size: int = 1024) -> list:
    if requests is None:
        raise RuntimeError("QRNG requires 'requests'. Run: pip install requests")
    sample_size = min(1024, max(1, sample_size))
    out = []
    for i in range(num_samples):
        try:
            r = requests.get(ANU_QRNG_URL, params={"length": sample_size, "type": "uint8"}, timeout=15)
            r.raise_for_status()
            data = r.json()
            if not data.get("success"):
                continue
            arr = np.array(data["data"], dtype=np.uint8)
            arr = arr.astype(np.float32) / 255.0
            out.append(arr)
            if num_samples > 1 and i < num_samples - 1:
                print(f"  QRNG sample {i+1}/{num_samples} ok; waiting {ANU_QRNG_DELAY_SEC}s for rate limit...")
                time.sleep(ANU_QRNG_DELAY_SEC)
        except Exception as e:
            print(f"  QRNG sample {i+1} failed: {e}")
            if i < num_samples - 1:
                time.sleep(ANU_QRNG_DELAY_SEC)
    return out


# -----------------------------------------------------------------------------
# Camera: dark-frame noise. Standalone implementation.
# -----------------------------------------------------------------------------
def _collect_camera_standalone(num_samples: int, camera_index: int = 0) -> list:
    if cv2 is None:
        raise RuntimeError("Camera requires 'opencv-python'. Run: pip install opencv-python")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Is it in use by another app?")
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(5):
            cap.read()  # warmup
        frames = []
        for i in range(num_samples):
            time.sleep(0.1)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Extract noise: subtract blurred to get high-freq component
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            noise = frame.astype(np.float32) - blurred.astype(np.float32)
            frames.append(noise.flatten().astype(np.float32))
        return frames
    finally:
        cap.release()


# -----------------------------------------------------------------------------
# Microphone: short recordings. Standalone implementation.
# -----------------------------------------------------------------------------
def _collect_microphone_standalone(
    num_samples: int, duration: float = 0.5, sample_rate: int = 44100
) -> list:
    if sd is None:
        raise RuntimeError(
            "Microphone requires 'sounddevice'. Run: pip install sounddevice"
        )
    out = []
    for i in range(num_samples):
        try:
            rec = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            arr = rec.flatten().astype(np.float32)
            if len(arr) > 2048:
                arr = arr[:2048]
            elif len(arr) < 1024 and len(arr) > 0:
                arr = np.pad(arr, (0, 1024 - len(arr)), mode="edge")
            out.append(arr)
        except Exception as e:
            print(f"  Mic sample {i+1} failed: {e}")
    return out


# -----------------------------------------------------------------------------
# Optional: use project noise_collection if this script is run from the repo
# -----------------------------------------------------------------------------
def _collect_qrng(num_samples: int, sample_size: int = 1024) -> list:
    try:
        from noise_collection import QRNGClient
        import os
        client = QRNGClient(api_key=os.getenv("QRNG_API_KEY"))
        # ANU allows 1 request/minute; fetch one at a time with delay
        samples = []
        for i in range(num_samples):
            try:
                s = client.fetch_quantum_noise(length=sample_size)
                arr = np.asarray(s, dtype=np.float32)
                if hasattr(s, "dtype") and np.issubdtype(s.dtype, np.integer):
                    arr = arr / 255.0
                samples.append(arr)
            except Exception as e:
                print(f"  QRNG sample {i+1} failed: {e}")
            if num_samples > 1 and i < num_samples - 1:
                print(f"  QRNG sample {i+1}/{num_samples} ok; waiting {ANU_QRNG_DELAY_SEC}s for rate limit...")
                time.sleep(ANU_QRNG_DELAY_SEC)
        return samples
    except ImportError:
        samples = _collect_qrng_standalone(num_samples, sample_size)
        return [np.asarray(s, dtype=np.float32) for s in samples]


def _collect_camera(num_samples: int) -> list:
    try:
        from noise_collection import CameraNoiseCollector
        collector = CameraNoiseCollector(camera_index=0)
        if not collector.initialize_camera():
            raise RuntimeError("Could not initialize camera")
        try:
            frames = collector.capture_multiple_frames(num_frames=num_samples, exposure_time=0.1)
            samples = [collector.extract_noise_features(f) for f in frames if f is not None]
            return [np.asarray(s, dtype=np.float32) for s in samples]
        finally:
            collector.release()
    except ImportError:
        return _collect_camera_standalone(num_samples)


def _collect_microphone(num_samples: int, duration: float = 0.5) -> list:
    try:
        from noise_collection import MicrophoneNoiseCollector
        collector = MicrophoneNoiseCollector(sample_rate=44100)
        samples = collector.capture_multiple_samples(num_samples=num_samples, duration=duration)
        out = []
        for s in samples:
            arr = np.asarray(s, dtype=np.float32).flatten()
            if len(arr) > 2048:
                arr = arr[:2048]
            elif len(arr) < 1024 and len(arr) > 0:
                arr = np.pad(arr, (0, 1024 - len(arr)), mode="edge")
            out.append(arr)
        return out
    except ImportError:
        return _collect_microphone_standalone(num_samples, duration)


def collect_and_save(
    output_dir: Path,
    device_name: str,
    sources: list[str],
    num_samples: int,
    create_zip: bool = False,
) -> Path:
    device_id = hashlib.sha256(
        f"{device_name}_{datetime.now().isoformat()}".encode()
    ).hexdigest()[:16]
    output_dir.mkdir(parents=True, exist_ok=True)
    collected = {}

    for source in sources:
        source_dir = output_dir / source
        source_dir.mkdir(exist_ok=True)
        print(f"\n--- Collecting {num_samples} samples from {source} ---")
        try:
            if source == "qrng":
                samples = _collect_qrng(num_samples)
            elif source == "camera":
                samples = _collect_camera(num_samples)
            elif source == "microphone":
                samples = _collect_microphone(num_samples)
            else:
                print(f"  Unknown source '{source}', skipping.")
                continue
            if not samples:
                print(f"  No samples collected from {source}, skipping.")
                continue
            for i, arr in enumerate(samples):
                np.save(source_dir / f"{i:03d}.npy", arr)
            collected[source] = len(samples)
            print(f"  Saved {len(samples)} samples to {source_dir}/")
        except Exception as e:
            print(f"  Error collecting {source}: {e}")
            continue

    if not collected:
        raise RuntimeError(
            "No data collected from any source. "
            "Check: pip install numpy requests opencv-python [sounddevice]"
        )

    manifest = {
        "device_id": device_id,
        "device_name": device_name,
        "sources": list(collected.keys()),
        "num_samples_per_source": num_samples,
        "actual_counts": collected,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "script_version": "1.0",
        "instructions": "Send this folder (or its .zip) to the project owner for training.",
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {output_dir / 'manifest.json'}")

    if create_zip:
        zip_path = output_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in output_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(output_dir.parent))
        print(f"Created {zip_path}")
        return zip_path
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Collect noise data for QNA-Auth training (standalone: run from any folder)."
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Participant/device name (e.g. 'Alice Laptop'). If omitted, will prompt.",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="qrng,camera,microphone",
        help="Comma-separated: qrng,camera,microphone (default: all three)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Samples per source (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder (default: ./qna_auth_collection_<name>_<timestamp> in current dir)",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Create a .zip of the output folder for easy sharing",
    )
    args = parser.parse_args()

    name = args.name
    if not name or not name.strip():
        name = input("Enter participant/device name (e.g. 'Alice Laptop'): ").strip() or "participant"
    name = name.strip()
    sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    if not sources:
        sources = ["qrng"]
    num_samples = max(1, min(200, args.num_samples))

    # Output in current working directory so running the script from anywhere creates data there
    cwd = Path.cwd()
    if not args.output_dir:
        safe_name = _sanitize(name)[:30]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = cwd / f"qna_auth_collection_{safe_name}_{ts}"
    output_dir = Path(args.output_dir).resolve()

    # Show what's available when running standalone
    print("QNA-Auth data collection (standalone)")
    print("  QRNG:       ", "ok" if requests else "missing (pip install requests)")
    print("  Camera:     ", "ok" if cv2 else "missing (pip install opencv-python)")
    print("  Microphone: ", "ok" if sd else "optional (pip install sounddevice)")
    print(f"  Name:       {name}")
    print(f"  Sources:    {sources}")
    print(f"  Samples:    {num_samples} per source")
    print(f"  Output:     {output_dir}")

    # Filter out sources we can't collect
    if not requests and "qrng" in sources:
        print("\nWarning: QRNG skipped (install requests). Using only: ", [s for s in sources if s != "qrng"])
        sources = [s for s in sources if s != "qrng"]
    if not cv2 and "camera" in sources:
        print("\nWarning: Camera skipped (install opencv-python).")
        sources = [s for s in sources if s != "camera"]
    if not sd and "microphone" in sources:
        print("\nWarning: Microphone skipped (install sounddevice).")
        sources = [s for s in sources if s != "microphone"]
    if not sources:
        print("No sources available. Install at least: pip install numpy requests")
        sys.exit(1)

    try:
        result = collect_and_save(
            output_dir=output_dir,
            device_name=name,
            sources=sources,
            num_samples=num_samples,
            create_zip=args.zip,
        )
    except RuntimeError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Done. Next steps:")
    print("  1. Send the folder (or the .zip) to the project owner.")
    print(f"     Folder: {output_dir}")
    if args.zip:
        print(f"     Zip:    {result}")
    print("  2. They will run: python scripts/ingest_collected_data.py <folder1> [folder2 ...]")
    print("     to merge your data into the training dataset.")
    print("=" * 60)


if __name__ == "__main__":
    main()
