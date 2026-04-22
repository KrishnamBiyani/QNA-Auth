#!/usr/bin/env python3
"""
Demo preflight checks for capstone review day.

Checks:
1) model checkpoint exists
2) DB can be opened
3) camera and microphone can capture at least one sample
4) API health endpoint responds
5) optional enroll/auth round-trip using captured client samples
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from noise_collection import CameraNoiseCollector, MicrophoneNoiseCollector


def _ok(label: str, detail: str = "") -> None:
    print(f"[OK] {label}" + (f" - {detail}" if detail else ""))


def _fail(label: str, detail: str = "") -> None:
    print(f"[FAIL] {label}" + (f" - {detail}" if detail else ""))


def check_model_exists() -> bool:
    model_path = Path(getattr(config, "MODEL_PATH", config.MODEL_CONFIG.get("model_path")))
    if model_path.exists():
        _ok("Model checkpoint found", str(model_path))
        return True
    _fail("Model checkpoint missing", str(model_path))
    return False


def check_db() -> bool:
    db_url = getattr(config, "DATABASE_URL", "sqlite:///./data/qna_auth.db")
    if not db_url.startswith("sqlite:///"):
        _ok("Database URL configured", db_url)
        return True
    db_path = Path(db_url.replace("sqlite:///", "", 1))
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.execute("SELECT 1;")
        conn.close()
        _ok("SQLite reachable", str(db_path))
        return True
    except Exception as exc:
        _fail("SQLite check failed", str(exc))
        return False


def check_camera() -> tuple[bool, np.ndarray | None]:
    collector = CameraNoiseCollector(camera_index=0)
    try:
        if not collector.initialize_camera():
            _fail("Camera", "could not initialize")
            return False, None
        frame = collector.capture_dark_frame(exposure_time=0.1)
        if frame is None:
            _fail("Camera", "no frame captured")
            return False, None
        noise = collector.extract_noise_features(frame).astype(np.float32)
        sample = noise.flatten()[:1024]
        _ok("Camera capture", f"samples={sample.shape[0]}")
        return True, sample
    except Exception as exc:
        _fail("Camera", str(exc))
        return False, None
    finally:
        collector.release()


def check_microphone() -> tuple[bool, np.ndarray | None]:
    collector = MicrophoneNoiseCollector(sample_rate=44100)
    try:
        sample = collector.capture_ambient_noise(duration=0.3)
        if sample is None:
            _fail("Microphone", "no audio captured")
            return False, None
        sample = sample.astype(np.float32).flatten()[:1024]
        _ok("Microphone capture", f"samples={sample.shape[0]}")
        return True, sample
    except Exception as exc:
        _fail("Microphone", str(exc))
        return False, None


def check_api_health(base_url: str) -> bool:
    try:
        resp = requests.get(f"{base_url}/health", timeout=10)
        if resp.status_code != 200:
            _fail("API /health", f"status={resp.status_code}")
            return False
        payload = resp.json()
        _ok("API /health", json.dumps(payload))
        return True
    except Exception as exc:
        _fail("API /health", str(exc))
        return False


def run_roundtrip(base_url: str, camera_sample: np.ndarray, mic_sample: np.ndarray) -> bool:
    try:
        camera_samples = [camera_sample.tolist() for _ in range(10)]
        mic_samples = [mic_sample.tolist() for _ in range(10)]
        enroll_payload = {
            "device_name": "PreflightDevice",
            "num_samples": 10,
            "sources": ["camera", "microphone"],
            "client_samples": {
                "camera": camera_samples,
                "microphone": mic_samples,
            },
        }
        enroll_resp = requests.post(f"{base_url}/enroll", json=enroll_payload, timeout=90)
        if enroll_resp.status_code != 201:
            _fail("API /enroll", f"status={enroll_resp.status_code} body={enroll_resp.text}")
            return False
        device_id = enroll_resp.json()["device_id"]
        _ok("API /enroll", f"device_id={device_id}")

        auth_payload = {
            "device_id": device_id,
            "sources": ["camera", "microphone"],
            "num_samples_per_source": 5,
            "client_samples": {
                "camera": [camera_sample.tolist() for _ in range(5)],
                "microphone": [mic_sample.tolist() for _ in range(5)],
            },
        }
        auth_resp = requests.post(f"{base_url}/authenticate", json=auth_payload, timeout=90)
        if auth_resp.status_code != 200:
            _fail("API /authenticate", f"status={auth_resp.status_code} body={auth_resp.text}")
            return False
        auth_data = auth_resp.json()
        if not auth_data.get("authenticated", False):
            _fail("API round-trip auth", f"rejected similarity={auth_data.get('similarity')}")
            return False
        _ok("API /authenticate", f"similarity={auth_data.get('similarity', 0.0):.4f}")
        return True
    except Exception as exc:
        _fail("API round-trip", str(exc))
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run capstone demo preflight checks")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--skip-roundtrip", action="store_true")
    args = parser.parse_args()

    print("=== QNA-Auth Demo Preflight ===")
    checks = []
    checks.append(check_model_exists())
    checks.append(check_db())
    cam_ok, camera_sample = check_camera()
    mic_ok, mic_sample = check_microphone()
    checks.extend([cam_ok, mic_ok])
    checks.append(check_api_health(args.base_url))

    if not args.skip_roundtrip and cam_ok and mic_ok and camera_sample is not None and mic_sample is not None:
        checks.append(run_roundtrip(args.base_url, camera_sample, mic_sample))

    passed = all(checks)
    print("\n=== Result ===")
    if passed:
        print("PRE-FLIGHT PASSED")
        return 0
    print("PRE-FLIGHT FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
