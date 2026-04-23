from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from scripts.data.collect_data_for_training import (
    _microphone_sample_quality,
    _normalize_microphone_array,
)


SUPPORTED_COLLECTOR_SOURCES = {"camera", "microphone"}


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._") or "device"


def stable_device_id(device_key: str) -> str:
    return hashlib.sha256(sanitize_name(device_key).lower().encode("utf-8")).hexdigest()[:16]


def _fingerprint_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).hexdigest()


def _prepare_camera_sample(sample: List[float]) -> np.ndarray:
    arr = np.asarray(sample, dtype=np.float32).flatten()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size < 64:
        raise ValueError("Camera sample is too small")
    return arr


def _prepare_microphone_sample(sample: List[float], seen_hashes: set[str]) -> Optional[np.ndarray]:
    arr = _normalize_microphone_array(np.asarray(sample, dtype=np.float32).flatten())
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    quality = _microphone_sample_quality(arr)
    if not quality["valid"]:
        return None
    sample_hash = _fingerprint_array(arr)
    if sample_hash in seen_hashes:
        return None
    seen_hashes.add(sample_hash)
    return arr


@dataclass
class SavedCollection:
    folder_path: Path
    manifest_path: Path
    zip_path: Optional[Path]
    device_id: str
    counts: Dict[str, int]


def save_browser_collection(
    base_dir: Path,
    device_name: str,
    device_key: str,
    source_samples: Dict[str, List[List[float]]],
    session_id: Optional[str] = None,
    environment_label: Optional[str] = None,
    notes: Optional[str] = None,
    operator: Optional[str] = None,
    create_zip: bool = False,
) -> SavedCollection:
    sanitized_name = sanitize_name(device_name or "device")
    sanitized_key = sanitize_name(device_key or sanitized_name)
    device_id = stable_device_id(sanitized_key)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    folder = base_dir / f"qna_auth_collection_{sanitized_name}_{timestamp}"
    folder.mkdir(parents=True, exist_ok=True)

    source_counts: Dict[str, int] = {}
    duplicate_rejections = 0
    invalid_rejections = 0

    for source, samples in source_samples.items():
        if source not in SUPPORTED_COLLECTOR_SOURCES:
            continue
        source_dir = folder / source
        source_dir.mkdir(parents=True, exist_ok=True)
        saved_index = 0
        seen_hashes: set[str] = set()

        for sample in samples:
            try:
                if source == "camera":
                    arr = _prepare_camera_sample(sample)
                else:
                    arr = _prepare_microphone_sample(sample, seen_hashes)
                    if arr is None:
                        fingerprint = _fingerprint_array(_normalize_microphone_array(np.asarray(sample, dtype=np.float32)))
                        if fingerprint in seen_hashes:
                            duplicate_rejections += 1
                        else:
                            invalid_rejections += 1
                        continue
                np.save(source_dir / f"{saved_index:03d}.npy", arr)
                saved_index += 1
            except Exception:
                invalid_rejections += 1
                continue

        if saved_index > 0:
            source_counts[source] = saved_index

    if not source_counts:
        raise ValueError("No valid samples were captured")

    actual_session_id = session_id or f"session_{timestamp}"
    manifest = {
        "device_id": device_id,
        "device_name": device_name,
        "device_key": sanitized_key,
        "session_id": actual_session_id,
        "sources": sorted(source_counts.keys()),
        "num_samples_per_source": max(source_counts.values()),
        "actual_counts": source_counts,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "script_version": "browser-collector-1.0",
        "collection_kind": "real",
        "protocol": {
            "protocol_version": "browser-collector-1.0",
            "collection_kind": "real",
            "device_name": device_name,
            "device_key": sanitized_key,
            "device_id": device_id,
            "session_id": actual_session_id,
            "environment_label": environment_label or "unspecified",
            "operator": operator or "unspecified",
            "notes": notes or "",
            "sources_requested": sorted(source_samples.keys()),
            "requested_samples_per_source": max(len(samples) for samples in source_samples.values() if samples),
            "collector": "browser_local_network",
        },
        "capture_summary": {
            "duplicate_rejections": duplicate_rejections,
            "invalid_rejections": invalid_rejections,
        },
        "instructions": "This folder was created by the LAN browser collector and can be ingested with scripts/data/ingest_collected_data.py",
    }
    manifest_path = folder / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    zip_path: Optional[Path] = None
    if create_zip:
        zip_path = folder.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in folder.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(folder.parent))

    return SavedCollection(
        folder_path=folder,
        manifest_path=manifest_path,
        zip_path=zip_path,
        device_id=device_id,
        counts=source_counts,
    )
