#!/usr/bin/env python3
"""
Ingest participant data collected with collect_data_for_training.py.

Merges one or more collection folders into the project's dataset (dataset/samples)
so you can run training. Each folder should contain manifest.json and subdirs
qrng/, camera/, microphone/ with 000.npy, 001.npy, ...

Usage (from project root):
  python scripts/data/ingest_collected_data.py path/to/collection_folder1 [folder2 ...]
  python scripts/data/ingest_collected_data.py ./qna_auth_collection_Alice_20250128_120000

If a folder is a .zip, it will be extracted to a temp dir and ingested.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_collection_folder(folder: Path) -> tuple[str, str | None, dict[str, list[np.ndarray]], dict]:
    """Load collection folder and return (device_id, device_name, samples_by_source, ingestion_metadata)."""
    folder = Path(folder).resolve()
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {folder}")

    with open(manifest_path) as f:
        manifest = json.load(f)
    device_id = manifest.get("device_id")
    device_name = manifest.get("device_name")
    if not device_id:
        device_id = manifest.get("device_name", "unknown") + "_" + folder.name

    samples_by_source: dict[str, list[np.ndarray]] = {}
    for source in ("qrng", "camera", "microphone"):
        source_dir = folder / source
        if not source_dir.is_dir():
            continue
        npy_files = sorted(source_dir.glob("*.npy"))
        if not npy_files:
            continue
        samples = []
        for p in npy_files:
            arr = np.load(p)
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            if arr.ndim > 1:
                arr = arr.flatten()
            samples.append(arr)
        samples_by_source[source] = samples
    if not samples_by_source:
        raise ValueError(f"No sample subdirs (qrng/camera/microphone) with .npy files in {folder}")
    created_at = manifest.get("created_at")
    session_id = manifest.get("session_id") or folder.name
    ingestion_metadata = {
        "session_id": session_id,
        "collection_created_at": created_at,
        "collection_folder": folder.name,
        "ingested_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "collector_script_version": manifest.get("script_version", "unknown"),
    }
    return device_id, device_name, samples_by_source, ingestion_metadata


def ingest_folders(folders: list[Path], dataset_dir: Path | None = None) -> None:
    if dataset_dir is None:
        dataset_dir = PROJECT_ROOT / "dataset" / "samples"
    from dataset.builder import DatasetBuilder
    builder = DatasetBuilder(base_dir=str(dataset_dir))
    for path in folders:
        path = Path(path).resolve()
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                extract_to = path.with_name(path.stem + "_extracted")
                extract_to.mkdir(exist_ok=True)
                zf.extractall(extract_to)
            path = extract_to
            # If zip had a single root folder, use it
            subdirs = [d for d in path.iterdir() if d.is_dir()]
            if len(subdirs) == 1 and (subdirs[0] / "manifest.json").exists():
                path = subdirs[0]
        if not path.is_dir():
            print(f"Skipping (not a directory): {path}")
            continue
        try:
            device_id, device_name, samples_by_source, ingestion_metadata = load_collection_folder(path)
            print(f"Ingesting {path.name} -> device_id={device_id}, name={device_name}, sources={list(samples_by_source)}")
            for source, samples in samples_by_source.items():
                builder.add_batch(
                    device_id=device_id,
                    noise_source=source,
                    samples=samples,
                    extra_metadata=ingestion_metadata,
                )
            print(f"  Added {sum(len(s) for s in samples_by_source.values())} samples.")
        except Exception as e:
            print(f"  Error ingesting {path}: {e}")
            continue
    print(f"Dataset written to {dataset_dir}")
    # Generate/update canonical manifest + quality gate report after ingestion.
    try:
        from scripts.data.build_dataset_manifest import build_manifest_and_quality
        manifest = build_manifest_and_quality(dataset_dir=dataset_dir)
        print(f"Dataset manifest updated: {manifest}")
    except Exception as e:
        print(f"Warning: failed to build dataset manifest: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/data/ingest_collected_data.py <folder1> [folder2 ...]")
        print("  Each folder (or .zip) should be from collect_data_for_training.py")
        sys.exit(1)
    ingest_folders([Path(p) for p in sys.argv[1:]])


if __name__ == "__main__":
    main()
