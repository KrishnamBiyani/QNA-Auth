#!/usr/bin/env python3
"""
Build canonical dataset manifest + quality gate report for dataset/samples.

This is the source of truth for experiment data versioning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any


def _sha256_of_strings(items: list[str]) -> str:
    h = hashlib.sha256()
    for item in sorted(items):
        h.update(item.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def build_manifest_and_quality(
    dataset_dir: Path,
    min_samples_per_device: int = 20,
    required_sources: tuple[str, ...] = ("camera", "microphone"),
) -> Path:
    dataset_dir = Path(dataset_dir)
    json_dir = dataset_dir / "json"
    if not json_dir.exists():
        raise FileNotFoundError(f"Dataset json directory not found: {json_dir}")

    records: list[Dict[str, Any]] = []
    missing_raw = 0
    malformed = 0
    file_fingerprints: list[str] = []
    per_device_count = Counter()
    per_source_count = Counter()
    device_sources = defaultdict(set)
    device_sessions = defaultdict(set)

    for jf in sorted(json_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            device_id = data["device_id"]
            source = str(data.get("noise_source", "")).lower()
            rel_path = str(data.get("raw_data_path", "")).lstrip("/\\")
            raw_path = dataset_dir / rel_path
            if not raw_path.exists():
                missing_raw += 1
                continue
            session_id = data.get("session_id") or data.get("collection_folder") or "unknown_session"
            records.append(data)
            file_fingerprints.append(f"{jf.name}:{device_id}:{source}:{rel_path}")
            per_device_count[device_id] += 1
            per_source_count[source] += 1
            device_sources[device_id].add(source)
            device_sessions[device_id].add(session_id)
        except Exception:
            malformed += 1

    dataset_fingerprint = _sha256_of_strings(file_fingerprints)
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    quality_failures: list[str] = []
    for device_id, count in per_device_count.items():
        if count < min_samples_per_device:
            quality_failures.append(
                f"{device_id}: insufficient samples ({count} < {min_samples_per_device})"
            )
        missing_sources = set(required_sources) - device_sources[device_id]
        if missing_sources:
            quality_failures.append(
                f"{device_id}: missing required sources {sorted(missing_sources)}"
            )

    manifest = {
        "manifest_version": "v1",
        "created_at": now,
        "dataset_dir": str(dataset_dir),
        "dataset_fingerprint": dataset_fingerprint,
        "counts": {
            "total_samples": len(records),
            "unique_devices": len(per_device_count),
            "sources": dict(per_source_count),
            "missing_raw_files": missing_raw,
            "malformed_json": malformed,
        },
        "devices": {
            device_id: {
                "samples": per_device_count[device_id],
                "sources": sorted(device_sources[device_id]),
                "sessions": sorted(device_sessions[device_id]),
            }
            for device_id in sorted(per_device_count.keys())
        },
        "quality_gates": {
            "min_samples_per_device": min_samples_per_device,
            "required_sources": list(required_sources),
            "passed": len(quality_failures) == 0 and missing_raw == 0 and malformed == 0,
            "failures": quality_failures,
        },
    }

    out_path = dataset_dir / "manifest.v1.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dataset manifest + quality gate report")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/samples"))
    parser.add_argument("--min-samples-per-device", type=int, default=20)
    parser.add_argument(
        "--required-sources",
        type=str,
        default="camera,microphone",
        help="Comma-separated required sources per device",
    )
    args = parser.parse_args()

    req_sources = tuple(s.strip().lower() for s in args.required_sources.split(",") if s.strip())
    out = build_manifest_and_quality(
        dataset_dir=args.dataset_dir,
        min_samples_per_device=args.min_samples_per_device,
        required_sources=req_sources,
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
