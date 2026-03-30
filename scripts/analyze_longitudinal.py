#!/usr/bin/env python3
"""
Longitudinal drift analysis on session-tagged samples.

Produces per-session similarity drift against enrollment baseline.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from scripts.experiment_utils import load_sample_records, features_from_split


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze longitudinal session drift")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/samples"))
    parser.add_argument("--baseline-session", type=str, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/longitudinal/summary.json"))
    args = parser.parse_args()

    records = load_sample_records(args.data_dir)
    by_session = defaultdict(list)
    for r in records:
        by_session[r.session_id].append(r)
    if args.baseline_session not in by_session:
        raise ValueError(f"Baseline session not found: {args.baseline_session}")

    session_features: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for session_id, recs in by_session.items():
        session_features[session_id] = features_from_split({"tmp": recs})["tmp"]

    baseline = session_features[args.baseline_session]
    baseline_centroids = {
        dev: np.mean(np.asarray(vecs), axis=0) for dev, vecs in baseline.items() if vecs
    }

    drift_rows = []
    for session_id, dev_map in session_features.items():
        for dev, vecs in dev_map.items():
            if dev not in baseline_centroids or not vecs:
                continue
            sims = [cosine(v, baseline_centroids[dev]) for v in vecs]
            drift_rows.append(
                {
                    "session_id": session_id,
                    "device_id": dev,
                    "n_samples": len(sims),
                    "mean_similarity_to_baseline": float(np.mean(sims)),
                    "std_similarity_to_baseline": float(np.std(sims)),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"baseline_session": args.baseline_session, "rows": drift_rows}, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
