#!/usr/bin/env python3
"""
Generate a reviewer-safe markdown brief from capstone evaluation artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _find_latest_results(base_dir: Path) -> Path:
    candidates = sorted(base_dir.glob("*/results.json"))
    if not candidates:
        raise FileNotFoundError(f"No results.json found under {base_dir}")
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate reviewer-safe markdown brief")
    parser.add_argument("--results", type=Path, default=None, help="Path to results.json")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/capstone_eval"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/review_brief.md"))
    args = parser.parse_args()

    results_path = args.results or _find_latest_results(args.artifacts_dir)
    data = json.loads(results_path.read_text(encoding="utf-8"))

    siamese = data["methods"]["siamese"]
    raw = data["methods"]["raw_feature_cosine"]
    mlp = data["methods"]["small_mlp_embedding"]
    attacks = data.get("attacks", {})

    brief = f"""# QNA-Auth Review Brief

## Scope and Claim Boundaries
- This is a **prototype noise-based device verification system**, not production-grade security.
- Stored embeddings are biometric-like feature templates used for similarity, not credentials.
- Evaluation uses the configured split policy and dataset subset in `results.json`.
- Results indicate comparative model behavior under this setup; they do not claim universal robustness.

## Core Metrics (Siamese)
- Optimal threshold: `{siamese['optimal_threshold']:.4f}`
- EER: `{siamese['eer']:.4f}`
- FAR: `{siamese['metrics']['far']:.4f}`
- FRR: `{siamese['metrics']['frr']:.4f}`
- Accuracy: `{siamese['metrics']['accuracy']:.4f}`

## Baseline Comparison
- Raw feature cosine EER: `{raw['eer']:.4f}`
- Small MLP embedding EER: `{mlp['eer']:.4f}`
- Siamese EER: `{siamese['eer']:.4f}`

## Attack Surface Snapshot
- Replay ASR mean: `{attacks.get('replay_asr', {}).get('mean', 0.0):.4f}`
- Impersonation ASR mean: `{attacks.get('impersonation_asr', {}).get('mean', 0.0):.4f}`
- Synthetic-stats ASR mean: `{attacks.get('synthetic_stats_asr', {}).get('mean', 0.0):.4f}`

## Live Demo Narrative
1. Enroll a device with camera + microphone samples.
2. Authenticate with same device samples (expected strong accept).
3. Authenticate with impostor profile (expected reject or uncertain).
4. Show confidence band, margin checks, and next action from API response details.

## Known Limitations
- Performance and metrics depend on data quality and source balance.
- Challenge hardening is server-side prototype hardening, not full client credential proof.
- Hardware and environment noise drift can change outcomes; demo fallback is required.
- Template stability should be justified with cross-session variance or score-distribution evidence rather than assumed.

## Attacker Model Snapshot
- Replayed samples are mitigated by single-use nonce-bound verification.
- Naive synthetic noise should usually fail because matching depends on higher-dimensional structure, not just simple summary statistics.
- Adaptive or learned mimicry could get closer and remains a residual risk.
- Same device under changed environmental conditions may fall into the uncertain band and require fallback auth.

## Drift Update Policy
- Rolling re-enrollment is limited to the strong-accept band.
- A stricter deployment can require multiple strong accepts before applying EMA updates to reduce gradual poisoning risk.

## Artifact Pointers
- Results JSON: `{results_path}`
- ROC/PR files: same run directory as `results.json`
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(brief, encoding="utf-8")
    print(f"Wrote review brief: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
