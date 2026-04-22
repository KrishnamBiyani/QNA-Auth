# QNA-Auth Future Reference (Consolidated)

This file consolidates the previous roadmap/status/vision/threat/report planning notes into one long-term reference.

## 1) Bounded thesis (canonical wording)

Use this framing in reports and presentations:

- QNA-Auth is a **sensor-based device verification** research prototype.
- QRNG is used as **fresh entropy support**, not as standalone device identity.
- Core measurable claim: under defined collection conditions, sensor-derived features + metric learning achieve target EER/FAR/FRR on held-out evaluation.
- Challenge/response endpoints are **experimental** and must not be presented as hardware-backed key authentication unless a real key hierarchy exists.

Out of scope claims:
- formal cryptographic security proof
- TEE/secure-enclave guarantees
- production-grade security/compliance

## 2) Current state snapshot

Implemented now:
- end-to-end enrollment/auth flow with FastAPI + React demo UI
- canonical feature extraction + Siamese embedding model
- dataset ingestion and manifest generation
- capstone evaluation runner with split artifacts, baselines, attacks, and CI-friendly outputs

Known practical limits:
- embedding storage is file-based unless hardened
- challenge/response is not equivalent to hardware trust anchor
- longitudinal conclusions require real multi-week data collection

## 3) Threat model boundaries

In-scope adversaries:
- replay of old samples
- impersonation from other devices
- synthetic statistical attempts

Out-of-scope adversaries:
- physical secure-enclave extraction
- full side-channel / fault-injection campaigns
- formal reduction-based crypto attacks

Reporting rule:
- every security/performance claim must map to an artifact (results JSON, split file, figure).

## 4) Evidence package (what to keep for defense)

Minimum artifacts:
1. `dataset/samples/manifest.v1.json`
2. `artifacts/splits/split_*.json`
3. `artifacts/capstone_eval/<run_id>/results.json`
4. `artifacts/capstone_eval/<run_id>/roc_siamese.png`
5. `artifacts/capstone_eval/<run_id>/pr_siamese.png`
6. `artifacts/repro_runs/<run_id>/run_metadata.json`
7. `artifacts/longitudinal/summary.json` (if session-tagged data exists)

Claim-to-evidence mapping:
- viability -> `methods.siamese` metrics in `results.json`
- baseline improvement -> `raw_feature_cosine` and `small_mlp_embedding` vs `siamese`
- leakage safety -> split artifact + test-only evaluation scope
- attack coverage -> `attacks` section (ASR values)
- temporal drift -> longitudinal summary and figures

## 5) Prioritized execution order (practical)

1. Data integrity and manifest quality gates
2. Leakage-safe split policy (device-held-out or session-held-out)
3. Baseline table on identical splits
4. Threshold policy from validation/EER and unified runtime config
5. Attack evaluation with repeatable artifacts
6. Longitudinal drift study with session metadata
7. Reproducibility one-command runner + CI
8. Final report packaging and slide narrative

## 6) Report checklist (condensed)

Before finalizing:
- Use only held-out test or LODO metrics for headline numbers
- Include EER + FAR/FRR at chosen operating threshold
- Include baseline comparison table
- Include limitations and explicit out-of-scope statements
- Avoid unsupported language (for example, "formally secure" or "production-ready")

## 7) Recommended command set

- Build/refresh manifest:
  - `python scripts/data/build_dataset_manifest.py --dataset-dir dataset/samples`
- Train + evaluate:
  - `python scripts/training/train_and_evaluate.py --data-dir dataset/samples --seed 42 --epochs 20`
- Capstone evaluation (splits + baselines + attacks):
  - `python scripts/training/run_capstone_evaluation.py --data-dir dataset/samples --seed 42`
- One-command reproducibility:
  - `python scripts/training/reproduce_capstone.py --data-dir dataset/samples --seed 42 --epochs 20`
- Longitudinal drift:
  - `python scripts/data/analyze_longitudinal.py --data-dir dataset/samples --baseline-session <session_id>`
