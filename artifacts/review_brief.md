# QNA-Auth Review Brief

## Scope and Claim Boundaries
- This is a **prototype device-authentication system**, not production-grade security.
- Evaluation uses the configured split policy and dataset subset in `results.json`.
- Results indicate comparative model behavior under this setup; they do not claim universal robustness.

## Core Metrics (Siamese)
- Optimal threshold: `1.0000`
- EER: `0.1000`
- FAR: `0.2000`
- FRR: `0.0000`
- Accuracy: `0.8333`

## Baseline Comparison
- Raw feature cosine EER: `0.0000`
- Small MLP embedding EER: `0.0000`
- Siamese EER: `0.1000`

## Attack Surface Snapshot
- Replay ASR mean: `0.0000`
- Impersonation ASR mean: `0.0000`
- Synthetic-stats ASR mean: `0.0000`

## Live Demo Narrative
1. Enroll a device with camera + microphone samples.
2. Authenticate with same device samples (expected pass).
3. Authenticate with impostor profile (expected reject).
4. Show threshold and margin checks from API response details.

## Known Limitations
- Performance and metrics depend on data quality and source balance.
- Challenge-response flow is experimental and should be framed as prototype hardening.
- Hardware and environment noise drift can change outcomes; demo fallback is required.

## Artifact Pointers
- Results JSON: `artifacts\capstone_eval\20260422T122948Z\results.json`
- ROC/PR files: same run directory as `results.json`
