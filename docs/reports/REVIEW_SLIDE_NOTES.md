# QNA-Auth Review Slide Notes

## Slide 1 - Problem
- Traditional device authentication relies on static credentials and can be replayed or stolen.
- Goal: evaluate whether sensor-noise fingerprints can act as a practical device-verification signal.

## Slide 2 - System Overview
- Data sources: camera sensor noise + microphone noise.
- Pipeline: noise capture -> feature extraction -> Siamese embedding -> similarity + margin decision.
- API endpoints: enroll/authenticate/challenge/verify.

## Slide 3 - What We Built
- End-to-end backend in FastAPI.
- Feature extractor and embedding model in PyTorch.
- Enrollment stores embeddings and metadata.
- Authentication uses:
  - source consistency checks
  - source profile guard
  - similarity threshold
  - nearest-impostor margin guard

## Slide 4 - Evaluation Method
- Leakage-aware split policy.
- Method comparison:
  - Siamese embeddings
  - raw feature cosine
  - small MLP baseline
- Metrics: FAR, FRR, EER, threshold sweep, ROC/PR, attack ASR.

## Slide 5 - Results
- Present values from latest `results.json` only.
- Show ROC and PR plots from same run.
- Emphasize comparative result and chosen operating threshold.

## Slide 6 - Live Demo
- Enroll a device.
- Authenticate same device (pass).
- Authenticate impostor profile (reject).
- Show decision diagnostics (`similarity`, `threshold`, `observed_margin`).

## Slide 7 - Limitations (say this clearly)
- Prototype, not production-grade cryptographic identity.
- Sensitive to environment/hardware drift.
- Current dataset scale is limited; broader validation still needed.

## Slide 8 - Future Work
- Larger multi-session dataset and stronger longitudinal tests.
- Better liveness/anti-spoofing checks.
- Threshold calibration by deployment environment.

## Reviewer-Safe Phrasing
- Say: "prototype device-verification pipeline."
- Say: "measured under our dataset and split policy."
- Avoid: "unbreakable security" or "production-ready."
