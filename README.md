# QNA-Auth

QNA-Auth is a capstone prototype for **noise-based device verification** using camera and microphone sensor fingerprints plus learned embeddings.
The current framing is deliberate:
- embeddings are **biometric-like templates used only for similarity**
- similarity is **not identity proof**
- the system produces **high-confidence device matching**, not cryptographic identity
## Current Model

The runtime now uses:
- per-source embeddings for `camera` and `microphone`
- weighted fusion at decision time with `camera=0.7`, `microphone=0.3`
- confidence bands instead of a single binary threshold
- rolling re-enrollment with EMA drift updates after strong matches
- nonce-bound HKDF hardening for `/challenge` -> `/verify`

Confidence bands:
- `>= 0.97`: strong accept
- `0.92 - 0.97`: uncertain, collect more samples or use fallback auth
- `< 0.92`: reject

Challenge hardening:
- the embedding is never used directly as a credential
- a stable feature template is derived from the live embedding
- the MAC key is derived with `HKDF(template || nonce || server_secret)`
- the nonce prevents replay and the server secret hardens against offline spoofing

Stability note:
- "stable" here means stable enough for repeated similarity matching after quantization and aggregation, not perfectly invariant
- the remaining stability question is empirical: reviewers may ask for cross-session variance, environmental sensitivity, and confidence-band behavior
- if you have evaluation plots or session-separated score distributions, present those as evidence rather than claiming full invariance

Drift update policy:
- rolling re-enrollment is gated to the `strong_accept` band only
- EMA updates are intended for gradual sensor drift, not for expanding identity claims
- a conservative deployment can require multiple strong accepts before applying an update to reduce gradual poisoning risk

## Attacker Model

This project explicitly models:
- replayed samples: blocked by nonce-bound single-use challenge flow
- naive synthetic noise: expected to fail because matching depends on higher-dimensional structure, not just simple summary statistics
- adaptive or learned mimicry: could get closer and should be treated as a real residual risk
- same device in a different environment: may produce borderline results and fall into the uncertain band

## Quick Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Create config.

```bash
# Windows
copy config.example.py config.py

# macOS/Linux
cp config.example.py config.py
```

4. Set a real server secret before review/demo use.

```bash
# PowerShell
$env:QNA_AUTH_SERVER_SECRET="replace-with-a-random-secret"
```

## Recommended Demo Config

Set these in `config.py`:
- `DEMO_MODE = True`
- `DEMO_ALLOWED_SOURCES = ["camera", "microphone"]`
- `DEMO_ENROLL_NUM_SAMPLES = 10`
- `DEMO_AUTH_NUM_SAMPLES = 5`
- `PREPROCESSING_FAST_MODE = True`
- `AUTH_CONFIDENCE_STRONG = 0.97`
- `AUTH_CONFIDENCE_UNCERTAIN = 0.92`
- `AUTH_IDENTIFICATION_MARGIN = 0.02`
- `AUTH_SOURCE_WEIGHTS = {"camera": 0.7, "microphone": 0.3}`

## Run

Backend:

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Useful URLs:
- `http://localhost:8000`
- `http://localhost:8000/docs`
- `http://localhost:8000/health`

## API

- `POST /enroll`
- `POST /authenticate`
- `POST /challenge`
- `POST /verify`
- `GET /devices`
- `GET /devices/{device_id}`
- `DELETE /devices/{device_id}`

Example auth payload:

```json
{
  "device_id": "abc123",
  "sources": ["camera", "microphone"],
  "num_samples_per_source": 5
}
```

## Evaluation and Review

Pre-demo reliability check:

```bash
python scripts/diagnostics/demo_preflight.py --base-url http://127.0.0.1:8000
```

Reviewer-safe brief:

```bash
python scripts/reporting/generate_review_brief.py
```

Important paths:
- `auth/device_embeddings/` - stored template bundles and metadata
- `artifacts/` - evaluation outputs and review material
- `docs/DEMO_RUNBOOK.md` - live demo guide

## Claim Boundaries

This system is not:
- a spoof-proof system
- a quantum-based identity system
- a replacement for cryptographic credentials

This system is:
- a multi-source sensor noise fingerprinting prototype
- a learned similarity system with statistical decision rules
- a security-aware capstone with optional challenge hardening
