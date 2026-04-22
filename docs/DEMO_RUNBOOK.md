# QNA-Auth Demo Runbook (Review Day)

## 1) One-Time Setup (before review day)
- Ensure dependencies are installed and server starts cleanly.
- Keep `config.py` with:
  - `DEMO_MODE = True`
  - `DEMO_ALLOWED_SOURCES = ["camera", "microphone"]`
  - `PREPROCESSING_FAST_MODE = True`
  - `AUTH_CONFIDENCE_STRONG = 0.97`
  - `AUTH_CONFIDENCE_UNCERTAIN = 0.92`
  - `AUTH_IDENTIFICATION_MARGIN = 0.02`
  - `AUTH_SOURCE_WEIGHTS = {"camera": 0.7, "microphone": 0.3}`
- Re-enroll demo devices after any auth/profile logic changes.

## 2) Preflight (30-60 min before demo)
1. Start backend:
   - `python -m uvicorn server.app:app --host 0.0.0.0 --port 8000`
2. Run preflight checks in another terminal:
   - `python scripts/diagnostics/demo_preflight.py --base-url http://127.0.0.1:8000`
3. Confirm you see:
   - model checkpoint OK
   - DB reachable
   - camera and microphone capture OK
   - API health OK
   - enroll/auth round-trip OK

If any preflight step fails, use fallback flow in section 5.

## 3) Live Demo Script (2-3 minutes)
1. Show `/health` and `/devices` quickly.
2. Enroll one device with camera + microphone samples.
3. Authenticate same device with fresh camera + microphone samples (expected strong match).
4. Attempt impostor authentication using a different sample profile (expected reject or uncertain).
5. Call out response details:
   - `similarity`
   - `confidence_band`
   - `observed_margin`
   - `margin_check_passed`
   - `recommended_action`
6. If asked about stability, say it is measured empirically and should be supported with cross-session score distributions rather than assumed invariance.
7. If asked about drift updates, say they are gated to `strong_accept` matches only and can be made stricter by requiring multiple strong accepts before updating.

## 4) Exact API Paths to Demonstrate
- `POST /enroll`
- `POST /authenticate`
- `GET /devices`
- `GET /health`

## 5) Fallback Demo (if live hardware is unstable)
1. Use previously captured client samples payloads (camera + microphone).
2. Replay payloads to:
   - successful genuine auth example
   - failed impostor auth example
3. Show offline evaluation artifacts:
   - `artifacts/capstone_eval/<run_id>/results.json`
   - `artifacts/capstone_eval/<run_id>/roc_siamese.png`
   - `artifacts/capstone_eval/<run_id>/pr_siamese.png`
   - `artifacts/review_brief.md`

## 6) Failure-to-Action Mapping
- Camera init fails -> switch to fallback payload demo immediately.
- Mic capture fails -> switch to fallback payload demo immediately.
- High false accepts -> increase `AUTH_CONFIDENCE_STRONG` by `+0.01` and retry preflight.
- Frequent false rejects -> lower `AUTH_CONFIDENCE_UNCERTAIN` or collect more samples before fallback.
- Concern about gradual poisoning -> disable drift updates for the demo or require repeated strong accepts before template update.

## 7) Rehearsal Checklist (run 3 times)
- [ ] Server starts cleanly
- [ ] Preflight passes
- [ ] Enroll succeeds
- [ ] Genuine auth succeeds
- [ ] Impostor auth rejected
- [ ] Metrics artifacts open locally
- [ ] Fallback demo can be executed in <90 seconds
