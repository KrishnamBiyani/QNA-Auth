# QNA-Auth: Report / Slides Template

Use this outline for your written report or presentation (problem, method, experiments, limitations, future work).

---

## 1. Problem and motivation

- **Problem**: Reliable device authentication (e.g. “is this the same laptop/phone?”) without passwords or tokens that can be stolen or replayed.
- **Idea**: Use **device-specific noise** (camera sensor, microphone, optional QRNG) as a “fingerprint” and a **Siamese embedding model** so the same device produces similar embeddings and different devices produce different ones.
- **Related work**: Device fingerprinting, PUFs, QRNG in security, metric learning for verification.

---

## 2. Method (pipeline)

- **Noise sources**: QRNG (ANU API), camera (dark-frame/sensor noise), microphone (ambient/self-noise). Same pipeline for enrollment and authentication.
- **Feature pipeline**: One canonical feature list (versioned); stats, entropy, FFT, autocorrelation, complexity. Same preprocessing in training and server.
- **Model**: Siamese network mapping feature vector → 128-D embedding; trained with triplet (or contrastive) loss on multi-device data so same-device pairs are close and different-device pairs are far.
- **Enrollment**: Collect N samples → features → embedder → mean embedding → store.
- **Authentication**: Fresh samples → features → embedder → compare to stored embedding (e.g. cosine similarity ≥ threshold).
- **Challenge/response**: Optional; server issues nonce; client responds with HMAC-like value using stored embedding + nonce; server still checks live embedding similarity.

Include a **pipeline diagram** (noise → features → embedder → similarity → accept/reject).

---

## 3. Experiments and results

- **Metrics**: FAR (false accept rate), FRR (false reject rate), EER (equal error rate), ROC, threshold sweep. Use the same metrics in the report as produced by `scripts/run_evaluation.py`.
- **Ablations**: Table comparing performance by noise source:
  - QRNG-only
  - Camera-only
  - Microphone-only
  - Combined (all sources)
  
  (Run `python scripts/run_evaluation.py` and paste the ablation table; add EER, FAR, FRR, accuracy columns.)
- **Reproducibility**: Fixed seed (e.g. 42), canonical feature version, single “run evaluation” path (`scripts/run_evaluation.py`). Mention: “We report metrics from `scripts/run_evaluation.py` with seed 42 and feature version X.”
- **Cross-device**: Evaluation uses pairs from different devices (negatives) and same device (positives); mention if you have multiple devices/sessions and how train/test or leave-one-device-out was done (if applicable).

---

## 4. Limitations

- **QRNG**: Not device-specific; same for everyone. Device identity comes mainly from camera/mic.
- **Security**: Prototype only; no formal security analysis; API key and rate limiting are optional; embeddings stored as files unless you add encryption/DB.
- **Data**: Small or single-session data may overfit; need multi-device, multi-session data for robust claims.
- **Failure cases**: Auth can fail under low samples, bad lighting, mic off, or model not trained on similar devices.

---

## 5. Future work

- Encrypt embeddings at rest; store in DB.
- Stronger API protection (JWT, rate limiting by user).
- Cross-session and cross-device evaluation (train on one session, test on another).
- Confidence intervals (bootstrap) for EER/FAR/FRR.
- Baseline comparison (e.g. raw-feature cosine similarity vs Siamese).

---

## 6. How to reproduce (short)

1. Install: `pip install -r requirements.txt`
2. Collect data: `scripts/collect_data_for_training.py` (participants); `scripts/ingest_collected_data.py` (merge).
3. Train: `scripts/train_and_evaluate.py --seed 42`
4. Evaluate: `scripts/run_evaluation.py`
5. Start server: `python server/app.py`

Document the seed, feature version, and dataset (path or version) used for the reported numbers.
