# QNA-Auth: Current State & What’s Left to Complete

This doc summarizes **what the project currently uses** and **what still needs to be done** to reach a research-level capstone. The full wishlist is in [ROADMAP.md](../ROADMAP.md).

---

## What We’re Using Right Now

### Backend & data
- **FastAPI server** (`server/app.py`) – enrollment, authentication, challenge/response, device list/delete, health.
- **SQLite database** (`db/`) – devices, challenges, audit logs. DB path from `config.DATABASE_URL` (default `./data/qna_auth.db`).
- **Challenge storage** – DB-backed (`DbChallengeStore`); no in-memory-only challenge dict.
- **Device embeddings** – Still stored as **plain `.pt` files** in `auth/device_embeddings/` (and metadata as JSON). Device list/delete and enrollment metadata use the DB; embedding blobs are file-based.
- **Config** – `config.py` / `config.example.py` for model path, DB URL, embeddings dir, etc. Server uses some of it; not every path is config-driven yet.

### Noise & features
- **Noise sources** – QRNG (ANU API, 1 req/min), camera (OpenCV), microphone (sounddevice). Used in enrollment, auth, and the data-collection script.
- **Feature pipeline** – Single canonical feature list + `FEATURE_VERSION` in `preprocessing/features.py`. Same pipeline in preprocessing, training, and server. Feature names and version are saved with the model checkpoint.
- **Preprocessing** – `NoisePreprocessor` (stats, entropy, FFT, autocorrelation, complexity) and `FeatureVector` (canonical order).

### Model & training
- **Siamese embedder** – `model/siamese_model.py`: maps feature vector → 128-D embedding. Loaded from `server/models/best_model.pt` (or `config.MODEL_PATH`) if present; otherwise **random weights** (auth will be poor).
- **Training** – `model/train.py`: triplet (or contrastive) loss, reproducible seeds, best checkpoint by validation loss, optional last-N checkpoints. Saves a server-ready checkpoint (with feature names + version) to `server/models/best_model.pt`.
- **Train + evaluate script** – `scripts/train_and_evaluate.py`: load dataset from `dataset/samples`, train with seeds, run evaluation (ROC, PR, optimal threshold), deploy checkpoint. Supports train/val split and `--save-last-n`.

### Data collection & dataset
- **Standalone collection script** – `scripts/collect_data_for_training.py`: runs without the rest of the repo; collects QRNG (with 65 s delay), camera, mic; writes a folder + `manifest.json`. Participants can zip and send.
- **Ingest script** – `scripts/ingest_collected_data.py`: merges one or more collection folders (or zips) into `dataset/samples/` for training.
- **Dataset layout** – `dataset/samples/`: `json/` (metadata per sample) and raw `.npy` files; built by `DatasetBuilder` and ingest script.

### Frontend
- **React + TypeScript** (Vite) – Home, Enroll, Authenticate, Devices. Calls backend for enroll, authenticate, list devices, get/delete device. No challenge/response in the UI yet.

### Auth flow
- **Enrollment** – Collect noise (qrng/camera/mic) → features → embedder → mean embedding → save `.pt` + metadata; register device in DB.
- **Authentication** – Fresh noise → features → embedder → cosine similarity vs stored embedding; threshold 0.85 (hardcoded in server).
- **Challenge/response** – Nonce from server, response from client (embedding + nonce); challenges stored in DB; server still checks live embedding similarity.

---

## What’s Not Done (To Complete the Project)

Below is a focused list. **High** = needed for a solid research-level capstone; **Medium** = strongly recommended; **Low** = nice to have.

### 1. Evaluation & reproducibility (High)
- **Single “run evaluation” path** – One script (e.g. `scripts/run_evaluation.py`) with a config that loads data + model and outputs FAR, FRR, EER, ROC, threshold sweep. Right now evaluation is inside `train_and_evaluate.py` and `model/evaluate.py` but there’s no one canonical “run the numbers” command.
- **Formal metrics everywhere** – FAR, FRR, EER, threshold sweep (vary threshold, plot FAR/FRR), and report EER + chosen operating point.
- **Cross-device / cross-session** – Train on one session (or N−1 devices), test on another (or held-out device); document in report.
- **Ablations** – Table: QRNG-only vs camera-only vs mic-only vs combined (accuracy/EER). Shows contribution of each source.

### 2. Security (High)
- **No hardcoded secrets** – Remove any API key / default secret from `auth/enrollment.py` (and elsewhere); use only env or `config.example` + README.
- **API protection** – At least API key or JWT for enroll/authenticate/delete; rate limiting on auth endpoints.
- **CORS** – Restrict to frontend origin(s); no `allow_origins=["*"]` with credentials in production.

### 3. Data & config (Medium)
- **Config-driven paths** – All paths (model, checkpoints, DB, embeddings, dataset) from config/env; no hardcoded `./auth/...` in core logic.
- **Embeddings storage** – Either store embeddings in DB (or as encrypted blobs) and document; or clearly state “embeddings as .pt files” as a known limitation.
- **Dataset versioning** – Version training/enrollment datasets (paths, hashes, or DVC) so you can say “model X was trained on dataset v1.2”.

### 4. Model & baselines (Medium)
- **Simple baselines** – e.g. cosine similarity on raw feature vectors (no NN), or a small MLP. One table: “Siamese vs baseline X”.
- **Pinned requirements** – `requirements.txt` with versions so `pip install -r requirements.txt` is reproducible.

### 5. Documentation & report (High)
- **README** – Install (venv, Python version, optional CUDA), dataset layout, how to collect data, train, run evaluation, start server. Point to PROJECT_OVERVIEW.md and this STATUS.
- **Report/slides** – Problem, related work (device auth / PUF / QRNG), method (pipeline diagram), experiments (tables/figures from evaluation), limitations, future work. This turns the code into a “research-level capstone”.

### 6. Frontend & UX (Medium / Low)
- **Clear error states** – “No devices enrolled,” “Authentication failed: similarity 0.72 (threshold 0.85),” “Camera unavailable.”
- **Challenge/response in UI** – Optional; if the report describes the protocol, having it in the UI helps the story.

### 7. Nice-to-have (Low)
- Confidence intervals (bootstrap or multiple runs) for EER/FAR/FRR.
- Failure analysis: when does auth fail? (low samples, bad lighting, mic off, etc.)
- Encrypt embeddings at rest (key from env).
- ONNX export for inference.

---

## Suggested Order to Finish

1. **Security** – Remove hardcoded secrets; add API auth (and optionally rate limiting); tighten CORS.
2. **Evaluation** – One `run_evaluation.py` (or equivalent) with config → FAR/FRR/EER, threshold sweep, ROC. Add cross-session or leave-one-device-out.
3. **Ablations** – Run eval for QRNG-only, camera-only, mic-only, combined; put results in a table.
4. **Docs** – README up to date; requirements pinned; then write the report with tables/figures from (2)–(3).
5. **Config & paths** – All paths from config; document embedding storage and dataset versioning.

---

## One-Line Summary

**Current:** Working end-to-end flow (enroll → auth, challenge/response, DB, canonical features, training + eval script, standalone data collection). **Still to do:** Single evaluation pipeline with FAR/FRR/EER and ablations, security cleanup (no secrets, API auth, CORS), config-driven paths, and a written report with tables/figures.
