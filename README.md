# QNA-Auth

QNA-Auth is a capstone project for device verification using noise signals (QRNG, camera, microphone) and a Siamese embedding model.

This README is the practical runbook: what is actively used, what is legacy, and exactly what to run.

## What Is Used vs Old

### Actively used (current flow)
- Backend API: `server/app.py`
- Core auth logic: `auth/`, `noise_collection/`, `preprocessing/`, `model/`
- Dataset pipeline: `scripts/data/collect_data_for_training.py` -> `scripts/data/ingest_collected_data.py` -> `dataset/samples/`
- Training/eval: `scripts/training/train_and_evaluate.py`, `scripts/training/run_evaluation.py`, `scripts/training/run_capstone_evaluation.py`
- Frontend: `frontend/`

### Legacy or generated artifacts (not source of truth)
- `qna_auth_collection_*` folders (participant/raw export dumps)
- `qna_auth_test_collection/` (old generated sample folder)
- `server/dataset/samples/` (old checked-in generated data; pipeline uses `dataset/samples/`)
- `scripts/tempCodeRunnerFile.py` (editor temp file)

Generated artifacts are now ignored in `.gitignore`.

---

## 1) Prerequisites

- Python `3.10+` recommended
- Node.js `18+` (for frontend)
- Optional hardware: webcam, microphone
- Optional GPU/CUDA for faster training

---

## 2) One-Time Setup

From project root:

```bash
python -m venv .venv
```

Activate virtual environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat

# macOS/Linux
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Create local config:

```bash
copy config.example.py config.py
```

On macOS/Linux:

```bash
cp config.example.py config.py
```

---

## 3) Run Backend + Frontend

### Backend

```bash
python server/app.py
```

Backend URLs:
- API root: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

### Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend URL:
- `http://localhost:3000`

---

## 4) End-to-End Data + Model Workflow

Use this sequence when building real data and training a better model.

### Step A: Collect participant data

Participants run:

```bash
python scripts/data/collect_data_for_training.py --name "Alice Laptop" --sources qrng,camera,microphone --num-samples 50 --zip
```

This creates:
- `qna_auth_collection_<name>_<timestamp>/`
- Optional zip archive if `--zip` is used

### Step B: Ingest collected folders/zips

Project owner runs:

```bash
python scripts/data/ingest_collected_data.py path/to/folder1 path/to/folder2.zip
```

This merges into `dataset/samples/` and updates dataset manifest metadata.

### Step C: Train model

```bash
python scripts/training/train_and_evaluate.py --data-dir dataset/samples --epochs 20 --seed 42
```

### Step D: Run evaluation

Quick evaluation:

```bash
python scripts/training/run_evaluation.py --data-dir dataset/samples --model-path server/models/best_model.pt
```

Capstone evaluation:

```bash
python scripts/training/run_capstone_evaluation.py --data-dir dataset/samples --seed 42
```

Reproducibility run:

```bash
python scripts/training/reproduce_capstone.py --data-dir dataset/samples --seed 42 --epochs 20
```

---

## 5) API Endpoints You Actually Use

- `GET /health` - service + DB status
- `POST /enroll` - enroll a device embedding
- `POST /authenticate` - verify a device against stored embedding
- `POST /challenge` and `POST /verify` - challenge/response flow
- `GET /devices` - list enrolled devices
- `GET /devices/{device_id}` - fetch metadata
- `DELETE /devices/{device_id}` - remove enrolled device

Example enroll payload:

```json
{
  "device_name": "my-laptop",
  "num_samples": 50,
  "sources": ["qrng", "camera", "microphone"]
}
```

Example authenticate payload:

```json
{
  "device_id": "abc123",
  "sources": ["camera", "microphone"],
  "num_samples_per_source": 5
}
```

---

## 6) Important Paths

- `dataset/samples/` - canonical training dataset
- `auth/device_embeddings/` - enrolled device embeddings + metadata
- `server/models/` - deployed model checkpoint for backend runtime
- `artifacts/` - evaluation/repro outputs

---

## 6.1) Script Organization (New)

- `scripts/data/` - participant collection, ingestion, and dataset manifest scripts
  - `scripts/data/collect_data_for_training.py`
  - `scripts/data/ingest_collected_data.py`
  - `scripts/data/build_dataset_manifest.py`
  - `scripts/data/analyze_longitudinal.py`
- `scripts/training/` - training and evaluation runners
  - `scripts/training/train_and_evaluate.py`
  - `scripts/training/run_evaluation.py`
  - `scripts/training/run_capstone_evaluation.py`
  - `scripts/training/reproduce_capstone.py`
- `scripts/db/` - database setup and maintenance scripts
  - `scripts/db/init_db.py`
  - `scripts/db/check_db.py`
  - `scripts/db/backfill_db_from_files.py`
- `scripts/diagnostics/` - manual verification/debug scripts
  - `scripts/diagnostics/test_hardware.py`
  - `scripts/diagnostics/test_collection.py`
  - `scripts/diagnostics/test_enrollment.py`
  - `scripts/diagnostics/test_cross_device.py`
  - `scripts/diagnostics/test_robustness.py`
  - `scripts/diagnostics/reproduce_issue.py`
  - `scripts/diagnostics/verify_cuda.py`
- `scripts/legacy/` - older demo/bootstrap scripts kept for reference
  - `scripts/legacy/collect_training_data.py`
  - `scripts/legacy/auto_collect.py`
  - `scripts/legacy/run_full_training.py`
- `scripts/reporting/` - report generation helpers
  - `scripts/reporting/generate_report.py`
- `docs/reports/` - generated report artifacts
  - `docs/reports/QNA_Auth_Project_Report.docx`
- `scripts/README.md` - command index for all script groups

---

## 7) Project Cleanup Policy

Keep Git focused on source code and reproducible scripts.

Do not commit:
- Raw collection exports (`qna_auth_collection_*`)
- Generated data dumps in temp/legacy folders
- Temporary editor files
- Local secrets (`config.py`, API keys, `.env`)

If old generated folders already exist locally, you can remove them safely once ingested:

```bash
# PowerShell example
Remove-Item -Recurse -Force qna_auth_collection_* -ErrorAction SilentlyContinue
```

---

## 8) Troubleshooting

- Camera/mic issues:
  - Run `python scripts/diagnostics/test_hardware.py`
  - Check OS permissions and device locks
- QRNG failures:
  - Check internet
  - Expect rate limits/timeouts
  - Set `QRNG_API_KEY` if needed
- Poor authentication quality:
  - Increase enrollment samples
  - Use multiple sources
  - Retrain model and tune threshold

---

## 9) Suggested Daily Workflow

1. Pull latest changes
2. Run backend + frontend
3. Collect or ingest new data
4. Retrain/evaluate if dataset changed
5. Keep generated artifacts out of git
6. Commit only source/docs/scripts changes

---

## 10) Notes

- `server/routes.py` exists as a placeholder and is not the active API entrypoint.
- Runtime model loading is controlled by `config.MODEL_PATH`.
- Similarity threshold is controlled by `config.SIMILARITY_THRESHOLD`.
