# Project Structure Guide

This file explains where code should live so the repository stays maintainable.

## Top-level modules

- `server/`: FastAPI app and API endpoints.
- `auth/`: enrollment/authentication domain logic.
- `noise_collection/`: QRNG/camera/microphone/sensor acquisition.
- `preprocessing/`: feature extraction and vectorization.
- `model/`: Siamese model, train/eval internals.
- `dataset/`: dataset builder and storage conventions.
- `db/`: SQLAlchemy models/session/challenge storage.
- `frontend/`: React client.
- `tests/`: automated test files.

## Scripts policy

- `scripts/data/`: collection, ingestion, and dataset preparation.
- `scripts/training/`: model training and evaluation.
- `scripts/db/`: database bootstrap and backfill helpers.
- `scripts/diagnostics/`: manual checks and debugging utilities.
- `scripts/legacy/`: old scripts kept only for backwards reference.
- `scripts/reporting/`: report/document generation helpers.

If a script is no longer part of the normal workflow but still useful historically,
move it to `scripts/legacy/` instead of keeping it at repo root.

## Root folder policy

Keep root minimal:

- setup/start convenience scripts (`setup.bat`, `setup.sh`, `start.bat`, `run_demo.ps1`)
- core docs (`README.md`, `requirements.txt`, `config.example.py`)
- core source directories only

Avoid adding one-off Python scripts at root. Place them under `scripts/` by purpose.
