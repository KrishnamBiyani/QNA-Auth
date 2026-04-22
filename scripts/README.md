# Scripts Command Index

This folder is organized by purpose to keep operational scripts separate from
experiments and legacy utilities.

## `scripts/data/` - Data collection and dataset prep

- `python scripts/data/collect_data_for_training.py`
  - Standalone participant collection script (creates `qna_auth_collection_*`).
- `python scripts/data/ingest_collected_data.py <folder_or_zip> [more...]`
  - Ingests participant collections into `dataset/samples/`.
- `python scripts/data/build_dataset_manifest.py --dataset-dir dataset/samples`
  - Rebuilds canonical manifest/quality summary.
- `python scripts/data/analyze_longitudinal.py --data-dir dataset/samples --baseline-session <session_id>`
  - Session drift analysis for longitudinal reporting.

## `scripts/training/` - Train and evaluate models

- `python scripts/training/train_and_evaluate.py --data-dir dataset/samples --seed 42 --epochs 20`
  - Main training + evaluation + model deployment workflow.
- `python scripts/training/run_evaluation.py --data-dir dataset/samples --model-path server/models/best_model.pt`
  - EER/FAR/FRR and ablation-style evaluation.
- `python scripts/training/run_capstone_evaluation.py --data-dir dataset/samples --seed 42`
  - Capstone-grade evaluation run with split artifacts and attack suite.
- `python scripts/training/reproduce_capstone.py --data-dir dataset/samples --seed 42 --epochs 20`
  - One-command reproducibility pipeline.

## `scripts/db/` - Database maintenance

- `python scripts/db/init_db.py`
  - Creates DB tables.
- `python scripts/db/check_db.py`
  - Prints DB health and row counts.
- `python scripts/db/backfill_db_from_files.py`
  - Populates DB from existing `auth/device_embeddings` files.

## `scripts/diagnostics/` - Manual diagnostics

Use these for local debugging and smoke checks:

- `scripts/diagnostics/test_hardware.py`
- `scripts/diagnostics/test_collection.py`
- `scripts/diagnostics/test_enrollment.py`
- `scripts/diagnostics/test_cross_device.py`
- `scripts/diagnostics/test_robustness.py`
- `scripts/diagnostics/reproduce_issue.py`
- `scripts/diagnostics/verify_cuda.py`

## `scripts/legacy/` - Historical scripts

Kept for reference only; not part of the canonical workflow:

- `scripts/legacy/collect_training_data.py`
- `scripts/legacy/auto_collect.py`
- `scripts/legacy/run_full_training.py`

## `scripts/reporting/` - Report generation helpers

- `python scripts/reporting/generate_report.py`
