#!/usr/bin/env python3
"""
One-command reproducibility pipeline for capstone evidence artifacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce capstone artifacts end-to-end")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/samples"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/repro_runs"))
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (ROOT / args.out_dir / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    _run([sys.executable, str(ROOT / "scripts" / "build_dataset_manifest.py"), "--dataset-dir", str(args.data_dir)])
    _run([sys.executable, str(ROOT / "scripts" / "train_and_evaluate.py"), "--data-dir", str(args.data_dir), "--seed", str(args.seed), "--epochs", str(args.epochs)])
    _run([sys.executable, str(ROOT / "scripts" / "run_capstone_evaluation.py"), "--data-dir", str(args.data_dir), "--seed", str(args.seed), "--out-dir", str(run_dir / "eval")])

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": args.seed,
        "epochs": args.epochs,
        "data_dir": str(args.data_dir),
        "commands": [
            "build_dataset_manifest.py",
            "train_and_evaluate.py",
            "run_capstone_evaluation.py",
        ],
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Done. Artifacts at {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
