"""
Train source-specific models with stronger defaults for a time-constrained final run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "scripts" / "training" / "train_and_evaluate.py"


def run_training(
    source: str,
    epochs: int,
    target_far: float,
    fast_features: bool,
    augment_camera_train: bool = False,
    camera_aug_copies: int = 3,
) -> int:
    output_stem = f"{source}_v2"
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--sources",
        source,
        "--epochs",
        str(epochs),
        "--batch-size",
        "64",
        "--num-workers",
        "2",
        "--save-last-n",
        "2",
        "--target-far",
        str(target_far),
        "--output-stem",
        output_stem,
        "--val-ratio",
        "0.0" if source == "camera" else "0.2",
    ]
    if fast_features:
        command.append("--fast-features")
    if augment_camera_train:
        command.extend(["--augment-camera-train", "--camera-aug-copies", str(camera_aug_copies)])

    print(f"\n=== Training {source} model ===")
    print(" ".join(command))
    result = subprocess.run(command, cwd=ROOT)
    return int(result.returncode)


def main() -> int:
    # Microphone has enough data to support a longer run.
    rc = run_training(source="microphone", epochs=30, target_far=0.10, fast_features=False)
    if rc != 0:
        return rc

    # Camera data is much smaller; keep the run conservative and faster.
    rc = run_training(
        source="camera",
        epochs=24,
        target_far=0.15,
        fast_features=False,
        augment_camera_train=True,
        camera_aug_copies=5,
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
