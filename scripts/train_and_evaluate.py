"""
Train on real multi-device data and run evaluation.

Usage:
  python scripts/train_and_evaluate.py [--data-dir dataset/samples] [--seed 42] [--epochs 20]

- Loads dataset from dataset/samples (or --data-dir).
- Uses canonical feature pipeline (FEATURE_VERSION) and reproducible seeds.
- Trains Siamese model; saves best by validation loss and optional last-N checkpoints.
- Runs evaluation (ROC, PR, optimal threshold, metrics) and saves report.
- Deploys server-style checkpoint (with feature_names + feature_version) to server/models/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from torch.utils.data import DataLoader

from model.train import (
    set_seed,
    load_real_dataset,
    TripletDataset,
    ModelTrainer,
)
from model.siamese_model import SiameseNetwork, DeviceEmbedder
from model.evaluate import ModelEvaluator
from preprocessing.features import FEATURE_VERSION, get_canonical_feature_names


def main():
    p = argparse.ArgumentParser(description="Train on real data and run evaluation")
    p.add_argument("--data-dir", type=str, default=None, help="Dataset dir (default: dataset/samples)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--epochs", type=int, default=20, help="Training epochs")
    p.add_argument("--save-last-n", type=int, default=3, help="Keep last N epoch checkpoints (0 = keep all)")
    p.add_argument("--eval-dir", type=str, default="model/evaluation", help="Where to save evaluation report/plots")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of data per device for validation (0 = no val)")
    args = p.parse_args()

    data_dir = args.data_dir or str(ROOT / "dataset" / "samples")
    set_seed(args.seed)

    # Load full dataset (no subsampling)
    features_by_device, input_dim = load_real_dataset(data_dir=data_dir, max_samples=None)

    if not features_by_device or len(features_by_device) < 2:
        print("Need at least 2 devices with samples. Run scripts/collect_data_for_training.py then scripts/ingest_collected_data.py")
        return 1

    n_devices = len(features_by_device)
    total_samples = sum(len(v) for v in features_by_device.values())
    print(f"Devices: {n_devices}, Total samples: {total_samples}, Input dim: {input_dim}")

    # Optional train/val split per device (deterministic by seed)
    rng = np.random.default_rng(args.seed)
    train_by_device = {}
    val_by_device = {}
    for dev_id, vecs in features_by_device.items():
        n = len(vecs)
        idx = rng.permutation(n)
        n_val = max(0, int(n * args.val_ratio)) if args.val_ratio > 0 else 0
        train_idx = idx[n_val:]
        val_idx = idx[:n_val]
        train_by_device[dev_id] = [vecs[i] for i in train_idx]
        if n_val > 0:
            val_by_device[dev_id] = [vecs[i] for i in val_idx]

    samples_per_epoch = max(100, total_samples * 2)
    train_dataset = TripletDataset(train_by_device, samples_per_epoch=samples_per_epoch)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(32, samples_per_epoch),
        shuffle=True,
        generator=g,
        worker_init_fn=lambda wid: np.random.seed(args.seed + wid),
    )

    val_loader = None
    if val_by_device and all(len(v) > 0 for v in val_by_device.values()):
        val_samples = max(50, sum(len(v) for v in val_by_device.values()) // 2)
        val_dataset = TripletDataset(val_by_device, samples_per_epoch=val_samples)
        g_val = torch.Generator()
        g_val.manual_seed(args.seed + 1)
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(32, val_samples),
            shuffle=False,
            generator=g_val,
        )
        print(f"Validation: {val_samples} triplets from {len(val_by_device)} devices")
    else:
        print("No validation split; best model will not be saved by val loss")

    embedding_dim = 128
    model = SiameseNetwork(input_dim=input_dim, embedding_dim=embedding_dim)
    trainer = ModelTrainer(model=model, loss_type="triplet", learning_rate=0.001)

    save_dir = str(ROOT / "model" / "checkpoints")
    save_last_n = args.save_last_n if args.save_last_n > 0 else None
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=save_dir,
        save_last_n=save_last_n,
    )

    # Evaluation on full features_by_device (or train_only if you prefer)
    best_pt = ROOT / "model" / "checkpoints" / "best_model.pt"
    if not best_pt.exists():
        best_pt = ROOT / "model" / "checkpoints" / "final_model.pt"
    if not best_pt.exists():
        print("No checkpoint found; skipping evaluation")
        return 0

    embedder = DeviceEmbedder(input_dim=input_dim, embedding_dim=embedding_dim)
    ckpt = torch.load(best_pt, map_location=embedder.device)
    embedder.model.load_state_dict(ckpt["model_state_dict"])
    embedder.model.eval()

    evaluator = ModelEvaluator(embedder)
    eval_dir = str(ROOT / args.eval_dir)
    report = evaluator.generate_report(features_by_device, save_dir=eval_dir)
    print("Evaluation report:", report)

    # Deploy server checkpoint with feature pipeline metadata
    server_path = ROOT / "server" / "models" / "best_model.pt"
    server_path.parent.mkdir(parents=True, exist_ok=True)
    server_ckpt = {
        "model_state_dict": ckpt["model_state_dict"],
        "embedding_dim": embedding_dim,
        "input_dim": input_dim,
        "feature_names": get_canonical_feature_names(),
        "feature_version": FEATURE_VERSION,
    }
    torch.save(server_ckpt, server_path)
    print(f"Deployed model (feature_version={FEATURE_VERSION}) to {server_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
