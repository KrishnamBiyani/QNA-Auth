"""
Train on real multi-device data and run evaluation.

Usage:
  python scripts/training/train_and_evaluate.py [--data-dir dataset/samples] [--seed 42] [--epochs 20]

- Loads dataset from dataset/samples (or --data-dir).
- Uses canonical feature pipeline (FEATURE_VERSION) and reproducible seeds.
- Trains Siamese model; saves best by validation loss and optional last-N checkpoints.
- Runs evaluation (ROC, PR, optimal threshold, metrics) and saves report.
- Deploys server-style checkpoint (with feature_names + feature_version) to server/models/.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from torch.utils.data import DataLoader
import config

from model.train import set_seed, TripletDataset, ModelTrainer
from model.siamese_model import SiameseNetwork, DeviceEmbedder
from model.evaluate import ModelEvaluator
from preprocessing.features import FEATURE_VERSION, get_canonical_feature_names
from scripts.training.experiment_utils import (
    load_sample_records,
    split_by_device,
    assert_no_leakage,
    save_split_artifacts,
    features_from_split,
    parse_sources_arg,
)


def main():
    p = argparse.ArgumentParser(description="Train on real data and run evaluation")
    p.add_argument("--data-dir", type=str, default=None, help="Dataset dir (default: dataset/samples)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--epochs", type=int, default=20, help="Training epochs")
    p.add_argument("--save-last-n", type=int, default=3, help="Keep last N epoch checkpoints (0 = keep all)")
    p.add_argument("--batch-size", type=int, default=64, help="DataLoader batch size")
    p.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1), help="DataLoader worker processes")
    p.add_argument("--samples-per-epoch", type=int, default=0, help="Triplets per epoch (0 = auto)")
    p.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max-norm (<=0 disables)")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use mixed precision on CUDA")
    p.add_argument("--eval-dir", type=str, default="model/evaluation", help="Where to save evaluation report/plots")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of data per device for validation (0 = no val)")
    p.add_argument("--sources", type=str, default="camera,microphone", help="Comma-separated sources to include in training/eval")
    p.add_argument("--fast-features", action="store_true", help="Skip expensive complexity features during feature extraction")
    args = p.parse_args()

    data_dir = args.data_dir or str(ROOT / "dataset" / "samples")
    set_seed(args.seed)
    source_filter = parse_sources_arg(args.sources)

    records = load_sample_records(Path(data_dir), source_filter=source_filter)
    splits = split_by_device(records, seed=args.seed, val_ratio=args.val_ratio, test_ratio=0.2)
    assert_no_leakage(splits)
    split_dir = ROOT / "artifacts" / "splits"
    split_path = save_split_artifacts(splits, split_dir, split_name=f"train_eval_seed_{args.seed}")
    feat_splits = features_from_split(splits, normalize=True, fast_features=args.fast_features)
    train_by_device = feat_splits["train"]
    val_by_device = feat_splits["val"]
    test_by_device = feat_splits["test"]
    input_dim = len(next(iter(train_by_device.values()))[0]) if train_by_device else 0

    if not train_by_device or len(train_by_device) < 2:
        print("Need at least 2 devices with samples. Run scripts/data/collect_data_for_training.py then scripts/data/ingest_collected_data.py")
        return 1

    n_devices = len(train_by_device)
    total_samples = sum(len(v) for v in train_by_device.values())
    print(f"Devices: {n_devices}, Total samples: {total_samples}, Input dim: {input_dim}")
    print(f"Sources: {source_filter or ['all']}")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    samples_per_epoch = args.samples_per_epoch if args.samples_per_epoch > 0 else max(100, total_samples * 2)
    train_dataset = TripletDataset(train_by_device, samples_per_epoch=samples_per_epoch)

    def _seed_worker(worker_id: int) -> None:
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader_kwargs = {
        "batch_size": min(args.batch_size, samples_per_epoch),
        "shuffle": True,
        "generator": g,
        "worker_init_fn": _seed_worker,
        "num_workers": max(0, args.num_workers),
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_dataset,
        **train_loader_kwargs,
    )

    val_loader = None
    if val_by_device and all(len(v) > 0 for v in val_by_device.values()):
        val_samples = max(50, sum(len(v) for v in val_by_device.values()) // 2)
        val_dataset = TripletDataset(val_by_device, samples_per_epoch=val_samples)
        g_val = torch.Generator()
        g_val.manual_seed(args.seed + 1)
        val_loader_kwargs = {
            "batch_size": min(args.batch_size, val_samples),
            "shuffle": False,
            "generator": g_val,
            "worker_init_fn": _seed_worker,
            "num_workers": max(0, args.num_workers),
            "pin_memory": torch.cuda.is_available(),
        }
        if args.num_workers > 0:
            val_loader_kwargs["persistent_workers"] = True
            val_loader_kwargs["prefetch_factor"] = 2
        val_loader = DataLoader(
            val_dataset,
            **val_loader_kwargs,
        )
        print(f"Validation: {val_samples} triplets from {len(val_by_device)} devices")
    else:
        print("No validation split; best model will not be saved by val loss")

    embedding_dim = 128
    model = SiameseNetwork(input_dim=input_dim, embedding_dim=embedding_dim)
    trainer = ModelTrainer(
        model=model,
        loss_type="triplet",
        learning_rate=0.001,
        use_amp=args.amp,
        grad_clip_norm=(args.grad_clip if args.grad_clip > 0 else None),
    )

    save_dir = str(ROOT / "model" / "checkpoints")
    save_last_n = args.save_last_n if args.save_last_n > 0 else None
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=save_dir,
        save_last_n=save_last_n,
    )

    # Evaluation on held-out test split only.
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
    report = evaluator.generate_report(test_by_device, save_dir=eval_dir)
    report["split_artifact"] = str(split_path)
    report["evaluation_scope"] = "test_only"
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
        "train_sources": source_filter or ["all"],
        "source_weights": getattr(config, "AUTH_SOURCE_WEIGHTS", {"camera": 0.7, "microphone": 0.3}),
        "runtime_alignment": "camera_microphone_weighted_matching",
    }
    torch.save(server_ckpt, server_path)
    print(f"Deployed model (feature_version={FEATURE_VERSION}) to {server_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
