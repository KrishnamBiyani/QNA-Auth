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
import functools
import json
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
from preprocessing.features import FEATURE_VERSION, FeatureVector, get_canonical_feature_names
from scripts.training.experiment_utils import (
    load_sample_records,
    split_by_device,
    split_by_device_session,
    assert_no_leakage,
    save_split_artifacts,
    features_from_split,
    parse_sources_arg,
)


def fit_feature_standardization(features_by_device: dict[str, list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    stacked = [
        np.asarray(vec, dtype=np.float32)
        for vectors in features_by_device.values()
        for vec in vectors
    ]
    if not stacked:
        raise ValueError("Cannot fit feature standardization on empty training set")
    matrix = np.vstack(stacked)
    feature_mean = matrix.mean(axis=0).astype(np.float32)
    feature_scale = matrix.std(axis=0).astype(np.float32)
    feature_scale = np.where(feature_scale < 1e-6, 1.0, feature_scale)
    return feature_mean, feature_scale


def apply_feature_standardization(
    features_by_device: dict[str, list[np.ndarray]],
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> dict[str, list[np.ndarray]]:
    transformed: dict[str, list[np.ndarray]] = {}
    for device_id, vectors in features_by_device.items():
        transformed[device_id] = [
            np.nan_to_num((np.asarray(vec, dtype=np.float32) - feature_mean) / feature_scale, nan=0.0, posinf=0.0, neginf=0.0)
            for vec in vectors
        ]
    return transformed


def seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def main():
    p = argparse.ArgumentParser(description="Train on real data and run evaluation")
    p.add_argument("--data-dir", type=str, default=None, help="Dataset dir (default: dataset/samples)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--epochs", type=int, default=20, help="Training epochs")
    p.add_argument("--save-last-n", type=int, default=3, help="Keep last N epoch checkpoints (0 = keep all)")
    p.add_argument("--batch-size", type=int, default=64, help="DataLoader batch size")
    default_num_workers = 0 if os.name == "nt" else min(4, os.cpu_count() or 1)
    p.add_argument("--num-workers", type=int, default=default_num_workers, help="DataLoader worker processes")
    p.add_argument("--samples-per-epoch", type=int, default=0, help="Triplets per epoch (0 = auto)")
    p.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max-norm (<=0 disables)")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use mixed precision on CUDA")
    p.add_argument("--learning-rate", type=float, default=0.001, help="Optimizer learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight decay")
    p.add_argument("--triplet-margin", type=float, default=1.0, help="Triplet/contrastive margin")
    p.add_argument("--hard-negative-k", type=int, default=4, help="Sample this many candidate negatives and keep the hardest")
    p.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension")
    p.add_argument("--eval-dir", type=str, default="model/evaluation", help="Where to save evaluation report/plots")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of data per device for validation (0 = no val)")
    p.add_argument("--sources", type=str, default="camera,microphone", help="Comma-separated sources to include in training/eval")
    p.add_argument("--collection-kinds", type=str, default="real", help="Comma-separated collection kinds to include, e.g. real or real,synthetic")
    p.add_argument("--split-policy", choices=["device", "device_session"], default="device", help="How to split train/val/test")
    p.add_argument("--fast-features", action="store_true", help="Skip expensive complexity features during feature extraction")
    p.add_argument("--output-stem", type=str, default=None, help="Subdirectory / file stem for checkpoints and evaluation artifacts")
    p.add_argument("--target-far", type=float, default=0.10, help="Target FAR used for threshold calibration")
    p.add_argument("--augment-camera-train", action=argparse.BooleanOptionalAction, default=False, help="Apply synthetic augmentation to camera samples in the training split only")
    p.add_argument("--camera-aug-copies", type=int, default=3, help="Number of synthetic camera variants per real training sample")
    args = p.parse_args()

    data_dir = args.data_dir or str(ROOT / "dataset" / "samples")
    set_seed(args.seed)
    source_filter = parse_sources_arg(args.sources)
    collection_kinds = parse_sources_arg(args.collection_kinds)
    output_stem = args.output_stem or "_".join(source_filter or ["all"])

    records = load_sample_records(
        Path(data_dir),
        source_filter=source_filter,
        collection_kinds=collection_kinds,
    )
    if args.split_policy == "device_session":
        splits = split_by_device_session(records, min_sessions_per_device=3, val_ratio=args.val_ratio)
    else:
        splits = split_by_device(records, seed=args.seed, val_ratio=args.val_ratio, test_ratio=0.2)
    assert_no_leakage(splits)
    split_dir = ROOT / "artifacts" / "splits"
    split_path = save_split_artifacts(splits, split_dir, split_name=f"{output_stem}_seed_{args.seed}")
    feat_splits = features_from_split(
        splits,
        normalize=True,
        fast_features=args.fast_features,
        augment_camera_train=args.augment_camera_train,
        camera_aug_copies=args.camera_aug_copies,
        seed=args.seed,
    )
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
    if args.augment_camera_train:
        print(f"Camera augmentation enabled: {args.camera_aug_copies} synthetic train variants per real camera sample")

    feature_mean, feature_scale = fit_feature_standardization(train_by_device)
    train_by_device = apply_feature_standardization(train_by_device, feature_mean, feature_scale)
    val_by_device = apply_feature_standardization(val_by_device, feature_mean, feature_scale) if val_by_device else {}
    test_by_device = apply_feature_standardization(test_by_device, feature_mean, feature_scale) if test_by_device else {}

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    samples_per_epoch = args.samples_per_epoch if args.samples_per_epoch > 0 else max(400, total_samples * 6)
    train_dataset = TripletDataset(
        train_by_device,
        samples_per_epoch=samples_per_epoch,
        hard_negative_k=args.hard_negative_k,
    )

    g = torch.Generator()
    g.manual_seed(args.seed)
    worker_init = functools.partial(seed_worker, base_seed=args.seed)
    train_loader_kwargs = {
        "batch_size": min(args.batch_size, samples_per_epoch),
        "shuffle": True,
        "generator": g,
        "worker_init_fn": worker_init,
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
    if len(val_by_device) >= 2 and all(len(v) > 0 for v in val_by_device.values()):
        val_samples = max(50, sum(len(v) for v in val_by_device.values()) // 2)
        val_dataset = TripletDataset(
            val_by_device,
            samples_per_epoch=val_samples,
            hard_negative_k=max(1, min(args.hard_negative_k, 4)),
        )
        g_val = torch.Generator()
        g_val.manual_seed(args.seed + 1)
        val_loader_kwargs = {
            "batch_size": min(args.batch_size, val_samples),
            "shuffle": False,
            "generator": g_val,
            "worker_init_fn": worker_init,
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

    embedding_dim = args.embedding_dim
    model = SiameseNetwork(input_dim=input_dim, embedding_dim=embedding_dim)
    trainer = ModelTrainer(
        model=model,
        loss_type="triplet",
        learning_rate=args.learning_rate,
        margin=args.triplet_margin,
        weight_decay=args.weight_decay,
        use_amp=args.amp,
        grad_clip_norm=(args.grad_clip if args.grad_clip > 0 else None),
    )

    save_dir = str(ROOT / "model" / "checkpoints" / output_stem)
    save_last_n = args.save_last_n if args.save_last_n > 0 else None
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=save_dir,
        save_last_n=save_last_n,
    )

    # Evaluation on held-out test split only.
    best_pt = ROOT / "model" / "checkpoints" / output_stem / "best_model.pt"
    if not best_pt.exists():
        best_pt = ROOT / "model" / "checkpoints" / output_stem / "final_model.pt"
    if not best_pt.exists():
        print("No checkpoint found; skipping evaluation")
        return 0

    embedder = DeviceEmbedder(input_dim=input_dim, embedding_dim=embedding_dim)
    ckpt = torch.load(best_pt, map_location=embedder.device)
    embedder.model.load_state_dict(ckpt["model_state_dict"])
    embedder.model.eval()

    evaluator = ModelEvaluator(embedder)
    eval_dir = str(ROOT / args.eval_dir / output_stem)
    calibration_threshold = None
    calibration_source = "test_fallback"
    calibration_report = None

    if len(val_by_device) >= 2:
        logger_target = "validation"
        val_embeddings = evaluator.compute_embeddings(val_by_device)
        val_scores, val_labels = evaluator.compute_similarity_scores(val_embeddings, metric="cosine")
        if val_scores and len(set(val_labels)) >= 2:
            calibration_report = evaluator.generate_score_report(
                val_scores,
                val_labels,
                target_far=args.target_far,
                optimal_metric="balanced_accuracy",
            )
            calibration_threshold = float(calibration_report["target_far_threshold"])
            calibration_source = "validation_target_far"
        else:
            logger_target = "test fallback"
        print(f"Threshold calibration source: {logger_target}")

    test_embeddings = evaluator.compute_embeddings(test_by_device)
    test_scores, test_labels = evaluator.compute_similarity_scores(test_embeddings, metric="cosine")
    if not test_scores or len(set(test_labels)) < 2:
        raise ValueError("Held-out test split must contain both genuine and impostor pairs")

    if calibration_threshold is None:
        fallback = evaluator.generate_score_report(
            test_scores,
            test_labels,
            target_far=args.target_far,
            optimal_metric="balanced_accuracy",
        )
        calibration_threshold = float(fallback["target_far_threshold"])

    report = evaluator.generate_score_report(
        test_scores,
        test_labels,
        target_far=args.target_far,
        optimal_metric="balanced_accuracy",
        deployed_threshold=calibration_threshold,
    )
    eval_path = Path(eval_dir)
    eval_path.mkdir(parents=True, exist_ok=True)
    evaluator.plot_roc_curve(test_scores, test_labels, eval_path / "roc_curve.png")
    evaluator.plot_precision_recall_curve(test_scores, test_labels, eval_path / "pr_curve.png")
    report["split_artifact"] = str(split_path)
    report["evaluation_scope"] = "test_only"
    report["source_filter"] = source_filter or ["all"]
    report["collection_kinds"] = collection_kinds or ["all"]
    report["split_policy"] = args.split_policy
    report["num_devices"] = len(test_by_device)
    report["total_samples"] = sum(len(v) for v in test_by_device.values())
    report["deployed_threshold_source"] = calibration_source
    report["calibration_target_far"] = float(args.target_far)
    if calibration_report is not None:
        report["calibration_report"] = calibration_report
    report["history"] = {
        "train_loss": history.get("train_loss", []),
        "val_loss": history.get("val_loss", []),
    }
    print("Evaluation report:", report)
    (Path(eval_dir) / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Deploy server checkpoint with feature pipeline metadata
    feature_converter = FeatureVector(get_canonical_feature_names(), feature_mean=feature_mean, feature_scale=feature_scale)
    server_path = ROOT / "server" / "models" / f"{output_stem}_best_model.pt"
    server_path.parent.mkdir(parents=True, exist_ok=True)
    server_ckpt = {
        "model_state_dict": ckpt["model_state_dict"],
        "embedding_dim": embedding_dim,
        "input_dim": input_dim,
        **feature_converter.to_metadata(),
        "feature_version": FEATURE_VERSION,
        "train_sources": source_filter or ["all"],
        "collection_kinds": collection_kinds or ["all"],
        "split_policy": args.split_policy,
        "source_weights": getattr(config, "AUTH_SOURCE_WEIGHTS", {"camera": 0.7, "microphone": 0.3}),
        "runtime_alignment": "camera_microphone_weighted_matching",
        "preprocessing_normalize": True,
        "preprocessing_fast_mode": bool(args.fast_features),
        "camera_train_augmentation": bool(args.augment_camera_train),
        "camera_aug_copies": int(args.camera_aug_copies),
        "hard_negative_k": int(args.hard_negative_k),
        "triplet_margin": float(args.triplet_margin),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "recommended_threshold": float(calibration_threshold),
        "target_far": float(args.target_far),
        "target_far_metrics": report.get("deployed_threshold_metrics", report["target_far_metrics"]),
        "output_stem": output_stem,
    }
    torch.save(server_ckpt, server_path)
    print(f"Deployed model (feature_version={FEATURE_VERSION}) to {server_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
