#!/usr/bin/env python3
"""
Capstone-grade evaluation runner:
- leakage-safe split artifacts
- test-only metrics
- baselines (raw cosine, small MLP embedding, Siamese)
- EER/FAR/FRR + threshold sweep + ROC/PR
- bootstrap CIs
- attack suite (replay, impersonation, synthetic-statistical)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import config

from model.evaluate import ModelEvaluator
from model.siamese_model import DeviceEmbedder
from scripts.training.experiment_utils import (
    assert_no_leakage,
    bootstrap_metric_ci,
    build_mlp_embeddings,
    features_from_split,
    load_sample_records,
    save_split_artifacts,
    split_by_device,
    split_by_device_session,
    split_by_session,
    parse_sources_arg,
)


def _eval_from_scores(evaluator: ModelEvaluator, scores: List[float], labels: List[int]) -> Dict:
    if not scores:
        return {"error": "No scores generated"}
    threshold, metrics = evaluator.find_optimal_threshold(scores, labels, metric="f1_score")
    eer, eer_threshold = evaluator.compute_eer(scores, labels)
    sweep = evaluator.threshold_sweep(scores, labels, n_thresholds=150)
    return {
        "optimal_threshold": float(threshold),
        "metrics": {k: float(v) if isinstance(v, (float, int)) else v for k, v in metrics.items()},
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "threshold_sweep": sweep,
    }


def _evaluate_attacks(evaluator: ModelEvaluator, vectors_by_device: Dict[str, List[np.ndarray]], threshold: float) -> Dict:
    rng = np.random.default_rng(42)
    # Replay: genuine pair reuse
    replay_successes = []
    for dev, vecs in vectors_by_device.items():
        if len(vecs) < 2:
            continue
        for _ in range(min(10, len(vecs) - 1)):
            i = int(rng.integers(len(vecs)))
            j = int(rng.integers(len(vecs)))
            a, b = vecs[i], vecs[j]
            sim = float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))
            replay_successes.append(1.0 if sim >= threshold else 0.0)

    # Impersonation: different devices attempt same ID
    impostor_successes = []
    devs = sorted(vectors_by_device.keys())
    for i, d1 in enumerate(devs):
        for d2 in devs[i + 1:]:
            if not vectors_by_device[d1] or not vectors_by_device[d2]:
                continue
            for _ in range(10):
                a = vectors_by_device[d1][int(rng.integers(len(vectors_by_device[d1])))]
                b = vectors_by_device[d2][int(rng.integers(len(vectors_by_device[d2])))]
                sim = float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))
                impostor_successes.append(1.0 if sim >= threshold else 0.0)

    # Synthetic: gaussian matching global mean/std of vectors
    all_vecs = np.vstack([v for vals in vectors_by_device.values() for v in vals]) if vectors_by_device else np.zeros((1, 1))
    mu = np.mean(all_vecs, axis=0)
    sigma = np.std(all_vecs, axis=0) + 1e-8
    synthetic_successes = []
    for d in devs:
        if not vectors_by_device[d]:
            continue
        enrolled = vectors_by_device[d][0]
        for _ in range(20):
            synth = rng.normal(mu, sigma).astype(np.float32)
            sim = float(np.dot(enrolled, synth) / ((np.linalg.norm(enrolled) + 1e-8) * (np.linalg.norm(synth) + 1e-8)))
            synthetic_successes.append(1.0 if sim >= threshold else 0.0)

    return {
        "replay_asr": bootstrap_metric_ci(replay_successes, seed=42),
        "impersonation_asr": bootstrap_metric_ci(impostor_successes, seed=43),
        "synthetic_stats_asr": bootstrap_metric_ci(synthetic_successes, seed=44),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capstone evaluation runner")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/samples"))
    parser.add_argument("--model-path", type=Path, default=Path("server/models/best_model.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/capstone_eval"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-policy", choices=["device", "session"], default="device")
    parser.add_argument("--collection-kinds", type=str, default="real", help="Comma-separated collection kinds to include")
    parser.add_argument("--train-sessions", type=str, default="")
    parser.add_argument("--test-sessions", type=str, default="")
    parser.add_argument("--source-filter", type=str, default=None, help="camera|microphone|None")
    parser.add_argument("--min-devices", type=int, default=2)
    parser.add_argument("--max-records", type=int, default=None, help="Cap total records for quick review runs")
    parser.add_argument("--fast-features", action="store_true", help="Skip expensive complexity features")
    parser.add_argument("--sources", type=str, default="camera,microphone", help="Comma-separated runtime sources to evaluate together")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Ensure manifest exists and is current.
    try:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "data" / "build_dataset_manifest.py"),
                "--dataset-dir",
                str(args.data_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        pass

    source_filter = parse_sources_arg(args.sources) if args.source_filter is None else [args.source_filter]
    collection_kinds = parse_sources_arg(args.collection_kinds)
    records = load_sample_records(
        args.data_dir,
        source_filter=source_filter,
        collection_kinds=collection_kinds,
        max_records=args.max_records,
        seed=args.seed
    )
    if len({r.device_id for r in records}) < args.min_devices:
        print("Need at least 2 devices for evaluation.")
        return 1

    if args.split_policy == "session":
        train_sessions = [s for s in args.train_sessions.split(",") if s]
        test_sessions = [s for s in args.test_sessions.split(",") if s]
        if train_sessions or test_sessions:
            splits = split_by_session(records, train_sessions=train_sessions, test_sessions=test_sessions)
            split_name = "session"
        else:
            splits = split_by_device_session(records, min_sessions_per_device=3, val_ratio=0.2)
            split_name = "device_session"
    else:
        splits = split_by_device(records, seed=args.seed, val_ratio=0.2, test_ratio=0.2)
        split_name = "device"
    assert_no_leakage(splits)
    split_path = save_split_artifacts(splits, run_dir, split_name=split_name)

    feat_splits = features_from_split(splits, normalize=True, fast_features=args.fast_features)
    train_feats = feat_splits["train"]
    test_feats = feat_splits["test"]
    if len(test_feats) < 2:
        print("Test split has fewer than 2 devices; cannot evaluate.")
        return 1

    # Siamese embeddings
    ckpt = torch.load(args.model_path, map_location="cpu")
    embedder = DeviceEmbedder(
        input_dim=int(ckpt.get("input_dim", 33)),
        embedding_dim=int(ckpt.get("embedding_dim", 128)),
    )
    embedder.model.load_state_dict(ckpt["model_state_dict"])
    embedder.model.eval()
    evaluator = ModelEvaluator(embedder)
    siamese_embeddings = evaluator.compute_embeddings(test_feats)
    s_scores, s_labels = evaluator.compute_similarity_scores(siamese_embeddings, metric="cosine", seed=args.seed)
    siamese_eval = _eval_from_scores(evaluator, s_scores, s_labels)
    evaluator.plot_roc_curve(s_scores, s_labels, run_dir / "roc_siamese.png")
    evaluator.plot_precision_recall_curve(s_scores, s_labels, run_dir / "pr_siamese.png")

    # Baseline 1: raw-feature cosine
    r_scores, r_labels = ModelEvaluator.compute_similarity_scores_from_vectors(test_feats, metric="cosine", seed=args.seed)
    raw_eval = _eval_from_scores(evaluator, r_scores, r_labels)

    # Baseline 2: small MLP embedding
    mlp_emb = build_mlp_embeddings(train_features=train_feats, target_features=test_feats, embedding_dim=128, seed=args.seed)
    m_scores, m_labels = ModelEvaluator.compute_similarity_scores_from_vectors(mlp_emb, metric="cosine", seed=args.seed)
    mlp_eval = _eval_from_scores(evaluator, m_scores, m_labels)

    # CIs on pair scores as lightweight stability proxy.
    ci = {
        "siamese_score_ci": bootstrap_metric_ci(s_scores, seed=args.seed),
        "raw_score_ci": bootstrap_metric_ci(r_scores, seed=args.seed + 1),
        "mlp_score_ci": bootstrap_metric_ci(m_scores, seed=args.seed + 2),
    }

    attacks = _evaluate_attacks(evaluator, test_feats, threshold=siamese_eval["optimal_threshold"])

    results = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": {
            "seed": args.seed,
            "split_policy": args.split_policy,
            "source_filter": args.source_filter,
            "sources": source_filter,
            "collection_kinds": collection_kinds,
            "max_records": args.max_records,
            "fast_features": args.fast_features,
            "model_path": str(args.model_path),
            "data_dir": str(args.data_dir),
            "source_weights": getattr(config, "AUTH_SOURCE_WEIGHTS", {"camera": 0.7, "microphone": 0.3}),
            "confidence_bands": {
                "strong_accept": float(getattr(config, "AUTH_CONFIDENCE_STRONG", 0.97)),
                "uncertain": float(getattr(config, "AUTH_CONFIDENCE_UNCERTAIN", 0.92)),
            },
        },
        "split_artifact": str(split_path),
        "methods": {
            "siamese": siamese_eval,
            "raw_feature_cosine": raw_eval,
            "small_mlp_embedding": mlp_eval,
        },
        "confidence_intervals": ci,
        "attacks": attacks,
    }

    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {run_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
