#!/usr/bin/env python3
"""
Run evaluation: FAR, FRR, EER, threshold sweep, ROC, and ablations by noise source.

Uses config for data path and model path. Outputs metrics and plots to model/evaluation/
and prints an ablation table (QRNG-only, camera-only, mic-only, combined).

Usage (from project root):
  python scripts/run_evaluation.py
  python scripts/run_evaluation.py --data-dir dataset/samples --model-path server/models/best_model.pt
  python scripts/run_evaluation.py --ablations-only  # skip full eval, only ablation table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from tqdm import tqdm

from model.siamese_model import DeviceEmbedder
from model.evaluate import ModelEvaluator
from preprocessing.features import (
    get_canonical_feature_names,
    NoisePreprocessor,
    FeatureVector,
)


def load_features_for_eval(
    data_dir: Path,
    source_filter: str | None = None,
    max_samples: int | None = None,
):
    """Load features by device from dataset/samples. Optionally filter by noise_source for ablations."""
    json_dir = data_dir / "json"
    if not json_dir.exists():
        return {}, 0

    preprocessor = NoisePreprocessor(normalize=True)
    converter = FeatureVector(get_canonical_feature_names())
    features_by_device = {}
    sample_count = 0

    json_files = list(json_dir.glob("*.json"))
    if max_samples is not None and len(json_files) > max_samples:
        import random
        random.shuffle(json_files)
        json_files = json_files[:max_samples]

    for json_file in tqdm(json_files, desc="Loading samples"):
        try:
            with open(json_file) as f:
                meta = json.load(f)
            device_id = meta["device_id"]
            noise_source = meta.get("noise_source", "").lower()
            if source_filter is not None and noise_source != source_filter:
                continue
            rel_path = meta.get("raw_data_path", "").lstrip("/\\")
            raw_path = data_dir / rel_path
            if not raw_path.exists():
                alt = data_dir.parent.parent / rel_path
                raw_path = alt if alt.exists() else raw_path
            if not raw_path.exists():
                continue
            raw_data = np.load(raw_path)
            features = preprocessor.extract_all_features(raw_data)
            vector = converter.to_vector(features)
            if device_id not in features_by_device:
                features_by_device[device_id] = []
            features_by_device[device_id].append(vector)
            sample_count += 1
        except Exception:
            continue

    input_dim = 0
    if features_by_device:
        input_dim = len(next(iter(features_by_device.values()))[0])
    return features_by_device, input_dim


def run_one_eval(
    embedder: DeviceEmbedder,
    features_by_device: dict,
    save_dir: Path | None,
    tag: str = "",
) -> dict:
    """Run evaluation (scores, EER, threshold sweep, ROC) and return metrics dict."""
    if len(features_by_device) < 2:
        return {"error": "Need at least 2 devices", "num_devices": len(features_by_device)}
    evaluator = ModelEvaluator(embedder)
    embeddings_by_device = evaluator.compute_embeddings(features_by_device)
    scores, labels = evaluator.compute_similarity_scores(embeddings_by_device, metric="cosine")
    if not scores:
        return {"error": "No pairs", "num_devices": len(features_by_device)}

    opt_threshold, opt_metrics = evaluator.find_optimal_threshold(scores, labels, metric="f1_score")
    eer_rate, eer_threshold = evaluator.compute_eer(scores, labels)
    sweep = evaluator.threshold_sweep(scores, labels, n_thresholds=100)

    out = {
        "num_devices": len(features_by_device),
        "total_samples": sum(len(v) for v in features_by_device.values()),
        "optimal_threshold": opt_threshold,
        "accuracy": opt_metrics["accuracy"],
        "precision": opt_metrics["precision"],
        "recall": opt_metrics["recall"],
        "f1_score": opt_metrics["f1_score"],
        "far": opt_metrics["far"],
        "frr": opt_metrics["frr"],
        "eer": eer_rate,
        "eer_threshold": eer_threshold,
    }

    if save_dir and tag:
        save_dir.mkdir(parents=True, exist_ok=True)
        evaluator.plot_roc_curve(scores, labels, str(save_dir / f"roc_{tag}.png"))
        evaluator.plot_precision_recall_curve(scores, labels, str(save_dir / f"pr_{tag}.png"))
        with open(save_dir / f"metrics_{tag}.json", "w") as f:
            json.dump({k: v for k, v in out.items() if isinstance(v, (int, float, str))}, f, indent=2)

    return out


def main():
    p = argparse.ArgumentParser(description="Run evaluation: FAR/FRR/EER, threshold sweep, ROC, ablations")
    p.add_argument("--data-dir", type=Path, default=None, help="Dataset dir (default: config STORAGE_CONFIG.dataset_dir)")
    p.add_argument("--model-path", type=Path, default=None, help="Model checkpoint (default: config MODEL_PATH)")
    p.add_argument("--output-dir", type=Path, default=None, help="Output dir (default: model/evaluation)")
    p.add_argument("--ablations-only", action="store_true", help="Only run ablation table by source")
    p.add_argument("--max-samples", type=int, default=None, help="Cap samples per run (for quick runs)")
    args = p.parse_args()

    try:
        import config
        data_dir = args.data_dir or Path(getattr(config, "DATA_DIR", None) or config.STORAGE_CONFIG.get("dataset_dir", ROOT / "dataset" / "samples"))
        model_path = args.model_path or Path(getattr(config, "MODEL_PATH", ROOT / "server" / "models" / "best_model.pt"))
        output_dir = args.output_dir or ROOT / "model" / "evaluation"
    except ImportError:
        data_dir = args.data_dir or ROOT / "dataset" / "samples"
        model_path = args.model_path or ROOT / "server" / "models" / "best_model.pt"
        output_dir = args.output_dir or ROOT / "model" / "evaluation"

    data_dir = Path(data_dir)
    model_path = Path(model_path)
    output_dir = Path(output_dir)

    if not model_path.exists():
        print(f"Model not found: {model_path}. Train first (e.g. scripts/train_and_evaluate.py).")
        return 1

    ckpt = torch.load(model_path, map_location="cpu")
    input_dim = int(ckpt.get("input_dim", 33))
    embedding_dim = int(ckpt.get("embedding_dim", 128))
    embedder = DeviceEmbedder(input_dim=input_dim, embedding_dim=embedding_dim)
    embedder.model.load_state_dict(ckpt["model_state_dict"])
    embedder.model.eval()

    print("Evaluation (FAR, FRR, EER, threshold sweep, ROC)")
    print(f"  Data:   {data_dir}")
    print(f"  Model:  {model_path}")
    print(f"  Output: {output_dir}")
    print(f"  Feature version: {ckpt.get('feature_version', '?')}")

    # Ablations: by source
    sources = [("combined", None), ("qrng", "qrng"), ("camera", "camera"), ("microphone", "microphone")]
    table_rows = []
    for label, source_filter in sources:
        features_by_device, _ = load_features_for_eval(data_dir, source_filter=source_filter, max_samples=args.max_samples)
        if not features_by_device:
            table_rows.append((label, {"error": "No data"}))
            continue
        res = run_one_eval(embedder, features_by_device, output_dir if not args.ablations_only else None, tag=label)
        table_rows.append((label, res))

    # Print ablation table
    print("\n" + "=" * 70)
    print("Ablation by noise source (for report table)")
    print("=" * 70)
    print(f"{'Source':<14} {'Devices':<8} {'Samples':<8} {'EER':<8} {'FAR':<8} {'FRR':<8} {'Accuracy':<8}")
    print("-" * 70)
    for label, res in table_rows:
        if "error" in res:
            print(f"{label:<14} -       -       {res.get('error', '-'):<8}")
            continue
        print(f"{label:<14} {res['num_devices']:<8} {res['total_samples']:<8} {res['eer']:.4f}   {res['far']:.4f}   {res['frr']:.4f}   {res['accuracy']:.4f}")
    print("=" * 70)

    # Full run with plots (unless ablations-only)
    if not args.ablations_only:
        features_by_device, _ = load_features_for_eval(data_dir, source_filter=None, max_samples=args.max_samples)
        if features_by_device and len(features_by_device) >= 2:
            full = run_one_eval(embedder, features_by_device, output_dir, tag="full")
            print("\nFull evaluation (all sources):")
            print(f"  EER: {full['eer']:.4f}  FAR: {full['far']:.4f}  FRR: {full['frr']:.4f}  Accuracy: {full['accuracy']:.4f}")
            print(f"  Plots: {output_dir}/roc_full.png, {output_dir}/pr_full.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
