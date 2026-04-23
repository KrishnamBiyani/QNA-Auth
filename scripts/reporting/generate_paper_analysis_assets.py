from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.evaluate import ModelEvaluator
from model.siamese_model import DeviceEmbedder
from preprocessing.features import FeatureVector, NoisePreprocessor


REPORT_IMAGES = ROOT / "report_images"
REPORT_TABLES = ROOT / "report_tables"

CAMERA_METRICS_PATH = ROOT / "model" / "evaluation" / "camera_real_aug_v2" / "metrics.json"
MIC_METRICS_PATH = ROOT / "model" / "evaluation" / "microphone_real_aug_v2" / "metrics.json"
CAMERA_MODEL_PATH = ROOT / "server" / "models" / "camera_real_aug_v2_best_model.pt"
CAMERA_SPLIT_PATH = ROOT / "artifacts" / "splits" / "split_camera_real_aug_v2_seed_42.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_camera_test_features() -> tuple[Dict[str, List[np.ndarray]], List[dict], dict]:
    split = _load_json(CAMERA_SPLIT_PATH)
    checkpoint = torch.load(CAMERA_MODEL_PATH, map_location="cpu")
    preprocessor = NoisePreprocessor(
        normalize=bool(checkpoint.get("preprocessing_normalize", True)),
        fast_mode=bool(checkpoint.get("preprocessing_fast_mode", False)),
    )
    converter = FeatureVector(
        checkpoint["feature_names"],
        feature_mean=np.asarray(checkpoint.get("feature_mean"), dtype=np.float32),
        feature_scale=np.asarray(checkpoint.get("feature_scale"), dtype=np.float32),
    )

    by_device: Dict[str, List[np.ndarray]] = {}
    ordered_samples: List[dict] = []
    for record in split["splits"]["test"]:
        raw = np.load(record["raw_path"]).astype(np.float32)
        features = preprocessor.extract_all_features(raw)
        vector = converter.to_vector(features)
        by_device.setdefault(record["device_id"], []).append(vector)
        ordered_samples.append(
            {
                "device_id": record["device_id"],
                "session_id": record["session_id"],
                "sample_id": record["sample_id"],
                "vector": vector,
            }
        )
    return by_device, ordered_samples, checkpoint


def _load_camera_embedder(checkpoint: dict) -> DeviceEmbedder:
    embedder = DeviceEmbedder(
        input_dim=int(checkpoint["input_dim"]),
        embedding_dim=int(checkpoint.get("embedding_dim", 128)),
        device="cpu",
    )
    embedder.load_model(str(CAMERA_MODEL_PATH))
    return embedder


def _save_confusion_matrix(camera_metrics: dict, output: Path) -> None:
    metrics = camera_metrics["deployed_threshold_metrics"]
    matrix = np.array([[metrics["tp"], metrics["fn"]], [metrics["fp"], metrics["tn"]]], dtype=float)
    labels = np.array([["TP", "FN"], ["FP", "TN"]])
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Predicted Accept", "Predicted Reject"])
    ax.set_yticks([0, 1], labels=["Actual Genuine", "Actual Impostor"])
    ax.set_title("Camera Pairwise Confusion Matrix at Deployed Threshold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i, j]}\n{int(matrix[i, j])}", ha="center", va="center", color="#08111b", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _save_threshold_sweep(scores: List[float], labels: List[int], evaluator: ModelEvaluator, deployed_threshold: float, output: Path) -> None:
    thresholds = np.unique(np.linspace(min(scores), max(scores), 220))
    fars = []
    frrs = []
    for threshold in thresholds:
        metrics = evaluator.evaluate_threshold(scores, labels, float(threshold))
        fars.append(metrics["far"])
        frrs.append(metrics["frr"])
    plt.figure(figsize=(9.6, 6.2))
    plt.plot(thresholds, fars, label="FAR", color="#0d6efd", linewidth=2.2)
    plt.plot(thresholds, frrs, label="FRR", color="#20c997", linewidth=2.2)
    plt.axvline(deployed_threshold, color="#ff8c42", linestyle="--", linewidth=2.0, label=f"Deployed threshold {deployed_threshold:.4f}")
    plt.ylim(0, 1.02)
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title("Camera Threshold Sweep")
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()


def _save_similarity_heatmap(ordered_samples: List[dict], embedder: DeviceEmbedder, output: Path) -> None:
    embeddings = []
    tick_positions = []
    tick_labels = []
    current_device = None
    for idx, sample in enumerate(ordered_samples):
        tensor = torch.from_numpy(sample["vector"]).float()
        embedding = embedder.embed(tensor).numpy()
        embeddings.append(embedding)
        if sample["device_id"] != current_device:
            tick_positions.append(idx)
            tick_labels.append(sample["device_id"][:8])
            current_device = sample["device_id"]
    matrix = np.matmul(np.vstack(embeddings), np.vstack(embeddings).T)
    plt.figure(figsize=(8.8, 7.4))
    plt.imshow(matrix, cmap="viridis", vmin=-1.0, vmax=1.0)
    plt.colorbar(fraction=0.046, pad=0.04, label="Cosine similarity")
    plt.xticks(tick_positions, tick_labels, rotation=45, ha="right")
    plt.yticks(tick_positions, tick_labels)
    plt.title("Camera Test-Embedding Similarity Heatmap")
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()


def _save_embedding_projection(ordered_samples: List[dict], embedder: DeviceEmbedder, output: Path) -> None:
    vectors = np.vstack([sample["vector"] for sample in ordered_samples])
    device_ids = [sample["device_id"] for sample in ordered_samples]
    embeddings = []
    for vector in vectors:
        tensor = torch.from_numpy(vector).float()
        embeddings.append(embedder.embed(tensor).numpy())
    emb_matrix = np.vstack(embeddings)

    pca = PCA(n_components=2, random_state=42)
    pca_points = pca.fit_transform(emb_matrix)

    perplexity = max(5, min(20, len(emb_matrix) - 1))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="random", learning_rate="auto", random_state=42)
    tsne_points = tsne.fit_transform(emb_matrix)

    unique_devices = sorted(set(device_ids))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_devices)))
    color_map = {device: colors[i] for i, device in enumerate(unique_devices)}

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))
    for ax, points, title in [
        (axes[0], pca_points, "PCA Projection"),
        (axes[1], tsne_points, "t-SNE Projection"),
    ]:
        for device in unique_devices:
            idxs = [i for i, d in enumerate(device_ids) if d == device]
            ax.scatter(points[idxs, 0], points[idxs, 1], s=34, alpha=0.85, color=color_map[device], label=device[:8])
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.22)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(unique_devices)), frameon=False)
    fig.suptitle("Camera Embedding Visualization on Held-Out Test Samples")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _write_operating_point_tables(camera_metrics: dict, mic_metrics: dict, output_md: Path, output_csv: Path) -> None:
    rows = [
        {
            "Configuration": "Camera only",
            "Deployed Threshold": round(camera_metrics["deployed_threshold"], 4),
            "Camera Weight": 1.00,
            "Microphone Weight": 0.00,
            "FAR": round(camera_metrics["deployed_threshold_metrics"]["far"], 4),
            "FRR": round(camera_metrics["deployed_threshold_metrics"]["frr"], 4),
            "TPR": round(camera_metrics["deployed_threshold_metrics"]["tpr"], 4),
            "TNR": round(camera_metrics["deployed_threshold_metrics"]["tnr"], 4),
            "Precision": round(camera_metrics["deployed_threshold_metrics"]["precision"], 4),
            "Recall": round(camera_metrics["deployed_threshold_metrics"]["recall"], 4),
        },
        {
            "Configuration": "Microphone only",
            "Deployed Threshold": round(mic_metrics["deployed_threshold"], 4),
            "Camera Weight": 0.00,
            "Microphone Weight": 1.00,
            "FAR": round(mic_metrics["deployed_threshold_metrics"]["far"], 4),
            "FRR": round(mic_metrics["deployed_threshold_metrics"]["frr"], 4),
            "TPR": round(mic_metrics["deployed_threshold_metrics"]["tpr"], 4),
            "TNR": round(mic_metrics["deployed_threshold_metrics"]["tnr"], 4),
            "Precision": round(mic_metrics["deployed_threshold_metrics"]["precision"], 4),
            "Recall": round(mic_metrics["deployed_threshold_metrics"]["recall"], 4),
        },
        {
            "Configuration": "Planned runtime fusion policy",
            "Deployed Threshold": "camera required",
            "Camera Weight": 0.85,
            "Microphone Weight": 0.15,
            "FAR": "N/A",
            "FRR": "N/A",
            "TPR": "N/A",
            "TNR": "N/A",
            "Precision": "N/A",
            "Recall": "N/A",
        },
    ]

    headers = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Final Operating Point Table",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    lines.extend(
        [
            "",
            "Caption: Final operating points for the camera and microphone models. The deployed runtime configuration uses camera-dominant weighted fusion (0.85 / 0.15) with camera as the required modality. Fusion FAR/FRR are marked N/A because fused performance was not benchmarked as a separate end-to-end experiment.",
        ]
    )
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    REPORT_IMAGES.mkdir(parents=True, exist_ok=True)
    REPORT_TABLES.mkdir(parents=True, exist_ok=True)

    camera_metrics = _load_json(CAMERA_METRICS_PATH)
    mic_metrics = _load_json(MIC_METRICS_PATH)

    features_by_device, ordered_samples, checkpoint = _load_camera_test_features()
    embedder = _load_camera_embedder(checkpoint)
    evaluator = ModelEvaluator(embedder)
    embeddings_by_device = evaluator.compute_embeddings(features_by_device)
    scores, labels = evaluator.compute_similarity_scores(embeddings_by_device, metric="cosine")

    _save_confusion_matrix(camera_metrics, REPORT_IMAGES / "camera_confusion_matrix.png")
    _save_threshold_sweep(scores, labels, evaluator, float(camera_metrics["deployed_threshold"]), REPORT_IMAGES / "camera_threshold_sweep.png")
    _save_similarity_heatmap(ordered_samples, embedder, REPORT_IMAGES / "camera_similarity_heatmap.png")
    _save_embedding_projection(ordered_samples, embedder, REPORT_IMAGES / "camera_embedding_projection.png")
    _write_operating_point_tables(
        camera_metrics,
        mic_metrics,
        REPORT_TABLES / "final_operating_point_table.md",
        REPORT_TABLES / "final_operating_point_table.csv",
    )

    print(f"Wrote analysis assets to {REPORT_IMAGES} and {REPORT_TABLES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
