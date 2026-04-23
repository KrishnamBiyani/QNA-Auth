"""
Model Evaluation and Testing
"""

import torch
import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from typing import Any, Dict, List, Literal, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from .siamese_model import DeviceEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates trained model performance"""
    
    def __init__(self, embedder: DeviceEmbedder):
        """
        Initialize evaluator
        
        Args:
            embedder: Trained DeviceEmbedder instance
        """
        self.embedder = embedder
    
    def compute_embeddings(
        self,
        features_by_device: Dict[str, List[np.ndarray]]
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Compute embeddings for all samples
        
        Args:
            features_by_device: Dictionary mapping device_id to feature arrays
            
        Returns:
            Dictionary mapping device_id to embedding tensors
        """
        embeddings_by_device = {}
        
        for device_id, features_list in features_by_device.items():
            embeddings = []
            for features in features_list:
                features_tensor = torch.from_numpy(features).float()
                embedding = self.embedder.embed(features_tensor)
                embeddings.append(embedding)
            
            embeddings_by_device[device_id] = embeddings
            logger.info(f"Computed {len(embeddings)} embeddings for {device_id}")
        
        return embeddings_by_device
    
    def compute_similarity_scores(
        self,
        embeddings_by_device: Dict[str, List[torch.Tensor]],
        metric: str = 'cosine',
        negative_pair_strategy: Literal["all", "sampled"] = "all",
        negative_pairs_per_device_pair: int = 10,
        positive_pair_strategy: Literal["all", "sampled"] = "all",
        max_positive_pairs_per_device: int | None = None,
        seed: int = 42,
    ) -> Tuple[List[float], List[int]]:
        """
        Compute similarity scores and labels for all pairs
        
        Args:
            embeddings_by_device: Dictionary of embeddings
            metric: Similarity metric
            negative_pair_strategy: Whether to score all impostor pairs or a sampled subset
            negative_pairs_per_device_pair: Number of negative pairs to sample per device pair
            positive_pair_strategy: Whether to score all genuine pairs or sample a subset
            max_positive_pairs_per_device: Maximum genuine pairs per device when sampling

        Returns:
            Tuple of (scores, labels) where label=1 for same device
        """
        scores = []
        labels = []
        rng = np.random.default_rng(seed)
        
        device_ids = list(embeddings_by_device.keys())
        
        # Positive pairs (same device)
        for device_id, embeddings in embeddings_by_device.items():
            positive_index_pairs = [
                (i, j)
                for i in range(len(embeddings))
                for j in range(i + 1, len(embeddings))
            ]
            if (
                positive_pair_strategy == "sampled"
                and max_positive_pairs_per_device is not None
                and len(positive_index_pairs) > max_positive_pairs_per_device
            ):
                sampled_indices = rng.choice(
                    len(positive_index_pairs),
                    size=max_positive_pairs_per_device,
                    replace=False,
                )
                positive_index_pairs = [positive_index_pairs[int(idx)] for idx in sampled_indices]
            for i, j in positive_index_pairs:
                score = self.embedder.compute_similarity(
                    embeddings[i],
                    embeddings[j],
                    metric=metric
                )
                scores.append(score)
                labels.append(1)

        # Negative pairs (different devices)
        for i, device1 in enumerate(device_ids):
            for device2 in device_ids[i+1:]:
                embeddings1 = embeddings_by_device[device1]
                embeddings2 = embeddings_by_device[device2]

                if negative_pair_strategy == "all":
                    negative_index_pairs = [
                        (idx1, idx2)
                        for idx1 in range(len(embeddings1))
                        for idx2 in range(len(embeddings2))
                    ]
                elif negative_pair_strategy == "sampled":
                    negative_index_pairs = []
                    num_samples = min(negative_pairs_per_device_pair, len(embeddings1), len(embeddings2))
                    for _ in range(num_samples):
                        idx1 = int(rng.integers(len(embeddings1)))
                        idx2 = int(rng.integers(len(embeddings2)))
                        negative_index_pairs.append((idx1, idx2))
                else:
                    raise ValueError(f"Unsupported negative_pair_strategy: {negative_pair_strategy}")

                for idx1, idx2 in negative_index_pairs:
                    score = self.embedder.compute_similarity(
                        embeddings1[idx1],
                        embeddings2[idx2],
                        metric=metric
                    )
                    scores.append(score)
                    labels.append(0)
        
        logger.info(f"Computed {len(scores)} similarity scores")
        logger.info(f"Positive pairs: {sum(labels)}, Negative pairs: {len(labels) - sum(labels)}")
        
        return scores, labels

    @staticmethod
    def compute_similarity_scores_from_vectors(
        vectors_by_device: Dict[str, List[np.ndarray]],
        metric: str = "cosine",
        negative_pair_strategy: Literal["all", "sampled"] = "all",
        negative_pairs_per_device_pair: int = 10,
        positive_pair_strategy: Literal["all", "sampled"] = "all",
        max_positive_pairs_per_device: int | None = None,
        seed: int = 42,
    ) -> Tuple[List[float], List[int]]:
        """Compute pairwise scores directly from vectors for non-Siamese baselines."""
        scores: List[float] = []
        labels: List[int] = []
        device_ids = list(vectors_by_device.keys())
        rng = np.random.default_rng(seed)

        def similarity(a: np.ndarray, b: np.ndarray) -> float:
            if metric == "cosine":
                na = np.linalg.norm(a) + 1e-8
                nb = np.linalg.norm(b) + 1e-8
                return float(np.dot(a, b) / (na * nb))
            if metric == "euclidean":
                return -float(np.linalg.norm(a - b))
            raise ValueError(f"Unsupported metric: {metric}")

        for device_id, vectors in vectors_by_device.items():
            positive_index_pairs = [
                (i, j)
                for i in range(len(vectors))
                for j in range(i + 1, len(vectors))
            ]
            if (
                positive_pair_strategy == "sampled"
                and max_positive_pairs_per_device is not None
                and len(positive_index_pairs) > max_positive_pairs_per_device
            ):
                sampled_indices = rng.choice(
                    len(positive_index_pairs),
                    size=max_positive_pairs_per_device,
                    replace=False,
                )
                positive_index_pairs = [positive_index_pairs[int(idx)] for idx in sampled_indices]
            for i, j in positive_index_pairs:
                scores.append(similarity(vectors[i], vectors[j]))
                labels.append(1)

        for i, d1 in enumerate(device_ids):
            for d2 in device_ids[i + 1:]:
                v1 = vectors_by_device[d1]
                v2 = vectors_by_device[d2]
                if negative_pair_strategy == "all":
                    negative_index_pairs = [
                        (idx1, idx2)
                        for idx1 in range(len(v1))
                        for idx2 in range(len(v2))
                    ]
                elif negative_pair_strategy == "sampled":
                    negative_index_pairs = []
                    num = min(negative_pairs_per_device_pair, len(v1), len(v2))
                    for _ in range(num):
                        idx1 = int(rng.integers(len(v1)))
                        idx2 = int(rng.integers(len(v2)))
                        negative_index_pairs.append((idx1, idx2))
                else:
                    raise ValueError(f"Unsupported negative_pair_strategy: {negative_pair_strategy}")
                for idx1, idx2 in negative_index_pairs:
                    scores.append(similarity(v1[idx1], v2[idx2]))
                    labels.append(0)
        return scores, labels

    @staticmethod
    def _candidate_thresholds(scores: List[float], n_thresholds: int = 200) -> np.ndarray:
        if not scores:
            raise ValueError("Scores are required to compute thresholds")
        thresholds = np.linspace(min(scores), max(scores), n_thresholds, dtype=np.float64)
        thresholds = np.unique(
            np.concatenate(
                [
                    np.asarray([min(scores) - 1e-6], dtype=np.float64),
                    thresholds,
                    np.asarray([max(scores) + 1e-6], dtype=np.float64),
                ]
            )
        )
        return thresholds

    @staticmethod
    def _score_distribution_summary(scores: List[float], labels: List[int]) -> Dict[str, Dict[str, float]]:
        score_arr = np.asarray(scores, dtype=np.float64)
        label_arr = np.asarray(labels, dtype=np.int32)

        def summarize(mask: np.ndarray) -> Dict[str, float]:
            subset = score_arr[mask]
            if subset.size == 0:
                return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            return {
                "count": int(subset.size),
                "mean": float(np.mean(subset)),
                "std": float(np.std(subset)),
                "min": float(np.min(subset)),
                "max": float(np.max(subset)),
            }

        return {
            "genuine": summarize(label_arr == 1),
            "impostor": summarize(label_arr == 0),
            "overall": summarize(np.ones_like(label_arr, dtype=bool)),
        }
    
    def evaluate_threshold(
        self,
        scores: List[float],
        labels: List[int],
        threshold: float
    ) -> Dict[str, float]:
        """
        Evaluate model at specific threshold
        
        Args:
            scores: Similarity scores
            labels: Ground truth labels
            threshold: Decision threshold
            
        Returns:
            Dictionary of metrics
        """
        predictions = [1 if s >= threshold else 0 for s in scores]
        
        tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
        tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
        fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
        
        accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Acceptance Rate
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Rejection Rate
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (recall + tnr) / 2.0

        metrics = {
            'threshold': threshold,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'far': far,
            'frr': frr,
            'tpr': recall,
            'tnr': tnr,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
        
        return metrics

    def find_threshold_for_target_far(
        self,
        scores: List[float],
        labels: List[int],
        target_far: float,
        n_thresholds: int = 200,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find the threshold that maximizes genuine acceptance while satisfying the FAR target.
        Falls back to the threshold with the smallest FAR gap if no threshold satisfies it.
        """
        candidates: List[Tuple[float, Dict[str, float]]] = []
        best_fallback: Tuple[float, Dict[str, float]] | None = None
        best_gap = float("inf")

        for threshold in self._candidate_thresholds(scores, n_thresholds=n_thresholds):
            metrics = self.evaluate_threshold(scores, labels, float(threshold))
            gap = abs(metrics["far"] - target_far)
            if gap < best_gap:
                best_gap = gap
                best_fallback = (float(threshold), metrics)
            if metrics["far"] <= target_far:
                candidates.append((float(threshold), metrics))

        if candidates:
            threshold, metrics = min(
                candidates,
                key=lambda item: (
                    item[1]["frr"],
                    -item[1]["balanced_accuracy"],
                    abs(item[1]["far"] - target_far),
                    item[0],
                ),
            )
            return threshold, metrics
        if best_fallback is None:
            raise ValueError("Unable to compute target FAR threshold")
        return best_fallback
    
    def find_optimal_threshold(
        self,
        scores: List[float],
        labels: List[int],
        metric: str = 'balanced_accuracy'
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal threshold based on metric
        
        Args:
            scores: Similarity scores
            labels: Ground truth labels
            metric: Metric to optimize ('f1_score', 'accuracy', etc.)
            
        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        thresholds = self._candidate_thresholds(scores, n_thresholds=200)
        best_threshold = 0.0
        best_score = float("-inf")
        best_metrics = {}
        
        for threshold in thresholds:
            metrics = self.evaluate_threshold(scores, labels, threshold)
            
            if (
                metrics[metric] > best_score
                or (
                    metrics[metric] == best_score
                    and metrics.get("balanced_accuracy", 0.0) > best_metrics.get("balanced_accuracy", float("-inf"))
                )
            ):
                best_score = metrics[metric]
                best_threshold = threshold
                best_metrics = metrics
        
        logger.info(f"Optimal threshold: {best_threshold:.4f} ({metric}={best_score:.4f})")

        return best_threshold, best_metrics

    def compute_eer(
        self,
        scores: List[float],
        labels: List[int],
        n_thresholds: int = 200,
    ) -> Tuple[float, float]:
        """
        Compute Equal Error Rate (EER): the rate at the threshold where FAR = FRR.
        Returns (eer_rate, eer_threshold).
        """
        if len(set(labels)) < 2:
            raise ValueError("EER requires both positive and negative labels")

        fpr, tpr, thresholds = roc_curve(labels, scores)
        fnr = 1.0 - tpr
        idx = int(np.argmin(np.abs(fpr - fnr)))
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
        eer_threshold = float(thresholds[idx])
        return eer, eer_threshold

    def threshold_sweep(
        self,
        scores: List[float],
        labels: List[int],
        n_thresholds: int = 100,
    ) -> List[Dict[str, float]]:
        """Return list of {threshold, far, frr, accuracy, ...} for plotting FAR/FRR vs threshold."""
        thresholds = self._candidate_thresholds(scores, n_thresholds=n_thresholds)
        rows = []
        for t in thresholds:
            m = self.evaluate_threshold(scores, labels, t)
            rows.append({"threshold": t, "far": m["far"], "frr": m["frr"], **m})
        return rows

    def generate_score_report(
        self,
        scores: List[float],
        labels: List[int],
        target_far: float = 0.10,
        optimal_metric: str = "balanced_accuracy",
        deployed_threshold: float | None = None,
    ) -> Dict[str, Any]:
        if not scores or len(set(labels)) < 2:
            raise ValueError("Evaluation requires both genuine and impostor pairs")

        optimal_threshold, optimal_metrics = self.find_optimal_threshold(
            scores, labels, metric=optimal_metric
        )
        target_far_threshold, target_far_metrics = self.find_threshold_for_target_far(
            scores, labels, target_far=target_far
        )
        eer, eer_threshold = self.compute_eer(scores, labels)

        labels_arr = np.asarray(labels, dtype=np.int32)
        scores_arr = np.asarray(scores, dtype=np.float64)
        roc_auc = float(roc_auc_score(labels_arr, scores_arr))
        pr_auc = float(average_precision_score(labels_arr, scores_arr))

        report: Dict[str, Any] = {
            "pair_counts": {
                "total": int(len(labels)),
                "genuine": int(np.sum(labels_arr == 1)),
                "impostor": int(np.sum(labels_arr == 0)),
            },
            "score_summary": self._score_distribution_summary(scores, labels),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "optimal_metric": optimal_metric,
            "optimal_threshold": float(optimal_threshold),
            "metrics": optimal_metrics,
            "target_far": float(target_far),
            "target_far_threshold": float(target_far_threshold),
            "target_far_metrics": target_far_metrics,
            "eer": float(eer),
            "eer_threshold": float(eer_threshold),
        }
        if deployed_threshold is not None:
            report["deployed_threshold"] = float(deployed_threshold)
            report["deployed_threshold_metrics"] = self.evaluate_threshold(scores, labels, float(deployed_threshold))
        return report

    def plot_roc_curve(
        self,
        scores: List[float],
        labels: List[int],
        save_path: str = None
    ):
        """
        Plot ROC curve
        
        Args:
            scores: Similarity scores
            labels: Ground truth labels
            save_path: Path to save plot
        """
        if len(set(labels)) < 2:
            logger.warning("Skipping ROC curve: need both positive and negative labels")
            return
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FAR)')
        plt.ylabel('True Positive Rate (1 - FRR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.close()
    
    def plot_precision_recall_curve(
        self,
        scores: List[float],
        labels: List[int],
        save_path: str = None
    ):
        """
        Plot precision-recall curve
        
        Args:
            scores: Similarity scores
            labels: Ground truth labels
            save_path: Path to save plot
        """
        if len(set(labels)) < 2:
            logger.warning("Skipping PR curve: need both positive and negative labels")
            return
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-recall curve saved to {save_path}")
        
        plt.close()
    
    def generate_report(
        self,
        features_by_device: Dict[str, List[np.ndarray]],
        save_dir: str = "./model/evaluation",
        target_far: float = 0.10,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        
        Args:
            features_by_device: Dictionary of features
            save_dir: Directory to save results
            
        Returns:
            Evaluation report dictionary
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Compute embeddings
        logger.info("Computing embeddings...")
        embeddings_by_device = self.compute_embeddings(features_by_device)
        
        # Compute similarity scores
        logger.info("Computing similarity scores...")
        scores, labels = self.compute_similarity_scores(embeddings_by_device)
        report = self.generate_score_report(
            scores,
            labels,
            target_far=target_far,
            optimal_metric="balanced_accuracy",
        )
        
        # Plot curves
        logger.info("Plotting ROC curve...")
        self.plot_roc_curve(scores, labels, save_path / "roc_curve.png")
        
        logger.info("Plotting precision-recall curve...")
        self.plot_precision_recall_curve(scores, labels, save_path / "pr_curve.png")
        
        # Extend report with dataset context
        report = {
            **report,
            'num_devices': len(features_by_device),
            'total_samples': sum(len(f) for f in features_by_device.values()),
        }
        
        logger.info("\n=== Evaluation Report ===")
        logger.info(f"Number of devices: {report['num_devices']}")
        logger.info(f"Total samples: {report['total_samples']}")
        logger.info(f"Pair counts: {report['pair_counts']}")
        logger.info(f"Optimal threshold: {report['optimal_threshold']:.4f}")
        logger.info(f"ROC-AUC: {report['roc_auc']:.4f}")
        logger.info(f"PR-AUC: {report['pr_auc']:.4f}")
        logger.info(f"Accuracy: {report['metrics']['accuracy']:.4f}")
        logger.info(f"Balanced Accuracy: {report['metrics']['balanced_accuracy']:.4f}")
        logger.info(f"Precision: {report['metrics']['precision']:.4f}")
        logger.info(f"Recall: {report['metrics']['recall']:.4f}")
        logger.info(f"F1 Score: {report['metrics']['f1_score']:.4f}")
        logger.info(f"FAR: {report['metrics']['far']:.4f}")
        logger.info(f"FRR: {report['metrics']['frr']:.4f}")
        logger.info(
            "Target FAR %.3f threshold %.4f => FAR %.4f, FRR %.4f",
            target_far,
            report["target_far_threshold"],
            report["target_far_metrics"]["far"],
            report["target_far_metrics"]["frr"],
        )
        
        return report


def main():
    """Test evaluation pipeline"""
    print("\n=== Evaluation Pipeline Test ===")
    
    # Create synthetic data
    input_dim = 50
    embedding_dim = 128
    num_devices = 3
    samples_per_device = 20
    
    # Generate synthetic features
    features_by_device = {}
    for i in range(num_devices):
        device_id = f"device_{i:03d}"
        features = [np.random.randn(input_dim) + i * 1.0 
                   for _ in range(samples_per_device)]
        features_by_device[device_id] = features
    
    # Create embedder (with random model)
    embedder = DeviceEmbedder(input_dim=input_dim, embedding_dim=embedding_dim)
    
    # Create evaluator
    evaluator = ModelEvaluator(embedder)
    
    # Generate report
    report = evaluator.generate_report(features_by_device)
    
    print("\n=== Report ===")
    print(report)


if __name__ == "__main__":
    main()
