"""
Model Evaluation and Testing
"""

import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from typing import Dict, List, Tuple
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
        metric: str = 'cosine'
    ) -> Tuple[List[float], List[int]]:
        """
        Compute similarity scores and labels for all pairs
        
        Args:
            embeddings_by_device: Dictionary of embeddings
            metric: Similarity metric
            
        Returns:
            Tuple of (scores, labels) where label=1 for same device
        """
        scores = []
        labels = []
        
        device_ids = list(embeddings_by_device.keys())
        
        # Positive pairs (same device)
        for device_id, embeddings in embeddings_by_device.items():
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
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
                
                # Sample a subset of negative pairs
                num_samples = min(10, len(embeddings1), len(embeddings2))
                for _ in range(num_samples):
                    idx1 = np.random.randint(len(embeddings1))
                    idx2 = np.random.randint(len(embeddings2))
                    
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
        
        metrics = {
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'far': far,
            'frr': frr,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
        
        return metrics
    
    def find_optimal_threshold(
        self,
        scores: List[float],
        labels: List[int],
        metric: str = 'f1_score'
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
        thresholds = np.linspace(min(scores), max(scores), 100)
        best_threshold = 0.0
        best_score = 0.0
        best_metrics = {}
        
        for threshold in thresholds:
            metrics = self.evaluate_threshold(scores, labels, threshold)
            
            if metrics[metric] > best_score:
                best_score = metrics[metric]
                best_threshold = threshold
                best_metrics = metrics
        
        logger.info(f"Optimal threshold: {best_threshold:.4f} ({metric}={best_score:.4f})")
        
        return best_threshold, best_metrics
    
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
        save_dir: str = "./model/evaluation"
    ) -> Dict[str, any]:
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
        
        # Find optimal threshold
        logger.info("Finding optimal threshold...")
        optimal_threshold, optimal_metrics = self.find_optimal_threshold(
            scores, labels, metric='f1_score'
        )
        
        # Plot curves
        logger.info("Plotting ROC curve...")
        self.plot_roc_curve(scores, labels, save_path / "roc_curve.png")
        
        logger.info("Plotting precision-recall curve...")
        self.plot_precision_recall_curve(scores, labels, save_path / "pr_curve.png")
        
        # Create report
        report = {
            'num_devices': len(features_by_device),
            'total_samples': sum(len(f) for f in features_by_device.values()),
            'optimal_threshold': optimal_threshold,
            'metrics': optimal_metrics
        }
        
        logger.info("\n=== Evaluation Report ===")
        logger.info(f"Number of devices: {report['num_devices']}")
        logger.info(f"Total samples: {report['total_samples']}")
        logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
        logger.info(f"Accuracy: {optimal_metrics['accuracy']:.4f}")
        logger.info(f"Precision: {optimal_metrics['precision']:.4f}")
        logger.info(f"Recall: {optimal_metrics['recall']:.4f}")
        logger.info(f"F1 Score: {optimal_metrics['f1_score']:.4f}")
        logger.info(f"FAR: {optimal_metrics['far']:.4f}")
        logger.info(f"FRR: {optimal_metrics['frr']:.4f}")
        
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
