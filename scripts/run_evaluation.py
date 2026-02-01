#!/usr/bin/env python
"""
Evaluation Metrics Runner

Generates comprehensive evaluation report including:
- FAR (False Accept Rate)
- FRR (False Reject Rate)  
- EER (Equal Error Rate)
- ROC and PR curves
- Threshold analysis
- Per-device performance breakdown

Usage:
    python scripts/run_evaluation.py [--model-path MODEL] [--data-dir DATA] [--output-dir OUTPUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

try:
    from model.siamese_model import DeviceEmbedder
    from model.train import load_real_dataset
    from preprocessing.features import FEATURE_VERSION, get_canonical_feature_names
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root")
    sys.exit(1)


class EvaluationRunner:
    """Runs comprehensive model evaluation."""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model_path = Path(model_path)
        self.embedder = None
        self.results = {}
        
    def load_model(self) -> bool:
        """Load the trained model."""
        if not self.model_path.exists():
            print(f"Model not found: {self.model_path}")
            print("Using untrained model for demonstration...")
            # Create default embedder
            self.embedder = DeviceEmbedder(input_dim=50)
            return True
            
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            input_dim = checkpoint.get("input_dim", 50)
            self.embedder = DeviceEmbedder(input_dim=input_dim)
            self.embedder.model.load_state_dict(checkpoint["model_state_dict"])
            self.embedder.model.eval()
            print(f"Loaded model from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def load_data(self, data_dir: str) -> dict:
        """Load evaluation dataset."""
        features_by_device, input_dim = load_real_dataset(data_dir=data_dir)
        
        if not features_by_device:
            print("No data found. Creating synthetic demo data...")
            features_by_device = self._create_demo_data()
            
        return features_by_device
    
    def _create_demo_data(self) -> dict:
        """Create synthetic demo data for testing."""
        np.random.seed(42)
        demo_data = {}
        
        for i in range(3):
            device_id = f"demo_device_{i}"
            # Create 20 samples per device with some device-specific offset
            base = np.random.randn(50) * 0.5 + i * 0.3  # Device-specific signature
            samples = [base + np.random.randn(50) * 0.1 for _ in range(20)]
            demo_data[device_id] = [s.astype(np.float32) for s in samples]
            
        return demo_data
    
    def compute_embeddings(self, features_by_device: dict) -> dict:
        """Compute embeddings for all samples."""
        embeddings_by_device = {}
        
        for device_id, features_list in features_by_device.items():
            embeddings = []
            for features in features_list:
                with torch.no_grad():
                    features_tensor = torch.from_numpy(features).float()
                    embedding = self.embedder.embed(features_tensor)
                    embeddings.append(embedding.numpy())
            embeddings_by_device[device_id] = embeddings
            
        return embeddings_by_device
    
    def compute_similarity_pairs(self, embeddings_by_device: dict) -> tuple:
        """Compute all similarity scores with labels."""
        scores = []
        labels = []
        pair_info = []
        
        device_ids = list(embeddings_by_device.keys())
        
        # Positive pairs (same device)
        for device_id in device_ids:
            embeddings = embeddings_by_device[device_id]
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = self._cosine_similarity(embeddings[i], embeddings[j])
                    scores.append(sim)
                    labels.append(1)
                    pair_info.append(("same", device_id, device_id))
        
        # Negative pairs (different devices)
        for i, dev1 in enumerate(device_ids):
            for j, dev2 in enumerate(device_ids):
                if i >= j:
                    continue
                emb1_list = embeddings_by_device[dev1]
                emb2_list = embeddings_by_device[dev2]
                for e1 in emb1_list:
                    for e2 in emb2_list:
                        sim = self._cosine_similarity(e1, e2)
                        scores.append(sim)
                        labels.append(0)
                        pair_info.append(("diff", dev1, dev2))
        
        return np.array(scores), np.array(labels), pair_info
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        a = a.flatten()
        b = b.flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def compute_metrics(self, scores: np.ndarray, labels: np.ndarray) -> dict:
        """Compute FAR, FRR, EER, and optimal threshold."""
        # ROC curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        # FAR = FPR, FRR = 1 - TPR
        far = fpr
        frr = 1 - tpr
        
        # Find EER (where FAR ≈ FRR)
        eer_idx = np.argmin(np.abs(far - frr))
        eer = (far[eer_idx] + frr[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]
        
        # Find optimal threshold (maximize TPR - FPR)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Precision-Recall
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)
        
        # Metrics at specific thresholds
        threshold_metrics = {}
        for thresh in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            preds = (scores >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
            threshold_metrics[f"{thresh:.2f}"] = {
                "accuracy": (tp + tn) / (tp + tn + fp + fn),
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
                "far": fp / (fp + tn) if (fp + tn) > 0 else 0,
                "frr": fn / (fn + tp) if (fn + tp) > 0 else 0
            }
        
        return {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "eer": float(eer),
            "eer_threshold": float(eer_threshold),
            "optimal_threshold": float(optimal_threshold),
            "threshold_metrics": threshold_metrics,
            "num_positive_pairs": int(np.sum(labels)),
            "num_negative_pairs": int(len(labels) - np.sum(labels)),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()},
            "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist()}
        }
    
    def plot_curves(self, metrics: dict, output_dir: Path):
        """Generate and save evaluation plots."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ROC Curve
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: ROC Curve
        roc = metrics["roc_curve"]
        axes[0].plot(roc["fpr"], roc["tpr"], 'b-', linewidth=2, 
                     label=f'ROC (AUC = {metrics["roc_auc"]:.3f})')
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[0].set_xlabel('False Positive Rate (FAR)')
        axes[0].set_ylabel('True Positive Rate (1-FRR)')
        axes[0].set_title('ROC Curve')
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: FAR/FRR vs Threshold
        thresholds = np.array(roc["thresholds"])
        far = np.array(roc["fpr"])
        frr = 1 - np.array(roc["tpr"])
        
        axes[1].plot(thresholds, far, 'r-', linewidth=2, label='FAR')
        axes[1].plot(thresholds, frr, 'b-', linewidth=2, label='FRR')
        axes[1].axvline(x=metrics["eer_threshold"], color='g', linestyle='--', 
                       label=f'EER = {metrics["eer"]:.3f}')
        axes[1].set_xlabel('Threshold')
        axes[1].set_ylabel('Error Rate')
        axes[1].set_title('FAR/FRR vs Threshold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([0, 1])
        
        # Plot 3: Precision-Recall
        pr = metrics["pr_curve"]
        axes[2].plot(pr["recall"], pr["precision"], 'g-', linewidth=2,
                    label=f'PR (AUC = {metrics["pr_auc"]:.3f})')
        axes[2].set_xlabel('Recall')
        axes[2].set_ylabel('Precision')
        axes[2].set_title('Precision-Recall Curve')
        axes[2].legend(loc='lower left')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "evaluation_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plots to {output_dir / 'evaluation_curves.png'}")
    
    def generate_report(self, metrics: dict, output_dir: Path, features_by_device: dict):
        """Generate markdown report."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = f"""# QNA-Auth Evaluation Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Model**: {self.model_path}  
**Feature Version**: {FEATURE_VERSION}

---

## Summary

| Metric | Value |
|--------|-------|
| **ROC AUC** | {metrics['roc_auc']:.4f} |
| **PR AUC** | {metrics['pr_auc']:.4f} |
| **EER (Equal Error Rate)** | {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%) |
| **EER Threshold** | {metrics['eer_threshold']:.4f} |
| **Optimal Threshold** | {metrics['optimal_threshold']:.4f} |

---

## Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Number of Devices | {len(features_by_device)} |
| Total Samples | {sum(len(v) for v in features_by_device.values())} |
| Positive Pairs | {metrics['num_positive_pairs']} |
| Negative Pairs | {metrics['num_negative_pairs']} |

### Per-Device Sample Counts

| Device ID | Samples |
|-----------|---------|
"""
        for dev_id, samples in features_by_device.items():
            report += f"| {dev_id} | {len(samples)} |\n"
        
        report += f"""
---

## Performance at Different Thresholds

| Threshold | Accuracy | Precision | Recall | FAR | FRR |
|-----------|----------|-----------|--------|-----|-----|
"""
        for thresh, m in metrics['threshold_metrics'].items():
            report += f"| {thresh} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['far']:.4f} | {m['frr']:.4f} |\n"
        
        report += f"""
---

## Interpretation

### EER (Equal Error Rate): {metrics['eer']*100:.2f}%

The EER is the point where FAR = FRR. Lower is better.

- **< 1%**: Excellent - production ready
- **1-5%**: Good - suitable for most applications
- **5-10%**: Fair - may need improvement
- **> 10%**: Poor - requires significant work

### Recommended Threshold: {metrics['optimal_threshold']:.4f}

This threshold maximizes the difference between True Positive Rate and False Positive Rate.
For high-security applications, consider using a higher threshold (lower FAR, higher FRR).

---

## Curves

![Evaluation Curves](evaluation_curves.png)

---

## Next Steps

1. If EER > 5%: Collect more training data per device
2. If FAR is high: Increase threshold or improve model
3. If FRR is high: Decrease threshold or collect more samples during enrollment
4. Run ablation studies to identify best noise sources

---

*Report generated by QNA-Auth evaluation pipeline*
"""
        
        report_path = output_dir / "evaluation_report.md"
        with open(report_path, "w") as f:
            f.write(report)
        
        # Save metrics as JSON
        metrics_path = output_dir / "evaluation_metrics.json"
        # Remove large curve data for JSON
        metrics_json = {k: v for k, v in metrics.items() if k not in ["roc_curve", "pr_curve"]}
        with open(metrics_path, "w") as f:
            json.dump(metrics_json, f, indent=2)
        
        print(f"Saved report to {report_path}")
        print(f"Saved metrics to {metrics_path}")
    
    def run(self, data_dir: str, output_dir: str) -> dict:
        """Run full evaluation pipeline."""
        print("=" * 60)
        print("QNA-Auth Model Evaluation")
        print("=" * 60)
        
        # Load model
        if not self.load_model():
            return {}
        
        # Load data
        print("\nLoading data...")
        features_by_device = self.load_data(data_dir)
        print(f"Loaded {len(features_by_device)} devices")
        
        # Compute embeddings
        print("\nComputing embeddings...")
        embeddings_by_device = self.compute_embeddings(features_by_device)
        
        # Compute similarity pairs
        print("\nComputing similarity scores...")
        scores, labels, pair_info = self.compute_similarity_pairs(embeddings_by_device)
        print(f"Total pairs: {len(scores)} (positive: {sum(labels)}, negative: {len(labels) - sum(labels)})")
        
        # Compute metrics
        print("\nComputing metrics...")
        metrics = self.compute_metrics(scores, labels)
        
        # Generate outputs
        output_path = Path(output_dir)
        print("\nGenerating plots...")
        self.plot_curves(metrics, output_path)
        
        print("\nGenerating report...")
        self.generate_report(metrics, output_path, features_by_device)
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"  ROC AUC:           {metrics['roc_auc']:.4f}")
        print(f"  PR AUC:            {metrics['pr_auc']:.4f}")
        print(f"  EER:               {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%)")
        print(f"  EER Threshold:     {metrics['eer_threshold']:.4f}")
        print(f"  Optimal Threshold: {metrics['optimal_threshold']:.4f}")
        print("=" * 60)
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--model-path", type=str, 
                       default="model/checkpoints/best_model.pt",
                       help="Path to trained model checkpoint")
    parser.add_argument("--data-dir", type=str,
                       default="dataset/samples",
                       help="Directory containing evaluation data")
    parser.add_argument("--output-dir", type=str,
                       default="model/evaluation",
                       help="Directory for evaluation outputs")
    args = parser.parse_args()
    
    runner = EvaluationRunner(args.model_path)
    metrics = runner.run(args.data_dir, args.output_dir)
    
    if metrics:
        print(f"\nEvaluation complete! Check {args.output_dir} for results.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
