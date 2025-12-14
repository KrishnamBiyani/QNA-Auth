"""
Dataset Builder for QNA-Auth
Creates labeled datasets from collected noise samples
"""

import os
import json
import csv
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Builds and manages labeled datasets for device authentication"""
    
    def __init__(self, base_dir: str = "./dataset/samples"):
        """
        Initialize dataset builder
        
        Args:
            base_dir: Base directory for storing datasets
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_file = self.base_dir / "noise_samples.csv"
        self.json_dir = self.base_dir / "json"
        self.json_dir.mkdir(exist_ok=True)
        
        # Initialize CSV if it doesn't exist
        if not self.csv_file.exists():
            self._initialize_csv()
    
    def _initialize_csv(self):
        """Create CSV file with headers"""
        headers = [
            'sample_id',
            'device_id',
            'timestamp',
            'noise_source',
            'sample_length',
            'mean',
            'std',
            'min',
            'max',
            'entropy',
            'raw_data_path'
        ]
        
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
        
        logger.info(f"Initialized CSV dataset at {self.csv_file}")
    
    def create_sample(
        self,
        device_id: str,
        noise_source: str,
        raw_noise_sample: np.ndarray,
        processed_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a labeled sample entry
        
        Args:
            device_id: Unique device identifier
            noise_source: Source of noise (qrng, camera, microphone, sensor)
            raw_noise_sample: Raw noise array
            processed_features: Optional pre-computed features
            
        Returns:
            Dictionary containing sample metadata
        """
        # Generate unique sample ID
        sample_id = f"{device_id}_{noise_source}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        timestamp = datetime.now().isoformat()
        
        # Compute basic statistics
        stats = {
            'mean': float(np.mean(raw_noise_sample)),
            'std': float(np.std(raw_noise_sample)),
            'min': float(np.min(raw_noise_sample)),
            'max': float(np.max(raw_noise_sample)),
            'length': len(raw_noise_sample)
        }
        
        # Compute entropy
        entropy = self._compute_entropy(raw_noise_sample)
        
        # Save raw data to numpy file
        raw_data_path = self.base_dir / f"{sample_id}_raw.npy"
        np.save(raw_data_path, raw_noise_sample)
        
        # Create sample metadata
        sample_data = {
            'sample_id': sample_id,
            'device_id': device_id,
            'timestamp': timestamp,
            'noise_source': noise_source,
            'sample_length': stats['length'],
            'mean': stats['mean'],
            'std': stats['std'],
            'min': stats['min'],
            'max': stats['max'],
            'entropy': entropy,
            'raw_data_path': str(raw_data_path.relative_to(self.base_dir))
        }
        
        # Add processed features if provided
        if processed_features:
            sample_data['processed_features'] = processed_features
        
        return sample_data
    
    def add_sample(self, sample_data: Dict[str, Any]):
        """
        Add a sample to the dataset
        
        Args:
            sample_data: Sample metadata dictionary
        """
        # Append to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sample_data.keys())
            # Write only CSV-compatible fields
            csv_data = {k: v for k, v in sample_data.items() if k != 'processed_features'}
            writer.writerow(csv_data)
        
        # Save full metadata to JSON
        json_path = self.json_dir / f"{sample_data['sample_id']}.json"
        with open(json_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        logger.info(f"Added sample: {sample_data['sample_id']}")
    
    def add_batch(
        self,
        device_id: str,
        noise_source: str,
        samples: List[np.ndarray],
        features_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add multiple samples in batch
        
        Args:
            device_id: Device identifier
            noise_source: Noise source type
            samples: List of noise arrays
            features_list: Optional list of feature dictionaries
            
        Returns:
            List of created sample IDs
        """
        sample_ids = []
        
        if features_list is None:
            features_list = [None] * len(samples)
        
        for i, (sample, features) in enumerate(zip(samples, features_list)):
            sample_data = self.create_sample(
                device_id=device_id,
                noise_source=noise_source,
                raw_noise_sample=sample,
                processed_features=features
            )
            self.add_sample(sample_data)
            sample_ids.append(sample_data['sample_id'])
            
            if (i + 1) % 10 == 0:
                logger.info(f"Added {i + 1}/{len(samples)} samples")
        
        logger.info(f"Batch complete: {len(sample_ids)} samples added")
        return sample_ids
    
    def get_samples_by_device(self, device_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all samples for a specific device
        
        Args:
            device_id: Device identifier
            
        Returns:
            List of sample metadata dictionaries
        """
        samples = []
        
        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['device_id'] == device_id:
                    samples.append(row)
        
        logger.info(f"Found {len(samples)} samples for device {device_id}")
        return samples
    
    def get_samples_by_source(self, noise_source: str) -> List[Dict[str, Any]]:
        """
        Retrieve all samples from a specific noise source
        
        Args:
            noise_source: Noise source type
            
        Returns:
            List of sample metadata dictionaries
        """
        samples = []
        
        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['noise_source'] == noise_source:
                    samples.append(row)
        
        logger.info(f"Found {len(samples)} samples from {noise_source}")
        return samples
    
    def load_raw_sample(self, sample_id: str) -> Optional[np.ndarray]:
        """
        Load raw noise data for a sample
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            Numpy array of raw noise data
        """
        # Find sample in JSON
        json_path = self.json_dir / f"{sample_id}.json"
        
        if not json_path.exists():
            logger.error(f"Sample {sample_id} not found")
            return None
        
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        
        # Load raw data
        raw_path = self.base_dir / metadata['raw_data_path']
        if not raw_path.exists():
            logger.error(f"Raw data file not found: {raw_path}")
            return None
        
        return np.load(raw_path)
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get overall dataset statistics
        
        Returns:
            Dictionary of dataset statistics
        """
        devices = set()
        sources = {}
        total_samples = 0
        
        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                devices.add(row['device_id'])
                source = row['noise_source']
                sources[source] = sources.get(source, 0) + 1
                total_samples += 1
        
        stats = {
            'total_samples': total_samples,
            'unique_devices': len(devices),
            'devices': list(devices),
            'samples_by_source': sources
        }
        
        return stats
    
    def _compute_entropy(self, data: np.ndarray) -> float:
        """
        Compute Shannon entropy of data
        
        Args:
            data: Input array
            
        Returns:
            Entropy value
        """
        # Quantize to reasonable bins
        hist, _ = np.histogram(data, bins=256)
        hist = hist[hist > 0]  # Remove zero bins
        
        # Normalize to probabilities
        probs = hist / hist.sum()
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs))
        
        return float(entropy)
    
    def export_for_training(
        self,
        output_dir: str,
        train_ratio: float = 0.8
    ) -> Dict[str, str]:
        """
        Export dataset split for training
        
        Args:
            output_dir: Output directory for train/test split
            train_ratio: Ratio of training samples
            
        Returns:
            Dictionary with paths to train/test files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Read all samples
        samples = []
        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            samples = list(reader)
        
        # Shuffle and split
        np.random.shuffle(samples)
        split_idx = int(len(samples) * train_ratio)
        
        train_samples = samples[:split_idx]
        test_samples = samples[split_idx:]
        
        # Save splits
        train_file = output_path / "train.csv"
        test_file = output_path / "test.csv"
        
        with open(train_file, 'w', newline='') as f:
            if train_samples:
                writer = csv.DictWriter(f, fieldnames=train_samples[0].keys())
                writer.writeheader()
                writer.writerows(train_samples)
        
        with open(test_file, 'w', newline='') as f:
            if test_samples:
                writer = csv.DictWriter(f, fieldnames=test_samples[0].keys())
                writer.writeheader()
                writer.writerows(test_samples)
        
        logger.info(f"Exported {len(train_samples)} train, {len(test_samples)} test samples")
        
        return {
            'train': str(train_file),
            'test': str(test_file)
        }


def main():
    """Test dataset builder"""
    builder = DatasetBuilder()
    
    print("\n=== Dataset Builder Test ===")
    
    # Create synthetic samples for testing
    print("\n=== Creating Test Samples ===")
    
    # Device 1 samples
    device1_samples = [
        np.random.rand(1024) for _ in range(5)
    ]
    builder.add_batch(
        device_id="device_001",
        noise_source="qrng",
        samples=device1_samples
    )
    
    # Device 2 samples
    device2_samples = [
        np.random.rand(512) for _ in range(3)
    ]
    builder.add_batch(
        device_id="device_002",
        noise_source="camera",
        samples=device2_samples
    )
    
    # Get statistics
    print("\n=== Dataset Statistics ===")
    stats = builder.get_dataset_statistics()
    print(json.dumps(stats, indent=2))
    
    # Query samples
    print("\n=== Query Samples ===")
    device1_data = builder.get_samples_by_device("device_001")
    print(f"Device 001 samples: {len(device1_data)}")
    
    qrng_data = builder.get_samples_by_source("qrng")
    print(f"QRNG samples: {len(qrng_data)}")
    
    # Export for training
    print("\n=== Exporting for Training ===")
    export_paths = builder.export_for_training("./dataset/training_data")
    print(f"Train file: {export_paths['train']}")
    print(f"Test file: {export_paths['test']}")


if __name__ == "__main__":
    main()
