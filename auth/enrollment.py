"""
Device Enrollment Module
Collects noise samples and creates device embeddings
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import json
import hashlib
from datetime import datetime
import logging

from model.siamese_model import DeviceEmbedder
from preprocessing.features import NoisePreprocessor, FeatureVector
from noise_collection import QRNGClient, CameraNoiseCollector, MicrophoneNoiseCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeviceEnroller:
    """Handles device enrollment process"""
    
    def __init__(
        self,
        embedder: DeviceEmbedder,
        preprocessor: NoisePreprocessor,
        feature_converter: FeatureVector,
        storage_dir: str = "./auth/device_embeddings",
        dataset_builder: Optional[object] = None
    ):
        """
        Initialize device enroller
        
        Args:
            embedder: Trained DeviceEmbedder
            preprocessor: Noise preprocessor
            feature_converter: Feature vector converter
            storage_dir: Directory to store device embeddings
            dataset_builder: Optional dataset builder for saving raw training data
        """
        self.embedder = embedder
        self.preprocessor = preprocessor
        self.feature_converter = feature_converter
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_builder = dataset_builder
        
        logger.info("DeviceEnroller initialized")
    
    def generate_device_id(self, device_name: Optional[str] = None) -> str:
        """
        Generate unique device ID
        
        Args:
            device_name: Optional device name
            
        Returns:
            Unique device ID
        """
        timestamp = datetime.now().isoformat()
        
        if device_name:
            identifier = f"{device_name}_{timestamp}"
        else:
            identifier = timestamp
        
        # Generate hash-based ID
        device_id = hashlib.sha256(identifier.encode()).hexdigest()[:16]
        
        return device_id
    
    def collect_noise_samples(
        self,
        num_samples: int = 50,
        sources: List[str] = ['qrng', 'camera', 'microphone']
    ) -> Dict[str, List[np.ndarray]]:
        """
        Collect noise samples from multiple sources
        
        Args:
            num_samples: Number of samples to collect per source
            sources: List of noise sources to use
            
        Returns:
            Dictionary mapping source name to list of noise arrays
        """
        noise_samples = {}
        
        # QRNG samples (with API key for authentic quantum noise)
        if 'qrng' in sources:
            logger.info(f"Collecting {num_samples} QRNG samples...")
            try:
                # Use API key from environment only (no hardcoded default)
                import os
                api_key = os.getenv('QRNG_API_KEY')
                qrng_client = QRNGClient(api_key=api_key)
                samples = qrng_client.fetch_multiple_samples(
                    num_samples=num_samples,
                    sample_size=1024
                )
                noise_samples['qrng'] = samples
                logger.info(f"Collected {len(samples)} authentic QRNG samples")
            except Exception as e:
                logger.error(f"Failed to collect QRNG samples: {e}")
        
        # Camera noise samples
        if 'camera' in sources:
            logger.info(f"Collecting {num_samples} camera noise samples...")
            try:
                camera_collector = CameraNoiseCollector(camera_index=0)
                if camera_collector.initialize_camera():
                    logger.info("Camera initialized successfully")
                    frames = camera_collector.capture_multiple_frames(
                        num_frames=num_samples,
                        exposure_time=0.1
                    )
                    logger.info(f"Captured {len(frames)} raw frames")
                    
                    # Extract noise features from frames
                    samples = [camera_collector.extract_noise_features(frame) 
                              for frame in frames if frame is not None]
                    noise_samples['camera'] = samples
                    camera_collector.release()
                    logger.info(f"Collected {len(samples)} camera samples with shapes: {[s.shape for s in samples[:3]]}")
                else:
                    logger.error("Failed to initialize camera")
            except Exception as e:
                logger.error(f"Failed to collect camera samples: {e}", exc_info=True)
        
        # Microphone noise samples
        if 'microphone' in sources:
            logger.info(f"Collecting {num_samples} microphone noise samples...")
            try:
                mic_collector = MicrophoneNoiseCollector(sample_rate=44100)
                logger.info("Microphone collector initialized")
                samples = mic_collector.capture_multiple_samples(
                    num_samples=num_samples,
                    duration=0.5
                )
                noise_samples['microphone'] = samples
                logger.info(f"Collected {len(samples)} microphone samples with shapes: {[s.shape for s in samples[:3]]}")
            except Exception as e:
                logger.error(f"Failed to collect microphone samples: {e}", exc_info=True)
        
        return noise_samples
    
    def process_noise_to_features(
        self,
        noise_samples: Dict[str, List[np.ndarray]]
    ) -> List[np.ndarray]:
        """
        Process noise samples into feature vectors
        
        Args:
            noise_samples: Dictionary of noise samples by source
            
        Returns:
            List of feature vectors
        """
        feature_vectors = []
        
        for source, samples in noise_samples.items():
            logger.info(f"Processing {len(samples)} samples from {source}...")
            
            for sample in samples:
                try:
                    # Extract features
                    features = self.preprocessor.extract_all_features(sample)
                    
                    # Convert to vector
                    feature_vector = self.feature_converter.to_vector(features)
                    feature_vectors.append(feature_vector)
                    
                except Exception as e:
                    logger.warning(f"Failed to process sample: {e}")
                    continue
        
        logger.info(f"Processed {len(feature_vectors)} feature vectors")
        return feature_vectors
    
    def create_device_embedding(
        self,
        feature_vectors: List[np.ndarray],
        method: str = 'mean'
    ) -> torch.Tensor:
        """
        Create device embedding from multiple feature vectors
        
        Args:
            feature_vectors: List of feature vectors
            method: Aggregation method ('mean', 'median', 'concat')
            
        Returns:
            Device embedding tensor
        """
        if not feature_vectors:
            raise ValueError("No feature vectors provided")
        
        # Generate embeddings for each feature vector
        embeddings = []
        for fv in feature_vectors:
            # Create tensor directly on the embedder's device to avoid CPU->CUDA transfer
            fv_tensor = torch.from_numpy(fv).float().to(self.embedder.device)
            embedding = self.embedder.embed(fv_tensor)
            embeddings.append(embedding)
        
        # Stack embeddings
        embedding_stack = torch.stack(embeddings)
        
        # Aggregate
        if method == 'mean':
            device_embedding = torch.mean(embedding_stack, dim=0)
        elif method == 'median':
            device_embedding = torch.median(embedding_stack, dim=0).values
        elif method == 'concat':
            # Take first N embeddings and concatenate (not recommended for large N)
            device_embedding = embedding_stack[:min(5, len(embeddings))].flatten()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        # Normalize
        device_embedding = torch.nn.functional.normalize(device_embedding, p=2, dim=0)
        
        logger.info(f"Created device embedding with shape {device_embedding.shape}")
        return device_embedding
    
    def save_device_embedding(
        self,
        device_id: str,
        embedding: torch.Tensor,
        metadata: Optional[Dict] = None
    ):
        """
        Save device embedding to storage
        
        Args:
            device_id: Unique device identifier
            embedding: Device embedding tensor
            metadata: Optional metadata dictionary
        """
        # Save embedding
        embedding_path = self.storage_dir / f"{device_id}_embedding.pt"
        torch.save(embedding, embedding_path)
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata['device_id'] = device_id
        metadata['enrollment_date'] = datetime.now().isoformat()
        metadata['embedding_shape'] = list(embedding.shape)
        
        metadata_path = self.storage_dir / f"{device_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved device embedding: {device_id}")
    
    def load_device_embedding(self, device_id: str) -> Optional[torch.Tensor]:
        """
        Load device embedding from storage
        
        Args:
            device_id: Device identifier
            
        Returns:
            Device embedding tensor or None if not found
        """
        embedding_path = self.storage_dir / f"{device_id}_embedding.pt"
        
        if not embedding_path.exists():
            logger.error(f"Device embedding not found: {device_id}")
            return None
        
        embedding = torch.load(embedding_path)
        logger.info(f"Loaded device embedding: {device_id}")
        
        return embedding
    
    def list_enrolled_devices(self) -> List[str]:
        """
        List all enrolled device IDs
        
        Returns:
            List of device IDs
        """
        device_ids = []
        
        for file in self.storage_dir.glob("*_embedding.pt"):
            device_id = file.stem.replace('_embedding', '')
            device_ids.append(device_id)
        
        return device_ids
    
    def enroll_device(
        self,
        device_name: Optional[str] = None,
        num_samples: int = 50,
        sources: List[str] = ['qrng', 'camera', 'microphone'],
        client_samples: Optional[Dict[str, List[List[float]]]] = None
    ) -> str:
        """
        Complete device enrollment process
        
        Args:
            device_name: Optional device name
            num_samples: Number of noise samples to collect
            sources: Noise sources to use
            client_samples: Optional raw noise samples provided by client
            
        Returns:
            Device ID
        """
        logger.info("Starting device enrollment...")
        
        # Generate device ID
        device_id = self.generate_device_id(device_name)
        logger.info(f"Generated device ID: {device_id}")
        
        # Collect noise samples
        if client_samples:
            logger.info("Using client-provided samples")
            noise_samples = {}
            for source, samples_list in client_samples.items():
                # Convert list of lists back to numpy arrays
                noise_samples[source] = [np.array(s) for s in samples_list]
        else:
            noise_samples = self.collect_noise_samples(
                num_samples=num_samples,
                sources=sources
            )
        
        # Validate samples were collected
        if not noise_samples:
            raise RuntimeError("No noise samples collected from any source")
        
        # Check if any source actually has samples
        total_samples = sum(len(samples) for samples in noise_samples.values())
        if total_samples == 0:
            sources_attempted = list(noise_samples.keys())
            raise RuntimeError(
                f"No samples collected from sources: {sources_attempted}. "
                "Check logs above for specific errors with camera/microphone/QRNG."
            )
        
        logger.info(f"Total samples collected: {total_samples} from {len(noise_samples)} sources")
        
        # Save raw data for training if builder is configured
        if self.dataset_builder:
            logger.info("Saving raw samples to dataset...")
            try:
                for source, samples in noise_samples.items():
                    self.dataset_builder.add_batch(
                        device_id=device_id,
                        noise_source=source,
                        samples=samples
                    )
                logger.info("Raw samples saved to dataset")
            except Exception as e:
                logger.error(f"Failed to save raw samples: {e}")

        # Process to feature vectors
        feature_vectors = self.process_noise_to_features(noise_samples)
        
        if not feature_vectors:
            raise RuntimeError(
                f"No feature vectors generated from {total_samples} samples. "
                "Feature extraction failed - check sample data format."
            )
        
        # Create device embedding
        device_embedding = self.create_device_embedding(
            feature_vectors,
            method='mean'
        )
        
        # Save embedding
        metadata = {
            'device_name': device_name,
            'num_samples': sum(len(v) for v in noise_samples.values()),
            'sources': list(noise_samples.keys()),
            'feature_dimension': len(feature_vectors[0])
        }
        
        self.save_device_embedding(
            device_id=device_id,
            embedding=device_embedding,
            metadata=metadata
        )
        
        logger.info(f"Device enrollment complete: {device_id}")
        return device_id


def main():
    """Test enrollment module"""
    print("\n=== Device Enrollment Test ===")
    
    # Create mock components
    from model.siamese_model import SiameseNetwork
    
    input_dim = 50  # Should match feature extractor output
    embedding_dim = 128
    
    # Create model and embedder
    model = SiameseNetwork(input_dim=input_dim, embedding_dim=embedding_dim)
    embedder = DeviceEmbedder(input_dim=input_dim, embedding_dim=embedding_dim)
    embedder.model = model
    
    # Create preprocessor and feature converter
    preprocessor = NoisePreprocessor(normalize=True)
    feature_converter = FeatureVector()
    
    # Create enroller
    enroller = DeviceEnroller(
        embedder=embedder,
        preprocessor=preprocessor,
        feature_converter=feature_converter,
        storage_dir="./auth/test_embeddings"
    )
    
    print("\n=== Testing with Simulated Noise ===")
    # Simulate noise samples (replace with real collection in production)
    simulated_noise = {
        'qrng': [np.random.rand(1024) for _ in range(10)],
        'camera': [np.random.rand(480*640) for _ in range(5)]
    }
    
    # Process to features
    feature_vectors = enroller.process_noise_to_features(simulated_noise)
    print(f"Generated {len(feature_vectors)} feature vectors")
    
    # Create embedding
    device_embedding = enroller.create_device_embedding(feature_vectors)
    print(f"Device embedding shape: {device_embedding.shape}")
    
    # Save embedding
    test_device_id = enroller.generate_device_id("TestDevice")
    enroller.save_device_embedding(test_device_id, device_embedding)
    
    # Load embedding
    loaded_embedding = enroller.load_device_embedding(test_device_id)
    print(f"Loaded embedding matches: {torch.allclose(device_embedding, loaded_embedding)}")
    
    # List enrolled devices
    devices = enroller.list_enrolled_devices()
    print(f"Enrolled devices: {devices}")


if __name__ == "__main__":
    main()
