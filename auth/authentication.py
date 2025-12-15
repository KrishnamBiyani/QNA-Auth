"""
Device Authentication Module
Authenticates devices using noise samples and embeddings
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple
import logging
from datetime import datetime

from model.siamese_model import DeviceEmbedder
from preprocessing.features import NoisePreprocessor, FeatureVector
from noise_collection import QRNGClient, CameraNoiseCollector, MicrophoneNoiseCollector
from .enrollment import DeviceEnroller

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeviceAuthenticator:
    """Handles device authentication"""
    
    def __init__(
        self,
        embedder: DeviceEmbedder,
        preprocessor: NoisePreprocessor,
        feature_converter: FeatureVector,
        enroller: DeviceEnroller,
        threshold: float = 0.85,
        metric: str = 'cosine'
    ):
        """
        Initialize device authenticator
        
        Args:
            embedder: Trained DeviceEmbedder
            preprocessor: Noise preprocessor
            feature_converter: Feature vector converter
            enroller: DeviceEnroller for loading stored embeddings
            threshold: Authentication threshold
            metric: Similarity metric ('cosine' or 'euclidean')
        """
        self.embedder = embedder
        self.preprocessor = preprocessor
        self.feature_converter = feature_converter
        self.enroller = enroller
        self.threshold = threshold
        self.metric = metric
        
        logger.info(f"DeviceAuthenticator initialized (threshold={threshold}, metric={metric})")
    
    def collect_authentication_sample(
        self,
        source: str = 'qrng',
        num_samples: int = 5
    ) -> Optional[np.ndarray]:
        """
        Collect fresh noise sample for authentication
        
        Args:
            source: Noise source to use
            num_samples: Number of samples to collect
            
        Returns:
            Noise array or None if collection failed
        """
        try:
            if source == 'qrng':
                qrng_client = QRNGClient()
                samples = qrng_client.fetch_multiple_samples(
                    num_samples=num_samples,
                    sample_size=1024
                )
                if samples:
                    # Concatenate or average samples
                    return np.mean(np.array(samples), axis=0)
            
            elif source == 'camera':
                camera_collector = CameraNoiseCollector(camera_index=0)
                if camera_collector.initialize_camera():
                    frame = camera_collector.capture_dark_frame(exposure_time=0.2)
                    camera_collector.release()
                    if frame is not None:
                        return camera_collector.extract_noise_features(frame)
            
            elif source == 'microphone':
                mic_collector = MicrophoneNoiseCollector(sample_rate=44100)
                audio = mic_collector.capture_ambient_noise(duration=1.0)
                return audio
            
        except Exception as e:
            logger.error(f"Failed to collect authentication sample from {source}: {e}")
        
        return None
    
    def generate_authentication_embedding(
        self,
        noise_samples: list
    ) -> Optional[torch.Tensor]:
        """
        Generate embedding from fresh noise samples
        
        Args:
            noise_samples: List of noise arrays
            
        Returns:
            Authentication embedding or None
        """
        try:
            feature_vectors = []
            
            for sample in noise_samples:
                # Extract features
                features = self.preprocessor.extract_all_features(sample)
                
                # Convert to vector
                feature_vector = self.feature_converter.to_vector(features)
                feature_vectors.append(feature_vector)
            
            # Generate embeddings
            embeddings = []
            for fv in feature_vectors:
                fv_tensor = torch.from_numpy(fv).float()
                embedding = self.embedder.embed(fv_tensor)
                embeddings.append(embedding)
            
            # Average embeddings
            auth_embedding = torch.mean(torch.stack(embeddings), dim=0)
            
            # Normalize
            auth_embedding = torch.nn.functional.normalize(auth_embedding, p=2, dim=0)
            
            return auth_embedding
            
        except Exception as e:
            logger.error(f"Failed to generate authentication embedding: {e}")
            return None
    
    def verify_device(
        self,
        device_id: str,
        auth_embedding: torch.Tensor
    ) -> Tuple[bool, float, Dict]:
        """
        Verify device by comparing embeddings
        
        Args:
            device_id: Device identifier to verify
            auth_embedding: Fresh authentication embedding
            
        Returns:
            Tuple of (is_authenticated, similarity_score, details)
        """
        # Load stored device embedding
        stored_embedding = self.enroller.load_device_embedding(device_id)
        
        if stored_embedding is None:
            return False, 0.0, {'error': 'Device not enrolled'}
        
        # Compute similarity
        similarity = self.embedder.compute_similarity(
            auth_embedding,
            stored_embedding,
            metric=self.metric
        )
        
        # Make decision
        is_authenticated = similarity >= self.threshold
        
        details = {
            'device_id': device_id,
            'similarity': similarity,
            'threshold': self.threshold,
            'metric': self.metric,
            'timestamp': datetime.now().isoformat(),
            'authenticated': is_authenticated
        }
        
        logger.info(f"Verification: device={device_id}, similarity={similarity:.4f}, "
                   f"authenticated={is_authenticated}")
        
        return is_authenticated, similarity, details
    
    def authenticate(
        self,
        device_id: str,
        sources: list = ['qrng'],
        num_samples_per_source: int = 5
    ) -> Tuple[bool, Dict]:
        """
        Complete authentication flow
        
        Args:
            device_id: Device identifier to authenticate
            sources: List of noise sources to use
            num_samples_per_source: Number of samples per source
            
        Returns:
            Tuple of (is_authenticated, details_dict)
        """
        logger.info(f"Starting authentication for device: {device_id}")
        
        # Collect noise samples
        noise_samples = []
        
        for source in sources:
            logger.info(f"Collecting samples from {source}...")
            for _ in range(num_samples_per_source):
                sample = self.collect_authentication_sample(source, num_samples=1)
                if sample is not None:
                    noise_samples.append(sample)
        
        if not noise_samples:
            return False, {'error': 'Failed to collect noise samples'}
        
        logger.info(f"Collected {len(noise_samples)} noise samples")
        
        # Generate authentication embedding
        auth_embedding = self.generate_authentication_embedding(noise_samples)
        
        if auth_embedding is None:
            return False, {'error': 'Failed to generate authentication embedding'}
        
        # Verify device
        is_authenticated, similarity, details = self.verify_device(
            device_id,
            auth_embedding
        )
        
        details['num_samples_collected'] = len(noise_samples)
        details['sources_used'] = sources
        
        return is_authenticated, details
    
    def identify_device(
        self,
        auth_embedding: torch.Tensor,
        top_k: int = 5
    ) -> list:
        """
        Identify device by finding closest matches (1:N matching)
        
        Args:
            auth_embedding: Authentication embedding
            top_k: Number of top matches to return
            
        Returns:
            List of (device_id, similarity_score) tuples
        """
        enrolled_devices = self.enroller.list_enrolled_devices()
        
        if not enrolled_devices:
            logger.warning("No enrolled devices found")
            return []
        
        # Compute similarities with all enrolled devices
        similarities = []
        
        for device_id in enrolled_devices:
            stored_embedding = self.enroller.load_device_embedding(device_id)
            if stored_embedding is not None:
                similarity = self.embedder.compute_similarity(
                    auth_embedding,
                    stored_embedding,
                    metric=self.metric
                )
                similarities.append((device_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        top_matches = similarities[:top_k]
        
        logger.info(f"Identification: top match = {top_matches[0][0]} "
                   f"(similarity={top_matches[0][1]:.4f})")
        
        return top_matches


class AuthenticationSession:
    """Manages an authentication session with retry logic"""
    
    def __init__(
        self,
        authenticator: DeviceAuthenticator,
        max_attempts: int = 3
    ):
        """
        Initialize authentication session
        
        Args:
            authenticator: DeviceAuthenticator instance
            max_attempts: Maximum authentication attempts
        """
        self.authenticator = authenticator
        self.max_attempts = max_attempts
        self.attempts = 0
        self.session_log = []
        
    def attempt_authentication(
        self,
        device_id: str,
        sources: list = ['qrng']
    ) -> Tuple[bool, Dict]:
        """
        Attempt authentication with retry logic
        
        Args:
            device_id: Device identifier
            sources: Noise sources to use
            
        Returns:
            Tuple of (is_authenticated, session_details)
        """
        self.attempts += 1
        
        if self.attempts > self.max_attempts:
            return False, {
                'error': 'Maximum authentication attempts exceeded',
                'attempts': self.attempts,
                'session_log': self.session_log
            }
        
        # Authenticate
        is_authenticated, details = self.authenticator.authenticate(
            device_id,
            sources=sources
        )
        
        # Log attempt
        attempt_log = {
            'attempt': self.attempts,
            'timestamp': datetime.now().isoformat(),
            'authenticated': is_authenticated,
            'details': details
        }
        self.session_log.append(attempt_log)
        
        if is_authenticated:
            details['session_log'] = self.session_log
            return True, details
        
        # Retry logic
        if self.attempts < self.max_attempts:
            logger.info(f"Authentication failed. Attempt {self.attempts}/{self.max_attempts}")
            return False, {'retry_available': True, 'attempts': self.attempts}
        else:
            logger.warning("Maximum authentication attempts reached")
            return False, {
                'error': 'Authentication failed after maximum attempts',
                'session_log': self.session_log
            }


def main():
    """Test authentication module"""
    print("\n=== Device Authentication Test ===")
    
    # Create mock components
    from model.siamese_model import SiameseNetwork
    
    input_dim = 50
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
    
    # Create authenticator
    authenticator = DeviceAuthenticator(
        embedder=embedder,
        preprocessor=preprocessor,
        feature_converter=feature_converter,
        enroller=enroller,
        threshold=0.8
    )
    
    print("\n=== Simulating Device Enrollment ===")
    # Simulate enrolling a device
    simulated_noise_enroll = {
        'qrng': [np.random.rand(1024) for _ in range(10)]
    }
    feature_vectors = enroller.process_noise_to_features(simulated_noise_enroll)
    device_embedding = enroller.create_device_embedding(feature_vectors)
    test_device_id = enroller.generate_device_id("TestDevice")
    enroller.save_device_embedding(test_device_id, device_embedding)
    print(f"Enrolled device: {test_device_id}")
    
    print("\n=== Simulating Authentication ===")
    # Simulate authentication with similar noise (should succeed)
    simulated_noise_auth = [np.random.rand(1024) for _ in range(5)]
    auth_embedding = authenticator.generate_authentication_embedding(simulated_noise_auth)
    
    if auth_embedding is not None:
        is_auth, similarity, details = authenticator.verify_device(test_device_id, auth_embedding)
        print(f"Authentication result: {is_auth}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Details: {details}")
    
    print("\n=== Testing Authentication Session ===")
    session = AuthenticationSession(authenticator, max_attempts=3)
    # This will likely fail with random noise, but demonstrates the API
    is_auth, session_details = session.attempt_authentication(test_device_id, sources=['qrng'])
    print(f"Session authenticated: {is_auth}")
    print(f"Session attempts: {session.attempts}")


if __name__ == "__main__":
    main()
