"""
Quantum Random Number Generator API Client
Fetches quantum noise from ANU QRNG service
"""

import requests
import numpy as np
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QRNGClient:
    """Client for fetching quantum random numbers from QRNG API"""
    
    # QRNG API endpoints
    ANU_API_URL = "https://qrng.anu.edu.au/API/jsonI.php"
    QRNG_ORG_URL = "https://api.qrng.org/api/v1.0/random"
    
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize QRNG client
        
        Args:
            api_url: Custom API URL (defaults to QRNG.org if key provided, else ANU)
            api_key: API key for authenticated QRNG service
        """
        self.api_key = api_key
        
        # Use QRNG.org if API key provided, otherwise ANU
        if api_key and not api_url:
            self.api_url = self.QRNG_ORG_URL
        else:
            self.api_url = api_url or self.ANU_API_URL
        
        # Setup headers for authenticated requests
        self.headers = {}
        if self.api_key:
            self.headers['Authorization'] = f'Bearer {self.api_key}'
            logger.info("Initialized QRNG client with API key authentication")
        
    def fetch_quantum_noise(
        self, 
        length: int = 1024, 
        data_type: str = 'uint8'
    ) -> np.ndarray:
        """
        Fetch quantum random numbers from the API
        
        Args:
            length: Number of random values to fetch (max 1024 for ANU)
            data_type: Type of random numbers ('uint8', 'uint16', 'hex16')
            
        Returns:
            numpy array of quantum random numbers
        """
        try:
            # Different API formats for different services
            if self.api_key:
                # QRNG.org format
                params = {
                    'length': min(length, 1024),
                    'type': 'uint8'
                }
                logger.info(f"Fetching {length} quantum random numbers (authenticated)...")
                response = requests.get(self.api_url, params=params, headers=self.headers, timeout=30)
            else:
                # ANU QRNG format
                params = {
                    'length': min(length, 1024),
                    'type': data_type
                }
                logger.info(f"Fetching {length} quantum random numbers...")
                response = requests.get(self.api_url, params=params, timeout=10)
            
            response.raise_for_status()
            data = response.json()
            
            # Handle different response formats
            if self.api_key:
                # QRNG.org response format
                if 'data' in data:
                    quantum_data = np.array(data['data'], dtype=np.uint8)
                    logger.info(f"Successfully fetched {len(quantum_data)} quantum values")
                    return quantum_data
                else:
                    raise ValueError(f"Unexpected API response format: {data}")
            else:
                # ANU response format
                if data.get('success'):
                    quantum_data = np.array(data['data'], dtype=np.uint8)
                    logger.info(f"Successfully fetched {len(quantum_data)} quantum values")
                    return quantum_data
                else:
                    raise ValueError(f"API returned unsuccessful response: {data}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch quantum noise: {e}")
            raise
    
    def fetch_multiple_samples(
        self, 
        num_samples: int = 10, 
        sample_size: int = 1024
    ) -> List[np.ndarray]:
        """
        Fetch multiple quantum noise samples (NO FALLBACK - ensures authenticity)
        
        Args:
            num_samples: Number of separate samples to collect
            sample_size: Size of each sample
            
        Returns:
            List of numpy arrays containing quantum noise
        """
        samples = []
        
        for i in range(num_samples):
            try:
                sample = self.fetch_quantum_noise(length=sample_size)
                samples.append(sample)
                logger.info(f"Collected sample {i+1}/{num_samples}")
            except Exception as e:
                logger.error(f"Failed to collect sample {i+1}: {e}")
                # NO FALLBACK - skip failed samples to ensure authenticity
                continue
                
        return samples
    
    def get_quantum_entropy(self, data: np.ndarray) -> float:
        """
        Calculate Shannon entropy of quantum noise
        
        Args:
            data: Quantum noise array
            
        Returns:
            Entropy value
        """
        # Calculate probability distribution
        values, counts = np.unique(data, return_counts=True)
        probabilities = counts / len(data)
        
        # Calculate Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return entropy


def main():
    """Test the QRNG client"""
    client = QRNGClient()
    
    # Fetch a single sample
    print("\n=== Fetching Quantum Noise Sample ===")
    noise = client.fetch_quantum_noise(length=1024)
    print(f"Sample shape: {noise.shape}")
    print(f"Sample mean: {noise.mean():.2f}")
    print(f"Sample std: {noise.std():.2f}")
    print(f"Sample range: [{noise.min()}, {noise.max()}]")
    
    # Calculate entropy
    entropy = client.get_quantum_entropy(noise)
    print(f"Shannon entropy: {entropy:.4f} bits")
    
    # Fetch multiple samples
    print("\n=== Fetching Multiple Samples ===")
    samples = client.fetch_multiple_samples(num_samples=5, sample_size=512)
    print(f"Collected {len(samples)} samples")
    
    # Compare entropy across samples
    print("\n=== Entropy Comparison ===")
    for i, sample in enumerate(samples):
        ent = client.get_quantum_entropy(sample)
        print(f"Sample {i+1} entropy: {ent:.4f} bits")


if __name__ == "__main__":
    main()
