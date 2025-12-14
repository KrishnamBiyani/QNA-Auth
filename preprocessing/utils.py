"""
Utility functions for preprocessing
"""

import numpy as np
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sliding_window(
    data: np.ndarray,
    window_size: int,
    stride: int = None
) -> List[np.ndarray]:
    """
    Create sliding windows over data
    
    Args:
        data: Input array
        window_size: Size of each window
        stride: Step size between windows (defaults to window_size)
        
    Returns:
        List of windowed arrays
    """
    if stride is None:
        stride = window_size
    
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i:i+window_size])
    
    return windows


def augment_noise_sample(
    data: np.ndarray,
    num_augmentations: int = 5,
    noise_level: float = 0.1
) -> List[np.ndarray]:
    """
    Augment noise sample with variations
    
    Args:
        data: Input noise array
        num_augmentations: Number of augmented samples to create
        noise_level: Amount of augmentation noise to add
        
    Returns:
        List of augmented samples
    """
    augmented = [data]  # Include original
    
    for _ in range(num_augmentations):
        # Add random noise
        noisy = data + noise_level * np.random.randn(*data.shape)
        augmented.append(noisy)
    
    return augmented


def downsample_signal(
    data: np.ndarray,
    factor: int = 2
) -> np.ndarray:
    """
    Downsample signal by factor
    
    Args:
        data: Input signal
        factor: Downsampling factor
        
    Returns:
        Downsampled signal
    """
    return data[::factor]


def pad_or_truncate(
    data: np.ndarray,
    target_length: int,
    pad_value: float = 0.0
) -> np.ndarray:
    """
    Pad or truncate array to target length
    
    Args:
        data: Input array
        target_length: Desired length
        pad_value: Value to use for padding
        
    Returns:
        Array with target length
    """
    if len(data) >= target_length:
        return data[:target_length]
    else:
        padding = np.full(target_length - len(data), pad_value)
        return np.concatenate([data, padding])


def batch_process(
    samples: List[np.ndarray],
    processor_func,
    **kwargs
) -> List:
    """
    Process multiple samples in batch
    
    Args:
        samples: List of input arrays
        processor_func: Function to apply to each sample
        **kwargs: Arguments to pass to processor_func
        
    Returns:
        List of processed results
    """
    results = []
    
    for i, sample in enumerate(samples):
        try:
            result = processor_func(sample, **kwargs)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process sample {i}: {e}")
            results.append(None)
    
    return results


def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio
    
    Args:
        signal: Signal array
        noise: Noise array
        
    Returns:
        SNR in dB
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    return float(snr)


def normalize_batch(
    samples: List[np.ndarray],
    method: str = 'standard'
) -> List[np.ndarray]:
    """
    Normalize a batch of samples
    
    Args:
        samples: List of arrays
        method: Normalization method
        
    Returns:
        List of normalized arrays
    """
    normalized = []
    
    for sample in samples:
        if method == 'standard':
            norm = (sample - np.mean(sample)) / (np.std(sample) + 1e-10)
        elif method == 'minmax':
            norm = (sample - np.min(sample)) / (np.ptp(sample) + 1e-10)
        else:
            norm = sample
        
        normalized.append(norm)
    
    return normalized


def merge_noise_sources(
    sources: List[np.ndarray],
    weights: List[float] = None
) -> np.ndarray:
    """
    Merge multiple noise sources with optional weighting
    
    Args:
        sources: List of noise arrays
        weights: Optional weights for each source
        
    Returns:
        Merged noise array
    """
    if weights is None:
        weights = [1.0] * len(sources)
    
    # Pad all sources to same length
    max_len = max(len(s) for s in sources)
    padded = [pad_or_truncate(s, max_len) for s in sources]
    
    # Weighted sum
    merged = np.zeros(max_len)
    for source, weight in zip(padded, weights):
        merged += weight * source
    
    # Normalize
    merged = merged / (sum(weights) + 1e-10)
    
    return merged


def main():
    """Test utility functions"""
    print("\n=== Preprocessing Utilities Test ===")
    
    # Test data
    data = np.random.randn(1000)
    
    # Sliding windows
    print("\n=== Sliding Windows ===")
    windows = sliding_window(data, window_size=100, stride=50)
    print(f"Created {len(windows)} windows")
    
    # Augmentation
    print("\n=== Augmentation ===")
    augmented = augment_noise_sample(data[:100], num_augmentations=3)
    print(f"Created {len(augmented)} augmented samples")
    
    # Pad/truncate
    print("\n=== Pad/Truncate ===")
    padded = pad_or_truncate(data[:50], target_length=100)
    print(f"Padded from 50 to {len(padded)}")
    
    truncated = pad_or_truncate(data, target_length=500)
    print(f"Truncated from 1000 to {len(truncated)}")
    
    # Merge sources
    print("\n=== Merge Sources ===")
    source1 = np.random.randn(100)
    source2 = np.random.randn(150)
    merged = merge_noise_sources([source1, source2], weights=[0.6, 0.4])
    print(f"Merged sources: length {len(merged)}")


if __name__ == "__main__":
    main()
