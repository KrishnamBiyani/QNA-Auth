"""
Feature Extraction and Preprocessing Pipeline
Implements filtering, normalization, entropy, FFT, and statistical features
"""

import numpy as np
from scipy import signal, stats
from scipy.fft import rfft, rfftfreq
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Canonical feature pipeline version and names (same everywhere: train, serve, evaluation)
FEATURE_VERSION = "1.0"

# Ordered list of feature names from extract_all_features (must match exactly)
_CANONICAL_FEATURE_NAMES: Optional[list] = None


def get_canonical_feature_names() -> list:
    """Return the canonical ordered feature list. Same for preprocessing, training, and server."""
    global _CANONICAL_FEATURE_NAMES
    if _CANONICAL_FEATURE_NAMES is None:
        preprocessor = NoisePreprocessor(normalize=True)
        dummy = np.random.RandomState(42).randn(1024)
        _CANONICAL_FEATURE_NAMES = sorted(preprocessor.extract_all_features(dummy).keys())
    return _CANONICAL_FEATURE_NAMES


class NoisePreprocessor:
    """Preprocessing and feature extraction for noise data"""
    
    def __init__(self, normalize: bool = True):
        """
        Initialize preprocessor
        
        Args:
            normalize: Whether to normalize features
        """
        self.normalize = normalize
    
    def apply_bandpass_filter(
        self,
        data: np.ndarray,
        lowcut: float = 0.1,
        highcut: float = 0.9,
        fs: float = 1.0
    ) -> np.ndarray:
        """
        Apply Butterworth bandpass filter
        
        Args:
            data: Input signal
            lowcut: Low cutoff frequency (normalized)
            highcut: High cutoff frequency (normalized)
            fs: Sampling frequency
            
        Returns:
            Filtered signal
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, data)
        
        return filtered
    
    def normalize_data(
        self,
        data: np.ndarray,
        method: str = 'standard'
    ) -> np.ndarray:
        """
        Normalize data
        
        Args:
            data: Input array
            method: Normalization method ('standard', 'minmax', 'robust')
            
        Returns:
            Normalized array
        """
        if method == 'standard':
            # Z-score normalization
            return (data - np.mean(data)) / (np.std(data) + 1e-10)
        
        elif method == 'minmax':
            # Min-max scaling to [0, 1]
            min_val = np.min(data)
            max_val = np.max(data)
            return (data - min_val) / (max_val - min_val + 1e-10)
        
        elif method == 'robust':
            # Robust scaling using median and IQR
            median = np.median(data)
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            return (data - median) / (iqr + 1e-10)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def compute_statistical_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Compute statistical features
        
        Args:
            data: Input array
            
        Returns:
            Dictionary of statistical features
        """
        rms = float(np.sqrt(np.mean(data**2)))
        
        features = {
            # Basic statistics
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'variance': float(np.var(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data)),
            'range': float(np.ptp(data)),
            
            # Distribution moments
            'skewness': float(stats.skew(data)),
            'kurtosis': float(stats.kurtosis(data)),
            
            # Quantiles
            'q25': float(np.percentile(data, 25)),
            'q75': float(np.percentile(data, 75)),
            'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
            
            # RMS
            'rms': rms,
            
            # Magnitude Encoding: Map amplitude to rotation
            # Use sin/cos of RMS to create circular features
            # Weights tuned to be robust to small noise (freq=100) but sensitive to large (weight=500)
            'rms_enc_sin': 500.0 * np.sin(rms * 100),
            'rms_enc_cos': 500.0 * np.cos(rms * 100),
            
            # Peak factor
            'peak_factor': float(np.max(np.abs(data)) / (rms + 1e-10))
        }
        
        return features
    
    def compute_entropy(self, data: np.ndarray, bins: int = 256) -> float:
        """
        Compute Shannon entropy
        
        Args:
            data: Input array
            bins: Number of histogram bins
            
        Returns:
            Entropy value in bits
        """
        hist, _ = np.histogram(data, bins=bins)
        hist = hist[hist > 0]  # Remove zero bins
        
        # Normalize to probabilities
        probs = hist / hist.sum()
        
        # Calculate Shannon entropy
        entropy = -np.sum(probs * np.log2(probs))
        
        return float(entropy)
    
    def compute_fft_features(
        self,
        data: np.ndarray,
        sample_rate: float = 1.0
    ) -> Dict[str, Any]:
        """
        Compute FFT-based frequency domain features
        
        Args:
            data: Input signal
            sample_rate: Sampling rate
            
        Returns:
            Dictionary of FFT features
        """
        # Compute FFT
        fft_vals = rfft(data)
        fft_mag = np.abs(fft_vals)
        fft_freq = rfftfreq(len(data), 1/sample_rate)
        
        # Power spectral density
        psd = fft_mag ** 2
        
        # Normalize PSD to probability distribution
        psd_norm = psd / (psd.sum() + 1e-10)
        
        # Spectral features
        features = {
            # Dominant frequency
            'dominant_freq': float(fft_freq[np.argmax(fft_mag)]),
            'dominant_magnitude': float(np.max(fft_mag)),
            
            # Spectral centroid (center of mass of spectrum)
            'spectral_centroid': float(np.sum(fft_freq * fft_mag) / (np.sum(fft_mag) + 1e-10)),
            
            # Spectral spread (standard deviation of spectrum)
            'spectral_spread': float(np.sqrt(np.sum(((fft_freq - np.sum(fft_freq * fft_mag) / 
                                     (np.sum(fft_mag) + 1e-10))**2) * fft_mag) / 
                                     (np.sum(fft_mag) + 1e-10))),
            
            # Spectral entropy
            'spectral_entropy': float(-np.sum(psd_norm[psd_norm > 0] * 
                                     np.log2(psd_norm[psd_norm > 0]))),
            
            # Spectral flatness (ratio of geometric mean to arithmetic mean)
            'spectral_flatness': float(stats.gmean(fft_mag + 1e-10) / 
                                      (np.mean(fft_mag) + 1e-10)),
            
            # Band power (energy in frequency bands)
            'low_freq_power': float(np.sum(psd[fft_freq < sample_rate * 0.25])),
            'mid_freq_power': float(np.sum(psd[(fft_freq >= sample_rate * 0.25) & 
                                               (fft_freq < sample_rate * 0.5)])),
            'high_freq_power': float(np.sum(psd[fft_freq >= sample_rate * 0.5]))
        }
        
        return features
    
    def compute_autocorrelation_features(
        self,
        data: np.ndarray,
        max_lag: int = 100
    ) -> Dict[str, float]:
        """
        Compute autocorrelation features
        
        Args:
            data: Input signal
            max_lag: Maximum lag for autocorrelation
            
        Returns:
            Dictionary of autocorrelation features
        """
        # Normalize data
        normalized = (data - np.mean(data)) / (np.std(data) + 1e-10)
        
        # Compute autocorrelation
        autocorr = np.correlate(normalized, normalized, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Trim to max_lag
        autocorr = autocorr[:min(max_lag, len(autocorr))]
        
        features = {
            # First zero crossing
            'first_zero_crossing': float(np.where(autocorr < 0)[0][0] 
                                        if np.any(autocorr < 0) else len(autocorr)),
            
            # Mean of first N lags
            'autocorr_mean_10': float(np.mean(autocorr[1:11])) if len(autocorr) > 10 else 0.0,
            'autocorr_mean_50': float(np.mean(autocorr[1:51])) if len(autocorr) > 50 else 0.0,
            
            # Decay rate (how fast autocorrelation decreases)
            'autocorr_decay': float(-np.polyfit(range(min(20, len(autocorr))), 
                                               autocorr[:min(20, len(autocorr))], 1)[0])
        }
        
        return features
    
    def compute_complexity_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Compute signal complexity features
        
        Args:
            data: Input signal
            
        Returns:
            Dictionary of complexity features
        """
        features = {}
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
        features['zero_crossing_rate'] = float(zero_crossings / len(data))
        
        # For large arrays, downsample before computing expensive features
        # Approximate entropy and Hurst exponent have O(n²) complexity
        if len(data) > 200:
            # Downsample to max 200 points for complexity calculations
            step = len(data) // 200
            data_downsampled = data[::step][:200]
            logger.info(f"Downsampled {len(data)} samples to {len(data_downsampled)} for complexity features")
        else:
            data_downsampled = data
        
        # Approximate entropy (regularity measure)
        logger.info("Computing approximate entropy...")
        features['approx_entropy'] = self._approximate_entropy(data_downsampled)
        logger.info("✓ Approximate entropy computed")
        
        # Hurst exponent (long-term memory)
        logger.info("Computing Hurst exponent...")
        features['hurst_exponent'] = self._hurst_exponent(data_downsampled)
        logger.info("✓ Hurst exponent computed")
        
        return features
    
    def _approximate_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        Compute approximate entropy
        
        Args:
            data: Input signal
            m: Pattern length
            r: Tolerance (fraction of std)
            
        Returns:
            Approximate entropy value
        """
        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        
        def _phi(m_val):
            x = [[data[j] for j in range(i, i + m_val - 1 + 1)] for i in range(N - m_val + 1)]
            C = []
            total = len(x)
            for idx, x_i in enumerate(x):
                # Log progress every 20%
                if idx % (total // 5) == 0 and idx > 0:
                    logger.info(f"  Progress: {idx}/{total} ({idx*100//total}%)")
                count = len([1 for x_j in x if _maxdist(x_i, x_j) <= r])
                C.append(count / (N - m_val + 1.0))
            return (N - m_val + 1.0)**(-1) * sum(np.log(C))
        
        N = len(data)
        r = r * np.std(data)
        
        logger.info(f"  Calculating phi for m={m}...")
        phi_m = _phi(m)
        logger.info(f"  Calculating phi for m={m+1}...")
        phi_m1 = _phi(m + 1)
        
        return abs(phi_m1 - phi_m)
    
    def _hurst_exponent(self, data: np.ndarray) -> float:
        """
        Compute Hurst exponent using R/S analysis
        
        Args:
            data: Input signal
            
        Returns:
            Hurst exponent (0.5 = random, >0.5 = trending, <0.5 = mean-reverting)
        """
        lags = range(2, min(100, len(data)//2))
        tau = []
        
        for lag in lags:
            # Divide series into chunks
            chunks = [data[i:i+lag] for i in range(0, len(data), lag) if len(data[i:i+lag]) == lag]
            
            if not chunks:
                continue
            
            # Compute R/S for each chunk
            rs_values = []
            for chunk in chunks:
                mean_chunk = np.mean(chunk)
                deviations = chunk - mean_chunk
                cumsum = np.cumsum(deviations)
                
                R = np.max(cumsum) - np.min(cumsum)
                S = np.std(chunk)
                
                if S > 0:
                    rs_values.append(R / S)
            
            if rs_values:
                tau.append(np.mean(rs_values))
        
        # Fit log-log plot
        if len(tau) > 2:
            poly = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
            return float(poly[0])
        else:
            return 0.5
    
    def extract_all_features(
        self,
        data: np.ndarray,
        sample_rate: float = 1.0
    ) -> Dict[str, Any]:
        """
        Extract all available features from noise data
        
        Args:
            data: Input noise array
            sample_rate: Sampling rate (for FFT features)
            
        Returns:
            Dictionary containing all features
        """
        logger.info("Extracting all features...")
        
        # Ensure data is 1D
        if data.ndim > 1:
            data = data.flatten()
        
        # Normalize if requested
        if self.normalize:
            data = self.normalize_data(data, method='standard')
        
        all_features = {}
        
        # Statistical features
        all_features.update(self.compute_statistical_features(data))
        
        # Entropy
        all_features['shannon_entropy'] = self.compute_entropy(data)
        
        # FFT features
        all_features.update(self.compute_fft_features(data, sample_rate))
        
        # Autocorrelation features
        all_features.update(self.compute_autocorrelation_features(data))
        
        # Complexity features
        all_features.update(self.compute_complexity_features(data))
        
        logger.info(f"Extracted {len(all_features)} features")
        
        return all_features


class FeatureVector:
    """Converts features dictionary to fixed-size feature vector. Uses canonical feature list by default for train/serve consistency."""

    def __init__(self, feature_names: Optional[list] = None):
        """
        Initialize feature vector converter.

        Args:
            feature_names: Ordered list of feature names. If None, uses get_canonical_feature_names() so preprocessing, training, and server use the same order.
        """
        self.feature_names = feature_names if feature_names is not None else get_canonical_feature_names()

    def to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Convert features dictionary to numpy vector.

        Args:
            features: Dictionary of features.

        Returns:
            Feature vector as numpy array (same order as self.feature_names).
        """
        vector = np.array([features.get(name, 0.0) for name in self.feature_names],
                         dtype=np.float32)
        return vector
    
    def from_vector(self, vector: np.ndarray) -> Dict[str, float]:
        """
        Convert feature vector back to dictionary
        
        Args:
            vector: Feature vector
            
        Returns:
            Dictionary of features
        """
        if self.feature_names is None:
            raise ValueError("Feature names not set")
        
        return {name: float(val) for name, val in zip(self.feature_names, vector)}


def main():
    """Test preprocessing and feature extraction"""
    preprocessor = NoisePreprocessor(normalize=True)
    
    print("\n=== Preprocessing & Feature Extraction Test ===")
    
    # Generate test signal
    t = np.linspace(0, 1, 1000)
    signal_test = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(1000)
    
    print(f"\nTest signal: {len(signal_test)} samples")
    
    # Apply bandpass filter
    print("\n=== Bandpass Filter ===")
    filtered = preprocessor.apply_bandpass_filter(signal_test)
    print(f"Filtered signal std: {filtered.std():.4f}")
    
    # Normalize
    print("\n=== Normalization ===")
    normalized = preprocessor.normalize_data(signal_test, method='standard')
    print(f"Normalized mean: {normalized.mean():.6f}")
    print(f"Normalized std: {normalized.std():.6f}")
    
    # Extract all features
    print("\n=== Feature Extraction ===")
    features = preprocessor.extract_all_features(signal_test, sample_rate=1000)
    
    print(f"\nTotal features: {len(features)}")
    print("\nSample features:")
    for i, (name, value) in enumerate(list(features.items())[:10]):
        print(f"  {name}: {value:.6f}")
    
    # Convert to feature vector
    print("\n=== Feature Vector Conversion ===")
    converter = FeatureVector()
    vector = converter.to_vector(features)
    print(f"Feature vector shape: {vector.shape}")
    print(f"Feature vector range: [{vector.min():.4f}, {vector.max():.4f}]")
    
    # Convert back
    reconstructed = converter.from_vector(vector)
    print(f"Reconstructed features: {len(reconstructed)}")


if __name__ == "__main__":
    main()
