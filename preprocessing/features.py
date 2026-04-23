"""
Feature extraction and preprocessing utilities shared by training and runtime.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from scipy import signal, stats
from scipy.fft import rfft, rfftfreq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Canonical feature pipeline version and names (same everywhere: train, serve, evaluation)
FEATURE_VERSION = "2.0"

_CANONICAL_FEATURE_NAMES: Optional[list] = None


def get_canonical_feature_names() -> list:
    """Return the canonical ordered feature list."""
    global _CANONICAL_FEATURE_NAMES
    if _CANONICAL_FEATURE_NAMES is None:
        preprocessor = NoisePreprocessor(normalize=True)
        dummy = np.random.RandomState(42).randn(1024)
        _CANONICAL_FEATURE_NAMES = sorted(preprocessor.extract_all_features(dummy).keys())
    return _CANONICAL_FEATURE_NAMES


class NoisePreprocessor:
    """Preprocessing and feature extraction for noise data."""

    def __init__(self, normalize: bool = True, fast_mode: bool = False, max_analysis_points: int = 16_384):
        self.normalize = normalize
        self.fast_mode = fast_mode
        self.max_analysis_points = max(1, int(max_analysis_points))

    @staticmethod
    def _ensure_1d_float_array(data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        if arr.size == 0:
            raise ValueError("Noise sample is empty")
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _safe_value(value: Any) -> float:
        scalar = float(np.asarray(value, dtype=np.float64))
        return scalar if np.isfinite(scalar) else 0.0

    @staticmethod
    def _prefix_features(prefix: str, features: Dict[str, float]) -> Dict[str, float]:
        return {f"{prefix}{name}": float(value) for name, value in features.items()}

    def _analysis_view(self, data: np.ndarray) -> np.ndarray:
        """Return a bounded-length view used by expensive transforms."""
        if data.size <= self.max_analysis_points:
            return data
        step = int(np.ceil(data.size / self.max_analysis_points))
        return np.ascontiguousarray(data[::step][: self.max_analysis_points], dtype=np.float32)

    def apply_bandpass_filter(
        self,
        data: np.ndarray,
        lowcut: float = 0.1,
        highcut: float = 0.9,
        fs: float = 1.0,
    ) -> np.ndarray:
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(4, [low, high], btype="band")
        filtered = signal.filtfilt(b, a, data)
        return np.asarray(filtered, dtype=np.float32)

    def normalize_data(self, data: np.ndarray, method: str = "standard") -> np.ndarray:
        if method == "standard":
            return (data - np.mean(data)) / (np.std(data) + 1e-10)
        if method == "minmax":
            min_val = np.min(data)
            max_val = np.max(data)
            return (data - min_val) / (max_val - min_val + 1e-10)
        if method == "robust":
            median = np.median(data)
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            return (data - median) / (iqr + 1e-10)
        raise ValueError(f"Unknown normalization method: {method}")

    def compute_statistical_features(self, data: np.ndarray) -> Dict[str, float]:
        rms = float(np.sqrt(np.mean(data ** 2)))
        features = {
            "mean": np.mean(data),
            "std": np.std(data),
            "variance": np.var(data),
            "min": np.min(data),
            "max": np.max(data),
            "median": np.median(data),
            "range": np.ptp(data),
            "skewness": stats.skew(data),
            "kurtosis": stats.kurtosis(data),
            "q25": np.percentile(data, 25),
            "q75": np.percentile(data, 75),
            "iqr": np.percentile(data, 75) - np.percentile(data, 25),
            "rms": rms,
            "peak_factor": np.max(np.abs(data)) / (rms + 1e-10),
        }
        return {name: self._safe_value(value) for name, value in features.items()}

    def compute_entropy(self, data: np.ndarray, bins: int = 256) -> float:
        hist, _ = np.histogram(data, bins=bins)
        hist = hist[hist > 0]
        probs = hist / max(hist.sum(), 1)
        entropy = -np.sum(probs * np.log2(probs))
        return self._safe_value(entropy)

    def compute_fft_features(self, data: np.ndarray, sample_rate: float = 1.0) -> Dict[str, float]:
        fft_vals = rfft(data)
        fft_mag = np.abs(fft_vals)
        fft_freq = rfftfreq(len(data), 1 / sample_rate)
        psd = fft_mag ** 2
        psd_norm = psd / (psd.sum() + 1e-10)

        spectral_centroid = np.sum(fft_freq * fft_mag) / (np.sum(fft_mag) + 1e-10)
        spectral_spread = np.sqrt(
            np.sum(((fft_freq - spectral_centroid) ** 2) * fft_mag) / (np.sum(fft_mag) + 1e-10)
        )
        features = {
            "dominant_freq": fft_freq[np.argmax(fft_mag)],
            "dominant_magnitude": np.max(fft_mag),
            "spectral_centroid": spectral_centroid,
            "spectral_spread": spectral_spread,
            "spectral_entropy": -np.sum(psd_norm[psd_norm > 0] * np.log2(psd_norm[psd_norm > 0])),
            "spectral_flatness": stats.gmean(fft_mag + 1e-10) / (np.mean(fft_mag) + 1e-10),
            "low_freq_power": np.sum(psd[fft_freq < sample_rate * 0.25]),
            "mid_freq_power": np.sum(psd[(fft_freq >= sample_rate * 0.25) & (fft_freq < sample_rate * 0.5)]),
            "high_freq_power": np.sum(psd[fft_freq >= sample_rate * 0.5]),
        }
        return {name: self._safe_value(value) for name, value in features.items()}

    def compute_autocorrelation_features(self, data: np.ndarray, max_lag: int = 100) -> Dict[str, float]:
        normalized = (data - np.mean(data)) / (np.std(data) + 1e-10)
        autocorr = np.correlate(normalized, normalized, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        autocorr = autocorr / max(autocorr[0], 1e-10)
        autocorr = autocorr[: min(max_lag, len(autocorr))]
        if len(autocorr) < 3:
            return {
                "first_zero_crossing": 0.0,
                "autocorr_mean_10": 0.0,
                "autocorr_mean_50": 0.0,
                "autocorr_decay": 0.0,
            }
        features = {
            "first_zero_crossing": np.where(autocorr < 0)[0][0] if np.any(autocorr < 0) else len(autocorr),
            "autocorr_mean_10": np.mean(autocorr[1:11]) if len(autocorr) > 10 else 0.0,
            "autocorr_mean_50": np.mean(autocorr[1:51]) if len(autocorr) > 50 else 0.0,
            "autocorr_decay": -np.polyfit(range(min(20, len(autocorr))), autocorr[: min(20, len(autocorr))], 1)[0],
        }
        return {name: self._safe_value(value) for name, value in features.items()}

    def compute_complexity_features(self, data: np.ndarray) -> Dict[str, float]:
        features = {
            "zero_crossing_rate": self._safe_value(np.sum(np.diff(np.sign(data)) != 0) / max(len(data), 1)),
        }
        if self.fast_mode:
            return features

        if len(data) > 200:
            step = max(1, len(data) // 200)
            data_downsampled = data[::step][:200]
        else:
            data_downsampled = data

        features["approx_entropy"] = self._approximate_entropy(data_downsampled)
        features["hurst_exponent"] = self._hurst_exponent(data_downsampled)
        return {name: self._safe_value(value) for name, value in features.items()}

    def _approximate_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        def _maxdist(x_i, x_j):
            return max(abs(ua - va) for ua, va in zip(x_i, x_j))

        def _phi(m_val: int) -> float:
            x = [[data[j] for j in range(i, i + m_val)] for i in range(N - m_val + 1)]
            c_vals = []
            for x_i in x:
                count = len([1 for x_j in x if _maxdist(x_i, x_j) <= tolerance])
                c_vals.append(count / (N - m_val + 1.0))
            return (N - m_val + 1.0) ** (-1) * sum(np.log(c_vals))

        N = len(data)
        tolerance = r * np.std(data)
        if N <= m + 1 or tolerance <= 0:
            return 0.0
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        return self._safe_value(abs(phi_m1 - phi_m))

    def _hurst_exponent(self, data: np.ndarray) -> float:
        lags = range(2, min(100, len(data) // 2))
        tau = []
        for lag in lags:
            chunks = [data[i : i + lag] for i in range(0, len(data), lag) if len(data[i : i + lag]) == lag]
            if not chunks:
                continue
            rs_values = []
            for chunk in chunks:
                mean_chunk = np.mean(chunk)
                deviations = chunk - mean_chunk
                cumsum = np.cumsum(deviations)
                r_val = np.max(cumsum) - np.min(cumsum)
                s_val = np.std(chunk)
                if s_val > 0:
                    rs_values.append(r_val / s_val)
            if rs_values:
                tau.append(np.mean(rs_values))
        if len(tau) <= 2:
            return 0.5
        poly = np.polyfit(np.log(list(lags)[: len(tau)]), np.log(tau), 1)
        return self._safe_value(poly[0])

    def extract_all_features(self, data: np.ndarray, sample_rate: float = 1.0) -> Dict[str, Any]:
        raw = self._ensure_1d_float_array(data)
        centered = raw - np.mean(raw)
        normalized = self.normalize_data(centered, method="standard") if self.normalize else centered
        analysis = self._analysis_view(normalized)

        features: Dict[str, float] = {}

        # Raw-domain features preserve amplitude and range information.
        raw_stats = self.compute_statistical_features(centered)
        for key in ("mean", "std", "variance", "min", "max", "median", "range", "q25", "q75", "iqr", "rms", "peak_factor"):
            features[f"raw_{key}"] = raw_stats[key]
        features["raw_shannon_entropy"] = self.compute_entropy(centered)

        # Normalized-domain features preserve shape and spectral structure.
        norm_stats = self.compute_statistical_features(normalized)
        for key in ("skewness", "kurtosis", "peak_factor"):
            features[f"norm_{key}"] = norm_stats[key]
        features["norm_shannon_entropy"] = self.compute_entropy(analysis)
        features.update(self._prefix_features("norm_", self.compute_fft_features(analysis, sample_rate)))
        features.update(self._prefix_features("norm_", self.compute_autocorrelation_features(analysis)))
        features.update(self._prefix_features("norm_", self.compute_complexity_features(analysis)))
        return {name: self._safe_value(value) for name, value in features.items()}


class FeatureVector:
    """Converts features dictionary to fixed-size feature vectors with optional standardization."""

    def __init__(
        self,
        feature_names: Optional[list] = None,
        feature_mean: Optional[np.ndarray] = None,
        feature_scale: Optional[np.ndarray] = None,
    ):
        self.feature_names = feature_names if feature_names is not None else get_canonical_feature_names()
        self.feature_mean = None if feature_mean is None else np.asarray(feature_mean, dtype=np.float32)
        self.feature_scale = None if feature_scale is None else np.asarray(feature_scale, dtype=np.float32)
        if self.feature_mean is not None and len(self.feature_mean) != len(self.feature_names):
            raise ValueError("feature_mean length must match feature_names")
        if self.feature_scale is not None and len(self.feature_scale) != len(self.feature_names):
            raise ValueError("feature_scale length must match feature_names")

    def to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        vector = np.array([features.get(name, 0.0) for name in self.feature_names], dtype=np.float32)
        if self.feature_mean is not None and self.feature_scale is not None:
            vector = (vector - self.feature_mean) / self.feature_scale
        return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

    def from_vector(self, vector: np.ndarray) -> Dict[str, float]:
        if self.feature_names is None:
            raise ValueError("Feature names not set")
        return {name: float(val) for name, val in zip(self.feature_names, vector)}

    def with_standardization(self, feature_mean: np.ndarray, feature_scale: np.ndarray) -> "FeatureVector":
        return FeatureVector(
            feature_names=list(self.feature_names),
            feature_mean=np.asarray(feature_mean, dtype=np.float32),
            feature_scale=np.asarray(feature_scale, dtype=np.float32),
        )

    def to_metadata(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"feature_names": list(self.feature_names)}
        if self.feature_mean is not None:
            metadata["feature_mean"] = self.feature_mean.tolist()
        if self.feature_scale is not None:
            metadata["feature_scale"] = self.feature_scale.tolist()
        return metadata


def main():
    preprocessor = NoisePreprocessor(normalize=True)
    print("\n=== Preprocessing & Feature Extraction Test ===")
    t = np.linspace(0, 1, 1000)
    signal_test = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(1000)
    features = preprocessor.extract_all_features(signal_test, sample_rate=1000)
    print(f"Total features: {len(features)}")
    converter = FeatureVector()
    vector = converter.to_vector(features)
    print(f"Feature vector shape: {vector.shape}")


if __name__ == "__main__":
    main()
