import numpy as np

from preprocessing.features import NoisePreprocessor


def test_extract_all_features_caps_analysis_length_for_heavy_transforms():
    preprocessor = NoisePreprocessor(normalize=True, fast_mode=True)
    observed = {}

    def fake_fft(data, sample_rate=1.0):
        observed["fft_len"] = len(data)
        return {
            "dominant_freq": 0.0,
            "dominant_magnitude": 0.0,
            "spectral_centroid": 0.0,
            "spectral_spread": 0.0,
            "spectral_entropy": 0.0,
            "spectral_flatness": 0.0,
            "low_freq_power": 0.0,
            "mid_freq_power": 0.0,
            "high_freq_power": 0.0,
        }

    def fake_autocorr(data, max_lag=100):
        observed["autocorr_len"] = len(data)
        return {
            "first_zero_crossing": 0.0,
            "autocorr_mean_10": 0.0,
            "autocorr_mean_50": 0.0,
            "autocorr_decay": 0.0,
        }

    preprocessor.compute_fft_features = fake_fft  # type: ignore[method-assign]
    preprocessor.compute_autocorrelation_features = fake_autocorr  # type: ignore[method-assign]

    huge_signal = np.random.default_rng(42).normal(size=1_000_000).astype(np.float32)
    _ = preprocessor.extract_all_features(huge_signal)

    assert observed["fft_len"] <= 16_384
    assert observed["autocorr_len"] <= 16_384
