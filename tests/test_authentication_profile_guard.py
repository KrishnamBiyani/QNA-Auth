import numpy as np
import torch

from auth.authentication import DeviceAuthenticator


class _DummyEmbedder:
    def compute_similarity(self, embedding1, embedding2, metric="cosine"):
        return 1.0


class _DummyPreprocessor:
    def extract_all_features(self, sample):
        return {"mean": 0.0, "std": 1.0, "spectral_entropy": 0.0, "shannon_entropy": 0.0}


class _DummyFeatureVector:
    feature_names = ["mean"]

    def to_vector(self, features):
        return np.array([0.0], dtype=np.float32)


class _DummyEnroller:
    storage_dir = "."

    def load_device_record(self, device_id):
        return {"combined_embedding": torch.tensor([1.0]), "source_embeddings": {}}

    def load_device_embedding(self, device_id):
        return torch.tensor([1.0])

    def load_device_metadata(self, device_id):
        return {}

    def get_effective_weights(self, sources):
        return {}

    def list_enrolled_devices(self):
        return []


def test_profile_guard_rejects_large_rms_shift():
    authenticator = DeviceAuthenticator(
        embedder=_DummyEmbedder(),
        preprocessor=_DummyPreprocessor(),
        feature_converter=_DummyFeatureVector(),
        enroller=_DummyEnroller(),
        threshold=0.85,
    )
    metadata = {
        "source_profiles": {
            "microphone": {"rms_mean": 0.02, "rms_std": 0.005}
        }
    }
    current_samples = {
        "microphone": [np.ones(1024, dtype=np.float32) * 0.30]
    }

    valid, reason, runtime_profiles = authenticator._validate_source_profile(metadata, current_samples)

    assert valid is False
    assert "RMS mismatch" in reason
    assert "microphone" in runtime_profiles


def test_profile_guard_accepts_similar_rms():
    authenticator = DeviceAuthenticator(
        embedder=_DummyEmbedder(),
        preprocessor=_DummyPreprocessor(),
        feature_converter=_DummyFeatureVector(),
        enroller=_DummyEnroller(),
        threshold=0.85,
    )
    metadata = {
        "source_profiles": {
            "microphone": {"rms_mean": 0.02, "rms_std": 0.005}
        }
    }
    current_samples = {
        "microphone": [np.ones(1024, dtype=np.float32) * 0.018]
    }

    valid, reason, runtime_profiles = authenticator._validate_source_profile(metadata, current_samples)

    assert valid is True
    assert reason == "Profile validation passed"
    assert runtime_profiles["microphone"]["rms_mean"] > 0.0

