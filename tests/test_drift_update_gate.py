from pathlib import Path

import torch

from auth.enrollment import DeviceEnroller


class _IdentityEmbedder:
    device = "cpu"

    def embed(self, tensor):
        return tensor


class _NoopPreprocessor:
    pass


class _NoopFeatureVector:
    pass


def test_drift_update_requires_multiple_strong_accepts(tmp_path: Path):
    enroller = DeviceEnroller(
        embedder=_IdentityEmbedder(),
        preprocessor=_NoopPreprocessor(),
        feature_converter=_NoopFeatureVector(),
        storage_dir=str(tmp_path),
    )
    enroller.drift_min_strong_matches = 2

    base = torch.nn.functional.normalize(torch.tensor([1.0, 0.0]), p=2, dim=0)
    auth = torch.nn.functional.normalize(torch.tensor([0.8, 0.2]), p=2, dim=0)

    enroller.save_device_embedding(
        "device_a",
        {
            "combined_embedding": base,
            "source_embeddings": {"camera": base},
        },
        metadata={"sources": ["camera"]},
    )

    auth_profile = {
        "combined_embedding": auth,
        "source_embeddings": {"camera": auth},
        "source_profiles": {"camera": {"rms_mean": 0.1, "rms_std": 0.01}},
    }

    first = enroller.update_device_template("device_a", auth_profile, similarity=0.99)
    assert first["drift_update_applied"] is False
    assert first["drift_model"]["pending_strong_accepts"] == 1

    second = enroller.update_device_template("device_a", auth_profile, similarity=0.99)
    assert second["drift_update_applied"] is True
    assert second["drift_model"]["pending_strong_accepts"] == 0
