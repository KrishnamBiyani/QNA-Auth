import torch

from auth.authentication import DeviceAuthenticator


class _ScalarEmbedder:
    def compute_similarity(self, embedding1, embedding2, metric="cosine"):
        return 1.0 - abs(float(embedding1[0]) - float(embedding2[0]))


class _NoopPreprocessor:
    pass


class _NoopFeatureVector:
    pass


class _MapEnroller:
    storage_dir = "."

    def __init__(self, store):
        self.store = store

    def load_device_record(self, device_id):
        embedding = self.store.get(device_id)
        if embedding is None:
            return None
        return {
            "combined_embedding": embedding,
            "source_embeddings": {},
        }

    def load_device_embedding(self, device_id):
        return self.store.get(device_id)

    def load_device_metadata(self, device_id):
        return {}

    def get_effective_weights(self, sources):
        return {}

    def list_enrolled_devices(self):
        return list(self.store.keys())


def _profile(value: float):
    tensor = torch.tensor([value])
    return {"combined_embedding": tensor, "source_embeddings": {}, "sources": []}


def test_margin_guard_rejects_when_impostor_too_close():
    enroller = _MapEnroller(
        {
            "target": torch.tensor([0.80]),
            "impostor": torch.tensor([0.79]),
        }
    )
    auth = DeviceAuthenticator(
        embedder=_ScalarEmbedder(),
        preprocessor=_NoopPreprocessor(),
        feature_converter=_NoopFeatureVector(),
        enroller=enroller,
        threshold=0.85,
    )
    auth.identification_margin = 0.02

    ok, similarity, details = auth.verify_device("target", _profile(0.80))

    assert similarity >= 0.85
    assert ok is False
    assert details["margin_check_passed"] is False


def test_margin_guard_accepts_when_separation_is_clear():
    enroller = _MapEnroller(
        {
            "target": torch.tensor([0.80]),
            "impostor": torch.tensor([0.60]),
        }
    )
    auth = DeviceAuthenticator(
        embedder=_ScalarEmbedder(),
        preprocessor=_NoopPreprocessor(),
        feature_converter=_NoopFeatureVector(),
        enroller=enroller,
        threshold=0.85,
    )
    auth.identification_margin = 0.02

    ok, similarity, details = auth.verify_device("target", _profile(0.80))

    assert similarity >= 0.85
    assert ok is True
    assert details["margin_check_passed"] is True

