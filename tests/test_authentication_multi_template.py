import torch

from auth.authentication import DeviceAuthenticator


class _ScalarEmbedder:
    def compute_similarity(self, embedding1, embedding2, metric="cosine"):
        return 1.0 - abs(float(embedding1[0]) - float(embedding2[0]))


class _NoopPreprocessor:
    pass


class _NoopFeatureVector:
    pass


class _TemplateEnroller:
    storage_dir = "."

    def __init__(self, records):
        self.records = records

    def load_device_record(self, device_id):
        return self.records.get(device_id)

    def load_device_embedding(self, device_id):
        record = self.records.get(device_id)
        return None if record is None else record["combined_embedding"]

    def load_device_metadata(self, device_id):
        return {}

    def get_effective_weights(self, sources):
        if not sources:
            return {}
        w = 1.0 / len(sources)
        return {source: w for source in sources}

    def list_enrolled_devices(self):
        return list(self.records.keys())


def _profile(value: float, sources: dict | None = None):
    combined = torch.tensor([value])
    source_embeddings = {k: torch.tensor([v]) for k, v in (sources or {}).items()}
    return {
        "combined_embedding": combined,
        "source_embeddings": source_embeddings,
        "sources": list(source_embeddings.keys()),
    }


def test_multi_template_matching_uses_best_templates():
    enroller = _TemplateEnroller(
        {
            "target": {
                "combined_embedding": torch.tensor([0.20]),
                "combined_templates": [torch.tensor([0.20]), torch.tensor([0.80])],
                "source_embeddings": {},
                "source_templates": {},
            },
            "impostor": {
                "combined_embedding": torch.tensor([0.10]),
                "combined_templates": [torch.tensor([0.10])],
                "source_embeddings": {},
                "source_templates": {},
            },
        }
    )
    auth = DeviceAuthenticator(
        embedder=_ScalarEmbedder(),
        preprocessor=_NoopPreprocessor(),
        feature_converter=_NoopFeatureVector(),
        enroller=enroller,
        threshold=0.85,
    )
    auth.template_top_k = 1
    auth.strong_accept_threshold = 0.95
    auth.uncertain_threshold = 0.90
    auth.identification_margin = 0.05

    ok, similarity, details = auth.verify_device("target", _profile(0.80))

    assert similarity == 1.0
    assert ok is True
    assert details["template_top_k"] == 1


def test_per_source_threshold_rejects_when_one_source_fails():
    enroller = _TemplateEnroller(
        {
            "target": {
                "combined_embedding": torch.tensor([0.0]),
                "combined_templates": [torch.tensor([0.0])],
                "source_embeddings": {
                    "camera": torch.tensor([0.80]),
                    "microphone": torch.tensor([0.80]),
                },
                "source_templates": {
                    "camera": [torch.tensor([0.80])],
                    "microphone": [torch.tensor([0.80])],
                },
            }
        }
    )
    auth = DeviceAuthenticator(
        embedder=_ScalarEmbedder(),
        preprocessor=_NoopPreprocessor(),
        feature_converter=_NoopFeatureVector(),
        enroller=enroller,
        threshold=0.85,
    )
    auth.strong_accept_threshold = 0.90
    auth.uncertain_threshold = 0.80
    auth.identification_margin = 0.0
    auth.source_thresholds = {
        "camera": {"strong": 0.90, "uncertain": 0.80},
        "microphone": {"strong": 0.95, "uncertain": 0.90},
    }

    ok, similarity, details = auth.verify_device(
        "target",
        _profile(0.0, sources={"camera": 0.80, "microphone": 0.70}),
    )

    assert similarity > 0.80
    assert ok is False
    assert details["per_source_check_passed"] is False
    assert details["per_source_similarity"]["microphone"]["band"] == "reject"
