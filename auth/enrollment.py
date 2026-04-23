"""
Device enrollment for noise-based device verification.

Embeddings are treated as biometric-like templates for similarity matching.
They are stored as verification features, not as standalone credentials.
"""

import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from itertools import zip_longest

import numpy as np
import torch

import config
from model.siamese_model import DeviceEmbedder
from noise_collection import CameraNoiseCollector, MicrophoneNoiseCollector
from preprocessing.features import FeatureVector, NoisePreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeviceEnroller:
    """Handles enrollment and persistence of verification templates."""

    def __init__(
        self,
        embedder: DeviceEmbedder,
        preprocessor: NoisePreprocessor,
        feature_converter: FeatureVector,
        source_embedders: Optional[Dict[str, DeviceEmbedder]] = None,
        source_feature_converters: Optional[Dict[str, FeatureVector]] = None,
        storage_dir: str = "./auth/device_embeddings",
        dataset_builder: Optional[object] = None,
    ):
        self.embedder = embedder
        self.preprocessor = preprocessor
        self.feature_converter = feature_converter
        self.source_embedders = source_embedders or {}
        self.source_feature_converters = source_feature_converters or {}
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_builder = dataset_builder
        self.source_weights = self._load_source_weights()
        self.template_chunk_size = int(getattr(config, "AUTH_TEMPLATE_CHUNK_SIZE", 5))
        self.max_templates_per_source = int(getattr(config, "AUTH_MAX_TEMPLATES_PER_SOURCE", 4))
        self.drift_ema_alpha = float(getattr(config, "AUTH_DRIFT_EMA_ALPHA", 0.2))
        self.drift_min_strong_matches = int(getattr(config, "AUTH_DRIFT_MIN_STRONG_MATCHES", 2))
        logger.info("DeviceEnroller initialized")

    def _get_embedder(self, source: Optional[str] = None) -> DeviceEmbedder:
        if source is not None and source in self.source_embedders:
            return self.source_embedders[source]
        return self.embedder

    def _get_feature_converter(self, source: Optional[str] = None) -> FeatureVector:
        if source is not None and source in self.source_feature_converters:
            return self.source_feature_converters[source]
        return self.feature_converter

    def _load_source_weights(self) -> Dict[str, float]:
        configured = getattr(
            config,
            "AUTH_SOURCE_WEIGHTS",
            {"camera": 0.7, "microphone": 0.3},
        )
        weights = {
            str(source): float(weight)
            for source, weight in configured.items()
            if float(weight) > 0.0
        }
        return weights or {"camera": 0.7, "microphone": 0.3}

    def get_effective_weights(self, sources: List[str]) -> Dict[str, float]:
        active = {
            source: float(self.source_weights.get(source, 0.0))
            for source in sources
            if float(self.source_weights.get(source, 0.0)) > 0.0
        }
        if not active:
            uniform = 1.0 / max(len(sources), 1)
            return {source: uniform for source in sources}
        total = sum(active.values())
        return {source: value / total for source, value in active.items()}

    def generate_device_id(self, device_name: Optional[str] = None) -> str:
        timestamp = datetime.now().isoformat()
        identifier = f"{device_name}_{timestamp}" if device_name else timestamp
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]

    def collect_noise_samples(
        self,
        num_samples: int = 50,
        sources: List[str] = ["camera", "microphone"],
    ) -> Dict[str, List[np.ndarray]]:
        """Collect runtime sources used for enrollment."""
        noise_samples: Dict[str, List[np.ndarray]] = {}

        if "camera" in sources:
            logger.info("Collecting %s camera noise samples...", num_samples)
            try:
                camera_collector = CameraNoiseCollector(camera_index=0)
                if camera_collector.initialize_camera():
                    frames = camera_collector.capture_multiple_frames(
                        num_frames=num_samples,
                        exposure_time=0.1,
                    )
                    samples = [
                        camera_collector.extract_noise_features(frame)
                        for frame in frames
                        if frame is not None
                    ]
                    noise_samples["camera"] = samples
                    camera_collector.release()
                    logger.info("Collected %s camera samples", len(samples))
                else:
                    logger.error("Failed to initialize camera")
            except Exception as exc:
                logger.error("Failed to collect camera samples: %s", exc, exc_info=True)

        if "microphone" in sources:
            logger.info("Collecting %s microphone noise samples...", num_samples)
            try:
                mic_collector = MicrophoneNoiseCollector(sample_rate=44100)
                samples = mic_collector.capture_multiple_samples(
                    num_samples=num_samples,
                    duration=0.5,
                )
                noise_samples["microphone"] = samples
                logger.info("Collected %s microphone samples", len(samples))
            except Exception as exc:
                logger.error("Failed to collect microphone samples: %s", exc, exc_info=True)

        unsupported = sorted(set(sources) - {"camera", "microphone"})
        if unsupported:
            logger.warning(
                "Ignoring unsupported runtime sources during enrollment: %s",
                unsupported,
            )

        return noise_samples

    def process_noise_to_features_by_source(
        self,
        noise_samples: Dict[str, List[np.ndarray]],
    ) -> Dict[str, List[np.ndarray]]:
        """Convert raw samples into feature vectors grouped by source."""
        features_by_source: Dict[str, List[np.ndarray]] = {}
        for source, samples in noise_samples.items():
            feature_vectors: List[np.ndarray] = []
            logger.info("Processing %s samples from %s...", len(samples), source)
            converter = self._get_feature_converter(source)
            for sample in samples:
                try:
                    features = self.preprocessor.extract_all_features(sample)
                    feature_vector = converter.to_vector(features)
                    feature_vectors.append(feature_vector)
                except Exception as exc:
                    logger.warning("Failed to process %s sample: %s", source, exc)
            if feature_vectors:
                features_by_source[source] = feature_vectors
        return features_by_source

    def process_noise_to_features(
        self,
        noise_samples: Dict[str, List[np.ndarray]],
    ) -> List[np.ndarray]:
        feature_vectors: List[np.ndarray] = []
        for vectors in self.process_noise_to_features_by_source(noise_samples).values():
            feature_vectors.extend(vectors)
        logger.info("Processed %s feature vectors", len(feature_vectors))
        return feature_vectors

    def _embed_feature_vector(
        self,
        feature_vector: np.ndarray,
        source: Optional[str] = None,
    ) -> torch.Tensor:
        embedder = self._get_embedder(source)
        fv_tensor = torch.from_numpy(feature_vector).float().to(embedder.device)
        return embedder.embed(fv_tensor)

    def create_device_embedding(
        self,
        feature_vectors: List[np.ndarray],
        source: Optional[str] = None,
        method: str = "mean",
    ) -> torch.Tensor:
        if not feature_vectors:
            raise ValueError("No feature vectors provided")

        embeddings = [self._embed_feature_vector(fv, source=source) for fv in feature_vectors]
        embedding_stack = torch.stack(embeddings)

        if method == "mean":
            device_embedding = torch.mean(embedding_stack, dim=0)
        elif method == "median":
            device_embedding = torch.median(embedding_stack, dim=0).values
        elif method == "concat":
            device_embedding = embedding_stack[: min(5, len(embeddings))].flatten()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        device_embedding = torch.nn.functional.normalize(device_embedding, p=2, dim=0)
        return device_embedding

    def create_source_embeddings(
        self,
        features_by_source: Dict[str, List[np.ndarray]],
    ) -> Dict[str, torch.Tensor]:
        source_embeddings: Dict[str, torch.Tensor] = {}
        for source, vectors in features_by_source.items():
            if vectors:
                source_embeddings[source] = self.create_device_embedding(vectors, source=source, method="mean")
        return source_embeddings

    def create_source_template_bank(
        self,
        features_by_source: Dict[str, List[np.ndarray]],
    ) -> Dict[str, List[torch.Tensor]]:
        source_templates: Dict[str, List[torch.Tensor]] = {}
        chunk_size = max(1, self.template_chunk_size)
        max_templates = max(1, self.max_templates_per_source)
        for source, vectors in features_by_source.items():
            if not vectors:
                continue
            templates: List[torch.Tensor] = []
            for start in range(0, len(vectors), chunk_size):
                chunk = vectors[start:start + chunk_size]
                if not chunk:
                    continue
                templates.append(self.create_device_embedding(chunk, source=source, method="mean"))
                if len(templates) >= max_templates:
                    break
            if templates:
                source_templates[source] = templates
        return source_templates

    def combine_source_template_bank(
        self,
        source_templates: Dict[str, List[torch.Tensor]],
    ) -> List[torch.Tensor]:
        if not source_templates:
            return []
        weights = self.get_effective_weights(list(source_templates.keys()))
        max_len = max(len(templates) for templates in source_templates.values())
        combined_templates: List[torch.Tensor] = []
        for idx in range(max_len):
            template_slice = {
                source: templates[min(idx, len(templates) - 1)]
                for source, templates in source_templates.items()
                if templates
            }
            if not template_slice:
                continue
            combined = None
            for source, embedding in template_slice.items():
                weighted = embedding * float(weights.get(source, 0.0))
                combined = weighted if combined is None else combined + weighted
            combined_templates.append(torch.nn.functional.normalize(combined, p=2, dim=0))
        return combined_templates

    def combine_source_embeddings(
        self,
        source_embeddings: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not source_embeddings:
            raise ValueError("No source embeddings provided")

        weights = self.get_effective_weights(list(source_embeddings.keys()))
        combined = None
        for source, embedding in source_embeddings.items():
            weighted = embedding * float(weights.get(source, 0.0))
            combined = weighted if combined is None else combined + weighted
        return torch.nn.functional.normalize(combined, p=2, dim=0)

    def _default_metadata(self, device_id: str, embedding: torch.Tensor) -> Dict:
        return {
            "device_id": device_id,
            "enrollment_date": datetime.now().isoformat(),
            "embedding_shape": list(embedding.shape),
            "template_role": "biometric-like feature template used only for similarity matching",
            "template_is_secret": False,
            "template_not_credential": True,
        }

    def save_device_embedding(
        self,
        device_id: str,
        embedding: torch.Tensor | Dict[str, torch.Tensor],
        metadata: Optional[Dict] = None,
    ):
        if isinstance(embedding, dict):
            payload = {
                "template_version": 2,
                "combined_embedding": embedding["combined_embedding"].detach().cpu(),
                "source_embeddings": {
                    source: tensor.detach().cpu()
                    for source, tensor in embedding.get("source_embeddings", {}).items()
                },
                "combined_templates": [
                    tensor.detach().cpu()
                    for tensor in embedding.get("combined_templates", [])
                ],
                "source_templates": {
                    source: [tensor.detach().cpu() for tensor in tensors]
                    for source, tensors in embedding.get("source_templates", {}).items()
                },
            }
            combined = payload["combined_embedding"]
        else:
            payload = embedding.detach().cpu()
            combined = payload

        embedding_path = self.storage_dir / f"{device_id}_embedding.pt"
        torch.save(payload, embedding_path)

        metadata = metadata or {}
        enriched = self._default_metadata(device_id, combined)
        enriched.update(metadata)

        metadata_path = self.storage_dir / f"{device_id}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(enriched, handle, indent=2)

        logger.info("Saved device template bundle: %s", device_id)

    def load_device_record(self, device_id: str) -> Optional[Dict]:
        embedding_path = self.storage_dir / f"{device_id}_embedding.pt"
        if not embedding_path.exists():
            logger.error("Device embedding not found: %s", device_id)
            return None

        payload = torch.load(embedding_path, map_location="cpu")
        if isinstance(payload, dict) and "combined_embedding" in payload:
            record = {
                "combined_embedding": payload["combined_embedding"].float(),
                "source_embeddings": {
                    source: tensor.float()
                    for source, tensor in payload.get("source_embeddings", {}).items()
                },
                "combined_templates": [
                    tensor.float() for tensor in payload.get("combined_templates", [])
                ] or [payload["combined_embedding"].float()],
                "source_templates": {
                    source: [tensor.float() for tensor in tensors]
                    for source, tensors in payload.get("source_templates", {}).items()
                },
                "template_version": int(payload.get("template_version", 2)),
            }
        else:
            record = {
                "combined_embedding": payload.float(),
                "source_embeddings": {},
                "combined_templates": [payload.float()],
                "source_templates": {},
                "template_version": 1,
            }
        return record

    def load_device_embedding(self, device_id: str) -> Optional[torch.Tensor]:
        record = self.load_device_record(device_id)
        if record is None:
            return None
        return record["combined_embedding"]

    def load_device_metadata(self, device_id: str) -> Dict:
        metadata_path = self.storage_dir / f"{device_id}_metadata.json"
        if not metadata_path.exists():
            return {}
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            logger.warning("Failed to load metadata for %s: %s", device_id, exc)
            return {}

    def update_device_template(
        self,
        device_id: str,
        auth_profile: Dict,
        similarity: float,
    ) -> Dict:
        """
        Apply rolling re-enrollment after enough strong matches using EMA updates.
        """
        record = self.load_device_record(device_id)
        if record is None:
            raise FileNotFoundError(f"Device not found: {device_id}")

        metadata = self.load_device_metadata(device_id)
        alpha = self.drift_ema_alpha
        min_strong_matches = max(self.drift_min_strong_matches, 1)
        drift_state = metadata.get("drift_model", {})
        pending_strong_accepts = int(drift_state.get("pending_strong_accepts", 0)) + 1
        update_applied = pending_strong_accepts >= min_strong_matches

        drift_state["ema_alpha"] = alpha
        drift_state["rolling_reenroll_enabled"] = True
        drift_state["min_strong_accepts"] = min_strong_matches
        drift_state["pending_strong_accepts"] = pending_strong_accepts
        drift_state["last_strong_accept_at"] = datetime.now().isoformat()
        drift_state["last_strong_accept_similarity"] = float(similarity)

        if not update_applied:
            metadata["drift_model"] = drift_state
            metadata["rolling_reenrollment_policy"] = "strong_accept_only_with_consecutive_gate"
            self.save_device_embedding(
                device_id=device_id,
                embedding={
                    "combined_embedding": record["combined_embedding"],
                    "source_embeddings": record.get("source_embeddings", {}),
                },
                metadata=metadata,
            )
            metadata["drift_update_applied"] = False
            return metadata

        updated_sources = dict(record.get("source_embeddings", {}))
        updated_source_templates = dict(record.get("source_templates", {}))
        for source, auth_embedding in auth_profile.get("source_embeddings", {}).items():
            existing = updated_sources.get(source)
            if existing is None:
                updated_sources[source] = auth_embedding.detach().cpu()
            else:
                updated_sources[source] = torch.nn.functional.normalize(
                    ((1.0 - alpha) * existing) + (alpha * auth_embedding.detach().cpu()),
                    p=2,
                    dim=0,
                )
            source_templates = [tensor.detach().cpu() for tensor in updated_source_templates.get(source, [])]
            source_templates.append(auth_embedding.detach().cpu())
            updated_source_templates[source] = source_templates[-max(1, self.max_templates_per_source):]

        combined = torch.nn.functional.normalize(
            ((1.0 - alpha) * record["combined_embedding"]) +
            (alpha * auth_profile["combined_embedding"].detach().cpu()),
            p=2,
            dim=0,
        )
        combined_templates = [tensor.detach().cpu() for tensor in record.get("combined_templates", [])]
        combined_templates.append(auth_profile["combined_embedding"].detach().cpu())
        combined_templates = combined_templates[-max(1, self.max_templates_per_source):]

        drift_state["updates"] = int(drift_state.get("updates", 0)) + 1
        drift_state["last_update_at"] = datetime.now().isoformat()
        drift_state["last_update_similarity"] = float(similarity)
        drift_state["pending_strong_accepts"] = 0

        source_profiles = metadata.get("source_profiles", {})
        runtime_profiles = auth_profile.get("source_profiles", {})
        for source, runtime_profile in runtime_profiles.items():
            enrolled = source_profiles.get(source, {})
            source_profiles[source] = {
                "rms_mean": float(
                    ((1.0 - alpha) * float(enrolled.get("rms_mean", runtime_profile["rms_mean"])))
                    + (alpha * float(runtime_profile["rms_mean"]))
                ),
                "rms_std": float(
                    ((1.0 - alpha) * float(enrolled.get("rms_std", runtime_profile["rms_std"])))
                    + (alpha * float(runtime_profile["rms_std"]))
                ),
            }

        metadata["source_profiles"] = source_profiles
        metadata["drift_model"] = drift_state
        metadata["last_strong_match_at"] = datetime.now().isoformat()
        metadata["rolling_reenrollment_policy"] = "strong_accept_only_with_consecutive_gate"

        self.save_device_embedding(
            device_id=device_id,
            embedding={
                "combined_embedding": combined,
                "source_embeddings": updated_sources,
                "combined_templates": combined_templates,
                "source_templates": updated_source_templates,
            },
            metadata=metadata,
        )
        metadata["drift_update_applied"] = True
        return metadata

    def list_enrolled_devices(self) -> List[str]:
        device_ids = []
        for file in self.storage_dir.glob("*_embedding.pt"):
            device_ids.append(file.stem.replace("_embedding", ""))
        return device_ids

    def _build_source_profiles(
        self,
        noise_samples: Dict[str, List[np.ndarray]],
    ) -> Dict[str, Dict[str, float]]:
        source_profiles: Dict[str, Dict[str, float]] = {}
        for source, samples in noise_samples.items():
            if not samples:
                continue
            rms_values = [float(np.sqrt(np.mean(np.asarray(sample) ** 2))) for sample in samples]
            source_profiles[source] = {
                "rms_mean": float(np.mean(rms_values)),
                "rms_std": float(np.std(rms_values)),
            }
        return source_profiles

    def _normalize_client_samples(
        self,
        client_samples: Dict[str, List[List[float]]],
        requested_sources: List[str],
    ) -> Dict[str, List[np.ndarray]]:
        noise_samples: Dict[str, List[np.ndarray]] = {}
        for source, samples_list in client_samples.items():
            if source not in requested_sources:
                continue
            noise_samples[source] = [np.asarray(sample, dtype=np.float32) for sample in samples_list]
        return noise_samples

    def enroll_device(
        self,
        device_name: Optional[str] = None,
        num_samples: int = 50,
        sources: List[str] = ["camera", "microphone"],
        client_samples: Optional[Dict[str, List[List[float]]]] = None,
    ) -> str:
        logger.info("Starting device enrollment...")
        device_id = self.generate_device_id(device_name)

        if client_samples:
            logger.info("Using client-provided samples")
            noise_samples = self._normalize_client_samples(client_samples, sources)
        else:
            noise_samples = self.collect_noise_samples(num_samples=num_samples, sources=sources)

        if not noise_samples:
            raise RuntimeError("No noise samples collected from any source")

        total_samples = sum(len(samples) for samples in noise_samples.values())
        if total_samples == 0:
            raise RuntimeError("No samples collected from any requested source")

        if self.dataset_builder:
            try:
                for source, samples in noise_samples.items():
                    self.dataset_builder.add_batch(
                        device_id=device_id,
                        noise_source=source,
                        samples=samples,
                    )
            except Exception as exc:
                logger.error("Failed to save raw samples: %s", exc)

        features_by_source = self.process_noise_to_features_by_source(noise_samples)
        if not features_by_source:
            raise RuntimeError("No feature vectors generated from the collected samples")

        source_embeddings = self.create_source_embeddings(features_by_source)
        source_templates = self.create_source_template_bank(features_by_source)
        combined_templates = self.combine_source_template_bank(source_templates)
        combined_embedding = self.combine_source_embeddings(source_embeddings)
        source_profiles = self._build_source_profiles(noise_samples)
        effective_weights = self.get_effective_weights(list(source_embeddings.keys()))
        feature_dimension = len(next(iter(features_by_source.values()))[0])

        metadata = {
            "device_name": device_name,
            "num_samples": total_samples,
            "sources": list(source_embeddings.keys()),
            "feature_dimension": feature_dimension,
            "source_profiles": source_profiles,
            "source_weights": effective_weights,
            "template_strategy": "multi_template_chunk_mean",
            "template_chunk_size": self.template_chunk_size,
            "max_templates_per_source": self.max_templates_per_source,
            "combined_template_count": len(combined_templates),
            "embedding_role": "feature_only",
            "match_semantics": "high-confidence device matching",
            "drift_model": {
                "ema_alpha": self.drift_ema_alpha,
                "rolling_reenroll_enabled": True,
                "updates": 0,
                "pending_strong_accepts": 0,
                "min_strong_accepts": self.drift_min_strong_matches,
            },
            "attacker_model": {
                "replay_samples": "blocked by per-request nonce and single-use challenge",
                "synthetic_noise": "partial success possible; similarity and margin checks still apply",
                "same_device_different_environment": "borderline outcomes handled via confidence bands",
            },
        }

        self.save_device_embedding(
            device_id=device_id,
            embedding={
                "combined_embedding": combined_embedding,
                "source_embeddings": source_embeddings,
                "combined_templates": combined_templates,
                "source_templates": source_templates,
            },
            metadata=metadata,
        )

        logger.info("Device enrollment complete: %s", device_id)
        return device_id


def main():
    print("\n=== Device Enrollment Test ===")
    from model.siamese_model import SiameseNetwork

    input_dim = 50
    embedding_dim = 128
    model = SiameseNetwork(input_dim=input_dim, embedding_dim=embedding_dim)
    embedder = DeviceEmbedder(input_dim=input_dim, embedding_dim=embedding_dim)
    embedder.model = model

    preprocessor = NoisePreprocessor(normalize=True)
    feature_converter = FeatureVector()

    enroller = DeviceEnroller(
        embedder=embedder,
        preprocessor=preprocessor,
        feature_converter=feature_converter,
        storage_dir="./auth/test_embeddings",
    )

    simulated_noise = {
        "camera": [np.random.rand(480 * 640) for _ in range(5)],
        "microphone": [np.random.rand(44100) for _ in range(5)],
    }

    features_by_source = enroller.process_noise_to_features_by_source(simulated_noise)
    source_embeddings = enroller.create_source_embeddings(features_by_source)
    combined_embedding = enroller.combine_source_embeddings(source_embeddings)

    test_device_id = enroller.generate_device_id("TestDevice")
    enroller.save_device_embedding(
        test_device_id,
        {
            "combined_embedding": combined_embedding,
            "source_embeddings": source_embeddings,
        },
    )
    loaded = enroller.load_device_record(test_device_id)
    print(f"Loaded combined embedding shape: {loaded['combined_embedding'].shape}")


if __name__ == "__main__":
    main()
