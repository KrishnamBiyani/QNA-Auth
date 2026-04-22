"""
Device authentication for noise-based device verification.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import config
from model.siamese_model import DeviceEmbedder
from noise_collection import CameraNoiseCollector, MicrophoneNoiseCollector
from preprocessing.features import FeatureVector, NoisePreprocessor
from .enrollment import DeviceEnroller

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeviceAuthenticator:
    """Authenticates devices by weighted multi-source template comparison."""

    def __init__(
        self,
        embedder: DeviceEmbedder,
        preprocessor: NoisePreprocessor,
        feature_converter: FeatureVector,
        enroller: DeviceEnroller,
        threshold: float = float(getattr(config, "SIMILARITY_THRESHOLD", 0.97)),
        metric: str = "cosine",
    ):
        self.embedder = embedder
        self.preprocessor = preprocessor
        self.feature_converter = feature_converter
        self.enroller = enroller
        self.threshold = threshold
        self.metric = metric
        self.profile_guard_z = float(getattr(config, "AUTH_PROFILE_GUARD_Z", 6.0))
        self.profile_guard_min_delta = float(getattr(config, "AUTH_PROFILE_GUARD_MIN_DELTA", 0.02))
        self.identification_margin = float(getattr(config, "AUTH_IDENTIFICATION_MARGIN", 0.02))
        self.strong_accept_threshold = float(getattr(config, "AUTH_CONFIDENCE_STRONG", 0.97))
        self.uncertain_threshold = float(getattr(config, "AUTH_CONFIDENCE_UNCERTAIN", 0.92))
        self.drift_update_enabled = bool(getattr(config, "AUTH_DRIFT_UPDATE_ENABLED", True))
        logger.info(
            "DeviceAuthenticator initialized (strong=%s, uncertain=%s, metric=%s)",
            self.strong_accept_threshold,
            self.uncertain_threshold,
            metric,
        )

    def _load_device_metadata(self, device_id: str) -> Dict:
        return self.enroller.load_device_metadata(device_id)

    def _validate_source_profile(
        self,
        metadata: Dict,
        samples_by_source: Dict[str, List[np.ndarray]],
    ) -> Tuple[bool, str, Dict[str, Dict[str, float]]]:
        source_profiles = metadata.get("source_profiles") or {}
        runtime_profiles: Dict[str, Dict[str, float]] = {}
        if not samples_by_source:
            return False, "No runtime samples provided", runtime_profiles

        for source, samples in samples_by_source.items():
            if not samples:
                continue

            current_rms = [float(np.sqrt(np.mean(np.asarray(sample) ** 2))) for sample in samples]
            current_mean = float(np.mean(current_rms))
            current_std = float(np.std(current_rms))
            runtime_profiles[source] = {
                "rms_mean": current_mean,
                "rms_std": current_std,
            }

            profile = source_profiles.get(source)
            if not profile:
                continue

            enrolled_mean = float(profile.get("rms_mean", 0.0))
            enrolled_std = float(profile.get("rms_std", 0.0))
            allowed_delta = max(
                self.profile_guard_min_delta,
                self.profile_guard_z * max(enrolled_std, 1e-6),
            )
            delta = abs(current_mean - enrolled_mean)
            if delta > allowed_delta:
                message = (
                    f"RMS mismatch for source={source}: "
                    f"enrolled={enrolled_mean:.6f}, current={current_mean:.6f}, "
                    f"delta={delta:.6f}, allowed={allowed_delta:.6f}"
                )
                logger.warning(message)
                return False, message, runtime_profiles

        return True, "Profile validation passed", runtime_profiles

    def _classify_similarity(self, similarity: float) -> Tuple[str, str]:
        if similarity >= self.strong_accept_threshold:
            return "strong_accept", "accept"
        if similarity >= self.uncertain_threshold:
            return "uncertain", "collect_more_samples_or_fallback_auth"
        return "reject", "reject"

    def _compute_identification_margin(
        self,
        claimed_device_id: str,
        auth_embedding: torch.Tensor,
    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        claimed_similarity = None
        best_other_similarity = None
        best_other_device_id = None

        for candidate_id in self.enroller.list_enrolled_devices():
            candidate_embedding = self.enroller.load_device_embedding(candidate_id)
            if candidate_embedding is None:
                continue
            similarity = self.embedder.compute_similarity(
                auth_embedding,
                candidate_embedding,
                metric=self.metric,
            )
            if candidate_id == claimed_device_id:
                claimed_similarity = similarity
            elif best_other_similarity is None or similarity > best_other_similarity:
                best_other_similarity = similarity
                best_other_device_id = candidate_id

        return claimed_similarity, best_other_similarity, best_other_device_id

    def collect_authentication_sample(
        self,
        source: str = "camera",
        num_samples: int = 1,
    ) -> Optional[np.ndarray]:
        try:
            if source == "camera":
                camera_collector = CameraNoiseCollector(camera_index=0)
                if camera_collector.initialize_camera():
                    frames = camera_collector.capture_multiple_frames(
                        num_frames=num_samples,
                        exposure_time=0.1,
                    )
                    camera_collector.release()
                    extracted = [
                        camera_collector.extract_noise_features(frame)
                        for frame in frames
                        if frame is not None
                    ]
                    if extracted:
                        return extracted[0]

            if source == "microphone":
                mic_collector = MicrophoneNoiseCollector(sample_rate=44100)
                return mic_collector.capture_ambient_noise(duration=1.0)
        except Exception as exc:
            logger.error("Failed to collect authentication sample from %s: %s", source, exc)

        if source == "qrng":
            logger.warning("QRNG is no longer part of the authentication feature pipeline")
        return None

    def _build_auth_profile(
        self,
        samples_by_source: Dict[str, List[np.ndarray]],
    ) -> Optional[Dict]:
        features_by_source = self.enroller.process_noise_to_features_by_source(samples_by_source)
        if not features_by_source:
            return None

        source_embeddings = self.enroller.create_source_embeddings(features_by_source)
        if not source_embeddings:
            return None

        combined_embedding = self.enroller.combine_source_embeddings(source_embeddings)
        _, _, source_profiles = self._validate_source_profile({}, samples_by_source)
        return {
            "combined_embedding": combined_embedding,
            "source_embeddings": source_embeddings,
            "source_profiles": source_profiles,
            "sources": list(source_embeddings.keys()),
        }

    def generate_authentication_embedding(
        self,
        noise_samples: list,
    ) -> Optional[torch.Tensor]:
        combined = self.generate_authentication_profile({"combined": [np.asarray(sample) for sample in noise_samples]})
        if combined is None:
            return None
        return combined["combined_embedding"]

    def generate_authentication_profile(
        self,
        samples_by_source: Dict[str, List[np.ndarray]],
    ) -> Optional[Dict]:
        try:
            normalized_samples: Dict[str, List[np.ndarray]] = {}
            for source, samples in samples_by_source.items():
                normalized_samples[source] = []
                for sample in samples:
                    arr = np.asarray(sample, dtype=np.float32)
                    if arr.size == 0:
                        continue
                    if source == "microphone":
                        rms = float(np.sqrt(np.mean(arr ** 2)))
                        if rms < 1e-6:
                            raise ValueError("Audio sample is silent; microphone capture is invalid.")
                    normalized_samples[source].append(arr)
            return self._build_auth_profile(normalized_samples)
        except Exception as exc:
            logger.error("Failed to generate authentication profile: %s", exc)
            return None

    def verify_device(
        self,
        device_id: str,
        auth_profile: Dict,
    ) -> Tuple[bool, float, Dict]:
        stored_record = self.enroller.load_device_record(device_id)
        if stored_record is None:
            return False, 0.0, {"error": "Device not enrolled"}

        metadata = self._load_device_metadata(device_id)
        source_weights = self.enroller.get_effective_weights(auth_profile.get("sources", []))
        weighted_scores: List[Tuple[str, float, float]] = []

        stored_sources = stored_record.get("source_embeddings", {})
        auth_sources = auth_profile.get("source_embeddings", {})
        common_sources = [source for source in auth_sources.keys() if source in stored_sources]

        if common_sources:
            for source in common_sources:
                similarity = self.embedder.compute_similarity(
                    auth_sources[source],
                    stored_sources[source],
                    metric=self.metric,
                )
                weighted_scores.append((source, similarity, float(source_weights.get(source, 0.0))))
            total_weight = sum(weight for _, _, weight in weighted_scores) or 1.0
            similarity = sum(score * weight for _, score, weight in weighted_scores) / total_weight
        else:
            similarity = self.embedder.compute_similarity(
                auth_profile["combined_embedding"],
                stored_record["combined_embedding"],
                metric=self.metric,
            )

        band, recommended_action = self._classify_similarity(similarity)
        claimed_similarity, best_other_similarity, best_other_device_id = self._compute_identification_margin(
            claimed_device_id=device_id,
            auth_embedding=auth_profile["combined_embedding"],
        )
        observed_margin = None
        margin_ok = True
        if claimed_similarity is not None and best_other_similarity is not None:
            observed_margin = claimed_similarity - best_other_similarity
            margin_ok = observed_margin >= self.identification_margin

        is_authenticated = band == "strong_accept" and margin_ok
        details = {
            "device_id": device_id,
            "similarity": similarity,
            "confidence_band": band,
            "recommended_action": recommended_action,
            "metric": self.metric,
            "strong_accept_threshold": self.strong_accept_threshold,
            "uncertain_threshold": self.uncertain_threshold,
            "required_margin": self.identification_margin,
            "observed_margin": observed_margin,
            "best_other_similarity": best_other_similarity,
            "best_other_device_id": best_other_device_id,
            "margin_check_passed": margin_ok,
            "per_source_similarity": {
                source: {
                    "similarity": score,
                    "weight": weight,
                }
                for source, score, weight in weighted_scores
            },
            "authenticated": is_authenticated,
            "timestamp": datetime.now().isoformat(),
        }
        return is_authenticated, similarity, details

    def _normalize_client_samples(
        self,
        sources: List[str],
        client_samples: Dict[str, List[List[float]]],
    ) -> Dict[str, List[np.ndarray]]:
        samples_by_source: Dict[str, List[np.ndarray]] = {}
        for source, samples_list in client_samples.items():
            if source not in sources:
                continue
            samples_by_source[source] = [np.asarray(sample, dtype=np.float32) for sample in samples_list]
        return samples_by_source

    def authenticate(
        self,
        device_id: str,
        sources: list = ["camera", "microphone"],
        num_samples_per_source: int = 5,
        client_samples: Optional[Dict[str, List[List[float]]]] = None,
    ) -> Tuple[bool, Dict]:
        logger.info("Starting authentication for device: %s", device_id)
        metadata = self._load_device_metadata(device_id)

        enrolled_sources = set(metadata.get("sources", []))
        requested_sources = set(sources)
        invalid_sources = requested_sources - enrolled_sources
        if invalid_sources:
            error_msg = (
                f"Source mismatch: enrolled with {sorted(enrolled_sources)}, "
                f"requested {sorted(requested_sources)}."
            )
            return False, {"error": error_msg}

        samples_by_source: Dict[str, List[np.ndarray]] = {}
        if client_samples:
            samples_by_source = self._normalize_client_samples(sources, client_samples)
        else:
            for source in sources:
                samples_by_source[source] = []
                for _ in range(num_samples_per_source):
                    sample = self.collect_authentication_sample(source, num_samples=1)
                    if sample is not None:
                        samples_by_source[source].append(sample)

        sample_count = sum(len(samples) for samples in samples_by_source.values())
        if sample_count == 0:
            return False, {"error": "Failed to collect noise samples"}

        profile_ok, profile_reason, runtime_profiles = self._validate_source_profile(metadata, samples_by_source)
        if not profile_ok:
            return False, {
                "error": "Authentication sample profile mismatch",
                "details": profile_reason,
                "sources_used": sources,
                "runtime_profiles": runtime_profiles,
            }

        auth_profile = self.generate_authentication_profile(samples_by_source)
        if auth_profile is None:
            return False, {"error": "Failed to generate authentication profile"}

        auth_profile["source_profiles"] = runtime_profiles
        is_authenticated, similarity, details = self.verify_device(device_id, auth_profile)
        details["num_samples_collected"] = sample_count
        details["sources_used"] = sources
        details["runtime_profiles"] = runtime_profiles
        details["match_semantics"] = "high-confidence device matching"

        if (
            is_authenticated
            and details.get("confidence_band") == "strong_accept"
            and self.drift_update_enabled
        ):
            updated_metadata = self.enroller.update_device_template(
                device_id=device_id,
                auth_profile=auth_profile,
                similarity=similarity,
            )
            details["rolling_reenrollment_applied"] = bool(updated_metadata.get("drift_update_applied", False))
            details["drift_model"] = updated_metadata.get("drift_model", {})
        else:
            details["rolling_reenrollment_applied"] = False

        return is_authenticated, details

    def identify_device(
        self,
        auth_embedding: torch.Tensor,
        top_k: int = 5,
    ) -> list:
        enrolled_devices = self.enroller.list_enrolled_devices()
        similarities = []
        for device_id in enrolled_devices:
            stored_embedding = self.enroller.load_device_embedding(device_id)
            if stored_embedding is None:
                continue
            similarity = self.embedder.compute_similarity(
                auth_embedding,
                stored_embedding,
                metric=self.metric,
            )
            similarities.append((device_id, similarity))

        similarities.sort(key=lambda item: item[1], reverse=True)
        return similarities[:top_k]


class AuthenticationSession:
    """Manages authentication attempts with retry logic."""

    def __init__(self, authenticator: DeviceAuthenticator, max_attempts: int = 3):
        self.authenticator = authenticator
        self.max_attempts = max_attempts
        self.attempts = 0
        self.session_log = []

    def attempt_authentication(
        self,
        device_id: str,
        sources: list = ["camera", "microphone"],
    ) -> Tuple[bool, Dict]:
        self.attempts += 1
        if self.attempts > self.max_attempts:
            return False, {
                "error": "Maximum authentication attempts exceeded",
                "attempts": self.attempts,
                "session_log": self.session_log,
            }

        is_authenticated, details = self.authenticator.authenticate(device_id, sources=sources)
        self.session_log.append(
            {
                "attempt": self.attempts,
                "timestamp": datetime.now().isoformat(),
                "authenticated": is_authenticated,
                "details": details,
            }
        )

        if is_authenticated:
            details["session_log"] = self.session_log
            return True, details

        if self.attempts < self.max_attempts:
            return False, {
                "retry_available": True,
                "attempts": self.attempts,
                "details": details,
            }
        return False, {
            "error": "Authentication failed after maximum attempts",
            "session_log": self.session_log,
        }


def main():
    print("\n=== Device Authentication Test ===")
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

    authenticator = DeviceAuthenticator(
        embedder=embedder,
        preprocessor=preprocessor,
        feature_converter=feature_converter,
        enroller=enroller,
        threshold=0.97,
    )

    simulated_enroll = {
        "camera": [np.random.rand(1024) for _ in range(5)],
        "microphone": [np.random.rand(1024) for _ in range(5)],
    }
    features = enroller.process_noise_to_features_by_source(simulated_enroll)
    source_embeddings = enroller.create_source_embeddings(features)
    enroller.save_device_embedding(
        "test_device",
        {
            "combined_embedding": enroller.combine_source_embeddings(source_embeddings),
            "source_embeddings": source_embeddings,
        },
        metadata={"sources": ["camera", "microphone"]},
    )

    profile = authenticator.generate_authentication_profile(simulated_enroll)
    print(authenticator.verify_device("test_device", profile))


if __name__ == "__main__":
    main()
