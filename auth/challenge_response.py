"""
Server-hardened challenge verification for noise-based device matching.

The embedding is treated as a feature template only. It is never used directly
as a credential. Challenge MAC keys are derived from a stable feature template,
the challenge nonce, and a server secret via HKDF.
"""

import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import torch

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChallengeResponseProtocol:
    """Implements nonce-bound challenge verification with HKDF key derivation."""

    def __init__(
        self,
        nonce_length: int = 32,
        challenge_expiry_seconds: int = 60,
        challenge_store: Optional[Any] = None,
        server_secret: Optional[str] = None,
    ):
        self.nonce_length = nonce_length
        self.challenge_expiry = timedelta(seconds=challenge_expiry_seconds)
        self.active_challenges = {}
        self.challenge_store = challenge_store
        configured_secret = server_secret or getattr(
            config,
            "CHALLENGE_SERVER_SECRET",
            "dev-only-qna-auth-server-secret-change-me",
        )
        self.server_secret = configured_secret.encode("utf-8")
        self.hkdf_info = b"noise-device-verification-challenge"
        logger.info(
            "ChallengeResponseProtocol initialized (nonce_length=%s, expiry=%ss, persisted=%s)",
            nonce_length,
            challenge_expiry_seconds,
            challenge_store is not None,
        )

    def generate_nonce(self) -> str:
        return secrets.token_bytes(self.nonce_length).hex()

    def create_challenge(self, device_id: str) -> Dict[str, str]:
        nonce = self.generate_nonce()
        created_at = datetime.now()
        challenge_id = hashlib.sha256(
            f"{device_id}_{nonce}_{created_at.isoformat()}".encode("utf-8")
        ).hexdigest()[:16]
        data = {
            "device_id": device_id,
            "nonce": nonce,
            "created_at": created_at,
            "expires_at": created_at + self.challenge_expiry,
        }
        if self.challenge_store is not None:
            self.challenge_store.put(challenge_id, data)
        else:
            self.active_challenges[challenge_id] = data
        return {
            "challenge_id": challenge_id,
            "nonce": nonce,
            "expires_at": data["expires_at"].isoformat(),
        }

    def _stable_feature_bytes(self, embedding: torch.Tensor) -> bytes:
        embedding_cpu = embedding.detach().cpu().float()
        # Coarse quantization creates a more drift-tolerant stable template.
        quantized = torch.round(embedding_cpu * 64.0).to(torch.int16).numpy()
        return quantized.tobytes()

    def _hkdf(self, ikm: bytes, salt: bytes, length: int = 32) -> bytes:
        prk = hmac.new(salt, ikm, hashlib.sha256).digest()
        okm = b""
        previous = b""
        counter = 1
        while len(okm) < length:
            previous = hmac.new(
                prk,
                previous + self.hkdf_info + bytes([counter]),
                hashlib.sha256,
            ).digest()
            okm += previous
            counter += 1
        return okm[:length]

    def derive_mac_key(self, embedding: torch.Tensor, nonce: str) -> bytes:
        nonce_bytes = bytes.fromhex(nonce)
        stable_feature = self._stable_feature_bytes(embedding)
        ikm = stable_feature + nonce_bytes + self.server_secret
        return self._hkdf(ikm=ikm, salt=nonce_bytes, length=32)

    def compute_response(
        self,
        embedding: torch.Tensor,
        nonce: str,
        challenge_id: str,
        device_id: str,
    ) -> str:
        key = self.derive_mac_key(embedding, nonce)
        message = f"{challenge_id}:{device_id}:{nonce}".encode("utf-8")
        return hmac.new(key, message, hashlib.sha256).hexdigest()

    def _load_challenge(self, challenge_id: str) -> Optional[Dict]:
        if self.challenge_store is not None:
            return self.challenge_store.get(challenge_id)
        return self.active_challenges.get(challenge_id)

    def _delete_challenge(self, challenge_id: str) -> None:
        if self.challenge_store is not None:
            self.challenge_store.delete(challenge_id)
        else:
            self.active_challenges.pop(challenge_id, None)

    def verify_response(
        self,
        challenge_id: str,
        presented_response: Optional[str],
        stored_embedding: torch.Tensor,
        auth_embedding: torch.Tensor,
    ) -> Tuple[bool, Dict]:
        challenge_data = self._load_challenge(challenge_id)
        if challenge_data is None:
            return False, {"error": "Challenge not found"}

        if datetime.now() > challenge_data["expires_at"]:
            self._delete_challenge(challenge_id)
            return False, {"error": "Challenge expired"}

        nonce = challenge_data["nonce"]
        device_id = challenge_data["device_id"]
        expected_response = self.compute_response(stored_embedding, nonce, challenge_id, device_id)
        auth_response = self.compute_response(auth_embedding, nonce, challenge_id, device_id)
        stable_feature_match = hmac.compare_digest(expected_response, auth_response)

        external_response_checked = False
        external_response_valid = True
        if presented_response and len(presented_response) == 64:
            try:
                int(presented_response, 16)
                external_response_checked = True
                external_response_valid = hmac.compare_digest(presented_response, expected_response)
            except ValueError:
                external_response_checked = False

        is_valid = stable_feature_match and external_response_valid
        self._delete_challenge(challenge_id)

        details = {
            "challenge_id": challenge_id,
            "device_id": device_id,
            "verified_at": datetime.now().isoformat(),
            "nonce_bound": True,
            "hkdf_hardened": True,
            "template_role": "feature_only",
            "stable_feature_match": stable_feature_match,
            "external_response_checked": external_response_checked,
            "external_response_valid": external_response_valid,
            "is_valid": is_valid,
        }
        return is_valid, details

    def cleanup_expired_challenges(self):
        now = datetime.now()
        expired = [
            challenge_id
            for challenge_id, challenge_data in self.active_challenges.items()
            if now > challenge_data["expires_at"]
        ]
        for challenge_id in expired:
            self._delete_challenge(challenge_id)

    def get_active_challenges_count(self) -> int:
        self.cleanup_expired_challenges()
        return len(self.active_challenges)


class SecureAuthenticationFlow:
    """Combines HKDF-hardened challenge verification and confidence bands."""

    def __init__(
        self,
        protocol: ChallengeResponseProtocol,
        strong_accept_threshold: float = float(getattr(config, "AUTH_CONFIDENCE_STRONG", 0.97)),
        uncertain_threshold: float = float(getattr(config, "AUTH_CONFIDENCE_UNCERTAIN", 0.92)),
    ):
        self.protocol = protocol
        self.strong_accept_threshold = strong_accept_threshold
        self.uncertain_threshold = uncertain_threshold

    def initiate_authentication(self, device_id: str) -> Dict[str, str]:
        return self.protocol.create_challenge(device_id)

    def _classify_similarity(self, similarity: float) -> Tuple[str, str]:
        if similarity >= self.strong_accept_threshold:
            return "strong_accept", "accept"
        if similarity >= self.uncertain_threshold:
            return "uncertain", "collect_more_samples_or_fallback_auth"
        return "reject", "reject"

    def complete_authentication(
        self,
        challenge_id: str,
        response: Optional[str],
        auth_embedding: torch.Tensor,
        stored_embedding: torch.Tensor,
    ) -> Tuple[bool, Dict]:
        response_valid, response_details = self.protocol.verify_response(
            challenge_id=challenge_id,
            presented_response=response,
            stored_embedding=stored_embedding,
            auth_embedding=auth_embedding,
        )
        if not response_valid:
            return False, response_details

        similarity = torch.nn.functional.cosine_similarity(
            auth_embedding.unsqueeze(0),
            stored_embedding.unsqueeze(0),
        ).item()
        confidence_band, recommended_action = self._classify_similarity(similarity)
        authenticated = confidence_band == "strong_accept"
        details = {
            **response_details,
            "embedding_similarity": similarity,
            "confidence_band": confidence_band,
            "recommended_action": recommended_action,
            "strong_accept_threshold": self.strong_accept_threshold,
            "uncertain_threshold": self.uncertain_threshold,
            "authenticated": authenticated,
        }
        return authenticated, details


class AntiReplayProtection:
    """Tracks recent responses when callers want an additional replay cache."""

    def __init__(self, window_seconds: int = 300):
        self.window = timedelta(seconds=window_seconds)
        self.used_responses = {}

    def check_and_record(self, response: str) -> bool:
        now = datetime.now()
        self.used_responses = {
            item: ts for item, ts in self.used_responses.items() if now - ts < self.window
        }
        if response in self.used_responses:
            logger.warning("Replay attack detected")
            return False
        self.used_responses[response] = now
        return True


def main():
    print("\n=== Challenge-Response Protocol Test ===")
    protocol = ChallengeResponseProtocol(nonce_length=32, challenge_expiry_seconds=60)
    device_id = "test_device_001"
    stored_embedding = torch.nn.functional.normalize(torch.randn(128), p=2, dim=0)
    auth_embedding = torch.nn.functional.normalize(stored_embedding + (0.001 * torch.randn(128)), p=2, dim=0)

    challenge = protocol.create_challenge(device_id)
    response = protocol.compute_response(auth_embedding, challenge["nonce"], challenge["challenge_id"], device_id)
    is_valid, details = protocol.verify_response(
        challenge["challenge_id"],
        response,
        stored_embedding,
        auth_embedding,
    )
    print(f"Valid: {is_valid}")
    print(details)


if __name__ == "__main__":
    main()
