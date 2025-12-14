"""
Challenge-Response Protocol
Implements secure challenge-response authentication with nonce
"""

import hashlib
import secrets
import hmac
from typing import Tuple, Dict, Optional
from datetime import datetime, timedelta
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChallengeResponseProtocol:
    """Implements secure challenge-response authentication"""
    
    def __init__(
        self,
        nonce_length: int = 32,
        challenge_expiry_seconds: int = 60
    ):
        """
        Initialize challenge-response protocol
        
        Args:
            nonce_length: Length of nonce in bytes
            challenge_expiry_seconds: Challenge expiration time
        """
        self.nonce_length = nonce_length
        self.challenge_expiry = timedelta(seconds=challenge_expiry_seconds)
        self.active_challenges = {}  # Map challenge_id -> challenge_data
        
        logger.info(f"ChallengeResponseProtocol initialized "
                   f"(nonce_length={nonce_length}, expiry={challenge_expiry_seconds}s)")
    
    def generate_nonce(self) -> str:
        """
        Generate cryptographically secure nonce
        
        Returns:
            Hex-encoded nonce string
        """
        nonce_bytes = secrets.token_bytes(self.nonce_length)
        nonce_hex = nonce_bytes.hex()
        return nonce_hex
    
    def create_challenge(self, device_id: str) -> Dict[str, str]:
        """
        Create authentication challenge
        
        Args:
            device_id: Device identifier
            
        Returns:
            Challenge dictionary with nonce and challenge_id
        """
        # Generate nonce
        nonce = self.generate_nonce()
        
        # Generate challenge ID
        challenge_id = hashlib.sha256(
            f"{device_id}_{nonce}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        # Store challenge
        self.active_challenges[challenge_id] = {
            'device_id': device_id,
            'nonce': nonce,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + self.challenge_expiry
        }
        
        logger.info(f"Created challenge {challenge_id} for device {device_id}")
        
        return {
            'challenge_id': challenge_id,
            'nonce': nonce,
            'expires_at': (datetime.now() + self.challenge_expiry).isoformat()
        }
    
    def compute_response(
        self,
        embedding: torch.Tensor,
        nonce: str
    ) -> str:
        """
        Compute challenge response from embedding and nonce
        
        Args:
            embedding: Device embedding tensor
            nonce: Challenge nonce
            
        Returns:
            Hex-encoded response signature
        """
        # Convert embedding to bytes
        embedding_bytes = embedding.numpy().tobytes()
        
        # Combine embedding hash with nonce
        embedding_hash = hashlib.sha256(embedding_bytes).hexdigest()
        
        # Create response using HMAC
        message = f"{embedding_hash}_{nonce}".encode()
        response = hmac.new(
            embedding_bytes[:32],  # Use first 32 bytes as key
            message,
            hashlib.sha256
        ).hexdigest()
        
        return response
    
    def verify_response(
        self,
        challenge_id: str,
        response: str,
        stored_embedding: torch.Tensor
    ) -> Tuple[bool, Dict]:
        """
        Verify challenge response
        
        Args:
            challenge_id: Challenge identifier
            response: Response signature from device
            stored_embedding: Stored device embedding
            
        Returns:
            Tuple of (is_valid, details)
        """
        # Check if challenge exists
        if challenge_id not in self.active_challenges:
            return False, {'error': 'Challenge not found'}
        
        challenge_data = self.active_challenges[challenge_id]
        
        # Check if challenge expired
        if datetime.now() > challenge_data['expires_at']:
            del self.active_challenges[challenge_id]
            return False, {'error': 'Challenge expired'}
        
        # Compute expected response
        expected_response = self.compute_response(
            stored_embedding,
            challenge_data['nonce']
        )
        
        # Compare responses (constant-time comparison)
        is_valid = hmac.compare_digest(response, expected_response)
        
        # Remove used challenge
        del self.active_challenges[challenge_id]
        
        details = {
            'challenge_id': challenge_id,
            'device_id': challenge_data['device_id'],
            'verified_at': datetime.now().isoformat(),
            'is_valid': is_valid
        }
        
        logger.info(f"Challenge verification: {challenge_id}, valid={is_valid}")
        
        return is_valid, details
    
    def cleanup_expired_challenges(self):
        """Remove expired challenges"""
        now = datetime.now()
        expired = [
            cid for cid, data in self.active_challenges.items()
            if now > data['expires_at']
        ]
        
        for cid in expired:
            del self.active_challenges[cid]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired challenges")
    
    def get_active_challenges_count(self) -> int:
        """Get number of active challenges"""
        self.cleanup_expired_challenges()
        return len(self.active_challenges)


class SecureAuthenticationFlow:
    """Complete secure authentication flow with challenge-response"""
    
    def __init__(
        self,
        protocol: ChallengeResponseProtocol,
        similarity_threshold: float = 0.85
    ):
        """
        Initialize secure authentication flow
        
        Args:
            protocol: ChallengeResponseProtocol instance
            similarity_threshold: Embedding similarity threshold
        """
        self.protocol = protocol
        self.similarity_threshold = similarity_threshold
    
    def initiate_authentication(
        self,
        device_id: str
    ) -> Dict[str, str]:
        """
        Step 1: Server initiates authentication
        
        Args:
            device_id: Device identifier
            
        Returns:
            Challenge dictionary
        """
        return self.protocol.create_challenge(device_id)
    
    def complete_authentication(
        self,
        challenge_id: str,
        response: str,
        auth_embedding: torch.Tensor,
        stored_embedding: torch.Tensor
    ) -> Tuple[bool, Dict]:
        """
        Step 2: Complete authentication with response and embedding verification
        
        Args:
            challenge_id: Challenge identifier
            response: Challenge response from device
            auth_embedding: Fresh authentication embedding
            stored_embedding: Stored device embedding
            
        Returns:
            Tuple of (is_authenticated, details)
        """
        # Verify challenge response
        response_valid, response_details = self.protocol.verify_response(
            challenge_id,
            response,
            stored_embedding
        )
        
        if not response_valid:
            return False, response_details
        
        # Verify embedding similarity
        similarity = torch.nn.functional.cosine_similarity(
            auth_embedding.unsqueeze(0),
            stored_embedding.unsqueeze(0)
        ).item()
        
        embedding_valid = similarity >= self.similarity_threshold
        
        details = {
            **response_details,
            'embedding_similarity': similarity,
            'similarity_threshold': self.similarity_threshold,
            'embedding_valid': embedding_valid,
            'authenticated': response_valid and embedding_valid
        }
        
        is_authenticated = response_valid and embedding_valid
        
        logger.info(f"Authentication complete: authenticated={is_authenticated}, "
                   f"similarity={similarity:.4f}")
        
        return is_authenticated, details


class AntiReplayProtection:
    """Prevents replay attacks"""
    
    def __init__(self, window_seconds: int = 300):
        """
        Initialize anti-replay protection
        
        Args:
            window_seconds: Time window for tracking used responses
        """
        self.window = timedelta(seconds=window_seconds)
        self.used_responses = {}  # Map response_hash -> timestamp
    
    def check_and_record(self, response: str) -> bool:
        """
        Check if response has been used and record it
        
        Args:
            response: Response signature
            
        Returns:
            True if response is fresh (not replayed), False if replayed
        """
        # Cleanup old entries
        now = datetime.now()
        self.used_responses = {
            r: t for r, t in self.used_responses.items()
            if now - t < self.window
        }
        
        # Check if response was used
        if response in self.used_responses:
            logger.warning(f"Replay attack detected!")
            return False
        
        # Record response
        self.used_responses[response] = now
        return True


def main():
    """Test challenge-response protocol"""
    print("\n=== Challenge-Response Protocol Test ===")
    
    # Create protocol
    protocol = ChallengeResponseProtocol(
        nonce_length=32,
        challenge_expiry_seconds=60
    )
    
    # Simulate device embedding
    device_id = "test_device_001"
    embedding = torch.randn(128)
    
    print("\n=== Step 1: Create Challenge ===")
    challenge = protocol.create_challenge(device_id)
    print(f"Challenge ID: {challenge['challenge_id']}")
    print(f"Nonce: {challenge['nonce'][:32]}...")
    print(f"Expires at: {challenge['expires_at']}")
    
    print("\n=== Step 2: Compute Response ===")
    response = protocol.compute_response(embedding, challenge['nonce'])
    print(f"Response: {response[:32]}...")
    
    print("\n=== Step 3: Verify Response ===")
    is_valid, details = protocol.verify_response(
        challenge['challenge_id'],
        response,
        embedding
    )
    print(f"Valid: {is_valid}")
    print(f"Details: {details}")
    
    print("\n=== Testing Secure Authentication Flow ===")
    flow = SecureAuthenticationFlow(protocol, similarity_threshold=0.85)
    
    # Initiate
    challenge = flow.initiate_authentication(device_id)
    print(f"Challenge initiated: {challenge['challenge_id']}")
    
    # Compute response
    response = protocol.compute_response(embedding, challenge['nonce'])
    
    # Complete authentication
    auth_embedding = embedding + 0.01 * torch.randn(128)  # Slightly different
    auth_embedding = torch.nn.functional.normalize(auth_embedding, p=2, dim=0)
    
    is_authenticated, auth_details = flow.complete_authentication(
        challenge['challenge_id'],
        response,
        auth_embedding,
        embedding
    )
    print(f"Authenticated: {is_authenticated}")
    print(f"Similarity: {auth_details['embedding_similarity']:.4f}")
    
    print("\n=== Testing Anti-Replay Protection ===")
    anti_replay = AntiReplayProtection(window_seconds=300)
    
    # First use - should pass
    is_fresh = anti_replay.check_and_record(response)
    print(f"First use - Fresh: {is_fresh}")
    
    # Second use - should fail (replay)
    is_fresh = anti_replay.check_and_record(response)
    print(f"Second use - Fresh: {is_fresh} (replay attack detected)")


if __name__ == "__main__":
    main()
