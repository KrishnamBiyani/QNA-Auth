"""
DB-backed challenge store for ChallengeResponseProtocol.
Replaces in-memory dict so challenges survive restarts and work across workers.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from .models import Challenge
from .session import get_session_factory


class DbChallengeStore:
    """Store challenges in the database. Interface: put(challenge_id, data), get(challenge_id), delete(challenge_id)."""

    def __init__(self):
        self._session_factory = get_session_factory()

    def put(self, challenge_id: str, data: Dict[str, Any]) -> None:
        """Store a challenge. data must have device_id, nonce, created_at, expires_at."""
        session = self._session_factory()
        try:
            row = Challenge(
                challenge_id=challenge_id,
                device_id=data["device_id"],
                nonce=data["nonce"],
                created_at=data["created_at"],
                expires_at=data["expires_at"],
            )
            session.add(row)
            session.commit()
        finally:
            session.close()

    def get(self, challenge_id: str) -> Optional[Dict[str, Any]]:
        """Return challenge data or None if not found."""
        session = self._session_factory()
        try:
            row = session.query(Challenge).filter(Challenge.challenge_id == challenge_id).first()
            if row is None:
                return None
            return {
                "device_id": row.device_id,
                "nonce": row.nonce,
                "created_at": row.created_at,
                "expires_at": row.expires_at,
            }
        finally:
            session.close()

    def delete(self, challenge_id: str) -> None:
        """Remove a challenge (after use or expiry)."""
        session = self._session_factory()
        try:
            session.query(Challenge).filter(Challenge.challenge_id == challenge_id).delete()
            session.commit()
        finally:
            session.close()
