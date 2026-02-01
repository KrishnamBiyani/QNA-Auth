"""
SQLAlchemy models for QNA-Auth.
Stores devices (metadata + embedding path), challenges, and optional audit log.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Integer, String, Text, Index
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Device(Base):
    """Enrolled device: metadata and path to embedding file."""

    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(64), unique=True, nullable=False, index=True)
    device_name = Column(String(255), nullable=True)
    embedding_path = Column(String(512), nullable=False)  # path relative to embeddings_dir or absolute
    metadata_json = Column(Text, nullable=True)  # JSON: num_samples, sources, feature_dimension, etc.
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<Device(device_id={self.device_id!r})>"


class Challenge(Base):
    """Active challenge for challenge-response auth (replaces in-memory dict)."""

    __tablename__ = "challenges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    challenge_id = Column(String(64), unique=True, nullable=False, index=True)
    device_id = Column(String(64), nullable=False, index=True)
    nonce = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)

    __table_args__ = (Index("ix_challenges_expires_at", "expires_at"),)

    def __repr__(self) -> str:
        return f"<Challenge(challenge_id={self.challenge_id!r}, device_id={self.device_id!r})>"


class AuditLog(Base):
    """Optional audit log for enrollment, auth, delete (for research/reproducibility)."""

    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String(32), nullable=False, index=True)  # enroll, authenticate, delete, challenge_create, verify
    device_id = Column(String(64), nullable=True, index=True)
    details = Column(Text, nullable=True)  # JSON
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<AuditLog(action={self.action!r}, device_id={self.device_id!r})>"
