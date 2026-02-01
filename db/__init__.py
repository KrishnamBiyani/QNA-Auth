"""
QNA-Auth database package.
Provides init_db(), get_db(), and models.
"""

from .models import Base, Device, Challenge, AuditLog
from .session import get_engine, get_session_factory, get_db


def init_db() -> None:
    """Create all tables. Safe to call on existing DB (no-op for existing tables)."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
