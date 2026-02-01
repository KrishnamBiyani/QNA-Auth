"""
Database engine and session for QNA-Auth.
Uses config.DATABASE_URL (SQLite by default).
"""

import logging
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


def get_engine():
    """Create or return the global engine. Thread-safe after first call."""
    global _engine
    if _engine is not None:
        return _engine
    try:
        import config
        url = config.DATABASE_URL
    except Exception:
        url = "sqlite:///./data/qna_auth.db"
        (Path(".").resolve() / "data").mkdir(parents=True, exist_ok=True)
    # SQLite: allow multi-threaded access (FastAPI may use multiple threads)
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    _engine = create_engine(url, connect_args=connect_args)
    logger.info("Database engine created: %s", url.split("?")[0])
    return _engine


def get_session_factory():
    """Return a session factory bound to the engine."""
    global _SessionLocal
    if _SessionLocal is not None:
        return _SessionLocal
    engine = get_engine()
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _SessionLocal


def get_db() -> Session:
    """Yield a DB session (for use as FastAPI dependency)."""
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
