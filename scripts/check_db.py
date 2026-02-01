"""
Verify the QNA-Auth database is working.
Run from project root: python scripts/check_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    from db import init_db
    from db.models import Device, Challenge
    from db.session import get_engine, get_session_factory

    print("Checking database...")
    try:
        import config
        print(f"  DB path: {getattr(config, 'DATABASE_URL', 'N/A')}")
    except Exception:
        pass
    init_db()
    engine = get_engine()
    print(f"  Using: {str(engine.url)}")
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        devices = session.query(Device).count()
        challenges = session.query(Challenge).count()
        print("  DB connection: OK")
        print(f"  Devices in DB: {devices}")
        print(f"  Active challenges: {challenges}")
        if devices > 0:
            for row in session.query(Device).limit(5):
                print(f"    - {row.device_id} (created {row.created_at})")
        print("Database is working.")
    except Exception as e:
        print(f"  DB error: {e}")
        sys.exit(1)
    finally:
        session.close()

if __name__ == "__main__":
    main()
