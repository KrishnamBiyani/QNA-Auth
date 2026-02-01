"""
Initialize the QNA-Auth database (create tables).
Run from project root: python scripts/init_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db import init_db

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
    print("Tables: devices, challenges, audit_log")
