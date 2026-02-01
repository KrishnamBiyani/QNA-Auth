"""
Backfill the database from existing enrollment files in auth/device_embeddings.
Use this if you enrolled devices before the DB was wired, or if check_db shows 0 devices
but you have *_embedding.pt / *_metadata.json files.

Run from project root: python scripts/backfill_db_from_files.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from db import init_db
    from db.models import Device
    from db.session import get_session_factory

    try:
        import config
        storage_dir = Path(config.EMBEDDINGS_DIR)
    except Exception:
        storage_dir = PROJECT_ROOT / "auth" / "device_embeddings"

    if not storage_dir.exists():
        print(f"Storage dir not found: {storage_dir}")
        sys.exit(1)

    init_db()
    SessionLocal = get_session_factory()
    session = SessionLocal()

    added = 0
    for emb_file in storage_dir.glob("*_embedding.pt"):
        device_id = emb_file.stem.replace("_embedding", "")
        meta_file = storage_dir / f"{device_id}_metadata.json"
        if not meta_file.exists():
            print(f"  Skip {device_id}: no metadata file")
            continue
        existing = session.query(Device).filter(Device.device_id == device_id).first()
        if existing:
            print(f"  Skip {device_id}: already in DB")
            continue
        with open(meta_file, "r") as f:
            metadata = json.load(f)
        session.add(Device(
            device_id=device_id,
            device_name=metadata.get("device_name"),
            embedding_path=f"{device_id}_embedding.pt",
            metadata_json=json.dumps(metadata)
        ))
        added += 1
        print(f"  Added: {device_id}")

    session.commit()
    session.close()
    print(f"Backfill done. Added {added} device(s). Run python scripts/check_db.py to verify.")


if __name__ == "__main__":
    main()
