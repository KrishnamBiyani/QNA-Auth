import json

import numpy as np

from server.collector_store import save_browser_collection


def test_save_browser_collection_writes_manifest_and_samples(tmp_path):
    camera_sample = list(np.linspace(-2.0, 2.0, 160 * 120, dtype=np.float32))
    microphone_sample = list(np.linspace(-0.02, 0.03, 2048, dtype=np.float32))

    saved = save_browser_collection(
        base_dir=tmp_path,
        device_name="Phone 1",
        device_key="phone_1",
        source_samples={
            "camera": [camera_sample, camera_sample[::-1]],
            "microphone": [microphone_sample, [0.0] * 2048, microphone_sample],
        },
        session_id="day1_evening",
        environment_label="desk_evening",
        create_zip=True,
    )

    manifest = json.loads(saved.manifest_path.read_text(encoding="utf-8"))
    assert saved.folder_path.exists()
    assert saved.zip_path is not None and saved.zip_path.exists()
    assert manifest["device_key"] == "phone_1"
    assert manifest["session_id"] == "day1_evening"
    assert manifest["actual_counts"]["camera"] == 2
    assert manifest["actual_counts"]["microphone"] == 1
    assert (saved.folder_path / "camera" / "000.npy").exists()
    assert (saved.folder_path / "microphone" / "000.npy").exists()
