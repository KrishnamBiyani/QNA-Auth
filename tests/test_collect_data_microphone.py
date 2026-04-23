import numpy as np

from scripts.data.collect_data_for_training import _microphone_sample_quality


def test_microphone_sample_quality_rejects_near_silent_signal():
    arr = np.zeros(2048, dtype=np.float32)
    quality = _microphone_sample_quality(arr)
    assert quality["valid"] is False
    assert quality["reason"] == "near_silent"


def test_microphone_sample_quality_accepts_nontrivial_signal():
    arr = np.linspace(-0.01, 0.02, 2048, dtype=np.float32)
    quality = _microphone_sample_quality(arr)
    assert quality["valid"] is True
    assert quality["reason"] == "ok"
