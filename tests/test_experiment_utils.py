import numpy as np

from scripts.training.experiment_utils import SampleRecord, bootstrap_metric_ci, features_from_split


def test_bootstrap_metric_ci_shape():
    vals = [0.1, 0.2, 0.3, 0.4]
    out = bootstrap_metric_ci(vals, seed=42, n_bootstrap=50)
    assert "mean" in out and "ci_low" in out and "ci_high" in out
    assert out["ci_low"] <= out["mean"] <= out["ci_high"]


def test_bootstrap_metric_ci_empty():
    out = bootstrap_metric_ci([], seed=42, n_bootstrap=10)
    assert out["mean"] == 0.0
    assert out["ci_low"] == 0.0
    assert out["ci_high"] == 0.0


def test_features_from_split_microphone_augmentation_increases_train_examples(tmp_path):
    mic_a = tmp_path / "mic_a.npy"
    mic_b = tmp_path / "mic_b.npy"
    np.save(mic_a, np.sin(np.linspace(0.0, 12.0, 512, dtype=np.float32)))
    np.save(mic_b, np.cos(np.linspace(0.0, 9.0, 512, dtype=np.float32)))

    splits = {
        "train": [
            SampleRecord(
                sample_id="a1",
                device_id="device_a",
                source="microphone",
                raw_path=mic_a,
                session_id="s1",
                timestamp="2026-04-23T10:00:00",
            ),
            SampleRecord(
                sample_id="b1",
                device_id="device_b",
                source="microphone",
                raw_path=mic_b,
                session_id="s1",
                timestamp="2026-04-23T10:00:00",
            ),
        ],
        "val": [],
        "test": [],
    }

    baseline = features_from_split(splits, normalize=True, fast_features=True, seed=42)
    augmented = features_from_split(
        splits,
        normalize=True,
        fast_features=True,
        augment_microphone_train=True,
        microphone_aug_copies=2,
        seed=42,
    )

    assert len(baseline["train"]["device_a"]) == 1
    assert len(baseline["train"]["device_b"]) == 1
    assert len(augmented["train"]["device_a"]) == 3
    assert len(augmented["train"]["device_b"]) == 3
