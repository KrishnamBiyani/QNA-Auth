from scripts.training.experiment_utils import SampleRecord, split_by_device_session


def _record(device_id: str, session_id: str, sample_id: str) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        device_id=device_id,
        source="camera",
        raw_path="dummy",
        session_id=session_id,
        timestamp="2026-01-01T00:00:00Z",
    )


def test_split_by_device_session_uses_last_session_for_test():
    records = [
        _record("dev_a", "s1", "a1"),
        _record("dev_a", "s2", "a2"),
        _record("dev_a", "s3", "a3"),
        _record("dev_b", "s1", "b1"),
        _record("dev_b", "s2", "b2"),
        _record("dev_b", "s3", "b3"),
    ]

    splits = split_by_device_session(records, min_sessions_per_device=3, val_ratio=0.2)

    test_ids = {r.sample_id for r in splits["test"]}
    train_ids = {r.sample_id for r in splits["train"]}
    val_ids = {r.sample_id for r in splits["val"]}

    assert test_ids == {"a3", "b3"}
    assert val_ids == {"a2", "b2"}
    assert train_ids == {"a1", "b1"}
