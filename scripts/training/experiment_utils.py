from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm

from preprocessing.features import NoisePreprocessor, FeatureVector, get_canonical_feature_names


def parse_sources_arg(value: str | None) -> List[str] | None:
    if value is None:
        return None
    parts = [part.strip().lower() for part in value.split(",") if part.strip()]
    return parts or None


@dataclass
class SampleRecord:
    sample_id: str
    device_id: str
    source: str
    raw_path: Path
    session_id: str
    timestamp: str


def load_sample_records(
    data_dir: Path,
    source_filter: str | List[str] | None = None,
    max_records: int | None = None,
    seed: int = 42,
) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    json_dir = data_dir / "json"
    allowed_sources = None
    if isinstance(source_filter, str):
        allowed_sources = {source_filter.lower()}
    elif source_filter is not None:
        allowed_sources = {str(item).lower() for item in source_filter}
    for jf in sorted(json_dir.glob("*.json")):
        try:
            meta = json.loads(jf.read_text(encoding="utf-8"))
            source = str(meta.get("noise_source", "")).lower()
            if allowed_sources is not None and source not in allowed_sources:
                continue
            rel = str(meta.get("raw_data_path", "")).lstrip("/\\")
            raw_path = data_dir / rel
            if not raw_path.exists():
                alt = data_dir.parent.parent / rel
                raw_path = alt if alt.exists() else raw_path
            if not raw_path.exists():
                continue
            records.append(
                SampleRecord(
                    sample_id=str(meta.get("sample_id", jf.stem)),
                    device_id=str(meta["device_id"]),
                    source=source,
                    raw_path=raw_path,
                    session_id=str(meta.get("session_id", meta.get("collection_folder", "unknown_session"))),
                    timestamp=str(meta.get("timestamp", meta.get("created_at", ""))),
                )
            )
        except Exception:
            continue
    if max_records is not None and len(records) > max_records:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(records), size=max_records, replace=False)
        records = [records[int(i)] for i in idx]
    return records


def build_features(records: List[SampleRecord], normalize: bool = True) -> Dict[str, List[Tuple[str, np.ndarray]]]:
    preprocessor = NoisePreprocessor(normalize=normalize)
    converter = FeatureVector(get_canonical_feature_names())
    out: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for r in tqdm(records, desc="Extracting features"):
        raw = np.load(r.raw_path)
        feat = preprocessor.extract_all_features(raw)
        vec = converter.to_vector(feat)
        out.setdefault(r.device_id, []).append((r.sample_id, vec))
    return out


def split_by_device(
    records: List[SampleRecord],
    seed: int,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> Dict[str, List[SampleRecord]]:
    rng = np.random.default_rng(seed)
    by_device: Dict[str, List[SampleRecord]] = {}
    for r in records:
        by_device.setdefault(r.device_id, []).append(r)
    device_ids = sorted(by_device.keys())
    rng.shuffle(device_ids)

    n = len(device_ids)
    n_test = max(2 if n >= 4 else 1, int(round(n * test_ratio)))
    n_test = min(n_test, max(1, n - 2)) if n >= 3 else min(n_test, n)
    n_val = max(1, int(round(n * val_ratio))) if n >= 5 and val_ratio > 0 else 0
    if n - n_test - n_val < 2:
        n_val = max(0, n - n_test - 2)
    test_devices = set(device_ids[:n_test])
    val_devices = set(device_ids[n_test:n_test + n_val])
    train_devices = set(device_ids[n_test + n_val:])
    if not train_devices:
        train_devices = set(device_ids[n_test:])
        val_devices = set()

    splits = {"train": [], "val": [], "test": []}
    for r in records:
        if r.device_id in test_devices:
            splits["test"].append(r)
        elif r.device_id in val_devices:
            splits["val"].append(r)
        else:
            splits["train"].append(r)
    return splits


def split_by_session(records: List[SampleRecord], train_sessions: List[str], test_sessions: List[str]) -> Dict[str, List[SampleRecord]]:
    train_set, test_set = set(train_sessions), set(test_sessions)
    splits = {"train": [], "val": [], "test": []}
    for r in records:
        if r.session_id in test_set:
            splits["test"].append(r)
        elif r.session_id in train_set:
            splits["train"].append(r)
    return splits


def assert_no_leakage(splits: Dict[str, List[SampleRecord]]) -> None:
    ids = {k: {s.sample_id for s in v} for k, v in splits.items()}
    for a in ids:
        for b in ids:
            if a >= b:
                continue
            overlap = ids[a].intersection(ids[b])
            if overlap:
                raise ValueError(f"Leakage detected between {a} and {b}: {len(overlap)} overlapping samples")


def save_split_artifacts(splits: Dict[str, List[SampleRecord]], output_dir: Path, split_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "split_name": split_name,
        "counts": {k: len(v) for k, v in splits.items()},
        "splits": {
            k: [
                {
                    "sample_id": s.sample_id,
                    "device_id": s.device_id,
                    "source": s.source,
                    "session_id": s.session_id,
                    "raw_path": str(s.raw_path),
                    "timestamp": s.timestamp,
                }
                for s in v
            ]
            for k, v in splits.items()
        },
    }
    path = output_dir / f"split_{split_name}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return path


def features_from_split(
    splits: Dict[str, List[SampleRecord]],
    normalize: bool = True,
    fast_features: bool = False,
    augment_camera_train: bool = False,
    camera_aug_copies: int = 3,
    seed: int = 42,
) -> Dict[str, Dict[str, List[np.ndarray]]]:
    split_feature_map: Dict[str, Dict[str, List[np.ndarray]]] = {}
    preprocessor = NoisePreprocessor(normalize=normalize, fast_mode=fast_features)
    converter = FeatureVector(get_canonical_feature_names())
    rng = np.random.default_rng(seed)

    def augment_camera_sample(raw: np.ndarray) -> List[np.ndarray]:
        variants: List[np.ndarray] = []
        base = np.asarray(raw, dtype=np.float32).reshape(-1)
        if base.size < 8:
            return variants

        for _ in range(max(0, camera_aug_copies)):
            arr = base.copy()

            # Mild gain / offset jitter.
            gain = float(rng.uniform(0.92, 1.08))
            offset = float(rng.normal(0.0, 0.02 * (np.std(arr) + 1e-6)))
            arr = (arr * gain) + offset

            # Small additive sensor-like noise.
            arr = arr + rng.normal(0.0, 0.01 * (np.std(arr) + 1e-6), size=arr.shape).astype(np.float32)

            # Mild blur / smoothing to mimic tiny exposure or denoise variation.
            if arr.size >= 5 and rng.random() < 0.7:
                kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
                kernel = kernel / kernel.sum()
                arr = np.convolve(arr, kernel, mode="same").astype(np.float32)

            # Slight circular shift to avoid exact alignment dependence.
            if arr.size >= 16 and rng.random() < 0.8:
                shift = int(rng.integers(-max(1, arr.size // 64), max(2, arr.size // 64 + 1)))
                arr = np.roll(arr, shift)

            # Random partial masking to mimic hot/dead regions or crop loss.
            if arr.size >= 32 and rng.random() < 0.5:
                width = max(4, arr.size // 40)
                start = int(rng.integers(0, max(1, arr.size - width)))
                arr[start:start + width] *= float(rng.uniform(0.7, 1.0))

            variants.append(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))

        return variants

    for split_name, recs in splits.items():
        by_device: Dict[str, List[np.ndarray]] = {}
        for r in recs:
            raw = np.load(r.raw_path)
            vec = converter.to_vector(preprocessor.extract_all_features(raw))
            by_device.setdefault(r.device_id, []).append(vec)

            if split_name == "train" and augment_camera_train and r.source == "camera":
                for aug_raw in augment_camera_sample(raw):
                    aug_vec = converter.to_vector(preprocessor.extract_all_features(aug_raw))
                    by_device.setdefault(r.device_id, []).append(aug_vec)
        split_feature_map[split_name] = by_device
    return split_feature_map


def build_mlp_embeddings(
    train_features: Dict[str, List[np.ndarray]],
    target_features: Dict[str, List[np.ndarray]],
    embedding_dim: int,
    seed: int,
) -> Dict[str, List[np.ndarray]]:
    """Fit a lightweight autoencoder-like MLPRegressor on train vectors; output hidden projection surrogate.

    This stays intentionally simple for capstone baseline comparisons.
    """
    x_train = []
    for vecs in train_features.values():
        x_train.extend(vecs)
    if not x_train:
        return {}
    X = np.asarray(x_train, dtype=np.float32)
    model = MLPRegressor(
        hidden_layer_sizes=(64, embedding_dim),
        activation="relu",
        solver="adam",
        random_state=seed,
        max_iter=200,
    )
    model.fit(X, X)
    out: Dict[str, List[np.ndarray]] = {}
    for dev, vecs in target_features.items():
        arr = np.asarray(vecs, dtype=np.float32)
        transformed = model.predict(arr)
        norms = np.linalg.norm(transformed, axis=1, keepdims=True) + 1e-8
        transformed = transformed / norms
        out[dev] = [row for row in transformed]
    return out


def bootstrap_metric_ci(values: List[float], seed: int, n_bootstrap: int = 500, alpha: float = 0.95) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    lo = np.percentile(means, (1 - alpha) / 2 * 100)
    hi = np.percentile(means, (1 + alpha) / 2 * 100)
    return {"mean": float(np.mean(arr)), "ci_low": float(lo), "ci_high": float(hi)}
