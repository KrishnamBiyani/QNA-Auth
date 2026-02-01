# QNA-Auth: Dataset Guide for Research-Level Evaluation

This document spells out **what data you need**, **how much**, **how to store it**, and **how it connects to training and evaluation** so the project can support a research-level capstone.

---

## 1. What the Dataset Is

Each **sample** is one noise capture (a 1D array of numbers) with:

- **Device ID** – which physical or logical “device” produced it (e.g. `device_001`, or “PC A webcam”).
- **Noise source** – `qrng`, `camera`, or `microphone`.
- **Session** (recommended) – e.g. “session_1” vs “session_2” on different days, so you can do cross-session evaluation.

One **record** = one sample + its metadata (device_id, source, timestamp, session_id, split, path to raw/features).

**Goal:** Train a model to map “noise from device X” → embedding, and evaluate: *same device* (genuine) vs *different device* (impostor). You need enough devices and enough samples per device to train and to compute FAR/FRR/EER reliably.

---

## 2. Minimum Scale for Research-Level Claims

These are **minimum** targets so you can report meaningful metrics and ablations.

| Item | Minimum | Better | Why |
|------|---------|--------|-----|
| **Devices** | 5 | 10+ | You need multiple “identities”; leave-one-device-out or 80/20 split by device. |
| **Samples per device per source** | 50 | 100+ | Enough for enrollment-like aggregation and for train/val/test per device. |
| **Noise sources** | 1 (e.g. QRNG only) | All 3 (QRNG, camera, mic) | Ablations: “QRNG only vs camera only vs combined.” |
| **Sessions** | 1 | 2+ (different days/times) | Cross-session = “train session 1, test session 2” shows generalization. |
| **Total samples (rough)** | 5 × 50 × 1 = 250 | 10 × 100 × 3 = 3000 | More is better for confidence intervals and ablations. |

**Practical suggestion:** Start with **5 devices × 50 samples × 1 source (QRNG)** so you can run the full pipeline and evaluation. Then add a second source (camera or mic) and/or more samples for ablations.

---

## 3. Recommended Directory Layout

Keep **raw** (sensor/API output) and **processed** (features, splits) separate, and version the dataset so you can say “model v1 was trained on dataset v1.0.”

```
dataset/
├── README.md                    # Short description + version + how to collect
├── manifest.json                # Dataset version, feature_version, creation date, split summary
├── raw/                         # Raw noise arrays (e.g. .npy or .npz)
│   └── v1/
│       ├── by_device/
│       │   ├── device_001/
│       │   │   ├── qrng/
│       │   │   │   ├── session_1/
│       │   │   │   │   ├── sample_000.npy
│       │   │   │   │   └── ...
│       │   │   │   └── session_2/
│       │   │   ├── camera/
│       │   │   └── microphone/
│       │   ├── device_002/
│       │   └── ...
│       └── index.csv             # sample_id, device_id, source, session_id, path, timestamp
├── processed/                   # Extracted features (same feature pipeline as server)
│   └── v1/
│       ├── features.npz         # Or per-device/source .npz; see below
│       ├── train.csv            # sample_id, device_id, source, session_id, split=train
│       ├── val.csv
│       └── test.csv
└── samples/                     # (Optional) keep existing builder output here for compatibility
    ├── noise_samples.csv
    └── json/
```

- **manifest.json** – e.g. `dataset_version`, `feature_version`, `num_devices`, `num_samples`, `splits` (counts), `created_at`. Lets you tie “Table 2” to a specific dataset version.
- **index.csv** – one row per raw sample: `sample_id`, `device_id`, `noise_source`, `session_id`, `path` (relative to `raw/v1/`), `timestamp`. Optional: `split` if you assign it at collection time.
- **processed/v1/** – features extracted with the **same** code and feature list as the server (see ROADMAP: “Fixed, versioned feature pipeline”). `train/val/test` splits by **device** (or by session) so evaluation is realistic.

You can keep using your existing `DatasetBuilder` and `dataset/samples/` (e.g. `noise_samples.csv` + JSON + `.npy`) and add `manifest.json` + `index.csv` + a script that builds `processed/v1/` from `samples/`; the important part is that evaluation and training always point at a **versioned** dataset (e.g. `processed/v1/`).

---

## 4. Metadata Schema (Per Sample)

Each sample should be traceable with at least:

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | string | Unique ID (e.g. `device_001_qrng_session_1_000`) |
| `device_id` | string | Device identity |
| `noise_source` | string | `qrng` \| `camera` \| `microphone` |
| `session_id` | string | e.g. `session_1`, `session_2` (optional but recommended) |
| `timestamp` | string (ISO) | When the sample was captured |
| `path` | string | Path to raw array (e.g. `.npy`) relative to dataset root |
| `split` | string | `train` \| `val` \| `test` (assigned by you, e.g. by device or by session) |
| `feature_version` | string | e.g. `v1` – must match the feature extractor used in training and server |

Optional: `sample_length`, `mean`, `std`, `entropy` (you already have these in the current CSV); add them to the index or keep them in a separate metadata table.

---

## 5. Splits and Evaluation Strategy

- **Split by device** – e.g. 80% of devices for train, 10% val, 10% test. No device appears in two splits. Prevents “same device, different samples” leakage and reflects “unseen device” at test time.
- **Leave-one-device-out (LODO)** – Train on devices 1..N-1, test on device N; rotate so each device is test once. Report mean ± std EER across folds. Strong for a capstone.
- **Cross-session** – If you have 2+ sessions: train on session 1, test on session 2 (same devices). Shows that the model generalizes across time/environment.

Recommended for the report:

1. **Primary:** Split by device (e.g. 60/20/20), train once, report FAR/FRR/EER and threshold sweep on the **test** set.
2. **Robustness:** LODO and/or cross-session (if you have 2 sessions), report EER (and optionally FAR/FRR at a fixed threshold).

Splits should be **deterministic** (e.g. seed + device list sorted) and recorded in `manifest.json` or a `splits.csv` so you can reproduce “Table 2.”

---

## 6. Collection Workflow

1. **Decide devices** – e.g. 5 PCs, or 5 “logical devices” (same PC, different browser/user profiles). Label them `device_001` … `device_005`.
2. **Decide sources** – Start with QRNG only (no hardware); add camera/mic if available.
3. **Per device, per source:**  
   - Collect 50–100 samples in one “session” (same day, same env).  
   - If possible, repeat on another day → second session.
4. **Save** – Use your existing `DatasetBuilder` or a small script that writes:
   - Raw arrays under `dataset/raw/v1/...` (or keep current `samples/` layout).
   - One row per sample in `index.csv` with `sample_id`, `device_id`, `noise_source`, `session_id`, `path`, `timestamp`.
5. **Assign splits** – e.g. script that assigns `train`/`val`/`test` by device (or by session) and writes `processed/v1/train.csv`, `val.csv`, `test.csv` (or a single CSV with a `split` column).
6. **Extract features** – Run the **same** feature pipeline as the server (versioned), write features to `processed/v1/` (e.g. `features.npz` keyed by `sample_id`, or one file per split). Record `feature_version` in manifest.
7. **Write manifest.json** – version, counts, feature_version, split counts.

This gives you a single, versioned dataset that training and evaluation scripts can load by path (e.g. `config.dataset_path = "dataset/processed/v1"`).

---

## 7. How the Dataset Feeds Training and Evaluation

- **Training** – Load `train` (and optionally `val`) by device. Build triplets (anchor, positive same device, negative different device) or pairs with labels. Train siamese model with fixed seed; checkpoint best model.
- **Evaluation** – Load `test` (or LODO fold):
  - **Genuine scores:** for each test device, compare “enrollment” embedding (e.g. mean of N samples) to “auth” embedding (other samples from same device).
  - **Impostor scores:** compare enrollment of device A to auth samples from device B (B ≠ A).
  - Vary threshold → compute FAR, FRR, EER; plot ROC and/or FAR/FRR vs threshold.
- **Ablations** – Repeat with subsets: “QRNG only,” “camera only,” “mic only,” “all sources.” Report EER (and optionally FAR/FRR) per condition in a table.

So: **one versioned dataset** → **one training run** → **one evaluation script** → **tables and figures** for the report.

---

## 8. Versioning and Reproducibility

- **Dataset version** – e.g. `v1`, `v1.1` (after adding more samples). Stored in `manifest.json` and (optionally) in config when you run training/eval.
- **Feature version** – Must match the feature list and preprocessing code (see ROADMAP). Store in manifest and in the model checkpoint so “model X + dataset Y + features Z” is unambiguous.
- **Optional:** Use DVC or a hash of `index.csv` + raw file list as a simple “dataset fingerprint” so you can re-download or verify the same data later.

---

## 9. Quick Checklist Before “Research-Level” Evaluation

- [ ] At least **5 devices**, **50+ samples per device** (per source you use).
- [ ] **Metadata** for every sample: device_id, source, session_id (if 2+ sessions), path, timestamp.
- [ ] **Splits** by device (or LODO) and optionally cross-session; splits recorded (e.g. in CSV or manifest).
- [ ] **Features** extracted with the **same** versioned pipeline as the server; stored under `processed/<version>/`.
- [ ] **manifest.json** (or equivalent) with dataset version, feature version, and split counts.
- [ ] README in `dataset/` describing how to collect and how to run the script that builds `processed/` from raw.

Once this is in place, you can plug the dataset into the **evaluation script** (FAR/FRR/EER, threshold sweep, ablations) and into the **report** (“We use dataset v1: N devices, M samples, split by device…”).
