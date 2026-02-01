# QNA-Auth (Quantum Noise Assisted Authentication)

QNA-Auth is an experimental **device authentication** system that derives a **device "fingerprint" embedding** from high-entropy noise sources (QRNG / camera sensor noise / microphone noise / system jitter) and uses a Siamese-style embedding model + similarity thresholding to authenticate a device.

This repository includes:
- A **FastAPI backend** that exposes enrollment/authentication APIs and persists enrolled device embeddings.
- A **React + TypeScript frontend** that calls the backend and provides an interactive UI.
- **Noise collection modules** for QRNG + hardware entropy sources.
- A **feature extraction pipeline** that converts raw noise arrays into fixed-length feature vectors.
- A **PyTorch Siamese embedding model** plus training/evaluation utilities.

> Note: This is a capstone-style research project. The security properties are **educational/prototypical** and should be reviewed carefully before any real-world use.

---

## What the system does (high-level)

### Enrollment (creates the "stored reference")
1. Collect \(N\) raw noise samples from one or more sources (e.g., `qrng`, `camera`, `microphone`).
2. For each sample: extract a **feature dictionary** (stats/entropy/FFT/autocorr/complexity).
3. Convert features to a **fixed feature vector** (sorted feature names) and run it through the embedder.
4. Aggregate embeddings across samples (mean by default) to produce a **single device embedding**.
5. Save:
   - `auth/device_embeddings/<device_id>_embedding.pt` (PyTorch tensor)
   - `auth/device_embeddings/<device_id>_metadata.json` (metadata for UI + debugging)

### Authentication (verifies a new "fresh" reading)
1. Collect fresh noise samples from the selected sources.
2. Generate a fresh authentication embedding (average across samples).
3. Compare fresh embedding to stored embedding using a similarity metric (default **cosine**).
4. Authenticate if similarity >= threshold (default **0.85** in the server).

### Challenge/Response (anti-replay "nonce" protocol)
The backend also exposes `/challenge` and `/verify` endpoints that implement:
- A server-generated **nonce** (short-lived)
- A response derived from the stored embedding + nonce (HMAC-like)
- A second factor check: **embedding similarity** must still meet threshold

---

## Repository structure (map)

```text
QNA-Auth/
  auth/                  # Enrollment + authentication + challenge/response logic
  noise_collection/      # QRNG + camera + microphone + system jitter collectors
  preprocessing/         # Feature extraction + preprocessing utilities
  model/                 # Siamese model, training, evaluation
  dataset/               # Dataset builder for storing raw samples/metadata
  server/                # FastAPI app exposing REST endpoints
  frontend/              # React/Vite UI
  *.py / *.bat / *.sh    # Scripts for setup, testing, and demos
```

---

## Quick start (recommended)

You can follow the dedicated quickstart at `QUICKSTART.md`, but here is the consolidated version.

### Prerequisites
- **Python**: 3.8+ (3.10+ recommended)
- **Node.js**: 18+ (frontend)
- Optional hardware:
  - Webcam (for `camera` source)
  - Microphone (for `microphone` source)

### Backend setup

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
# source venv/bin/activate

pip install -r requirements.txt
python server/app.py
```

Backend runs on `http://localhost:8000`
- OpenAPI/Swagger UI: `http://localhost:8000/docs`

### Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:3000`

---

## Install, data, train, evaluate, and run server

### Install
- **Python**: 3.8+ (3.10+ recommended). Create a venv, activate it, then:
  ```bash
  pip install -r requirements.txt
  ```
- **Node.js**: 18+ for the frontend (`cd frontend && npm install && npm run dev`).
- Optional: CUDA for faster training (`pip install torch ... --index-url https://download.pytorch.org/whl/cu124`).

### Dataset layout
- Training/enrollment data lives under **`dataset/samples/`** (config: `config.DATA_DIR` or `STORAGE_CONFIG["dataset_dir"]`).
- Layout: `dataset/samples/json/` (one JSON metadata file per sample) and raw `.npy` files referenced in the JSON (e.g. `*_raw.npy`).
- Each JSON has `device_id`, `noise_source` (qrng/camera/microphone), `raw_data_path`, and other metadata.

### Collect data (for training)
- **Participants** (no repo needed): run the standalone script; install deps and run:
  ```bash
  pip install numpy requests opencv-python sounddevice
  python scripts/collect_data_for_training.py
  ```
  Then zip the created folder and send it. See `scripts/collect_data_for_training.py` docstring and `docs/DATA_COLLECTION.md` if present.
- **Ingest** received folders into the project dataset:
  ```bash
  python scripts/ingest_collected_data.py path/to/collection_folder1 [folder2 ...]
  ```
  This merges into `dataset/samples/`.

### Train
- Train on the dataset (canonical features, reproducible seeds, best checkpoint by validation loss):
  ```bash
  python scripts/train_and_evaluate.py --data-dir dataset/samples [--epochs 20] [--seed 42]
  ```
  Or quick demo: `python -m model.train` (subsamples to 20 samples).
- Trained model is written to `server/models/best_model.pt` (and feature pipeline metadata is saved with it).

### Run evaluation
- FAR, FRR, EER, threshold sweep, ROC, and ablation by noise source (QRNG / camera / mic / combined):
  ```bash
  python scripts/run_evaluation.py [--data-dir dataset/samples] [--model-path server/models/best_model.pt]
  ```
  Output: metrics and plots under `model/evaluation/` and an ablation table in the console. Use `--ablations-only` to skip full plots.

### Start server
- From project root:
  ```bash
  python server/app.py
  ```
  Server uses `config.MODEL_PATH` (default `server/models/best_model.pt`), `config.EMBEDDINGS_DIR`, and `config.DATABASE_URL`. See `config.example.py` and `docs/PROJECT_STATUS.md`.

---

## Configuration

### `config.example.py` (copy to `config.py`)
Copy `config.example.py` to `config.py` and adjust paths, CORS origins, and optional API protection:
- **CORS**: `CORS_CONFIG["allow_origins"]` — restrict to your frontend origin(s) in production (e.g. `["http://localhost:3000", "http://localhost:5173"]`).
- **API protection**: Set `API_KEY` (e.g. from `os.environ.get("QNA_AUTH_API_KEY")`) to require an `X-API-Key` header on enroll, authenticate, challenge, verify, and delete. Leave `None` for local dev without API key.
- **Rate limiting**: `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_WINDOW_SEC` limit auth-related requests per IP; set to `0` to disable.

### QRNG API key
`auth/enrollment.py` attempts to read `QRNG_API_KEY` from the environment, otherwise it uses a fallback string.

Set an environment variable to override:

```bash
# PowerShell
$env:QRNG_API_KEY="your_key_here"

# bash/zsh
export QRNG_API_KEY="your_key_here"
```

Security note: **do not commit real API keys**. Prefer `.env` + environment variables.

---

## API reference (backend)

Base URL: `http://localhost:8000`

### Health
- `GET /health`
  - Returns whether the model and core services initialized successfully.

### Enrollment
- `POST /enroll`
  - Body:
    - `device_name` (optional)
    - `num_samples` (10?200)
    - `sources` (list; e.g. `["qrng"]`, `["camera","microphone"]`)
  - Returns: `device_id`, status/message, and stored metadata.

### Authentication (simple)
- `POST /authenticate`
  - Body:
    - `device_id`
    - `sources`
    - `num_samples_per_source` (1?20)
  - Returns: authenticated boolean + details, or 401 on failure.

### Challenge/response
- `POST /challenge`
  - Body: `device_id`
  - Returns: `challenge_id`, `nonce`, `expires_at`

- `POST /verify`
  - Body:
    - `challenge_id`
    - `device_id`
    - `response` (signature string)
    - `noise_samples` (list of float arrays; fresh raw noise)
  - Returns: authenticated boolean + similarity + protocol details, or 401 on failure.

### Device management
- `GET /devices` ? list device IDs
- `GET /devices/{device_id}` ? read stored metadata JSON
- `DELETE /devices/{device_id}` ? delete `*_embedding.pt` + `*_metadata.json`

---

## Frontend behavior

The frontend calls backend endpoints via `axios`:
- Default base URL is `http://localhost:8000` (or `VITE_API_URL` if set).

Pages:
- `/` Home
- `/enroll` Enroll a device
- `/authenticate` Authenticate a device
- `/devices` List/view/delete enrolled devices

---

## How data flows through the code (end-to-end)

### Enrollment flow (important call chain)
1. `server/app.py` ? `POST /enroll`
2. `auth/enrollment.py` ? `DeviceEnroller.enroll_device(...)`
3. `noise_collection/*` ? collects raw samples (`np.ndarray`)
4. `preprocessing/features.py` ? `NoisePreprocessor.extract_all_features(...)`
5. `preprocessing/features.py` ? `FeatureVector.to_vector(...)`
6. `model/siamese_model.py` ? `DeviceEmbedder.embed(...)`
7. `auth/enrollment.py` ? aggregate embeddings + save files

### Authentication flow (simple endpoint)
1. `server/app.py` ? `POST /authenticate`
2. `auth/authentication.py` ? `DeviceAuthenticator.authenticate(...)`
3. Collect fresh noise per source (same noise collectors)
4. Generate fresh embedding ? compute similarity vs stored embedding

### Challenge/response flow
1. `server/app.py` ? `POST /challenge` ? `auth/challenge_response.py` creates nonce and stores it temporarily in-memory
2. `server/app.py` ? `POST /verify`:
   - loads stored embedding
   - generates fresh embedding from `noise_samples`
   - verifies nonce response signature and embedding similarity

---

## Scripts and common tasks

### Verify CUDA
- `python verify_cuda.py`

### Test hardware access
- `python test_hardware.py`
  - Uses OpenCV and `sounddevice` directly to confirm camera/mic permissions.

### Test collection + feature extraction
- `python test_collection.py`

### Test enrollment/auth via HTTP
- `python test_enrollment.py`
  - Note: this script sends `noise_sources` in JSON, but the server expects `sources`.
  - If it fails, update the payload keys or use the frontend/UI.

### Collect data via API for training
- `python collect_training_data.py`
  - Enrolls multiple ?devices? via repeated enrollment calls to build up stored embeddings.

### Train on real data and run evaluation
- `python scripts/train_and_evaluate.py [--data-dir dataset/samples] [--seed 42] [--epochs 20]`
  - Loads dataset from `dataset/samples` (canonical features, no subsampling).
  - Reproducible seeds; optional train/val split; saves best by validation loss and last-N checkpoints.
  - Runs evaluation (ROC, PR, optimal threshold) and deploys server checkpoint with feature pipeline metadata.

### Start scripts
- `setup.bat` / `setup.sh`: create venv, install deps, create folders, copy `config.py`
- `start.bat`: starts backend + frontend, but **contains a hard-coded Python path** (`D:/QNA-Auth/venv/...`). You will likely need to edit it to point to your local `venv`.

---

## Detailed file-by-file guide (what each file does)

This section explains the ?meaningful? files (source code + scripts). Generated artifacts (e.g. `frontend/package-lock.json`) are not described in depth.

### Root
- `README.md`: This documentation (project overview + guide).
- `QUICKSTART.md`: Shorter quickstart instructions.
- `requirements.txt`: Python dependencies (FastAPI, PyTorch, OpenCV, sounddevice, etc.).
- `config.example.py`: Example configuration template (copy to `config.py`).
- `setup.bat`: Windows bootstrap (venv + pip install + folder creation).
- `setup.sh`: macOS/Linux bootstrap (same intent as above).
- `start.bat`: Convenience starter for backend + frontend (hard-coded venv path to fix).
- `verify_cuda.py`: Prints CUDA availability and GPU details.
- `test_hardware.py`: Tests whether camera/mic can be opened/recorded.
- `test_collection.py`: Smoke test: camera/mic noise ? feature extraction ? vectorization.
- `test_enrollment.py`: HTTP smoke test for backend endpoints (payload keys may not match).
- `collect_training_data.py`: Uses the running backend to enroll multiple devices for training/demo.

### `server/` (FastAPI backend)
- `server/app.py`:
  - Creates the FastAPI app, configures CORS, initializes global services on startup.
  - Builds the preprocessing pipeline and detects the feature vector dimension dynamically.
  - Creates a `DeviceEmbedder` (loads `./model/checkpoints/best_model.pt` if it exists).
  - Exposes endpoints: `/health`, `/enroll`, `/authenticate`, `/challenge`, `/verify`, `/devices`, `/devices/{id}`, deletion.
- `server/routes.py`:
  - Placeholder router for future endpoints like `/stats` and `/metrics` (not currently used by `app.py`).

### `auth/` (authentication domain logic)
- `auth/__init__.py`: Re-exports the main classes for convenience imports.
- `auth/enrollment.py`:
  - `DeviceEnroller`: collect samples, extract features, embed, aggregate, and persist device embeddings + metadata.
  - Knows where embeddings are stored (`./auth/device_embeddings` by default).
- `auth/authentication.py`:
  - `DeviceAuthenticator`: collects fresh samples, produces an auth embedding, compares vs stored embedding (cosine by default).
  - `AuthenticationSession`: optional retry/session wrapper.
- `auth/challenge_response.py`:
  - `ChallengeResponseProtocol`: creates nonce challenges and verifies responses (in-memory active challenge store).
  - `SecureAuthenticationFlow`: combines challenge-response validity + embedding similarity check.
  - `AntiReplayProtection`: additional replay protection utility (not wired into the server endpoints by default).

### `noise_collection/` (entropy sources)
- `noise_collection/__init__.py`: Exposes collector classes: `QRNGClient`, `CameraNoiseCollector`, `MicrophoneNoiseCollector`, `SensorNoiseCollector`.
- `noise_collection/qrng_api.py`:
  - `QRNGClient`: fetches quantum random bytes (ANU by default; QRNG.org supported if key provided).
  - `fetch_multiple_samples` intentionally **does not** fallback to pseudo-random values.
- `noise_collection/camera_noise.py`:
  - `CameraNoiseCollector`: captures dark frames, extracts noise by subtracting a blurred image, returns flattened noise arrays.
- `noise_collection/mic_noise.py`:
  - `MicrophoneNoiseCollector`: records audio noise via `sounddevice` and returns raw sample arrays (or lists).
- `noise_collection/sensor_noise.py`:
  - `SensorNoiseCollector`: collects timing jitter + CPU usage + memory access timing + disk/network stats; can concatenate into a composite signature.
- `noise_collection/camera_noise.py`, `mic_noise.py`, `sensor_noise.py` each have a `main()` for manual testing.

### `preprocessing/` (feature extraction)
- `preprocessing/features.py`:
  - `NoisePreprocessor`: statistical features + Shannon entropy + FFT features + autocorrelation + complexity (approx entropy + Hurst exponent).
  - `FEATURE_VERSION` and `get_canonical_feature_names()`: canonical feature list used everywhere (preprocessing, training, server); saved with the model so serve matches train.
  - `FeatureVector`: converts feature dicts to fixed vectors; defaults to canonical feature names.
- `preprocessing/utils.py`:
  - Utility helpers: sliding window, augmentation, pad/truncate, batch processing, SNR, merging multiple sources, etc.

### `model/` (ML model + training/evaluation)
- `model/siamese_model.py`:
  - `EmbeddingNetwork` and `SiameseNetwork`
  - `TripletLoss` + `ContrastiveLoss`
  - `DeviceEmbedder`: wraps the model and provides `embed()` + `compute_similarity()`.
- `model/train.py`:
  - `TripletDataset` and `PairDataset` to generate training tuples/pairs from per-device feature vectors.
  - `ModelTrainer`: training loop, best checkpoint by validation loss, optional last-N checkpoints (`save_last_n`).
  - `set_seed()`: reproducible training (torch, numpy, DataLoader generator).
  - Canonical feature names and `FEATURE_VERSION` are saved with the deployed model for train/serve consistency.
- `model/evaluate.py`:
  - `ModelEvaluator`: generates embeddings, computes similarity distributions, finds a threshold, plots ROC/PR curves, and produces a report.

### `dataset/` (dataset builder / storage)
- `dataset/builder.py`:
  - `DatasetBuilder`: stores raw noise arrays as `.npy`, metadata as JSON, and a summary row in CSV.
  - Useful for building training corpora beyond the ?store one embedding per device? approach.

### `frontend/` (React UI)
- `frontend/package.json`: frontend scripts (`dev`, `build`, `preview`, `lint`) and dependencies.
- `frontend/vite.config.ts`: Vite dev server config; runs on port 3000 and defines a proxy for `/api` (not required by current UI).
- `frontend/src/services/api.ts`: axios client + typed API wrappers (`enrollDevice`, `authenticateDevice`, device management).
- `frontend/src/App.tsx`: router + navigation.
- `frontend/src/pages/HomePage.tsx`: landing page.
- `frontend/src/pages/EnrollPage.tsx`: enrollment form + status display.
- `frontend/src/pages/AuthenticatePage.tsx`: device dropdown + authenticate flow.
- `frontend/src/pages/DevicesPage.tsx`: list devices, view metadata, delete device.
- `frontend/src/components/ui/background-beams.tsx`: background visual component used by layout.

---

## Troubleshooting

### Camera / Microphone not working
- Run `python test_hardware.py` to diagnose device availability and permissions.
- For camera: ensure no other app is using the webcam.
- For mic: ensure Windows microphone permissions allow Python to record input.

### QRNG failures
- Ensure you have internet connectivity.
- Expect timeouts/rate limits; retry later.
- If you have an API key, set `QRNG_API_KEY` and `noise_collection/qrng_api.py` will use authenticated requests.

### Low authentication accuracy
- Increase samples during enrollment (`num_samples`).
- Use multiple sources (camera + microphone typically adds more device-specific entropy than QRNG alone).
- Consider training and loading a better checkpoint in `model/checkpoints/best_model.pt`.
- Tune similarity threshold in `server/app.py` (currently hard-coded).

---

## Notes on limitations (important)

- The backend uses a **database** (SQLite by default) for devices, challenges, and audit logs; embeddings are still stored as `.pt` files in `auth/device_embeddings/`.
- `server/app.py` loads a model from `config.MODEL_PATH` (default `server/models/best_model.pt`) if it exists; otherwise it runs with random weights.
- Optional API key and rate limiting are configured via `config.py`; set `API_KEY` to require `X-API-Key` on protected endpoints.

# QNA-Auth - Quantum Noise Assisted Authentication

A novel authentication system that uses quantum noise samples and machine learning to authenticate devices in a secure, non-reproducible way.

## ?? Features

- **Quantum Random Number Generation**: Fetches true quantum noise from ANU QRNG service
- **Multi-Source Noise Collection**: Camera dark frames, microphone ambient noise, and system sensors
- **Siamese Neural Network**: Creates unique, non-invertible device embeddings
- **Challenge-Response Protocol**: Secure authentication with nonce-based verification
- **FastAPI Backend**: RESTful API for enrollment and authentication
- **React TypeScript Frontend**: Modern UI for device management

## ??? Project Structure

```
qna-auth/
??? noise_collection/          # Noise sampling modules
?   ??? qrng_api.py           # Quantum RNG client
?   ??? camera_noise.py       # Camera dark frame capture
?   ??? mic_noise.py          # Microphone noise capture
?   ??? sensor_noise.py       # System sensor jitter
??? dataset/                   # Dataset management
?   ??? builder.py            # Dataset builder
?   ??? samples/              # Stored samples
??? preprocessing/             # Feature extraction
?   ??? features.py           # Statistical & FFT features
?   ??? utils.py              # Preprocessing utilities
??? model/                     # ML models
?   ??? siamese_model.py      # Siamese network architecture
?   ??? train.py              # Training script
?   ??? evaluate.py           # Model evaluation
??? auth/                      # Authentication modules
?   ??? enrollment.py         # Device enrollment
?   ??? authentication.py     # Authentication logic
?   ??? challenge_response.py # Challenge-response protocol
??? server/                    # FastAPI backend
?   ??? app.py               # Main application
?   ??? routes.py            # Additional routes
??? frontend/                  # React TypeScript UI
    ??? src/
    ?   ??? pages/           # Page components
    ?   ??? services/        # API service
    ?   ??? App.tsx          # Main app
    ??? package.json

```

## ?? Getting Started

### Prerequisites

- Python 3.8+
- Node.js 18+ (for frontend)
- Webcam (optional, for camera noise)
- Microphone (optional, for audio noise)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd QNA-Auth
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install frontend dependencies**
```bash
cd frontend
npm install
```

### Running the Application

1. **Start the backend server**
```bash
python server/app.py
```
The API will be available at `http://localhost:8000`

2. **Start the frontend (in a new terminal)**
```bash
cd frontend
npm run dev
```
The UI will be available at `http://localhost:3000`

## ?? Usage

### 1. Enroll a Device

Navigate to the Enroll page and:
- Choose noise sources (QRNG, camera, microphone)
- Set number of samples (50-100 recommended)
- Click "Enroll Device"
- Save the generated device ID

### 2. Authenticate a Device

Navigate to the Authenticate page and:
- Select an enrolled device
- Choose the same noise sources used during enrollment
- Click "Authenticate"
- View authentication result and similarity score

### 3. Manage Devices

Navigate to the Devices page to:
- View all enrolled devices
- See device metadata
- Delete devices

## ?? API Endpoints

- `GET /health` - Health check
- `POST /enroll` - Enroll a new device
- `POST /authenticate` - Authenticate a device
- `POST /challenge` - Create authentication challenge
- `POST /verify` - Verify challenge response
- `GET /devices` - List enrolled devices
- `GET /devices/{id}` - Get device details
- `DELETE /devices/{id}` - Delete device

## ?? Testing Individual Modules

Each module can be tested independently:

```bash
# Test quantum noise collection
python -m noise_collection.qrng_api

# Test camera noise collection
python -m noise_collection.camera_noise

# Test microphone noise collection
python -m noise_collection.mic_noise

# Test preprocessing
python -m preprocessing.features

# Test model
python -m model.siamese_model

# Test enrollment
python -m auth.enrollment

# Test authentication
python -m auth.authentication
```

## ?? Training the Model

To train the Siamese network with your own data:

1. Collect data from multiple devices:
```python
from dataset.builder import DatasetBuilder
from noise_collection import QRNGClient

builder = DatasetBuilder()
qrng = QRNGClient()

# Collect samples for each device
for device_id in device_ids:
    samples = qrng.fetch_multiple_samples(num_samples=50)
    builder.add_batch(device_id, 'qrng', samples)
```

2. Train the model:
```python
from model.train import ModelTrainer, TripletDataset
from model.siamese_model import SiameseNetwork
import torch

# Create model
model = SiameseNetwork(input_dim=50, embedding_dim=128)

# Create dataset
dataset = TripletDataset(features_by_device, samples_per_epoch=1000)
loader = DataLoader(dataset, batch_size=32)

# Train
trainer = ModelTrainer(model, loss_type='triplet')
trainer.train(loader, epochs=50)
```

### Reproducible training

To reproduce results (e.g. for papers or ablations), the training pipeline fixes all random seeds:

- **Torch**: `torch.manual_seed(seed)` and `torch.cuda.manual_seed_all(seed)` if CUDA is used.
- **NumPy**: `np.random.seed(seed)`.
- **DataLoader**: a `torch.Generator` with `manual_seed(seed)` is passed so shuffle order is deterministic; with `num_workers > 0`, `worker_init_fn` seeds each worker with `seed + worker_id`.

Default seed is `42`. When you run `python -m model.train` or `scripts/train_and_evaluate.py` with the same data and seed, you should get the same training curves and checkpoint metrics. Document the seed and feature pipeline version (e.g. `FEATURE_VERSION` in `preprocessing/features.py`) when reporting results.

## ?? Security Considerations

- **Quantum Randomness**: Uses ANU QRNG for true quantum random numbers
- **Non-Invertible Embeddings**: Embeddings cannot be reversed to recover original noise
- **Challenge-Response**: Prevents replay attacks with time-limited nonces
- **Threshold-Based**: Configurable similarity threshold for authentication
- **Multi-Factor**: Combines multiple noise sources for robustness

## ?? Configuration

Copy `config.example.py` to `config.py` and adjust settings:

```python
# Authentication threshold
AUTH_CONFIG = {
    "similarity_threshold": 0.85,  # Adjust based on requirements
    "similarity_metric": "cosine",
}

# Model architecture
MODEL_CONFIG = {
    "input_dim": 50,
    "embedding_dim": 128,
    "hidden_dims": [256, 256, 128]
}
```

## ?? Architecture

### Noise Collection
- **QRNG**: Fetches quantum random bits from ANU quantum service
- **Camera**: Captures sensor noise from dark frames
- **Microphone**: Records ambient noise and self-noise
- **Sensors**: Collects timing jitter and system noise

### Feature Extraction
- Statistical features (mean, std, skewness, kurtosis)
- FFT-based frequency features
- Entropy calculations
- Autocorrelation analysis

### Machine Learning
- Siamese network with shared weights
- Triplet loss or contrastive loss
- L2-normalized embeddings
- Cosine similarity for verification

### Authentication
- Multi-sample collection for robustness
- Embedding aggregation (mean/median)
- Challenge-response protocol
- Configurable threshold

## ?? Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ?? License

This project is licensed under the MIT License.

## ?? Acknowledgments

- ANU Quantum Random Numbers: https://qrng.anu.edu.au/
- PyTorch team for the deep learning framework
- FastAPI for the modern web framework

## ?? Contact

For questions or support, please open an issue on GitHub.

---

**Built with quantum randomness and machine learning** ???
