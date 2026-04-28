# QNA-Auth Project Knowledge Base

This document is a teaching guide for understanding and defending the QNA-Auth capstone project. Read it in order if you are preparing for an external review or viva.

## 1. Project in One Minute

QNA-Auth is a noise-based device verification system. Instead of authenticating a person directly, it verifies whether a device looks similar to a previously enrolled device by comparing sensor-noise fingerprints from the camera and microphone.

The system collects raw sensor samples, extracts numerical signal features, passes those features through a Siamese neural network to create embeddings, stores enrollment templates, and later compares fresh authentication samples against the stored templates. The result is not a cryptographic identity proof. It is a statistical similarity decision with confidence bands.

The strongest way to describe the project is:

> QNA-Auth is a multi-source sensor fingerprinting prototype that uses learned embeddings and statistical decision rules to verify whether a live device resembles an enrolled device.

Do not claim it is spoof-proof, quantum identity, or a replacement for passwords, passkeys, or certificates.

## 2. Problem Statement

Traditional authentication often depends on what the user knows, has, or is: passwords, OTPs, biometrics, or hardware tokens. QNA-Auth explores a supporting factor: whether the client device itself has measurable sensor-noise characteristics that can be reused for verification.

Real cameras and microphones contain small hardware and environmental imperfections. These imperfections can appear as repeatable patterns in captured data. The project tries to learn those patterns well enough to distinguish same-device samples from different-device samples.

The practical problem is:

1. Enroll a device by collecting several camera and microphone samples.
2. Convert those samples into stable feature templates.
3. Later collect fresh samples from a claiming device.
4. Compare fresh embeddings to stored templates.
5. Accept, reject, or mark the result uncertain.

## 3. Core Concepts

### Device Verification vs Identification

Verification answers: "Is this the claimed device?"

Identification answers: "Which device is this among all enrolled devices?"

QNA-Auth mainly performs verification. The user sends a `device_id`, and the system checks whether fresh samples match that device. The code also includes an identification-margin guard that compares the claimed device against other enrolled devices, but the main flow is still verification.

### Sensor Fingerprint

A sensor fingerprint is a statistical pattern produced by a physical sensor. For this project, it is not a visible image or an audio recording. It is a numerical feature representation extracted from noise-like camera and microphone data.

### Feature Vector

A feature vector is a fixed-length numerical representation of a raw sample. QNA-Auth extracts statistical, entropy, spectral, autocorrelation, and complexity features.

### Embedding

An embedding is the output of the neural network. It is a compact vector, currently 128 dimensions by default, where samples from the same device should be close and samples from different devices should be farther apart.

### Template

A template is a stored enrollment embedding or group of embeddings. The project treats embeddings as biometric-like templates used only for similarity matching, not as secret credentials.

### Similarity Score

A similarity score measures how close a fresh authentication embedding is to the stored template. QNA-Auth primarily uses cosine similarity. Higher score means stronger match.

### Confidence Band

The project does not only return true or false. It classifies results into:

- `strong_accept`: high-confidence match.
- `uncertain`: borderline result; collect more samples or use fallback authentication.
- `reject`: low-confidence match.

This is important because real-world sensor data varies with lighting, microphone noise, browser behavior, and environment.

## 4. Repository Map

Important directories:

- `server/`: FastAPI application and REST endpoints.
- `auth/`: enrollment, authentication, challenge-response logic.
- `preprocessing/`: raw-signal cleaning and feature extraction.
- `model/`: Siamese network, training, and evaluation.
- `frontend/`: React/Vite client for enrollment, authentication, and device management.
- `noise_collection/`: Python-side camera and microphone collection.
- `dataset/`: sample storage and dataset manifests.
- `db/`: SQLAlchemy models for devices, challenges, and audit logs.
- `scripts/`: diagnostics, data collection, training, evaluation, and reporting utilities.
- `tests/`: pytest tests for authentication guards, drift gates, evaluation utilities, and feature behavior.
- `docs/reports/`: project report, runbook, readiness, dataset, and review material.

## 5. Runtime Architecture

The runtime architecture has these layers:

1. Frontend collects camera and microphone samples in the browser.
2. FastAPI receives samples through REST endpoints.
3. Preprocessor converts raw arrays into fixed feature vectors.
4. DeviceEmbedder converts feature vectors into neural embeddings.
5. Enroller stores template banks and metadata.
6. Authenticator compares fresh embeddings against stored templates.
7. Challenge-response layer optionally adds nonce-bound HKDF verification.
8. Database stores device metadata, active challenges, and audit logs.

Text flow:

```text
Browser sensors
  -> client_samples JSON
  -> FastAPI endpoint
  -> NoisePreprocessor
  -> FeatureVector
  -> Siamese/DeviceEmbedder
  -> template comparison
  -> confidence band and API response
```

## 6. Frontend Layer

The frontend is a React/Vite app in `frontend/`.

Main files:

- `frontend/src/pages/EnrollPage.tsx`: collects samples and enrolls a device.
- `frontend/src/pages/AuthenticatePage.tsx`: collects fresh samples and authenticates a device.
- `frontend/src/pages/DevicesPage.tsx`: lists and manages enrolled devices.
- `frontend/src/services/api.ts`: wraps backend calls using Axios.
- `frontend/src/services/collectors.ts`: browser camera and microphone collectors.

The browser collectors use Web APIs:

- Camera: `navigator.mediaDevices.getUserMedia({ video: ... })`
- Microphone: `navigator.mediaDevices.getUserMedia({ audio: true })`
- Camera samples are converted to grayscale and downsampled.
- Microphone samples are captured from an `AudioContext` and downsampled.

The frontend does not make the authentication decision. It only collects samples and sends them to the backend.

## 7. Backend API Layer

The backend is a FastAPI app in `server/app.py`.

Main endpoints:

- `GET /health`: checks model/component status.
- `GET /stats`: returns basic runtime statistics.
- `POST /enroll`: enrolls a device.
- `POST /authenticate`: performs direct similarity-based authentication.
- `POST /challenge`: creates a nonce-bound challenge.
- `POST /verify`: verifies challenge response plus fresh sensor evidence.
- `GET /devices`: lists enrolled devices.
- `GET /devices/{device_id}`: returns metadata for one device.
- `DELETE /devices/{device_id}`: deletes a device.
- `POST /collector/api/save`: saves browser-collected raw datasets.

The API includes optional protection:

- CORS configured through `CORS_CONFIG`.
- Optional `API_KEY` via `X-API-Key`.
- Per-IP rate limiting for sensitive endpoints.
- Audit logging through the `AuditLog` table.

## 8. Enrollment Flow

Enrollment is implemented mainly in `auth/enrollment.py`.

Step-by-step:

1. User chooses device name and sources.
2. System collects or receives camera/microphone samples.
3. Each raw sample is converted to features using `NoisePreprocessor`.
4. Features are converted to a canonical ordered vector using `FeatureVector`.
5. Feature vectors are embedded using `DeviceEmbedder`.
6. Embeddings are aggregated into source-level and combined templates.
7. Metadata is saved, including source profiles and template information.
8. Device row is stored in SQLite through SQLAlchemy.

Why multiple samples are used:

- Single samples are noisy.
- Multiple samples allow averaging or template-bank creation.
- Aggregation reduces random noise and improves repeatability.

Template-bank idea:

- Instead of storing only one mean embedding, the system can store multiple templates per source.
- During authentication, it compares against a bank and averages the top-k scores.
- This handles normal variation better than one rigid vector.

## 9. Authentication Flow

Authentication is implemented mainly in `auth/authentication.py`.

Step-by-step:

1. Client sends claimed `device_id` and fresh samples.
2. Backend loads stored record and metadata.
3. Runtime samples are checked against simple source profiles, such as RMS statistics.
4. Fresh samples are converted to feature vectors.
5. Fresh feature vectors are embedded.
6. Per-source embeddings are compared to stored source templates.
7. Scores are fused with source weights.
8. Result is classified into `strong_accept`, `uncertain`, or `reject`.
9. Drift update may run only after sufficiently strong matches.

Important runtime safeguards:

- Profile guard catches obvious RMS mismatch.
- Required-source logic can enforce that certain sources must be present.
- Identification-margin logic can reject if another enrolled device scores too close.
- Drift update is gated to strong matches to reduce poisoning risk.

## 10. Multi-Source Fusion

QNA-Auth supports camera and microphone as active runtime sources.

The general fusion formula is:

```text
combined_score = sum(source_score * source_weight) / sum(active_weights)
```

The README describes a recommended demo framing of camera `0.7` and microphone `0.3`. The current `config.py` uses camera `0.85` and microphone `0.15`. If asked, explain that weights are configuration parameters and can be tuned from validation metrics. Camera receives more weight because current evaluation indicates it is stronger than microphone in this dataset.

Why not give both equal weight?

- Different sensors have different stability.
- If microphone is more environment-sensitive, overweighting it can increase false reject or false accept risk.
- Weighting allows the stronger modality to dominate while still using the weaker modality as supporting evidence.

## 11. Feature Extraction

Feature extraction is implemented in `preprocessing/features.py`.

The canonical feature pipeline version is `FEATURE_VERSION = "2.0"`.

The preprocessor:

1. Converts input to a 1D float array.
2. Replaces NaN and infinity with safe values.
3. Centers data by subtracting the mean.
4. Optionally normalizes using standard normalization.
5. Uses bounded analysis length for expensive transforms.
6. Extracts raw-domain and normalized-domain features.

Feature categories:

- Statistical: mean, standard deviation, variance, min, max, median, range, RMS, peak factor.
- Distributional: skewness, kurtosis, quartiles, IQR.
- Entropy: Shannon entropy.
- Frequency-domain: FFT dominant frequency, spectral centroid, spectral spread, spectral entropy, spectral flatness, low/mid/high frequency power.
- Autocorrelation: first zero crossing, short-lag averages, decay.
- Complexity: zero-crossing rate, approximate entropy, Hurst exponent.

Why feature extraction is needed:

- Raw sensor arrays have variable size and are too large.
- Neural network expects fixed-size input.
- Features preserve useful statistical structure while reducing noise and dimensionality.

## 12. Model Design

The neural model is implemented in `model/siamese_model.py`.

Main classes:

- `EmbeddingNetwork`: MLP that maps feature vectors to embeddings.
- `SiameseNetwork`: shared embedding network for anchor, positive, and negative samples.
- `TripletLoss`: encourages anchor-positive distance to be smaller than anchor-negative distance by a margin.
- `ContrastiveLoss`: alternative pairwise metric-learning loss.
- `DeviceEmbedder`: high-level wrapper for embedding and similarity computation.

Default model configuration:

- Input dimension: 50.
- Embedding dimension: 128.
- Hidden dimensions: `[256, 256, 128]`.
- Activation: Tanh.
- Output: L2-normalized embedding.

Why Siamese learning?

The goal is not classification into fixed known devices. The goal is similarity learning. A Siamese/triplet setup learns an embedding space where same-device samples are close and different-device samples are farther apart. This is better for verification because new devices can be enrolled without retraining the final classifier.

## 13. Training Pipeline

Training code is in `model/train.py` and `scripts/training/`.

Main idea:

1. Load dataset records.
2. Extract or load feature vectors.
3. Build triplets or pairs.
4. Train the Siamese network.
5. Save model checkpoint.
6. Evaluate similarity scores on held-out data.

Triplet format:

- Anchor: sample from device A.
- Positive: different sample from device A.
- Negative: sample from device B.

Triplet loss:

```text
loss = max(distance(anchor, positive) - distance(anchor, negative) + margin, 0)
```

If the negative is already far enough, the loss is zero. If the negative is too close, the model is penalized.

## 14. Evaluation Metrics

Evaluation code is in `model/evaluate.py` and `scripts/training/`.

Important metrics:

- Accuracy: total correct decisions.
- Precision: among accepted samples, how many were genuine.
- Recall or TPR: how many genuine samples were accepted.
- FAR: false acceptance rate; impostors accepted as genuine.
- FRR: false rejection rate; genuine users rejected.
- EER: equal error rate; point where FAR and FRR are approximately equal.
- ROC-AUC: ranking quality across thresholds.
- PR-AUC: precision-recall quality across thresholds.

Why FAR matters for security:

False accepts are dangerous because an impostor gets in.

Why FRR matters for usability:

False rejects are bad because legitimate devices fail.

QNA-Auth uses high thresholds and uncertainty bands because accepting uncertain sensor matches is risky.

## 15. Current Evaluation Snapshot

From existing metrics files:

- Camera model `camera_real_aug_v2`: ROC-AUC about `0.858`, EER about `0.268`, deployed FAR about `0.062` at threshold about `0.980`.
- Microphone model `microphone_real_aug_v2`: ROC-AUC about `0.710`, EER about `0.364`, deployed FAR about `0.305` at threshold about `0.970`.

These numbers show the prototype is promising but not production-grade. Camera performs better than microphone in the current data. Microphone is more sensitive to environment and has weaker separation.

Important defense statement:

> The project demonstrates a working research prototype and reports its limitations honestly. It should be used as an auxiliary signal, not as a standalone production authenticator.

## 16. Confidence Bands and Thresholds

Current confidence defaults:

- Strong accept: `>= 0.97`.
- Uncertain: `0.92` to `0.97`.
- Reject: `< 0.92`.

Why use bands?

- Sensor readings vary.
- A single threshold hides uncertainty.
- Borderline decisions should not silently become security decisions.

Recommended action:

- Strong accept: allow the device match.
- Uncertain: collect more samples or use fallback authentication.
- Reject: deny the device match.

## 17. Drift and Re-Enrollment

Sensor behavior can drift over time due to:

- Lighting changes.
- Microphone environment.
- Camera exposure and focus changes.
- Hardware aging.
- Browser/device pipeline changes.

QNA-Auth includes rolling re-enrollment with EMA updates. EMA means exponential moving average:

```text
new_template = alpha * fresh_embedding + (1 - alpha) * old_template
```

The system only updates after strong matches and can require multiple strong matches. This is important because updating on weak or malicious samples could poison the stored template.

## 18. Challenge-Response Hardening

Challenge-response is implemented in `auth/challenge_response.py`.

Purpose:

- Prevent replay of old responses.
- Bind verification to a fresh server nonce.
- Avoid treating the embedding directly as a password.

Flow:

1. Server creates a challenge with a random nonce.
2. Challenge is stored with expiry.
3. Fresh authentication embedding is generated.
4. Stable feature bytes are derived through coarse quantization.
5. HKDF derives a MAC key from stable template bytes, nonce, and server secret.
6. HMAC validates that the fresh sample matches the stored template under the current nonce.
7. Challenge is deleted after use.

Key statement:

> The embedding is not used directly as a credential. It is transformed into a stable feature template and combined with a nonce and server secret through HKDF.

Replay defense:

- Nonces are random.
- Challenges expire.
- Challenges are single-use.
- A captured old response should not work for a new nonce.

Residual risk:

- If an attacker can generate live samples close enough to the enrolled sensor fingerprint, the statistical matching layer may still be attacked.

## 19. Database Layer

Database code is in `db/`.

Tables:

- `devices`: device ID, name, embedding path, metadata JSON, creation time.
- `challenges`: active challenge ID, device ID, nonce, expiry.
- `audit_log`: enrollment/auth/delete/challenge/verify events.

The default database is SQLite at `data/qna_auth.db`, configured through `DATABASE_URL`.

Why store challenge state in DB?

- In-memory challenges disappear on restart.
- DB-backed challenges support more reliable operation and multiple workers.

## 20. Data Collection

Dataset documentation is in `docs/reports/DATASET.md`.

Good dataset practice:

- Collect multiple devices.
- Collect multiple samples per device.
- Record source, session, timestamp, path, and metadata.
- Use train/test splits that avoid leakage.
- Prefer cross-session testing for realism.

Current active runtime sources are camera and microphone. Older QRNG references are historical or dataset-related and are not the active runtime verification pipeline.

## 21. Security Model

The project considers:

- Replay attacks: mitigated by nonce-bound challenge flow.
- Naive synthetic noise: expected to fail if it does not reproduce higher-dimensional structure.
- Adaptive mimicry: still a real residual risk.
- Same device in different environment: may become uncertain or rejected.
- Template leakage: mitigated partly because templates are similarity features, but still sensitive and should be protected.

What the system does not solve:

- Malware on the client.
- Fully adaptive model-inversion attacks.
- Perfect spoof resistance.
- Cryptographic proof of physical identity.

## 22. Privacy and Ethics

Sensor samples can be sensitive because they may reveal information about a user's environment or device. Good practice:

- Store only what is necessary.
- Prefer templates over raw recordings in production.
- Protect templates like biometric data.
- Explain consent and collection purpose.
- Provide deletion through `DELETE /devices/{device_id}`.

## 23. Testing Strategy

Tests are in `tests/`.

Existing test themes:

- Authentication margin guard.
- Multi-template authentication.
- Profile guard.
- Collector store.
- Drift update gate.
- Experiment/session split leakage.
- Feature extraction scaling.
- Model evaluation behavior.

These tests support the main review story: the project is not only a demo UI; it has guardrails and reproducibility checks.

## 24. Demo Workflow

Typical demo:

1. Start backend:

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

2. Start frontend:

```bash
cd frontend
npm install
npm run dev
```

3. Open frontend.
4. Enroll a device with camera and microphone.
5. Authenticate the same device.
6. Show result confidence band.
7. Optionally show `/docs`, `/health`, `/devices`, and generated report artifacts.

Pre-demo check:

```bash
python scripts/diagnostics/demo_preflight.py --base-url http://127.0.0.1:8000
```

## 25. How to Explain the Full System in Viva

Use this answer:

> The project verifies devices using camera and microphone noise fingerprints. During enrollment, the frontend or backend collects multiple sensor samples. The backend extracts fixed statistical and spectral features, converts them into neural embeddings using a Siamese network, and stores source-specific templates. During authentication, fresh samples go through the same pipeline and are compared against stored templates using cosine similarity. The per-source scores are fused with configurable weights. The final score is mapped to strong accept, uncertain, or reject. For replay resistance, the challenge endpoint creates a nonce and the verify endpoint derives an HMAC key using HKDF from the stable template, nonce, and server secret.

## 26. Most Important Limitations

Be direct about limitations:

- Current dataset is small for production claims.
- Microphone performance is weaker and environment-sensitive.
- Sensor fingerprints are statistical, not cryptographic.
- Adaptive spoofing remains a residual risk.
- Browser capture pipelines can vary by device, OS, and browser.
- More cross-session and cross-environment evaluation is needed.

Strong defense:

> The project intentionally uses confidence bands and claim boundaries because a realistic biometric-like system should expose uncertainty instead of pretending every score is absolute proof.

## 27. Future Work

Good future work items:

- Larger dataset with more physical devices.
- Multi-day cross-session evaluation.
- Better microphone denoising and calibration.
- Threshold calibration per source.
- Template encryption at rest.
- Stronger anti-poisoning policy for drift updates.
- Compare against classical ML baselines.
- Add passkey/password fallback in uncertain band.
- Add production-grade API authentication and deployment hardening.

## 28. Deep Technical System Walkthrough

This section explains the project at implementation depth. Use it when the examiner asks how the system actually works internally, not just what it does at a high level.

The backend application starts in `server/app.py`. On startup, it initializes the ML pipeline, database, enroller, authenticator, and challenge-response protocol. The system first checks whether source-specific model paths exist in `SOURCE_MODEL_PATHS`. If camera and microphone checkpoints are available, each source can get its own `DeviceEmbedder` and its own `FeatureVector` standardization parameters. This matters because camera and microphone signals have different distributions, so a single model may not represent both equally well.

If source-specific checkpoints are not available, the backend falls back to a shared model path from `MODEL_PATH` or `MODEL_CONFIG["model_path"]`. If no trained model exists, it can initialize a model randomly, but that is only useful for wiring tests or demos of the software path. For meaningful verification, a trained checkpoint must exist.

The startup process also loads checkpoint metadata:

- `feature_names`: the exact ordered list of input features used during training.
- `feature_mean`: training-set mean for feature standardization.
- `feature_scale`: training-set scale or standard deviation for feature standardization.
- `input_dim`: number of model input features.
- `embedding_dim`: output embedding dimension.
- preprocessing flags such as normalization and fast mode.

This is important because training and inference must be identical. If runtime extracts the same feature names but in a different order, the neural network receives invalid inputs. If runtime standardization differs from training standardization, similarity scores become unreliable. The checkpoint metadata prevents those issues.

The application then creates:

- `NoisePreprocessor`: converts raw arrays to feature dictionaries.
- `FeatureVector`: converts feature dictionaries to ordered standardized arrays.
- `DeviceEnroller`: handles collection, feature extraction, embedding creation, template storage, metadata, and drift updates.
- `DeviceAuthenticator`: handles runtime samples, source-profile checks, similarity scoring, confidence-band decisions, margin checks, and optional drift updates.
- `ChallengeResponseProtocol`: creates nonce challenges and validates HKDF/HMAC responses.
- `SecureAuthenticationFlow`: combines challenge verification with similarity-band interpretation.

The result is a stateful FastAPI app where the API layer is thin and the core logic lives in the `auth`, `preprocessing`, and `model` modules.

## 29. Detailed Data Flow: Enrollment

Enrollment starts at `POST /enroll`.

Input schema:

```json
{
  "device_name": "Alice Laptop",
  "num_samples": 50,
  "sources": ["camera", "microphone"],
  "client_samples": {
    "camera": [[...], [...]],
    "microphone": [[...], [...]]
  }
}
```

The endpoint accepts two collection modes. In client-sample mode, the frontend captures arrays and sends them directly. In backend-collection mode, Python collectors can collect from local camera and microphone hardware. In the browser demo, client-sample mode is normally the important path.

Technical enrollment pipeline:

1. Validate API controls, rate limits, and demo-mode source restrictions.
2. Determine sample count. In demo mode, configured demo sample count can override request count.
3. Normalize incoming client samples into `np.float32` arrays.
4. Optionally save raw samples through `DatasetBuilder` for research dataset construction.
5. Extract features source-by-source using `NoisePreprocessor.extract_all_features`.
6. Convert feature dictionaries to ordered vectors using the source's `FeatureVector`.
7. Convert each vector into an embedding using the source's `DeviceEmbedder`.
8. Create source-level mean embeddings.
9. Create source-level template banks by chunking feature vectors.
10. Create combined templates by weighted combination of source templates.
11. Create one combined embedding by weighted source fusion.
12. Build source profiles such as RMS mean and RMS standard deviation.
13. Save the template bundle as a PyTorch file.
14. Save metadata JSON.
15. Add device metadata to the database.

The template bundle is versioned. A version-2 template bundle contains:

```text
combined_embedding
source_embeddings
combined_templates
source_templates
template_version
```

This is better than storing only one vector because it preserves both fused and source-specific evidence. During authentication, source-specific matching can tell whether camera, microphone, or both are responsible for the score.

## 30. Detailed Data Flow: Authentication

Authentication starts at `POST /authenticate`.

Input schema:

```json
{
  "device_id": "abc123",
  "sources": ["camera", "microphone"],
  "num_samples_per_source": 5,
  "client_samples": {
    "camera": [[...], [...]],
    "microphone": [[...], [...]]
  }
}
```

Technical authentication pipeline:

1. Validate API controls and source restrictions.
2. Load metadata for the claimed `device_id`.
3. Reject if requested sources do not match enrolled sources.
4. Normalize client samples to `np.float32`.
5. Count samples and reject if no samples are available.
6. Run source-profile validation against enrollment metadata.
7. Build a fresh authentication profile from runtime samples.
8. Load stored template record from disk.
9. Score fresh source embeddings against stored source template banks.
10. Fuse source scores using effective source weights.
11. Compute confidence band from fused score.
12. Compare claimed device score against best other device if margin guard is enabled.
13. Check per-source decisions and required-source policy.
14. Return authenticated only if the fused band is `strong_accept`, margin passes, and per-source checks pass.
15. If authenticated and drift update is enabled, update templates after the configured strong-match gate.

The important detail is that `authenticated = true` is not based only on `similarity >= threshold`. It also depends on margin and per-source validity. This prevents the system from accepting a high fused score if one required source is missing or rejected.

## 31. Raw Browser Collection Details

The frontend browser collectors are in `frontend/src/services/collectors.ts`.

Camera collection:

- Requests video permission using `navigator.mediaDevices.getUserMedia`.
- Prefers rear/environment camera on mobile.
- Waits briefly so auto-exposure can stabilize.
- Draws a frame into a canvas.
- Reads pixel data using `getImageData`.
- Converts RGB to grayscale using luminosity weighting.
- Downsamples the flattened grayscale array to a fixed length.

The grayscale conversion is approximately:

```text
gray = 0.299 * R + 0.587 * G + 0.114 * B
```

Microphone collection:

- Requests audio permission.
- Creates an `AudioContext`.
- Connects a `MediaStreamAudioSourceNode`.
- Uses a script processor to collect samples for a duration.
- Routes through a zero-gain node to avoid audible feedback.
- Downsamples collected audio to a fixed length.

Downsampling is used for practical reasons:

- Smaller JSON payloads.
- Faster feature extraction.
- More predictable runtime.
- Same approximate input length across devices.

The backend still performs feature extraction and normalization. The frontend collection step should be treated as data acquisition, not as trusted preprocessing.

## 32. Feature Engineering in Detail

The feature extractor intentionally combines raw-domain features and normalized-domain features.

Raw-domain features preserve amplitude information. This can capture sensor gain, brightness, exposure artifacts, or audio energy differences. Examples include:

- `raw_std`
- `raw_variance`
- `raw_range`
- `raw_rms`
- `raw_peak_factor`
- `raw_shannon_entropy`

Normalized-domain features preserve shape independent of absolute scale. This helps when the same device is captured under slightly different amplitude conditions. Examples include:

- `norm_skewness`
- `norm_kurtosis`
- `norm_spectral_centroid`
- `norm_spectral_entropy`
- `norm_autocorr_decay`
- `norm_zero_crossing_rate`

The extractor first ensures the input is a safe 1D float array. It handles bad numerical values by replacing NaN and infinity with zero. This prevents feature extraction from crashing and prevents invalid values from reaching the model.

Centering:

```text
centered = raw - mean(raw)
```

Standard normalization:

```text
normalized = (centered - mean(centered)) / (std(centered) + epsilon)
```

The small epsilon avoids division by zero.

Entropy:

Entropy is computed from a histogram. A highly concentrated signal has lower entropy; a more spread-out signal has higher entropy. In this project, entropy is not used as "randomness proof." It is simply one statistical descriptor among many.

FFT features:

FFT converts the signal into frequency components. The system computes spectral centroid, spread, entropy, flatness, and frequency-band power. These features help detect repeated periodic structure or spectral shape differences between sensors.

Autocorrelation:

Autocorrelation measures how much a signal resembles shifted versions of itself. Sensor artifacts may have short-lag correlation patterns. The project captures first zero crossing, average autocorrelation over short windows, and decay.

Complexity features:

Zero-crossing rate captures how often the signal changes sign after normalization. Approximate entropy and Hurst exponent are more expensive descriptors of irregularity and long-range behavior. In fast mode, expensive features can be skipped or limited for demo speed.

## 33. Feature Standardization and Checkpoint Compatibility

The `FeatureVector` class does more than create a list of numbers. It enforces a stable feature order and can apply saved training-set standardization.

Conceptually:

```text
vector[i] = features[feature_names[i]]
standardized[i] = (vector[i] - feature_mean[i]) / feature_scale[i]
```

This is a major reproducibility point. If a model was trained using standardized features and runtime uses unstandardized features, the embedding distribution changes. The model may still produce vectors, but similarity scores become meaningless.

Exam answer:

> The checkpoint stores the feature list and standardization parameters, and the server reconstructs `FeatureVector` from that metadata. This keeps train-time and runtime preprocessing consistent.

## 34. Embedding Network Internals

The embedding network is a multilayer perceptron:

```text
input feature vector
  -> Linear
  -> Tanh
  -> Linear
  -> Tanh
  -> Linear
  -> Tanh
  -> Linear
  -> L2 normalization
  -> embedding
```

The network is not trying to reconstruct the signal. It learns a metric space. The output vector is useful only because distances or similarities between embeddings should reflect same-device vs different-device relationships.

L2 normalization means:

```text
z_normalized = z / ||z||
```

For two normalized vectors, cosine similarity becomes closely related to their dot product:

```text
cosine(a, b) = (a . b) / (||a|| ||b||)
```

Since both norms are one:

```text
cosine(a, b) = a . b
```

This simplifies similarity interpretation.

## 35. Triplet Learning in More Depth

Triplet learning is used because the model should learn relative similarity.

For each triplet:

- Anchor `A`: one sample from a device.
- Positive `P`: another sample from the same device.
- Negative `N`: a sample from another device.

The desired condition is:

```text
distance(A, P) + margin < distance(A, N)
```

Triplet loss:

```text
L = max(d(A, P) - d(A, N) + margin, 0)
```

Interpretation:

- If negative is far enough, loss is zero.
- If negative is too close, loss is positive.
- Training pushes same-device embeddings together and different-device embeddings apart.

Why this is better than ordinary classification:

- Classification would learn labels for known training devices.
- Verification needs to enroll new devices after training.
- Metric learning creates a reusable embedding space where new templates can be added without retraining the final layer.

## 36. Template Aggregation and Chunking

Enrollment creates multiple embeddings from multiple samples. Aggregation reduces noise.

Mean aggregation:

```text
template = normalize(mean(embedding_1, embedding_2, ..., embedding_n))
```

Template chunking:

```text
samples -> chunks of size AUTH_TEMPLATE_CHUNK_SIZE
each chunk -> mean embedding template
keep at most AUTH_MAX_TEMPLATES_PER_SOURCE
```

Why chunking helps:

- A single mean can over-smooth different valid capture conditions.
- Multiple templates represent multiple local modes of the same device.
- Top-k matching can tolerate session variation.

During scoring, the authenticator compares a fresh embedding against all templates for that source, sorts the scores, takes the top `AUTH_TEMPLATE_TOP_K`, and averages them. This means the system asks: "Does the fresh sample match any of the known valid template regions strongly enough?"

## 37. Source Fusion Mathematics

Source fusion combines camera and microphone evidence.

Let:

- `s_camera` = camera similarity.
- `s_microphone` = microphone similarity.
- `w_camera` = camera weight.
- `w_microphone` = microphone weight.

Then:

```text
score = (s_camera * w_camera + s_microphone * w_microphone) /
        (w_camera + w_microphone)
```

The implementation normalizes effective weights over available sources. If only one valid source is active, it gets full effective weight. If no configured positive weights exist, the fallback is uniform weighting.

Important nuance:

The system also stores per-source band decisions. A good fused score is not always enough if a required source is missing or if per-source policy rejects a modality. This is useful for stricter deployment modes.

## 38. Confidence Decision Logic

The basic confidence function is:

```text
if score >= strong_threshold:
    band = "strong_accept"
elif score >= uncertain_threshold:
    band = "uncertain"
else:
    band = "reject"
```

But final authentication is stricter:

```text
authenticated =
    band == "strong_accept"
    and margin_check_passed
    and per_source_check_passed
```

This is a useful defense point. It shows the project separates scoring from final policy. The model produces evidence; the authenticator applies security policy.

## 39. Profile Guard Internals

The profile guard is a simple statistical sanity check before accepting an embedding comparison.

For each source, enrollment metadata stores:

- RMS mean.
- RMS standard deviation.

At authentication, runtime samples produce:

- current RMS mean.
- current RMS standard deviation.

The guard computes:

```text
allowed_delta = max(AUTH_PROFILE_GUARD_MIN_DELTA,
                    AUTH_PROFILE_GUARD_Z * enrolled_std)
delta = abs(current_mean - enrolled_mean)
```

If:

```text
delta > allowed_delta
```

then the sample profile is rejected.

Why this exists:

- It catches silent microphone captures.
- It catches completely different amplitude regimes.
- It blocks some obvious invalid samples before ML comparison.

Limitation:

RMS is not a full fingerprint. It is only a cheap guardrail, not the main authentication method.

## 40. Identification Margin Guard

The claimed-device verification score can be high, but another enrolled device may score similarly. The margin guard compares:

```text
observed_margin = claimed_similarity - best_other_similarity
```

If the observed margin is below the configured margin, authentication can be blocked. This protects against ambiguous cases.

The current config has `AUTH_IDENTIFICATION_MARGIN = -1.0`, which effectively disables strict margin checking because a negative required margin is always easy to pass. If challenged, explain that the feature is implemented but can be relaxed for demo stability; a production or stronger research setting should calibrate this margin on validation data.

## 41. Drift Update Mechanics

Drift handling lives in `DeviceEnroller.update_device_template`.

The update only happens after:

- authentication succeeds,
- confidence band is `strong_accept`,
- drift update is enabled,
- enough consecutive strong accepts have occurred.

The metadata tracks:

- pending strong accepts,
- minimum strong accepts required,
- EMA alpha,
- last strong accept timestamp,
- update count,
- last update similarity.

When update is applied:

```text
updated = normalize((1 - alpha) * old + alpha * fresh)
```

For source profiles:

```text
new_rms_mean = (1 - alpha) * old_rms_mean + alpha * runtime_rms_mean
```

Why EMA instead of replacement:

- Replacement is too sensitive to one session.
- EMA gradually adapts without discarding history.
- Small alpha reduces abrupt template movement.

Main risk:

If attackers can repeatedly get strong accepts, they may slowly poison templates. That is why strong gating, limited template count, audit logs, and fallback checks are important.

## 42. Challenge-Response Cryptographic Layer

The challenge-response layer is not the same as normal password challenge-response. It binds the statistical sensor template to a nonce.

Challenge creation:

```text
nonce = random bytes
challenge_id = hash(device_id || nonce || timestamp)
expires_at = now + challenge_expiry
store challenge
```

Stable feature bytes:

```text
embedding_cpu = float embedding
quantized = round(embedding_cpu * 64).astype(int16)
stable_bytes = quantized bytes
```

Quantization exists because two live embeddings from the same device may not be bit-identical. Coarse quantization makes the derived bytes more tolerant to small embedding drift. It is a tradeoff:

- Too fine: genuine attempts fail because bytes differ.
- Too coarse: different embeddings may collide more easily.

HKDF input material:

```text
IKM = stable_feature_bytes || nonce || server_secret
salt = nonce
info = "noise-device-verification-challenge"
```

HMAC message:

```text
message = challenge_id || device_id || nonce
```

Final verification checks whether the response computed from the stored embedding matches the response computed from the fresh authentication embedding. It can also check an external presented response if provided.

Important claim boundary:

This layer improves replay resistance. It does not convert a noisy sensor fingerprint into a perfect cryptographic key.

## 43. API Endpoint Behavior in Detail

`GET /health`

Returns model and component initialization status plus database reachability. Useful before demos.

`GET /stats`

Returns runtime counters: enrollments, deletions, challenge count, verification calls, auth calls, confidence-band counts, active challenges, and devices in DB.

`POST /collector/api/save`

Stores browser-collected data for dataset building. It sanitizes names, creates a stable device ID, writes arrays/metadata, and can create a zip.

`POST /enroll`

Creates a new device template. It returns `device_id`, status, message, and metadata.

`POST /authenticate`

Runs direct sensor matching. It returns authenticated status and detailed similarity information.

`POST /challenge`

Creates a single-use nonce challenge for an enrolled device.

`POST /verify`

Runs challenge response verification using fresh sensor samples and returns whether the challenge-auth flow passed.

`GET /devices`

Lists enrolled devices, usually for UI dropdowns.

`GET /devices/{device_id}`

Returns detailed metadata for one enrolled device.

`DELETE /devices/{device_id}`

Deletes stored device template and metadata. In a privacy discussion, this is the basic deletion mechanism.

## 44. Configuration Parameters You Should Know

Model:

- `MODEL_PATH`: default model checkpoint path.
- `SOURCE_MODEL_PATHS`: per-source model checkpoints.
- `MODEL_CONFIG["input_dim"]`: expected feature vector size.
- `MODEL_CONFIG["embedding_dim"]`: embedding size.

Authentication:

- `AUTH_CONFIDENCE_STRONG`: strong accept threshold.
- `AUTH_CONFIDENCE_UNCERTAIN`: uncertain threshold.
- `AUTH_SOURCE_WEIGHTS`: camera/microphone fusion weights.
- `AUTH_SOURCE_THRESHOLDS`: source-specific thresholds.
- `AUTH_REQUIRED_SOURCES`: sources that must pass.
- `AUTH_TEMPLATE_CHUNK_SIZE`: enrollment chunk size for templates.
- `AUTH_MAX_TEMPLATES_PER_SOURCE`: maximum stored templates.
- `AUTH_TEMPLATE_TOP_K`: number of best template scores averaged.
- `AUTH_PROFILE_GUARD_Z`: multiplier for RMS profile guard.
- `AUTH_PROFILE_GUARD_MIN_DELTA`: minimum allowed RMS difference.
- `AUTH_DRIFT_EMA_ALPHA`: drift update alpha.
- `AUTH_DRIFT_MIN_STRONG_MATCHES`: gate before drift update.

Challenge:

- `CHALLENGE_CONFIG["nonce_length"]`: random nonce size.
- `CHALLENGE_CONFIG["challenge_expiry_seconds"]`: expiration window.
- `CHALLENGE_SERVER_SECRET`: server-side secret required outside demo mode.

Server:

- `DATABASE_URL`: SQLAlchemy database URL.
- `CORS_CONFIG`: allowed frontend origins.
- `API_KEY`: optional API key.
- `RATE_LIMIT_REQUESTS`: request limit.
- `RATE_LIMIT_WINDOW_SEC`: rate-limit window.

Demo:

- `DEMO_MODE`: review/demo behavior.
- `DEMO_ALLOWED_SOURCES`: allowed sources in demo.
- `DEMO_ENROLL_NUM_SAMPLES`: enrollment sample count in demo.
- `DEMO_AUTH_NUM_SAMPLES`: auth sample count in demo.

## 45. Why the Server Secret Matters

When `DEMO_MODE` is false, startup rejects the default server secret. This is deliberate. A hardcoded default secret would weaken the challenge-response design because attackers could compute expected HMAC behavior if they know the template.

Good viva answer:

> The server secret is required so that the template and nonce alone are not enough to derive the MAC key. The project refuses to run in non-demo mode with the default development secret.

## 46. Dataset and Split Discipline

For this project, evaluation quality depends heavily on dataset discipline.

Bad split:

- Randomly split individual samples from the same device/session across train and test.
- This can leak session-specific patterns and inflate metrics.

Better split:

- Split by device when testing unseen-device generalization.
- Split by session when testing time/environment robustness.
- Use leave-one-device-out for small datasets.

Best report framing:

> The model is evaluated on held-out pairs using explicit split artifacts. The dataset is still small, so results should be interpreted as prototype evidence, not production proof.

The dataset manifest should record:

- device count,
- sample count,
- source count,
- session labels,
- split policy,
- feature version,
- generation timestamp,
- raw file paths.

## 47. Evaluation Interpretation at Examiner Level

The examiner may challenge why the metrics are not extremely high. The correct response is not to hide it.

Camera metrics show useful separation but still significant error. Microphone metrics show weaker separation. This is expected because:

- Dataset size is small.
- Real sensors are noisy.
- Browser collection adds variability.
- Environmental effects are strong.
- Microphone signal is especially environment-dependent.

How to defend:

> The purpose of the capstone is to design and evaluate a complete verification pipeline. The results show that the signal exists but also show the limitations. That is why the system uses confidence bands, fallback behavior, and claim boundaries.

Avoid saying:

- "It is fully secure."
- "It cannot be spoofed."
- "It uniquely identifies every device."
- "The model proves identity."

Say instead:

- "It provides statistical evidence."
- "It can be an auxiliary factor."
- "It requires larger evaluation for deployment."
- "Replay is mitigated, but adaptive spoofing remains a risk."

## 48. Security Threat Model in More Depth

Replay attacker:

The attacker records old samples or responses and tries to reuse them. The nonce-bound challenge mitigates response replay because every challenge has a fresh nonce and expires.

Naive synthetic attacker:

The attacker sends random arrays, silence, or simple generated noise. These should usually fail because feature distributions and embeddings will not match stored templates.

Same-environment confusion:

Two devices in the same environment may share environmental microphone patterns. This is why microphone should not dominate the score and why source-specific evaluation matters.

Adaptive mimicry attacker:

The attacker learns the target's sensor distribution and generates close samples. This is the strongest realistic attack and remains a residual risk.

Template theft:

The attacker obtains stored templates. Templates are not raw credentials, but they are sensitive. Production should encrypt them and monitor access.

Client compromise:

If malware controls the browser or device, it can manipulate samples. This project does not solve compromised-client security.

Server compromise:

If the server secret and templates are stolen, challenge-response protection weakens. Standard server hardening is required.

## 49. Privacy and Data Protection Details

Camera and microphone collection raises privacy concerns even if the system uses downsampled arrays.

Camera:

The frontend captures grayscale pixel arrays. Even if intended as noise, raw frames could contain scene information if not carefully controlled. A production system should avoid storing raw frames unless strictly necessary.

Microphone:

Raw audio may contain speech or environmental information. A production system should process locally, avoid storing raw audio, or collect only controlled ambient/noise segments with clear consent.

Templates:

Embeddings are safer than raw samples, but they are still biometric-like identifiers. Treat them as sensitive personal/device data.

Recommended privacy measures:

- Consent screen before collection.
- Clear explanation of purpose.
- Store templates, not raw samples, in production.
- Encrypt templates at rest.
- Delete device endpoint.
- Audit access.
- Avoid using samples for unrelated purposes.

## 50. Failure Modes and Debugging

Common failure: model not loaded.

Check:

- `/health`
- model paths in `config.py`
- whether checkpoint files exist in `server/models`

Common failure: browser cannot access camera/microphone.

Check:

- HTTPS or localhost requirement.
- browser permissions.
- mobile browser support.
- active camera/microphone usage by another app.

Common failure: microphone sample is silent.

The authenticator rejects near-zero RMS microphone samples. Check permissions and input device.

Common failure: authentication returns uncertain.

Possible causes:

- too few samples,
- lighting/noise changed,
- microphone environment changed,
- source weights or thresholds are strict,
- enrollment was poor quality.

Common failure: source mismatch.

If a device was enrolled with camera and microphone, authenticating with only microphone or an unexpected source may be rejected depending on metadata and policy.

Common failure: default secret outside demo.

Set `QNA_AUTH_SERVER_SECRET` to a real random secret.

## 51. How to Read Important Files

Start here:

- `README.md`: project claims, setup, endpoints, demo config.
- `config.py`: thresholds, weights, model paths, challenge settings.
- `server/app.py`: API and startup wiring.
- `auth/enrollment.py`: template creation and persistence.
- `auth/authentication.py`: scoring and final decision logic.
- `auth/challenge_response.py`: nonce, HKDF, HMAC, replay resistance.
- `preprocessing/features.py`: signal feature extraction.
- `model/siamese_model.py`: embedding model and losses.
- `frontend/src/services/collectors.ts`: browser sensor capture.
- `frontend/src/services/api.ts`: frontend-backend API calls.
- `db/models.py`: persistent tables.
- `tests/`: guardrail tests.

If explaining to an examiner, do not start from the UI. Start from the security and ML pipeline, then mention the UI as a collection interface.

## 52. End-to-End Example With Numbers

Suppose a device is enrolled with:

- 10 camera samples.
- 10 microphone samples.
- chunk size 5.
- max templates per source 4.

Enrollment creates:

- 2 camera templates.
- 2 microphone templates.
- source mean embeddings for camera and microphone.
- combined templates by weighted source fusion.
- one combined embedding.

During authentication:

- fresh camera samples produce a camera auth embedding.
- fresh microphone samples produce a microphone auth embedding.
- camera auth embedding is scored against camera template bank.
- microphone auth embedding is scored against microphone template bank.

Example:

```text
camera score = 0.985
microphone score = 0.930
camera weight = 0.85
microphone weight = 0.15
combined = 0.985 * 0.85 + 0.930 * 0.15
combined = 0.97675
```

With strong threshold `0.97`, the fused score is strong accept. But final authentication still depends on profile guard, per-source checks, and margin guard.

If microphone score were rejected and required-source policy required microphone, the final decision could still fail even if the camera is strong.

## 53. Production Deployment Gaps

The capstone prototype has several production gaps:

- No production identity provider integration.
- No passkey/password fallback implementation.
- In-memory rate limiting is not distributed.
- Template files are not encrypted by default.
- Model drift and poisoning controls need stronger policy.
- Dataset scale is too small.
- Browser collection is variable across platforms.
- Challenge-response is useful but not sufficient against adaptive live spoofing.
- Monitoring and alerting are basic.
- Secrets need proper secret-manager integration.

Good answer:

> I would not deploy this as standalone authentication. I would deploy it only as an auxiliary risk signal after more data collection, threshold calibration, template encryption, and integration with a standard authentication mechanism.

## 54. Strong Capstone Defense Framing

The best way to defend the project is to separate achievement from claim boundary.

Achievement:

- Complete end-to-end system.
- Multi-source sensor collection.
- Feature extraction pipeline.
- Siamese embedding model.
- Template-bank enrollment.
- Similarity-based authentication.
- Confidence bands.
- Drift update gate.
- Replay-resistant challenge flow.
- Database and audit support.
- Evaluation metrics and documentation.

Claim boundary:

- Statistical verification only.
- Not cryptographic identity.
- Not spoof-proof.
- Not production-ready as standalone auth.
- Needs larger cross-session dataset.

This framing is technically honest and usually stronger than overclaiming.

## 55. Five-Minute Technical Viva Script

Use this if asked to explain the whole project in detail.

> QNA-Auth is a device verification prototype based on sensor-noise fingerprints. The frontend collects camera and microphone samples using browser media APIs. Camera frames are converted to grayscale arrays and microphone samples are captured through an audio context; both are downsampled before being sent to the backend.
>
> The FastAPI backend validates the request and sends samples through a common preprocessing pipeline. The preprocessor converts each raw sample into a fixed feature vector containing raw statistical features, entropy, FFT features, autocorrelation features, and complexity features. The feature vector order and standardization parameters are loaded from the model checkpoint so that runtime inference matches training.
>
> A Siamese neural network maps these feature vectors to L2-normalized embeddings. It is trained with metric learning, mainly triplet loss, so same-device samples are close and different-device samples are farther apart. During enrollment, multiple embeddings are aggregated into source-specific templates and combined templates. During authentication, fresh embeddings are scored against stored template banks using cosine similarity.
>
> Camera and microphone scores are fused using configurable source weights. The final score is classified into strong accept, uncertain, or reject. Final authentication also checks source-profile consistency, per-source status, and an optional identification margin. Strong matches can update the template using a gated EMA drift policy.
>
> For replay resistance, the system also supports a challenge-response flow. The server creates a nonce, derives stable feature bytes from the stored and fresh embeddings through quantization, and uses HKDF plus HMAC with a server secret. This makes old responses unusable for new challenges. The project is still statistical verification, not a cryptographic proof of identity, so it should be treated as an auxiliary authentication signal.

## 56. Terms You Must Be Able to Define

Verification:

Checking whether the presented sample matches a claimed identity or device.

Identification:

Finding which identity or device from a set best matches the sample.

Feature:

A measurable numerical property extracted from raw data.

Embedding:

A learned vector representation used for similarity comparison.

Template:

Stored enrollment representation used for future matching.

Cosine similarity:

Vector similarity based on angle, usually between `-1` and `1`.

FAR:

Rate at which impostor attempts are accepted.

FRR:

Rate at which genuine attempts are rejected.

EER:

Operating point where FAR and FRR are approximately equal.

Nonce:

Random value used once to prevent replay.

HKDF:

Key derivation function that derives a cryptographic key from input material, salt, and context info.

HMAC:

Keyed hash used to authenticate a message.

Drift:

Gradual change in sensor behavior over time.

Template poisoning:

Corrupting a stored template by updating it with malicious or low-quality samples.

## 57. Appendix: Exact Runtime Object Responsibilities

This section breaks the runtime classes down by responsibility. This is useful when an examiner asks "which file does what?" or "where exactly is this handled?"

`server.app.AppState`

This is the runtime container for initialized components. It stores the active embedder, preprocessor, feature converter, source-specific embedders, enroller, authenticator, challenge protocol, auth flow, rate-limit buckets, and lightweight stats. In a production system, dependency injection would be cleaner, but for a capstone prototype this global state keeps the app simple.

`NoisePreprocessor`

This class converts raw arrays into feature dictionaries. It is intentionally independent from FastAPI and the model so that the same preprocessing can be used in training, evaluation, and serving. Its job is not to decide whether a device matches; it only extracts signal descriptors.

`FeatureVector`

This class converts feature dictionaries into fixed-order arrays. It is one of the most important reproducibility pieces. Neural networks do not know feature names; they only see positions. If `raw_rms` was feature index 7 during training, it must still be index 7 during runtime. `FeatureVector` enforces that mapping.

`DeviceEmbedder`

This class wraps the PyTorch Siamese model. It handles device placement, eval mode, single-sample shape handling, no-gradient inference, CPU return, model loading, model saving, and similarity computation.

`DeviceEnroller`

This class owns enrollment. It collects or accepts samples, extracts features, creates source embeddings, creates template banks, combines sources, saves templates, saves metadata, lists enrolled devices, and updates templates during drift handling.

`DeviceAuthenticator`

This class owns authentication policy. It builds fresh auth profiles, validates source profiles, scores template banks, fuses scores, classifies confidence bands, checks identification margin, checks required sources, and triggers drift update only after strong accept.

`ChallengeResponseProtocol`

This class owns nonce creation, challenge storage, HKDF key derivation, HMAC computation, response verification, challenge expiry, and challenge deletion.

`DbChallengeStore`

This class adapts the database to the challenge protocol. The protocol only needs `put`, `get`, and `delete`; the store hides SQLAlchemy details.

## 58. Appendix: API Request and Response Anatomy

The API is designed around JSON payloads. The key design choice is that raw sample arrays can be supplied by the client, so the backend does not need direct hardware access during normal browser demos.

Enrollment request:

```json
{
  "device_name": "My Laptop",
  "num_samples": 10,
  "sources": ["camera", "microphone"],
  "client_samples": {
    "camera": [[0.1, 0.2, 0.3]],
    "microphone": [[0.01, -0.02, 0.03]]
  }
}
```

Enrollment response:

```json
{
  "device_id": "16_character_id",
  "status": "success",
  "message": "Device enrolled successfully for future high-confidence matching",
  "metadata": {
    "sources": ["camera", "microphone"],
    "feature_dimension": 50,
    "template_strategy": "multi_template_chunk_mean"
  }
}
```

Authentication request:

```json
{
  "device_id": "16_character_id",
  "sources": ["camera", "microphone"],
  "num_samples_per_source": 5,
  "client_samples": {
    "camera": [[...], [...]],
    "microphone": [[...], [...]]
  }
}
```

Authentication success response:

```json
{
  "authenticated": true,
  "device_id": "16_character_id",
  "similarity": 0.978,
  "details": {
    "confidence_band": "strong_accept",
    "recommended_action": "accept",
    "per_source_similarity": {
      "camera": {
        "similarity": 0.985,
        "band": "strong_accept"
      },
      "microphone": {
        "similarity": 0.942,
        "band": "uncertain"
      }
    }
  }
}
```

Authentication failure response can still be `200 OK` with `authenticated: false` for clean client behavior. This is deliberate in parts of the app because a similarity rejection is not necessarily a server error. A server error means the system failed to process the request; an authentication failure means the system processed it and rejected the match.

## 59. Appendix: HTTP Status Code Reasoning

The project uses different types of failure:

`400 Bad Request`

Used when the request is structurally invalid, such as no samples provided or invalid demo-mode sources.

`401 Unauthorized`

Used when API key protection is enabled and the key is missing or invalid.

`429 Too Many Requests`

Used when the rate limiter blocks excessive requests from an IP.

`500 Internal Server Error`

Used when enrollment or storage fails unexpectedly.

`503 Service Unavailable`

Used when core services such as the enroller or authenticator are not initialized.

Authentication mismatch is different from these. A genuine "device does not match" result is a domain result, not necessarily an HTTP error.

## 60. Appendix: Model Checkpoint Contents

A trained checkpoint can contain more than only model weights. The project benefits from storing metadata alongside weights.

Typical checkpoint fields:

```text
model_state_dict
embedding_dim
input_dim
feature_names
feature_version
feature_mean
feature_scale
preprocessing_normalize
preprocessing_fast_mode
```

Why each matters:

- `model_state_dict`: actual learned PyTorch parameters.
- `embedding_dim`: needed to reconstruct output layer size.
- `input_dim`: needed to reconstruct input layer size.
- `feature_names`: guarantees runtime feature order.
- `feature_version`: documents feature pipeline compatibility.
- `feature_mean`: training-set standardization mean.
- `feature_scale`: training-set standardization scale.
- preprocessing flags: ensure runtime signal processing matches training.

If an examiner asks "what happens if checkpoint metadata is missing?", answer:

> The app falls back to canonical feature names and config defaults, but the best practice is to store feature metadata inside the checkpoint. Without it, train/serve mismatch risk increases.

## 61. Appendix: Feature Math Reference

Mean:

```text
mean = sum(x_i) / n
```

Standard deviation:

```text
std = sqrt(sum((x_i - mean)^2) / n)
```

Variance:

```text
variance = std^2
```

RMS:

```text
rms = sqrt(mean(x_i^2))
```

Peak factor:

```text
peak_factor = max(abs(x)) / rms
```

Skewness:

Measures asymmetry of the distribution. Positive skew means a longer right tail; negative skew means a longer left tail.

Kurtosis:

Measures tail heaviness or sharpness compared with a normal distribution.

Shannon entropy:

```text
H = -sum(p_i * log2(p_i))
```

Spectral centroid:

```text
centroid = sum(f_i * magnitude_i) / sum(magnitude_i)
```

Spectral entropy:

```text
P_i = power_i / total_power
H_spectral = -sum(P_i * log2(P_i))
```

Spectral flatness:

```text
flatness = geometric_mean(magnitude) / arithmetic_mean(magnitude)
```

Autocorrelation:

```text
R(k) = sum(x_t * x_{t-k})
```

Zero-crossing rate:

```text
ZCR = count(sign changes) / signal_length
```

Hurst exponent:

Summarizes long-range dependence. Around `0.5` suggests random-walk-like behavior; higher values suggest persistence; lower values suggest anti-persistence.

Exam framing:

> No single feature proves identity. The model uses a combination of descriptors, and the learned embedding decides which combinations are useful for similarity.

## 62. Appendix: Why Raw and Normalized Features Both Exist

Using only raw features can make the model sensitive to irrelevant scale differences. For example, microphone loudness may change because of room noise or browser gain. Using only normalized features can remove useful amplitude information. The project keeps both:

- Raw features preserve energy and range.
- Normalized features preserve shape and structure.

This gives the model more information and lets training learn which parts matter.

Example:

Two camera captures may have different brightness but similar noise texture. Normalized spectral/autocorrelation features can still match. But if the overall raw profile is completely different, raw RMS/range can flag a mismatch.

## 63. Appendix: Similarity Metrics

The code supports cosine similarity and negative Euclidean distance.

Cosine:

```text
higher = more similar
range roughly -1 to 1
good for normalized embeddings
```

Euclidean:

```text
lower distance = more similar
implementation returns negative distance so higher is still better
```

Why cosine is preferred here:

- Embeddings are L2-normalized.
- Cosine becomes simple angular similarity.
- Thresholds are easier to think about as high-similarity values.

If asked why threshold is high, explain:

> With normalized embeddings and a security-sensitive decision, high thresholds reduce false accepts. The tradeoff is higher false rejection, which the uncertain band handles.

## 64. Appendix: Training Pair and Triplet Construction

The training scripts build examples from grouped samples.

Pair mode:

- Same-device pair gets label `1`.
- Different-device pair gets label `0`.
- Contrastive loss pulls same pairs together and pushes different pairs apart.

Triplet mode:

- Anchor and positive are same-device.
- Negative is different-device.
- Triplet loss enforces relative ranking.

Triplet mode is usually easier to explain for verification because the desired behavior is direct:

```text
same-device distance < different-device distance
```

Important dataset requirement:

Each device needs multiple samples. If a device has only one sample, it cannot form anchor-positive pairs.

## 65. Appendix: Threshold Calibration

A threshold should ideally be chosen on validation data, not guessed.

Process:

1. Generate genuine scores from same-device pairs.
2. Generate impostor scores from different-device pairs.
3. Sweep thresholds across score range.
4. At each threshold compute FAR and FRR.
5. Pick threshold based on security target.

For high-security mode:

```text
choose threshold where FAR is very low
accept higher FRR
```

For usability mode:

```text
choose threshold with lower FRR
accept higher FAR
```

For balanced research reporting:

```text
report EER and ROC-AUC
```

QNA-Auth uses confidence bands because one threshold cannot represent all operational risk.

## 66. Appendix: Understanding FAR, FRR, TP, TN, FP, FN

Definitions in this project:

- Genuine attempt: fresh sample is from the claimed enrolled device.
- Impostor attempt: fresh sample is from a different device.
- True positive: genuine accepted.
- False negative: genuine rejected.
- True negative: impostor rejected.
- False positive: impostor accepted.

FAR:

```text
FAR = FP / (FP + TN)
```

FRR:

```text
FRR = FN / (FN + TP)
```

Precision:

```text
precision = TP / (TP + FP)
```

Recall:

```text
recall = TP / (TP + FN)
```

Balanced accuracy:

```text
balanced_accuracy = (TPR + TNR) / 2
```

Why balanced accuracy matters:

If genuine and impostor counts are imbalanced, normal accuracy can be misleading. Balanced accuracy treats both classes more evenly.

## 67. Appendix: Why the Microphone Is Hard

Microphone fingerprinting is technically difficult because the signal is heavily affected by environment and software processing.

Factors:

- background fan noise,
- human speech,
- room echo,
- automatic gain control,
- noise suppression,
- browser audio processing,
- microphone placement,
- sample-rate conversion,
- OS audio drivers.

This explains why microphone metrics are weaker in current evaluation. A good defense is:

> Microphone is useful as supporting evidence, but the current data suggests it should not dominate the fused decision. That is why the system supports source weights and source-specific thresholds.

## 68. Appendix: Why the Camera Is Also Hard

Camera fingerprinting also has complications:

- lighting changes,
- auto-exposure,
- auto-focus,
- compression or browser processing,
- sensor temperature,
- motion blur,
- image content leaking into features,
- different camera resolutions.

The frontend waits briefly for exposure stabilization, converts to grayscale, and downscales. However, a stronger production version would use controlled dark frames, fixed exposure settings where possible, and stronger noise-residual extraction.

If asked "is the camera capturing the scene or the sensor?", answer carefully:

> In the browser path, the sample comes from camera frames and is converted to grayscale. The goal is to capture sensor-noise characteristics, but uncontrolled scene content can influence features. This is a limitation and future work should use more controlled capture or stronger residual extraction.

## 69. Appendix: Noise Residual vs Raw Signal

A pure sensor-fingerprint system usually tries to isolate noise residual:

```text
residual = observed_signal - estimated_clean_signal
```

For images, this can involve denoising filters, flat-field captures, dark frames, or high-pass filtering. For audio, this can involve filtering, silence detection, or controlled ambient capture.

QNA-Auth currently uses practical browser-compatible arrays and backend feature extraction. It is a prototype compromise:

- Easier to demo across devices.
- Less control over physical capture.
- More environmental influence.
- Still enough structure to evaluate the concept.

This is a good limitation to admit.

## 70. Appendix: Anti-Replay Flow With Example

Assume device `D1` is enrolled.

Step 1: Client asks for challenge.

```json
{
  "device_id": "D1"
}
```

Step 2: Server returns:

```json
{
  "challenge_id": "c123",
  "nonce": "random_hex",
  "expires_at": "timestamp"
}
```

Step 3: Client submits fresh samples to `/verify`.

```json
{
  "challenge_id": "c123",
  "device_id": "D1",
  "client_samples": {
    "camera": [[...]],
    "microphone": [[...]]
  }
}
```

Step 4: Server computes:

```text
expected = HMAC(HKDF(stored_template, nonce, server_secret), message)
fresh = HMAC(HKDF(fresh_template, nonce, server_secret), message)
```

Step 5: Server compares them and deletes challenge.

Why replay fails:

An old response was computed for an old nonce. A new challenge uses a new nonce, so the HMAC key and message change.

## 71. Appendix: What "Stable Feature Template" Means

The word stable does not mean perfectly identical every time. It means stable enough after quantization and aggregation for repeated matching.

Sensor readings are naturally noisy. Two captures from the same device will differ. The system handles this through:

- multiple samples,
- feature aggregation,
- learned embeddings,
- L2 normalization,
- template banks,
- top-k scoring,
- confidence bands,
- coarse quantization in challenge flow.

Correct examiner answer:

> Stability here is empirical and statistical. It must be measured with cross-session genuine score distributions, not assumed.

## 72. Appendix: Why Not Store the Embedding as a Password?

Passwords require exact or cryptographically transformed matching. Sensor embeddings are fuzzy. A fresh genuine embedding may be close but not identical.

If the embedding were treated as a password:

- small natural variation would break authentication,
- replay risk would be high,
- stolen templates could be misused,
- it would invite incorrect security claims.

QNA-Auth treats embeddings as biometric-like templates:

- compare by similarity,
- protect as sensitive,
- do not expose as credentials,
- bind challenge responses to nonce and server secret.

## 73. Appendix: Audit Logging Value

Audit logs are not required for model accuracy, but they are useful for security and research.

They record:

- enrollments,
- authentication attempts,
- challenge creation,
- verification calls,
- deletions.

Use cases:

- demo transparency,
- debugging failed attempts,
- detecting repeated attacks,
- reproducing review behavior,
- supporting future security monitoring.

Production improvement:

Audit logs should be immutable or tamper-evident, and sensitive details should be minimized.

## 74. Appendix: Rate Limiting and Abuse Control

The current rate limiter is simple:

```text
state.request_buckets[client_ip] = list of request timestamps
remove timestamps older than window
if count >= limit: reject
otherwise append current timestamp
```

This helps against brute-force API abuse during demos. It is not production-grade because:

- state is in memory,
- state resets on restart,
- multiple workers do not share buckets,
- IP addresses can be shared or spoofed behind proxies.

Production improvement:

Use Redis, an API gateway, or reverse-proxy rate limiting.

## 75. Appendix: CORS and Browser Security

CORS controls which browser origins can call the API. The config allows localhost frontend origins such as `http://localhost:3000` and `http://localhost:5173`.

Why this matters:

- Browser apps are origin-restricted.
- Without CORS, the frontend cannot call the backend.
- With overly broad CORS, untrusted websites may interact with the API from a user's browser.

Production rule:

Do not use wildcard CORS with credentials. Restrict origins to known frontend domains.

## 76. Appendix: Demo Mode vs Real Mode

Demo mode exists to make review-day behavior predictable.

Demo mode can:

- restrict allowed sources,
- reduce enrollment sample count,
- reduce authentication sample count,
- allow development secret behavior depending on config,
- prioritize speed and reliability.

Real mode should:

- use a real server secret,
- use stronger sample counts,
- use stricter thresholds,
- require TLS,
- use proper API auth,
- protect templates,
- log and monitor attempts.

If asked why demo mode exists:

> Demo mode separates controlled presentation settings from stricter deployment expectations. It prevents the review from depending on slow or fragile sample counts while keeping production claim boundaries clear.

## 77. Appendix: How to Present Weak Results Honestly

If challenged on metric quality, use this structure:

1. Acknowledge the limitation.
2. Explain the technical cause.
3. Show what the system does to manage risk.
4. State future work.

Example:

> The microphone EER is not production-grade. That is expected because microphone data is environment-sensitive and the dataset is small. The system manages this by using source-specific thresholds, giving microphone lower fusion weight, and returning uncertain instead of accepting borderline scores. Future work is larger cross-session data and better microphone preprocessing.

This is stronger than pretending the metric is perfect.

## 78. Appendix: Common Examiner Traps

Trap: "So this proves the user's identity?"

Correct answer:

> No. It verifies device similarity. User identity still needs a separate authentication factor.

Trap: "Can it never be spoofed?"

Correct answer:

> No. Replay is mitigated, but adaptive spoofing remains a residual risk.

Trap: "Are embeddings passwords?"

Correct answer:

> No. They are biometric-like templates for similarity matching and must be protected.

Trap: "Why is your dataset small?"

Correct answer:

> It is a capstone-scale prototype. The architecture and evaluation pipeline are complete, but stronger claims require more devices and sessions.

Trap: "Why use ML when handcrafted features already exist?"

Correct answer:

> Handcrafted features describe the signal. Metric learning learns how to combine them into a verification space where same-device samples are close and different-device samples are farther.

Trap: "Why not use cryptographic device identity?"

Correct answer:

> Cryptographic identity is stronger for production. This project explores physical sensor fingerprinting as an auxiliary signal, not a replacement for certificates or passkeys.

## 79. Appendix: What to Draw on a Whiteboard

If asked to draw architecture, draw this:

```text
Camera/Mic
   |
Browser Collector
   |
client_samples JSON
   |
FastAPI
   |
NoisePreprocessor
   |
FeatureVector + standardization
   |
Siamese Embedder
   |
Source Templates
   |
Weighted Fusion
   |
Confidence Band
   |
Accept / Uncertain / Reject
```

If asked to draw challenge-response, draw this:

```text
/challenge -> nonce + challenge_id
client captures fresh samples
/verify -> fresh embedding
stored embedding + fresh embedding
quantize -> HKDF(nonce, server_secret)
HMAC compare
single-use challenge deleted
```

If asked to draw training:

```text
dataset samples
   |
feature extraction
   |
anchor-positive-negative triplets
   |
Siamese network
   |
triplet loss
   |
embedding space
```

## 80. Appendix: One-Sentence Answers for Rapid Fire

What is the project?

> A sensor-noise-based device verification prototype using camera and microphone embeddings.

What is the model?

> A Siamese neural network that maps signal features to normalized embeddings.

What is the main metric?

> FAR, FRR, EER, and ROC-AUC are the most important evaluation metrics.

What is the main security feature?

> Nonce-bound HKDF/HMAC challenge verification mitigates replay.

What is the biggest limitation?

> Dataset scale and environmental variability.

What is the safest deployment role?

> Auxiliary risk signal alongside conventional authentication.

Why confidence bands?

> Because sensor matching is statistical and borderline scores should not become silent accepts.

Why source weights?

> Because camera and microphone have different reliability.

Why template banks?

> They represent multiple valid enrollment conditions better than one mean vector.

Why drift update?

> To adapt gradually to sensor changes while gating updates to strong matches.

