# QNA-Auth External Examiner Question Bank

Use this as viva preparation. Answers are written in a direct defense style.

## A. Project Basics

### 1. What is your project?

QNA-Auth is a noise-based device verification prototype. It uses camera and microphone sensor noise to create learned embeddings for a device, then compares fresh samples against stored templates during authentication.

### 2. Is this user authentication or device authentication?

It is device verification. It checks whether the current device resembles an enrolled device. It does not directly prove the human user's identity.

### 3. What problem are you solving?

The project explores whether device sensor noise can act as an auxiliary authentication signal. It can support existing authentication by adding evidence that the same physical device is being used.

### 4. Why is the project called QNA-Auth?

The project name refers to using noise-like signals as an authentication factor. In the current implementation, the active runtime sources are camera and microphone noise.

### 5. What are the active sensor sources?

The active runtime sources are camera and microphone. Older QRNG references are historical or dataset-related, not the current runtime authentication path.

### 6. What is the main output of the system?

The system outputs whether the device is authenticated, a similarity score, and a confidence band such as `strong_accept`, `uncertain`, or `reject`.

### 7. Is this production-ready?

No. It is a research-level capstone prototype. It demonstrates a working pipeline and honest evaluation, but it needs larger datasets, stronger deployment hardening, and more cross-session testing before production use.

## B. Architecture

### 8. Explain the architecture end to end.

The frontend collects camera and microphone samples. The FastAPI backend receives samples, extracts features, converts features into embeddings with a Siamese model, compares fresh embeddings with stored enrollment templates, fuses per-source scores, and returns a confidence-band decision.

### 9. What are the major modules?

`frontend` handles UI and browser collection, `server` exposes FastAPI endpoints, `auth` handles enrollment/authentication/challenge logic, `preprocessing` extracts features, `model` trains and runs the Siamese network, `db` stores metadata and challenges, and `scripts` supports collection, evaluation, diagnostics, and reporting.

### 10. Why did you use FastAPI?

FastAPI is suitable because it provides typed request/response models through Pydantic, automatic OpenAPI docs, async endpoint support, and quick development for ML-backed APIs.

### 11. Why did you use React?

React makes it practical to build interactive enrollment and authentication pages. It also allows browser APIs like camera and microphone capture to be integrated into a usable client.

### 12. Where is the decision made?

The backend makes the decision. The frontend only collects samples and displays results.

### 13. What is stored for each enrolled device?

The system stores embedding/template files, metadata JSON, source profiles, and a database row containing device ID, device name, embedding path, and timestamps.

### 14. Why do you have a database if embeddings are files?

The files store heavier template data. The database stores searchable metadata, active challenges, and audit logs. This separation keeps template storage simple while still supporting API operations.

## C. Enrollment

### 15. What happens during enrollment?

The system collects multiple camera and microphone samples, extracts features from each sample, embeds those features with the model, aggregates embeddings into templates, stores source-specific templates, and records metadata.

### 16. Why collect multiple samples?

Sensor readings are noisy. Multiple samples help average out random variation and create a more stable representation of the device.

### 17. How is the device ID generated?

The enroller hashes the device name plus timestamp or a timestamp-only identifier and truncates the hash to a 16-character ID.

### 18. What is a template bank?

A template bank is a set of stored embeddings rather than a single embedding. Authentication compares fresh embeddings against the bank and can use top-k matching to tolerate normal variation.

### 19. Why not store raw camera and microphone data?

Raw data is larger and more privacy-sensitive. Templates and metadata are more appropriate for verification. Raw data may be kept only for research datasets when consent and storage controls are clear.

### 20. What metadata is useful at enrollment?

Useful metadata includes device name, sources, number of samples, feature version, source RMS profiles, template count, and enrollment timestamp.

## D. Authentication

### 21. What happens during authentication?

The client sends fresh samples and a claimed device ID. The backend loads stored templates, extracts fresh features, creates fresh embeddings, compares them with stored templates, fuses scores, and returns a confidence band.

### 22. What is cosine similarity?

Cosine similarity measures the angle between two vectors. Because embeddings are L2-normalized, cosine similarity is a natural way to compare whether two embeddings point in a similar direction.

### 23. What thresholds do you use?

The common confidence bands are strong accept at `>= 0.97`, uncertain from `0.92` to `0.97`, and reject below `0.92`. Source-specific thresholds can also be configured.

### 24. Why have an uncertain band?

Sensor data is variable. The uncertain band prevents the system from making overconfident decisions on borderline scores and allows fallback authentication or extra sample collection.

### 25. What is weighted fusion?

Weighted fusion combines source scores using configured weights. For example, camera can receive higher weight than microphone if camera is more stable in evaluation.

### 26. Why is camera weighted more than microphone?

Current evaluation shows camera has better separation than microphone. Microphone is more affected by environmental sound, device audio pipelines, and ambient conditions.

### 27. What if one source is missing?

The system can compute effective weights over available sources. Configuration can also require specific sources if stricter behavior is needed.

### 28. What is the profile guard?

The profile guard compares simple runtime statistics such as RMS against enrolled source profiles. It catches obvious mismatches before relying only on embedding similarity.

### 29. What is the identification margin?

It checks whether the claimed device score is sufficiently better than the best other enrolled device. This reduces confusion when two devices score similarly.

### 30. Can the same device be rejected?

Yes. Environmental changes, poor capture quality, sensor instability, or too few samples can cause false rejection. That is why the system has an uncertain band.

## E. Feature Extraction

### 31. Why do you need feature extraction?

Raw sensor arrays are large and variable. Feature extraction converts them into fixed-size numerical vectors that preserve statistical and spectral patterns useful for matching.

### 32. What features do you extract?

The system extracts statistical features, entropy, FFT/spectral features, autocorrelation features, and complexity features such as zero-crossing rate, approximate entropy, and Hurst exponent.

### 33. What is RMS and why is it useful?

RMS is root mean square amplitude. It summarizes signal energy and helps detect large differences between enrolled and runtime samples.

### 34. What is FFT used for?

FFT converts the signal from time domain to frequency domain. It helps capture spectral properties such as dominant frequency, spectral centroid, flatness, and frequency-band power.

### 35. What is entropy in this context?

Entropy measures unpredictability or spread in the signal distribution. Noise-like signals often have useful entropy characteristics.

### 36. Why normalize data?

Normalization reduces scale differences and makes features more comparable. The system also preserves some raw-domain features so amplitude information is not completely lost.

### 37. What is the feature version?

The feature pipeline declares `FEATURE_VERSION = "2.0"`. Versioning matters because training, evaluation, and runtime must use the same feature order and extraction logic.

### 38. What happens if feature order changes?

The model would receive the wrong meaning at each input index, causing unreliable predictions. That is why `FeatureVector` uses a canonical ordered feature list.

## F. Machine Learning

### 39. Why use a Siamese network?

A Siamese network learns similarity instead of only classifying fixed labels. This fits device verification because new devices can be enrolled by storing embeddings without retraining a classifier for every new device.

### 40. What is an embedding?

An embedding is a learned vector representation. In this project, it should place samples from the same device close together and samples from different devices farther apart.

### 41. What is triplet loss?

Triplet loss uses anchor, positive, and negative samples. It penalizes the model when the anchor is not closer to the positive than to the negative by a configured margin.

### 42. What are anchor, positive, and negative?

Anchor is a sample from one device. Positive is another sample from the same device. Negative is a sample from a different device.

### 43. Why not use a normal classifier?

A classifier learns fixed device classes. Verification needs to compare new enrolled devices without retraining the entire model. Metric learning is more flexible for this use case.

### 44. What is L2 normalization?

L2 normalization scales embeddings to unit length. It stabilizes similarity comparison and makes cosine similarity more meaningful.

### 45. What model architecture do you use?

The model is a multilayer perceptron with hidden dimensions `[256, 256, 128]`, Tanh activations, and a 128-dimensional normalized embedding output.

### 46. Why Tanh instead of ReLU?

The current code uses Tanh to avoid collapse issues seen with ReLU in this prototype. It keeps activations bounded and deterministic for the embedding pipeline.

### 47. What is overfitting risk here?

The dataset is small, so the model may learn session-specific or environment-specific artifacts instead of stable device patterns. Device/session-separated evaluation helps detect this.

## G. Evaluation

### 48. What metrics do you report?

The project reports FAR, FRR, EER, accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and score distributions.

### 49. What is FAR?

FAR is false acceptance rate. It is the fraction of impostor attempts incorrectly accepted. It is critical for security.

### 50. What is FRR?

FRR is false rejection rate. It is the fraction of genuine attempts incorrectly rejected. It is critical for usability.

### 51. What is EER?

EER is equal error rate, where FAR and FRR are approximately equal. Lower EER means better separation.

### 52. Which source performs better currently?

Camera performs better in the current evaluation. Existing metrics show camera ROC-AUC around `0.858` and EER around `0.268`, while microphone ROC-AUC is around `0.710` and EER around `0.364`.

### 53. Are those metrics strong enough for production?

No. They are useful for a capstone prototype but not sufficient for standalone production authentication.

### 54. Why is microphone weaker?

Microphone samples are affected by ambient sound, room acoustics, OS audio processing, browser audio pipelines, and background noise.

### 55. What is data leakage?

Data leakage happens when training and test sets share information that would not be available in real deployment. For this project, mixing samples from the same session or device incorrectly can inflate results.

### 56. How do you avoid leakage?

Use deterministic splits by device or by session and store split artifacts. The project has experiment utilities and tests for split leakage.

### 57. What dataset improvement would strengthen the project most?

More devices, more sessions per device, different days/environments, and controlled train/test splits would strengthen the claims most.

## H. Challenge-Response and Security

### 58. Why do you need challenge-response?

Plain similarity matching could be vulnerable to replayed samples. Challenge-response binds verification to a fresh nonce so old responses cannot be reused directly.

### 59. What is a nonce?

A nonce is a random value used once. It ensures every challenge is fresh.

### 60. How does HKDF help?

HKDF derives a MAC key from stable template bytes, the nonce, and the server secret. This avoids directly using the embedding as a credential and binds the response to server-side secret material.

### 61. What is HMAC?

HMAC is a keyed hash used to verify message authenticity. Here it signs challenge data using a key derived from the template, nonce, and server secret.

### 62. Does challenge-response prove identity cryptographically?

Not fully. It prevents replay and hardens the flow, but the underlying match is still statistical sensor verification.

### 63. What happens to a challenge after verification?

It is deleted. Challenges are single-use and also expire after a configured time.

### 64. What attacks are considered?

Replay, naive synthetic noise, adaptive mimicry, environment changes, and template leakage are considered. Replay is directly mitigated. Adaptive mimicry remains a residual risk.

### 65. Can an attacker spoof the sensor?

A naive attacker may fail, but an adaptive attacker with enough data and modeling capability could get closer. This is why the system should be an auxiliary factor.

### 66. Is the embedding secret?

It should be treated as sensitive, similar to a biometric template. It is not a password, but leakage can still harm security and privacy.

### 67. What would you add for production security?

Template encryption at rest, stronger API authentication, TLS-only deployment, strict rate limiting, monitoring, secure key management, and fallback authentication.

## I. Drift and Robustness

### 68. What is sensor drift?

Sensor drift is gradual change in sensor behavior over time due to hardware, software, or environmental factors.

### 69. How does your system handle drift?

It supports EMA-based rolling template updates after strong matches. Updates are gated to avoid poisoning from weak or malicious attempts.

### 70. What is template poisoning?

Template poisoning occurs when bad or attacker-controlled samples are used to update the stored template, gradually moving it away from the genuine device.

### 71. How do you reduce poisoning risk?

Only update after strong accepts, require multiple strong matches, and use conservative EMA alpha.

### 72. What can cause false rejection?

Poor lighting, microphone noise, browser permission problems, too few samples, hardware changes, or environment shifts.

### 73. What can cause false acceptance?

Weak threshold calibration, similar devices, environment artifacts, or an adaptive attacker generating similar signal characteristics.

## J. Implementation Details

### 74. Where are thresholds configured?

Thresholds are in `config.py`, including `AUTH_CONFIDENCE_STRONG`, `AUTH_CONFIDENCE_UNCERTAIN`, `AUTH_SOURCE_THRESHOLDS`, and `AUTH_SOURCE_WEIGHTS`.

### 75. What is demo mode?

Demo mode constrains sources and sample counts for reliable review-day behavior. It is controlled by `DEMO_MODE` and related config values.

### 76. What is the health endpoint for?

`/health` checks whether models and core components are loaded. It helps diagnose whether the backend is ready.

### 77. What does `/stats` show?

It tracks counts such as enrollments, authentication calls, challenges, deletes, and confidence-band outcomes.

### 78. How do you collect browser data?

The frontend uses `getUserMedia` for camera and microphone, converts data into arrays, downsamples them, and sends them as `client_samples`.

### 79. Why downsample browser samples?

Downsampling reduces payload size and makes processing faster while retaining enough signal structure for feature extraction.

### 80. What is the role of `server/collector_store.py`?

It saves browser-collected sample sessions to dataset storage with sanitized names, stable device IDs, manifests, and optional zip archives.

### 81. What is in `auth/device_embeddings/`?

It contains saved enrollment templates and metadata for enrolled devices.

### 82. What is in `model/evaluation/`?

It stores evaluation metrics JSON files for different model/source experiments.

### 83. What is the role of `scripts/diagnostics/demo_preflight.py`?

It checks model availability, database, hardware, API health, and a basic roundtrip before a demo.

## K. Database and API

### 84. Which database do you use?

SQLite by default through SQLAlchemy, configured as `sqlite:///./data/qna_auth.db`.

### 85. Can it use another database?

Yes. `DATABASE_URL` can be changed to another SQLAlchemy-supported database such as PostgreSQL.

### 86. What tables exist?

The main tables are `devices`, `challenges`, and `audit_log`.

### 87. Why use Pydantic models?

Pydantic validates API payloads and documents the request/response schema in FastAPI.

### 88. How is rate limiting done?

The backend stores timestamps per client IP in memory and rejects requests beyond the configured count within a time window.

### 89. Is in-memory rate limiting production-grade?

No. For production, use Redis or an API gateway so rate limits work across workers and restarts.

## L. Limitations and Defense

### 90. What is the biggest limitation?

The biggest limitation is dataset scale and real-world variability. More devices and cross-session evaluation are needed for stronger claims.

### 91. Why should the examiner trust your result?

The project includes a complete pipeline, stored metrics, tests for important guardrails, documented claim boundaries, and reproducible scripts. It does not overclaim production-grade security.

### 92. What if the examiner says the EER is high?

I would agree that it is high for standalone authentication. The correct interpretation is that this is a prototype auxiliary signal. The value is in the architecture, evaluation, and security-aware design, not a claim of production readiness.

### 93. Why not just use passwords?

Passwords authenticate knowledge, not the physical device. This project explores an additional passive device signal that could supplement stronger methods, not replace them.

### 94. Why not use device certificates?

Certificates are stronger cryptographic credentials. QNA-Auth is different: it investigates physical sensor fingerprints as a biometric-like device signal. In production, certificates would be preferable for strong identity, while sensor fingerprints could be risk scoring.

### 95. What is novel in your project?

The project integrates multi-source sensor noise collection, feature extraction, Siamese embedding learning, template-bank matching, confidence bands, drift gating, and nonce-bound challenge hardening in a complete capstone prototype.

### 96. What would you improve first?

I would improve dataset quality first: more physical devices, more sessions, stronger split discipline, and source-specific threshold calibration.

### 97. What is your strongest technical contribution?

The strongest contribution is the end-to-end verification architecture with explicit claim boundaries: feature extraction, learned embeddings, multi-source fusion, confidence bands, drift controls, and challenge-response hardening.

### 98. What is your weakest technical area?

The weakest area is empirical scale. Current data is enough for prototype demonstration but not enough for strong production security claims.

### 99. If the model fails, what fallback exists?

The system can return `uncertain` and recommend collecting more samples or using fallback authentication. In production, that fallback should be password, passkey, OTP, or admin review depending on context.

### 100. Summarize your project in 30 seconds.

QNA-Auth verifies devices using camera and microphone noise fingerprints. It extracts fixed signal features, maps them into embeddings with a Siamese network, stores enrollment templates, and compares fresh samples during authentication. It combines camera and microphone scores, returns confidence bands, and optionally hardens the flow with nonce-bound HKDF challenge-response. It is a security-aware prototype, not a production replacement for cryptographic authentication.

## M. Quick Oral Defense Lines

- "This is device verification, not direct user identity proof."
- "The embedding is a biometric-like template for similarity, not a password."
- "The uncertain band is intentional because sensor matching is statistical."
- "Camera currently performs better than microphone, so weighting is justified."
- "Challenge-response mitigates replay but does not remove all spoofing risk."
- "The most important future work is larger cross-session evaluation."
- "I would deploy this only as an auxiliary risk signal with fallback authentication."

