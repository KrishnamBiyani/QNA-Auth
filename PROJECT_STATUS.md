# QNA-Auth Project Status & Guide

**Last Updated**: February 1, 2026  
**Current Branch**: omkar  
**Status**: Research Prototype - Active Development

---

## � Abstract

**QNA-Auth (Quantum Noise Assisted Authentication)** is a novel device authentication system that uses the natural randomness from quantum sources and hardware sensors to create a unique "fingerprint" for each device. Unlike traditional authentication methods that rely on passwords, tokens, or biometrics—all of which can be stolen, copied, or faked—QNA-Auth leverages something that cannot be reproduced: the unique way each device generates and processes random noise.

Here's the simple idea: Every electronic device has subtle physical differences in how its camera sensor, microphone, and timing circuits behave. When we collect random noise from these sources (including true quantum random numbers from specialized APIs), each device produces a slightly different "signature" in the statistical patterns of that noise. We use a machine learning model (Siamese neural network) to learn these patterns and create a compact mathematical representation called an "embedding." During enrollment, we store this embedding. During authentication, we collect fresh noise, generate a new embedding, and check if it's similar enough to the stored one. Since every authentication uses brand-new random data, attackers cannot simply replay old data or synthesize fake noise—they would need to physically replicate the device's exact hardware characteristics, which is practically impossible.

---

## �📊 What's Been Done (Recent Updates)

### ✅ Core System Implementation
- **Device Fingerprinting**: Multi-source noise collection (QRNG, camera, microphone, system jitter)
- **ML Model**: Siamese neural network architecture for embedding generation
- **Feature Extraction**: Comprehensive pipeline (stats, FFT, entropy, autocorrelation, complexity measures)
- **Backend API**: FastAPI server with RESTful endpoints
- **Frontend UI**: React + TypeScript interface for device management
- **Challenge-Response Protocol**: Anti-replay protection framework

### ✅ Database Layer (NEW)
- **SQLAlchemy Models**: Structured data storage for devices, challenges, and audit logs
- **Device Management**: Proper metadata tracking and embedding path management
- **Challenge Storage**: Persistent challenge storage replacing in-memory dictionary
- **Audit Trail**: Logging system for all authentication events

### ✅ Documentation & Planning
- **VISION.md**: Core innovation and scientific approach
- **IMPROVEMENTS.md**: Detailed gap analysis and security enhancements (875 lines)
- **ROADMAP.md**: Research-level priorities and implementation guide
- **DATASET.md**: Data collection and organization guidelines
- **DATA_COLLECTION.md**: Detailed data gathering procedures

### ✅ Scripts & Automation
- **Database Scripts**: `init_db.py`, `check_db.py`, `backfill_db_from_files.py`
- **Data Collection**: `collect_data_for_training.py`, `auto_collect.py`, `ingest_collected_data.py`
- **Training Pipeline**: `train_and_evaluate.py`, `run_full_training.py`
- **Testing Suite**: `test_enrollment.py`, `test_cross_device.py`, `test_robustness.py`
- **Demo**: `run_demo.ps1` for Windows automation
- **Evaluation**: `run_evaluation.py` - FAR/FRR/EER metrics and ROC curves
- **Synthetic Data**: `generate_synthetic_data.py` - Generate test data

### ✅ Frontend Enhancements
- **Data Collectors Service**: New `collectors.ts` for unified noise collection
- **Improved UI Pages**: Enhanced enrollment and authentication interfaces
- **Accessibility Fixes**: Added ARIA labels and proper form elements

### ✅ Security & Infrastructure (NEW)
- **Rate Limiting**: `rate_limiter.py` - Protection against brute-force attacks
- **API Key Auth**: `api_auth.py` - Optional API key authentication
- **Configuration**: `config.py` with environment variable support
- **Environment Files**: `.env` and `.env.example` for secrets management
- **Docker Support**: `Dockerfile` and `docker-compose.yml` for containerization
- **Run Scripts**: `run.py` - Unified CLI for setup, start, train, evaluate

---

## 🎯 Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React/TS)                       │
│  - Enrollment UI  - Authentication UI  - Device Management   │
└────────────────┬────────────────────────────────────────────┘
                 │ REST API (HTTP/JSON)
┌────────────────▼────────────────────────────────────────────┐
│                  Backend (FastAPI + SQLAlchemy)              │
│  - /enroll  - /authenticate  - /challenge  - /verify         │
└────┬──────────┬──────────────┬──────────────────────────────┘
     │          │               │
     ▼          ▼               ▼
┌─────────┐ ┌──────────┐ ┌─────────────────────────────┐
│ SQLite  │ │ PyTorch  │ │   Noise Collection Modules  │
│   DB    │ │  Model   │ │  - QRNG API                 │
└─────────┘ └──────────┘ │  - Camera sensor noise       │
                         │  - Microphone noise          │
                         │  - System jitter             │
                         └─────────────────────────────┘
```

---

## 🚀 What's Working Right Now

### Enrollment Flow
1. ✅ Collect N samples from selected noise sources
2. ✅ Extract features (160+ dimensions per sample)
3. ✅ Generate embeddings via Siamese model
4. ✅ Aggregate embeddings (mean pooling)
5. ✅ Store to database + file system
6. ✅ UI feedback and progress tracking

### Authentication Flow
1. ✅ Collect fresh noise samples
2. ✅ Generate authentication embedding
3. ✅ Compute cosine similarity with stored embedding
4. ✅ Threshold-based decision (default: 0.85)
5. ✅ Return authentication result

### Challenge-Response Flow
1. ✅ Generate nonce challenge
2. ✅ Store in persistent database with TTL
3. ✅ Verify response with embedding similarity
4. ✅ Prevent replay attacks

---

## ⚠️ Known Limitations & Gaps

### 🔴 Critical (Security)
- ❌ **No trained model**: System runs with random weights (placeholder)
- ❌ **No encryption**: Embeddings stored as plaintext `.pt` files
- ❌ **No HTTPS**: API runs on HTTP only (use reverse proxy for production)
- ✅ ~~**No authentication**~~: API key auth available (`API_KEY_ENABLED=true`)
- ✅ ~~**No rate limiting**~~: Rate limiting middleware added
- ✅ ~~**Secrets in code**~~: Now uses `.env` file and config.py

### 🟡 Medium (Functionality)
- ⚠️ **No real training data**: Need multi-device dataset for proper training
- ✅ ~~**Feature pipeline not versioned**~~: FEATURE_VERSION now tracked
- ⚠️ **Threshold not optimized**: Using arbitrary 0.85 threshold
- ✅ ~~**No metrics**~~: FAR/FRR/EER evaluation script added
- ⚠️ **Limited testing**: Cross-device validation incomplete

### 🟢 Low (Polish)
- ⚠️ Frontend error handling could be improved
- ⚠️ No real-time progress for long collections
- ⚠️ UI styling could be more polished

---

## 📋 What Needs to Be Done Next

### Phase 1: Data & Training (HIGH PRIORITY)
**Goal**: Get a properly trained model with real performance metrics

1. **Collect Multi-Device Dataset**
   - [ ] Run `auto_collect.py` on 5-10 different devices
   - [ ] Or generate synthetic data: `python scripts/generate_synthetic_data.py`
   - [ ] Gather 100+ samples per device across multiple sessions
   - [ ] Organize in `dataset/samples/` with proper structure
   - [ ] Document device specs and collection conditions

2. **Train the Model**
   - [ ] Run `python scripts/train_and_evaluate.py`
   - [x] Train/validation/test splits implemented
   - [x] Checkpointing and early stopping available
   - [x] Random seeds set for reproducibility

3. **Measure Performance**
   - [x] Run `python scripts/run_evaluation.py` for metrics
   - [x] FAR/FRR/EER calculation implemented
   - [x] ROC and PR curves generated
   - [x] Threshold sweep with optimal point recommendation
   - [ ] Test cross-device generalization with real data

4. **Feature Pipeline**
   - [x] Feature list versioned (FEATURE_VERSION)
   - [x] Feature configuration saved with model
   - [x] Same features at train and inference time

### Phase 2: Security Hardening (MOSTLY COMPLETE)
**Goal**: Make the system secure for demonstration/testing

1. **Encryption**
   - [ ] Implement AES-256-GCM for embedding encryption
   - [x] Keys read from environment variables
   - [ ] Add key rotation mechanism

2. **API Security**
   - [x] API key authentication available (`server/api_auth.py`)
   - [x] Rate limiting implemented (`server/rate_limiter.py`)
   - [ ] Add HTTPS/TLS support (use reverse proxy)
   - [x] CORS configurable via environment

3. **Secrets Management**
   - [x] All secrets in `.env` file
   - [x] `config.py` reads from environment
   - [x] `.env.example` documents required variables
   - [x] Config validation function added

4. **Audit & Monitoring**
   - [x] Audit logging in database (AuditLog model)
   - [x] Structured logging configured
   - [x] Rate limiter tracks failed attempts
   - [ ] Alert on suspicious patterns (future)

### Phase 3: Research Validation (MEDIUM PRIORITY)
**Goal**: Prove the scientific claims

1. **Ablation Studies**
   - [ ] Test QRNG-only authentication
   - [ ] Test camera-only authentication
   - [ ] Test microphone-only authentication
   - [ ] Test combined (prove improvement)

2. **Attack Resistance**
   - [ ] Test replay attack resistance
   - [ ] Test synthetic noise generation attacks
   - [ ] Test device cloning resistance
   - [ ] Document failure modes

3. **Statistical Analysis**
   - [ ] Bootstrap confidence intervals for metrics
   - [ ] Multiple runs with different seeds
   - [ ] Distribution analysis of similarity scores
   - [ ] Failure case analysis

4. **Comparison Baselines**
   - [ ] Raw feature cosine similarity (no NN)
   - [ ] Simple MLP instead of Siamese
   - [ ] Traditional hash-based approach
   - [ ] Show Siamese improvement

### Phase 4: Documentation & Publication (MEDIUM PRIORITY)
**Goal**: Make findings reproducible and publishable

1. **Technical Documentation**
   - [ ] Complete API documentation
   - [ ] Architecture diagrams
   - [ ] Data flow diagrams
   - [ ] Security threat model

2. **Research Report**
   - [ ] Problem statement and motivation
   - [ ] Related work (PUF, QRNG, device fingerprinting)
   - [ ] Method description with equations
   - [ ] Experimental results (tables and figures)
   - [ ] Limitations and future work
   - [ ] Reproducibility section

3. **Reproducibility Package**
   - [ ] Single command to reproduce all results
   - [ ] Docker container with fixed environment
   - [ ] Example dataset included
   - [ ] Pre-trained model checkpoint

### Phase 5: Polish & Demo (LOW PRIORITY)
**Goal**: Make it presentable for demos and users

1. **Frontend Improvements**
   - [ ] Real-time collection progress
   - [ ] Better error messages
   - [ ] Visual feedback for similarity scores
   - [ ] Device comparison visualizations

2. **Performance Optimization**
   - [ ] Optimize `approximate_entropy` (currently O(N²))
   - [ ] Cache feature extraction results
   - [ ] Async collection where possible
   - [ ] Model quantization for faster inference

---

## 🛠️ Quick Start Guide

### Setup (First Time) - RECOMMENDED

```powershell
# One-command setup (creates venv, installs deps, initializes db)
python run.py setup

# Start the server
python run.py start
```

### Alternative Manual Setup

```powershell
# Backend setup
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Initialize database
python scripts/init_db.py

# Frontend setup
cd frontend
npm install
cd ..
```

### Development Workflow

```powershell
# Option 1: Use run.py (recommended)
python run.py dev  # Starts both backend and frontend

# Option 2: Manual (separate terminals)
# Terminal 1: Start backend
.\venv\Scripts\activate
python server/app.py
# Backend runs on http://localhost:8000

# Terminal 2: Start frontend
cd frontend
npm run dev
# Frontend runs on http://localhost:3000
```

### Device Fingerprint Collection (AUTHENTIC DATA ONLY)

```powershell
# Interactive collection (recommended)
python run.py collect

# Direct script with options
python scripts/collect_device_fingerprint.py --device-name "My Laptop" --samples 50

# Quick test (10 samples)
python scripts/collect_device_fingerprint.py --device-name "Test Device" --samples 10

# Specific sources only
python scripts/collect_device_fingerprint.py --device-name "Server" --sources qrng,system

# Use the frontend UI
# Navigate to http://localhost:3000 and click "Enroll New Device"
```

**Important**: All data is collected from YOUR REAL device. No synthetic or fake data.
Each device generates a unique fingerprint based on its actual hardware noise patterns.

### Training

```powershell
# Train model on collected fingerprint data
python scripts/train_and_evaluate.py

# Or full pipeline
python run_full_training.py

# Or use run.py
python run.py train
```

### Evaluation

```powershell
# Run comprehensive evaluation
python scripts/run_evaluation.py

# Or use run.py
python run.py evaluate

# Results saved to model/evaluation/
```

### Testing

```powershell
# Run all tests
python run.py test

# Individual tests
python test_enrollment.py

# Test cross-device authentication
python test_cross_device.py

# Test robustness
python test_robustness.py
```

---

## 📁 Project Structure

```
QNA-Auth/
├── auth/                          # Core authentication logic
│   ├── enrollment.py              # Device enrollment
│   ├── authentication.py          # Authentication verification
│   ├── challenge_response.py      # Challenge-response protocol
│   └── device_embeddings/         # Stored embeddings (*.pt, *.json)
│
├── db/                            # Database layer
│   ├── models.py                  # SQLAlchemy models
│   ├── session.py                 # Database session management
│   └── challenge_store.py         # Challenge persistence
│
├── noise_collection/              # Entropy sources (REAL HARDWARE)
│   ├── qrng_api.py                # Quantum random numbers
│   ├── camera_noise.py            # Camera sensor noise
│   ├── mic_noise.py               # Microphone noise
│   └── sensor_noise.py            # System timing jitter
│
├── preprocessing/                 # Feature extraction
│   ├── features.py                # Feature computation (versioned)
│   └── utils.py                   # Helper functions
│
├── model/                         # Machine learning
│   ├── siamese_model.py           # Siamese network architecture
│   ├── train.py                   # Training loop
│   ├── evaluate.py                # Evaluation metrics
│   ├── checkpoints/               # Saved model weights
│   └── evaluation/                # Evaluation reports & plots
│
├── dataset/                       # Data management
│   ├── builder.py                 # Dataset construction
│   ├── samples/                   # Collected device fingerprints
│   └── processed/                 # Train/test splits
│
├── server/                        # Backend API
│   ├── app.py                     # FastAPI application
│   ├── routes.py                  # API endpoints
│   ├── rate_limiter.py            # Rate limiting middleware
│   └── api_auth.py                # API key authentication
│
├── frontend/                      # Web UI
│   ├── src/
│   │   ├── pages/                 # UI pages (Enroll, Authenticate, Devices)
│   │   ├── services/              # API client + collectors
│   │   └── components/            # Reusable components
│   └── package.json
│
├── scripts/                       # Automation scripts
│   ├── init_db.py                 # Database initialization
│   ├── collect_device_fingerprint.py  # Real device fingerprint collection
│   ├── train_and_evaluate.py      # Training pipeline
│   ├── run_evaluation.py          # FAR/FRR/EER metrics
│   └── backfill_db_from_files.py  # Migrate old data
│
├── docs/                          # Documentation
│   ├── DATASET.md                 # Dataset guidelines
│   └── DATA_COLLECTION.md         # Collection procedures
│
├── config.py                      # Configuration (from env)
├── run.py                         # Unified CLI runner
├── .env.example                   # Environment template
├── VISION.md                      # Core innovation explanation
├── IMPROVEMENTS.md                # Gap analysis (875 lines)
├── ROADMAP.md                     # Research priorities
├── README.md                      # Main documentation
└── PROJECT_STATUS.md              # This file
```

---

## 🔬 Research Focus

This is **not a production web app**. This is a **research project** proving that:

1. **Quantum noise can reliably fingerprint devices**
2. **ML can learn device-specific randomness patterns**
3. **The system is resistant to replay/cloning attacks**
4. **Performance meets practical security requirements** (low FAR/FRR)

### Success Criteria (Research-Level)
- ✅ Working end-to-end system
- ✅ Reproducible evaluation pipeline (`run_evaluation.py`)
- ✅ Authentic device fingerprint collection
- ⏳ Trained model with <5% EER on multi-device dataset
- ⏳ ROC curves and threshold analysis
- ⏳ Ablation studies showing multi-source improvement
- ⏳ Security analysis documenting attack resistance
- ⏳ Technical report with findings

---

## 🤝 Contributing & Next Steps

### For Team Members

**Current Status**:
- ✅ Frontend accessibility fixes
- ✅ Latest main branch changes merged
- ✅ Security middleware added (rate limiting, API auth)
- ✅ Evaluation pipeline ready
- ✅ Docker deployment ready

**Immediate Next Actions**:
1. Generate synthetic data: `python scripts/generate_synthetic_data.py`
2. Train model: `python scripts/train_and_evaluate.py`
3. Run evaluation: `python scripts/run_evaluation.py`
4. Test with real devices when available

### For Future Research

- Investigate hardware-specific QRNG sources
- Explore alternative embedding architectures
- Test on mobile devices (iOS/Android)
- Extend to continuous authentication
- Publish findings in security/crypto venue

---

## 📞 Support & Resources

- **Main README**: Comprehensive setup guide
- **VISION.md**: Understanding the core innovation
- **IMPROVEMENTS.md**: Detailed implementation guide
- **ROADMAP.md**: Research priorities
- **API Docs**: http://localhost:8000/docs (when server running)

---

## 🏆 Project Vision

**Goal**: Prove that quantum noise + ML = secure, physics-based authentication

**Not a goal**: Production-ready web service (it's a prototype!)

**Impact**: Novel authentication primitive for research community and industry

---

*This document provides a snapshot of project status. Update as major milestones are achieved.*
