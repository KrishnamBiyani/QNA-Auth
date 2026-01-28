# QNA-Auth: Research & Development Roadmap

> **Note**: This project is a **research prototype**, not a production web application. Priorities below are reordered to emphasize the core quantum authentication innovation over web development polish.

This document provides a **detailed gap analysis** comparing the current implementation against the stated mission, and outlines specific improvements needed to realize the full vision of a "physics-based, AI-validated, cryptographically reinforced" authentication system.

**Primary goal**: Prove that quantum noise can reliably fingerprint devices and publish findings.  
**Secondary goal**: Provide a reference implementation for researchers and industry.

---

## Research-first priorities (reordered)

### What matters most
1. **Core authentication mechanism** - Prove quantum noise can fingerprint devices
2. **Statistical validation** - Demonstrate distinguishability and stability
3. **Security analysis** - Prove resistance to attacks (replay, cloning, synthesis)
4. **Model optimization** - Achieve <1% FAR/FRR with minimal samples
5. **Academic publication** - Document findings in peer-reviewed venue

### What matters less (for research project)
- Production API scalability (this is a prototype)
- Web UI polish (Jupyter notebooks are sufficient)
- User management and multi-tenancy (one researcher at a time)
- Deployment automation (Docker for reproducibility is enough)
- Monitoring dashboards (experiment logs are sufficient)

**If you have limited time**: Focus on sections 2 (ML & Model), 3 (Anti-Attack), and the data collection/analysis workflow. The web stack (FastAPI/React) is just a demo harness.

---

## Mission statement recap

> "QNA-Auth is a novel authentication system that uses quantum noise patterns combined with lightweight AI models to provide highly secure, non-reproducible device authentication... The system does not store raw noise; instead, it stores a secure, non-invertible embedding... Because each authentication uses new, unpredictable noise data, replay and synthetic attacks are ineffective."

---

## Current state assessment

### ✅ What's working
- ✅ Multi-source noise collection (QRNG, camera, microphone, system jitter)
- ✅ Feature extraction pipeline (stats + FFT + entropy + autocorrelation)
- ✅ Siamese embedding model architecture
- ✅ FastAPI backend with enrollment/authentication endpoints
- ✅ React frontend for device management
- ✅ Challenge-response protocol foundation
- ✅ Embeddings stored (not raw noise)

### ❌ Critical gaps (blockers for production)

1. **No trained model** - Current system runs with random weights
2. **No encryption at rest** - Device embeddings stored as plaintext PyTorch tensors
3. **No secure channel** - HTTP-only API (no TLS/HTTPS)
4. **No authentication/authorization** - Anyone can enroll/delete devices
5. **In-memory storage only** - No persistent database for challenges/sessions
6. **Limited anti-replay protection** - Challenge protocol exists but not fully integrated
7. **No rate limiting** - Vulnerable to brute-force enrollment/authentication attempts
8. **No audit logging** - No forensic trail for security events
9. **No model security** - Model weights not protected from extraction/tampering

---

## Detailed improvements by category

### 1. Security & Cryptography (CRITICAL)

#### 1.1 Encryption at rest
**Current:** Embeddings saved as plaintext `.pt` files  
**Required:**
- Encrypt device embeddings using AES-256-GCM or ChaCha20-Poly1305
- Derive encryption keys from a hardware security module (HSM) or secure key management service
- Store initialization vectors (IVs) separately from ciphertext
- Implement key rotation without re-enrolling all devices

**Implementation:**
```python
# auth/encryption.py (new file)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend
import os

class EmbeddingEncryptor:
    def __init__(self, master_key: bytes):
        self.aesgcm = AESGCM(master_key)
    
    def encrypt_embedding(self, embedding_bytes: bytes) -> tuple[bytes, bytes]:
        nonce = os.urandom(12)
        ciphertext = self.aesgcm.encrypt(nonce, embedding_bytes, None)
        return nonce, ciphertext
    
    def decrypt_embedding(self, nonce: bytes, ciphertext: bytes) -> bytes:
        return self.aesgcm.decrypt(nonce, ciphertext, None)
```

#### 1.2 Secure communication (TLS/HTTPS)
**Current:** Backend runs on HTTP  
**Required:**
- Deploy behind nginx/Apache with TLS 1.3
- Use Let's Encrypt or enterprise CA for certificates
- Enforce HTTPS-only in production
- Implement certificate pinning in frontend for additional security

**Implementation:**
```bash
# nginx config
server {
    listen 443 ssl http2;
    ssl_certificate /etc/letsencrypt/live/yourdomain/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain/privkey.pem;
    ssl_protocols TLSv1.3;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header X-Forwarded-Proto https;
    }
}
```

#### 1.3 API authentication & authorization
**Current:** Endpoints are completely open  
**Required:**
- Implement OAuth2/JWT for API authentication
- Role-based access control (RBAC): admin vs. device user
- API key management for programmatic access
- Device-specific access tokens (device can only authenticate itself)

**Implementation:**
```python
# server/auth.py (new file)
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_device(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        device_id: str = payload.get("sub")
        if device_id is None:
            raise HTTPException(status_code=401)
        return device_id
    except JWTError:
        raise HTTPException(status_code=401)

# Apply to endpoints:
@app.post("/authenticate")
async def authenticate(
    request: AuthRequest,
    current_device: str = Depends(get_current_device)
):
    if request.device_id != current_device:
        raise HTTPException(status_code=403, detail="Cannot authenticate other devices")
    # ... rest of logic
```

#### 1.4 Enhanced challenge-response protocol
**Current:** Basic nonce generation, not fully integrated  
**Required:**
- Implement mutual authentication (server also proves identity to device)
- Add timestamp validation to prevent time-shift attacks
- Cryptographically bind challenge to device identity using HMAC
- Store challenge history in persistent database (not in-memory)

**Implementation:**
```python
# auth/challenge_response.py (enhance existing)
class SecureChallengeProtocol:
    def create_challenge(self, device_id: str) -> dict:
        nonce = secrets.token_bytes(32)
        timestamp = int(time.time())
        
        # Bind challenge to device using HMAC
        challenge_data = f"{device_id}:{nonce.hex()}:{timestamp}".encode()
        signature = hmac.new(self.server_secret, challenge_data, hashlib.sha256).digest()
        
        challenge_id = hashlib.sha256(challenge_data + signature).hexdigest()
        
        # Store in persistent DB (Redis/PostgreSQL)
        self.db.store_challenge(challenge_id, {
            'device_id': device_id,
            'nonce': nonce.hex(),
            'timestamp': timestamp,
            'signature': signature.hex(),
            'expires_at': timestamp + 60
        })
        
        return {
            'challenge_id': challenge_id,
            'nonce': nonce.hex(),
            'server_signature': signature.hex(),
            'timestamp': timestamp
        }
```

---

### 2. Machine Learning & Model Security

#### 2.1 Train a production model
**Current:** Model uses random initialization  
**Required:**
- Collect diverse training data from 50+ real devices
- Train with triplet/contrastive loss for 50-100 epochs
- Perform cross-validation and hyperparameter tuning
- Achieve >95% verification accuracy and <1% FAR/FRR

**Training workflow:**
```bash
# 1. Collect training data
python collect_training_data.py --devices 50 --samples-per-device 100

# 2. Train model
python model/train.py \
    --data dataset/training_data \
    --epochs 100 \
    --batch-size 64 \
    --loss triplet \
    --margin 1.0 \
    --lr 0.001

# 3. Evaluate
python model/evaluate.py \
    --checkpoint model/checkpoints/best_model.pt \
    --test-data dataset/test_data
```

#### 2.2 Model security & integrity
**Current:** Model checkpoint is a regular file  
**Required:**
- Sign model weights with digital signature (prevent tampering)
- Verify signature on model load
- Encrypt model weights to prevent reverse engineering
- Implement model versioning and rollback capability

**Implementation:**
```python
# model/security.py (new file)
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key

class ModelSigner:
    def sign_model(self, model_path: str, private_key_path: str):
        # Load model bytes
        with open(model_path, 'rb') as f:
            model_bytes = f.read()
        
        # Load private key
        with open(private_key_path, 'rb') as f:
            private_key = load_pem_private_key(f.read(), password=None)
        
        # Sign
        signature = private_key.sign(
            model_bytes,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )
        
        # Save signature
        with open(f"{model_path}.sig", 'wb') as f:
            f.write(signature)
    
    def verify_model(self, model_path: str, public_key_path: str) -> bool:
        # Implementation similar to above but using public_key.verify()
        pass
```

#### 2.3 Adaptive threshold calibration
**Current:** Fixed threshold (0.85)  
**Required:**
- Per-device threshold calibration based on enrollment quality
- Adaptive threshold adjustment based on authentication history
- Confidence scoring (not just binary accept/reject)

**Implementation:**
```python
# auth/adaptive_threshold.py (new file)
class AdaptiveThreshold:
    def calibrate_for_device(self, device_id: str, enrollment_samples: list):
        # Compute intra-device similarity distribution
        similarities = []
        for i in range(len(enrollment_samples)):
            for j in range(i+1, len(enrollment_samples)):
                sim = compute_similarity(enrollment_samples[i], enrollment_samples[j])
                similarities.append(sim)
        
        # Set threshold at mean - 2*std (allow 95% of legitimate samples)
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        threshold = mean_sim - 2 * std_sim
        
        # Clamp to reasonable range
        return max(0.7, min(0.95, threshold))
```

---

### 3. Anti-Replay & Anti-Spoofing

#### 3.1 Enhanced replay protection
**Current:** Basic challenge expiry  
**Required:**
- Persistent challenge history (prevent reuse across server restarts)
- Device-specific challenge counters (monotonic, prevent rollback)
- Bind challenges to client IP/session fingerprint

**Implementation:**
```python
# auth/anti_replay.py (new file)
class EnhancedAntiReplay:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def check_and_record_challenge(self, challenge_id: str, device_id: str) -> bool:
        # Check if challenge already used
        key = f"challenge_used:{challenge_id}"
        if self.redis.exists(key):
            logger.warning(f"Replay attack detected: {challenge_id}")
            return False
        
        # Mark as used (with 1-hour TTL)
        self.redis.setex(key, 3600, "1")
        
        # Increment device counter
        counter_key = f"device_counter:{device_id}"
        self.redis.incr(counter_key)
        
        return True
```

#### 3.2 Liveness detection
**Current:** No liveness checks  
**Required:**
- For camera: require active user interaction (e.g., movement detection)
- For microphone: analyze temporal patterns (synthetic audio detection)
- Time-based freshness validation (reject stale samples)

**Implementation:**
```python
# noise_collection/liveness.py (new file)
class LivenessDetector:
    def detect_camera_liveness(self, frames: list) -> bool:
        # Compute frame-to-frame differences
        diffs = []
        for i in range(len(frames)-1):
            diff = np.abs(frames[i+1] - frames[i]).sum()
            diffs.append(diff)
        
        # Synthetic/replayed video will have low variance
        variance = np.var(diffs)
        return variance > LIVENESS_THRESHOLD
    
    def detect_microphone_liveness(self, audio: np.ndarray) -> bool:
        # Analyze spectral characteristics of genuine noise
        # vs. replayed/synthetic audio
        pass
```

#### 3.3 Synthetic noise detection
**Current:** No defense against AI-generated noise  
**Required:**
- Train a discriminator network to detect synthetic noise samples
- Analyze statistical properties that differ between real and synthetic noise
- Implement ensemble detection (multiple checks)

---

### 4. Database & Persistence

#### 4.1 Replace in-memory storage
**Current:** Challenges and sessions stored in Python dicts  
**Required:**
- PostgreSQL for device metadata, audit logs, user accounts
- Redis for active challenges, rate limiting, session management
- Proper transaction handling and ACID guarantees

**Schema:**
```sql
-- devices table
CREATE TABLE devices (
    device_id VARCHAR(64) PRIMARY KEY,
    device_name VARCHAR(255),
    enrollment_date TIMESTAMP NOT NULL,
    last_auth_date TIMESTAMP,
    auth_count INTEGER DEFAULT 0,
    status VARCHAR(32) DEFAULT 'active',
    metadata JSONB
);

-- challenges table
CREATE TABLE challenges (
    challenge_id VARCHAR(64) PRIMARY KEY,
    device_id VARCHAR(64) REFERENCES devices(device_id),
    nonce VARCHAR(128) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    used BOOLEAN DEFAULT FALSE,
    used_at TIMESTAMP
);

-- audit_log table
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    event_type VARCHAR(64) NOT NULL,
    device_id VARCHAR(64),
    ip_address INET,
    user_agent TEXT,
    result VARCHAR(32),
    details JSONB
);
```

#### 4.2 Implement proper database layer
```python
# server/database.py (new file)
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Device(Base):
    __tablename__ = 'devices'
    device_id = Column(String(64), primary_key=True)
    device_name = Column(String(255))
    enrollment_date = Column(DateTime, nullable=False)
    # ... rest of fields

class Challenge(Base):
    __tablename__ = 'challenges'
    challenge_id = Column(String(64), primary_key=True)
    # ... rest of fields

# Connection management
engine = create_engine('postgresql://user:pass@localhost/qnaauth')
SessionLocal = sessionmaker(bind=engine)
```

---

### 5. Performance & Scalability

#### 5.1 Model optimization
**Current:** Full PyTorch model inference (slow on CPU)  
**Required:**
- Convert to ONNX for faster inference
- Quantize model (INT8) for edge deployment
- Batch authentication requests for throughput

**Implementation:**
```python
# model/optimize.py (new file)
import torch
import onnx
import onnxruntime as ort

def convert_to_onnx(model_path: str, output_path: str):
    model = torch.load(model_path)
    model.eval()
    
    dummy_input = torch.randn(1, input_dim)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        input_names=['input'],
        output_names=['output']
    )

# Use ONNX Runtime for inference
class ONNXEmbedder:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
    
    def embed(self, features):
        return self.session.run(None, {'input': features})[0]
```

#### 5.2 Caching layer
**Current:** No caching  
**Required:**
- Cache device embeddings in Redis for fast lookup
- Cache feature extraction results for repeated samples
- Implement TTL and invalidation strategies

#### 5.3 Async processing
**Current:** Synchronous enrollment (blocks for minutes)  
**Required:**
- Background task queue (Celery/RQ) for enrollment
- Websocket progress updates to frontend
- Parallel noise collection from multiple sources

---

### 6. Hardware & IoT Considerations

#### 6.1 Lightweight model variants
**Current:** Single model size  
**Required:**
- Model quantization for edge devices (TensorFlow Lite, ONNX Mobile)
- Pruned models for constrained environments
- Feature subset selection based on available hardware

#### 6.2 Fallback strategies
**Current:** Requires all selected sources to work  
**Required:**
- Graceful degradation (authenticate with fewer sources)
- Source prioritization (QRNG > camera > mic > sensors)
- Adaptive sampling (collect more if initial confidence is low)

#### 6.3 Power efficiency
**Required:**
- Lazy loading of ML models
- Feature computation on-device (reduce bandwidth)
- Configurable authentication frequency vs. security tradeoff

---

### 7. Monitoring & Observability

#### 7.1 Comprehensive logging
**Current:** Basic `logger.info()` statements  
**Required:**
- Structured logging (JSON format)
- Centralized log aggregation (ELK stack, Splunk)
- Security event monitoring (failed auth attempts, suspicious patterns)

**Implementation:**
```python
# server/logging_config.py (new file)
import logging
import json
from datetime import datetime

class StructuredLogger:
    def log_event(self, event_type: str, device_id: str = None, **kwargs):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'device_id': device_id,
            **kwargs
        }
        logging.info(json.dumps(log_entry))

# Usage:
logger.log_event('enrollment_started', device_id='abc123', sources=['qrng', 'camera'])
logger.log_event('authentication_failed', device_id='abc123', similarity=0.72, threshold=0.85)
```

#### 7.2 Metrics & alerting
**Required:**
- Prometheus metrics export
- Grafana dashboards
- Alert on: high failure rates, unusual enrollment patterns, model degradation

**Metrics to track:**
```python
# server/metrics.py (new file)
from prometheus_client import Counter, Histogram, Gauge

enrollment_counter = Counter('qna_enrollments_total', 'Total enrollments', ['status'])
auth_counter = Counter('qna_authentications_total', 'Total auth attempts', ['result'])
similarity_histogram = Histogram('qna_similarity_scores', 'Similarity score distribution')
active_devices_gauge = Gauge('qna_active_devices', 'Number of enrolled devices')
```

---

### 8. Testing & Quality Assurance

#### 8.1 Unit tests
**Current:** No automated tests  
**Required:**
- Unit tests for all core modules (pytest)
- Mock noise sources for deterministic testing
- Test edge cases (empty samples, corrupted data)

**Test structure:**
```
tests/
  test_noise_collection.py
  test_preprocessing.py
  test_model.py
  test_auth.py
  test_challenge_response.py
  test_api.py
```

#### 8.2 Integration tests
**Required:**
- End-to-end enrollment + authentication flows
- Challenge-response protocol testing
- Multi-device scenarios
- Load testing (concurrent authentication requests)

#### 8.3 Security testing
**Required:**
- Penetration testing
- Fuzzing of API endpoints
- Replay attack simulation
- Model extraction attempts

---

### 9. Documentation & Deployment

#### 9.1 Enhanced documentation
**Current:** README + QUICKSTART  
**Required:**
- API documentation (OpenAPI/Swagger)
- Security whitepaper
- Deployment guide (Docker, Kubernetes)
- Compliance documentation (GDPR, SOC2)

#### 9.2 Docker & container orchestration
**Required:**
```yaml
# docker-compose.yml
version: '3.8'
services:
  backend:
    build: .
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/qnaauth
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
  
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
  
  db:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
```

#### 9.3 CI/CD pipeline
**Required:**
```yaml
# .github/workflows/test-and-deploy.yml
name: Test and Deploy
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: pytest tests/
      - name: Security scan
        run: bandit -r .
  
  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: # deployment script
```

---

## Prioritized roadmap (research-focused)

### Phase 1: Core authentication validation (3-4 weeks) ⭐⭐⭐
**Goal**: Prove the fundamental hypothesis
1. ✅ Collect data from 50+ diverse devices (phones, laptops, IoT)
2. ✅ Implement comprehensive feature extraction (beyond basic stats)
3. ✅ Train multiple model architectures (compare triplet vs. contrastive)
4. ✅ Measure intra-device vs. inter-device similarity distributions
5. ✅ Achieve >95% accuracy with <1% FAR/FRR

**Deliverable**: Evidence that quantum noise can reliably fingerprint devices

### Phase 2: Security & attack resistance (2-3 weeks) ⭐⭐⭐
**Goal**: Demonstrate robustness against realistic attacks
1. ✅ Implement replay attack simulation (reuse old samples)
2. ✅ Implement synthesis attack (GAN-generated noise)
3. ✅ Implement cloning attack (identical hardware)
4. ✅ Enhanced challenge-response protocol with cryptographic binding
5. ✅ Model inversion resistance analysis

**Deliverable**: Security evaluation showing >99.9% attack failure rate

### Phase 3: Longitudinal stability study (6-12 weeks, parallel) ⭐⭐
**Goal**: Prove stability over time
1. ✅ Daily re-authentication for 90 days
2. ✅ Measure similarity drift and identify factors (temperature, battery, etc.)
3. ✅ Develop drift compensation strategies
4. ✅ Test re-enrollment triggers

**Deliverable**: Evidence that device signatures remain stable

### Phase 4: Paper writing & publication (4-6 weeks) ⭐⭐⭐
**Goal**: Document findings for academic community
1. ✅ Write paper (IEEE S&P, CCS, NDSS, or USENIX Security format)
2. ✅ Create figures (ROC curves, t-SNE embeddings, attack results)
3. ✅ Prepare artifact (code + data for reproducibility)
4. ✅ Submit to top-tier venue

**Deliverable**: Accepted paper at major security conference

### Phase 5: Reference implementation polish (2-3 weeks) ⭐
**Goal**: Make it usable by other researchers
1. ✅ Clean up code and add documentation
2. ✅ Create Jupyter notebooks for demos
3. ✅ Package as pip-installable library
4. ✅ Minimal web demo (optional, for presentations)
5. ✅ Docker for reproducibility

**Deliverable**: Open-source reference implementation on GitHub

### Phase 6: Production hardening (only if commercializing) ⭐
**Note**: Skip this phase if goal is purely research
1. ⏳ Encryption at rest for embeddings
2. ⏳ TLS/HTTPS deployment
3. ⏳ PostgreSQL + Redis integration
4. ⏳ Monitoring & alerting
5. ⏳ Security audit & penetration testing

---

## Metrics for success

### Security metrics
- **Zero successful replay attacks** in penetration testing
- **FAR (False Acceptance Rate) < 0.01%**
- **FRR (False Rejection Rate) < 1%**
- **Model extraction resistance** (confirmed by red team)

### Performance metrics
- **Enrollment time < 60 seconds**
- **Authentication latency < 2 seconds**
- **Support 10,000+ concurrent authentications**
- **Model inference < 50ms on CPU**

### Reliability metrics
- **99.9% uptime**
- **Zero data loss** (encrypted backups)
- **Automatic failover** for high availability

---

## Conclusion

The current implementation provides a **solid proof-of-concept** but needs focus on the **core research contribution** rather than production web development.

### For a research project (recommended focus)
The most critical gaps are:
1. **Trained model**: Need real data from 50+ devices to prove the hypothesis
2. **Statistical validation**: Measure and document distinguishability rigorously
3. **Security analysis**: Demonstrate attack resistance with formal threat model
4. **Academic paper**: Document methodology and results for peer review

**Estimated effort**: 10-15 weeks with 1-2 researchers  
**Next action**: Start Phase 1 (data collection and model training)

### For a production system (only if commercializing)
Additional requirements:
1. **Security hardening**: Encryption, TLS, authentication, audit logging
2. **Infrastructure**: PostgreSQL, Redis, monitoring, high availability
3. **Compliance**: GDPR, SOC2, penetration testing
4. **Scalability**: Load balancing, horizontal scaling, CDN

**Estimated effort**: Additional 10-14 weeks with 2-3 engineers  
**Next action**: Complete research validation first, then implement Phase 6

---

## Recommended focus

**If this is a capstone/thesis project**: Focus on Phases 1-4 (research validation + paper)  
**If this is a startup/product**: Complete Phases 1-4 first to prove the concept, then Phase 6 for production

**The web UI (React/FastAPI) should remain minimal** - it's just a demo harness for the core authentication primitive. Jupyter notebooks are sufficient for research visualization.
