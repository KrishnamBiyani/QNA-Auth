# Research Focus: What to Work On Next

## TL;DR - If you only read one thing

**Stop working on the web UI. Start working on the core quantum authentication science.**

The React frontend and FastAPI polishing is **not the innovation**. The innovation is proving that quantum noise can reliably fingerprint devices.

---

## What makes this project valuable (the research contribution)

### Novel claims
1. **Quantum noise contains device-specific patterns** that can be learned by ML models
2. **These patterns are stable enough** for authentication (across days/weeks)
3. **But unpredictable enough** to resist replay and cloning attacks
4. **Non-invertible embeddings** prevent recovery of raw device characteristics

### Why this matters
- First practical use of quantum randomness for device authentication
- Combines physics (quantum unpredictability) + AI (metric learning) + crypto (challenge-response)
- Hardware-agnostic (works on commodity devices)
- No stored secrets (unlike passwords, keys, biometric templates)

---

## What to focus on (research tasks)

### 1. Data collection (Week 1-2)
**Task**: Collect comprehensive device corpus
```bash
# Target: 50+ devices, 200 samples each, multiple sources
devices_to_collect = [
    "Your laptop",
    "Your phone", 
    "Lab desktop computers (ask lab mates)",
    "Friends' phones",
    "Raspberry Pis",
    "Arduino boards",
    # ... get 50+ diverse devices
]

for device in devices_to_collect:
    enroll_device(device, sources=['qrng', 'camera', 'microphone'], samples=200)
```

**Why this matters**: Without diverse real-world data, you can't prove the hypothesis. The current "random initialization" model proves nothing.

**Success metric**: ≥50 devices enrolled with ≥100 samples each

### 2. Statistical analysis (Week 2-3)
**Task**: Prove devices are distinguishable
```python
# research/analyze_device_signatures.py

# Load all device samples
devices = load_device_corpus()

# Compute similarity matrices
intra_device_sims = []  # Same device, different samples
inter_device_sims = []  # Different devices

for d in devices:
    samples = d.get_samples()
    # Compute pairwise similarities within device
    intra_device_sims.extend(pairwise_similarities(samples))
    
    for other in devices:
        if other.id != d.id:
            # Compute similarity between devices
            inter_device_sims.append(similarity(d, other))

# Statistical test: Are distributions significantly different?
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(intra_device_sims, inter_device_sims)

print(f"Intra-device similarity: {mean(intra_device_sims):.3f} ± {std(intra_device_sims):.3f}")
print(f"Inter-device similarity: {mean(inter_device_sims):.3f} ± {std(inter_device_sims):.3f}")
print(f"T-test: t={t_stat:.2f}, p={p_value:.2e}")

# Create figure for paper
plot_similarity_distributions(intra_device_sims, inter_device_sims)
```

**Why this matters**: This is the **core hypothesis** of your project. If devices aren't distinguishable, the whole approach fails.

**Success metric**: Clear bimodal distribution with p < 0.001

### 3. Model training (Week 3-4)
**Task**: Train a model that actually works
```python
# research/train_models.py

# Compare architectures
architectures = [
    {'embedding_dim': 64, 'hidden': [128, 128]},
    {'embedding_dim': 128, 'hidden': [256, 256, 128]},
    {'embedding_dim': 256, 'hidden': [512, 512, 256]},
]

# Compare loss functions
losses = ['triplet', 'contrastive', 'arcface']

# Grid search
for arch in architectures:
    for loss in losses:
        model = train_model(
            data=device_corpus,
            architecture=arch,
            loss_function=loss,
            epochs=50,
            batch_size=32
        )
        
        results = evaluate_model(model, test_data)
        print(f"{arch} + {loss}: FAR={results.far:.4f}, FRR={results.frr:.4f}")
```

**Why this matters**: The current random model is a placeholder. You need a trained model to prove this actually works.

**Success metric**: ≥95% accuracy, <1% FAR, <1% FRR

### 4. Security evaluation (Week 4-5)
**Task**: Prove resistance to attacks
```python
# research/attack_simulations.py

# Attack 1: Replay (use old samples)
def replay_attack(device, stored_samples):
    # Attacker has access to N old authentication samples
    # Try to authenticate using those
    success_rate = 0
    for old_sample in stored_samples:
        result = authenticate(device_id, noise=old_sample)
        if result.authenticated:
            success_rate += 1
    return success_rate / len(stored_samples)

# Attack 2: Synthesis (GAN-generated noise)
def synthesis_attack(device, embedding):
    # Train GAN to generate noise that produces similar embedding
    gan = train_gan_to_match_embedding(embedding)
    synthetic_samples = gan.generate(n=100)
    
    success_rate = 0
    for sample in synthetic_samples:
        result = authenticate(device_id, noise=sample)
        if result.authenticated:
            success_rate += 1
    return success_rate / len(synthetic_samples)

# Attack 3: Cloning (identical hardware)
def cloning_attack(target_device):
    # Get same model phone/laptop
    clone = get_identical_hardware(target_device.model)
    
    # Try to enroll clone with target's device_id
    success = can_clone_authenticate(clone, target_device.id)
    return success

# Run all attacks
attacks = {
    'Replay': replay_attack,
    'Synthesis': synthesis_attack,
    'Cloning': cloning_attack,
}

for attack_name, attack_fn in attacks.items():
    success_rate = attack_fn(test_devices)
    print(f"{attack_name} attack success rate: {success_rate:.2%}")
    # Should be < 0.1% for all attacks
```

**Why this matters**: Security claims require evidence. Can't just say "it's secure" - need to demonstrate attack failure.

**Success metric**: All attacks fail >99.9% of the time

### 5. Write the paper (Week 5-8)
**Task**: Document everything for peer review
```latex
% paper/qna_auth.tex

\section{Introduction}
We propose QNA-Auth, a novel device authentication system that derives 
fingerprints from quantum noise patterns using machine learning...

\section{Threat Model}
We consider an adversary with the following capabilities:
- Access to N historical authentication samples
- Ability to measure quantum noise from identical hardware
- White-box access to the ML model and stored embeddings
- Computational power bounded by...

\section{Method}
\subsection{Noise Collection}
We collect quantum noise from four sources: ...

\subsection{Feature Extraction}
We extract D-dimensional feature vectors comprising statistical, 
spectral, and complexity features: ...

\subsection{Embedding Model}
We train a Siamese network with triplet loss to learn a metric 
space where same-device samples cluster: ...

\section{Evaluation}
\subsection{Experimental Setup}
We enrolled 50 devices (25 smartphones, 15 laptops, 10 IoT devices)
and collected 200 samples per device over 90 days...

\subsection{Device Distinguishability}
Figure X shows the distribution of intra-device vs inter-device 
similarities. We observe clear separation (t=X.XX, p<0.001)...

\subsection{Attack Resistance}
Table Y summarizes attack success rates. All attacks failed >99.9%...

\section{Discussion}
Limitations: ...
Future work: ...
```

**Why this matters**: The paper is the **primary deliverable** for a research project. Without publication, this is just a hobby project.

**Success metric**: Submitted to IEEE S&P, CCS, NDSS, or USENIX Security

---

## What NOT to focus on (web dev tasks)

### Stop doing these (low priority for research)
- ❌ Styling the React UI with better CSS
- ❌ Adding more API endpoints (only need enroll + authenticate)
- ❌ User authentication for the web UI (only one researcher uses it)
- ❌ Database schema design (flat files are fine for research)
- ❌ Rate limiting and DDoS protection (not running a public service)
- ❌ Deployment automation and CI/CD (manual is fine)
- ❌ Monitoring dashboards (just use logs)
- ❌ Multi-tenancy (one user at a time is fine)

### Keep minimal
- ✅ Simple FastAPI endpoint (50 lines total)
- ✅ Jupyter notebooks for analysis (not React)
- ✅ Command-line tools for experiments
- ✅ Docker for reproducibility (not for scaling)

---

## Time allocation (for a 15-week project)

| Task | Weeks | % of time |
|------|-------|-----------|
| Data collection | 2 | 13% |
| Statistical analysis | 1 | 7% |
| Model training & tuning | 2 | 13% |
| Security evaluation | 2 | 13% |
| Longitudinal stability study | 4 (parallel) | 27% |
| Paper writing | 3 | 20% |
| Code cleanup & documentation | 1 | 7% |
| **Total** | **15** | **100%** |

**Note**: Web UI polish is <1% of time (maybe 1-2 hours to make a simple demo page)

---

## Tools you should be using

### For research (use these)
- **Jupyter notebooks** for data exploration and visualization
- **Matplotlib/Seaborn** for figures in the paper
- **Scikit-learn** for baseline comparisons and metrics
- **PyTorch** for model training
- **NumPy/SciPy** for statistical analysis
- **Pandas** for data management

### For web dev (minimize usage)
- **FastAPI** (keep to <100 lines for basic endpoints)
- **React** (optional, only for demo presentations)

---

## Metrics that matter (research success)

### Core authentication metrics
- [x] **Device distinguishability**: Intra-device similarity >> inter-device similarity (effect size d' > 3)
- [ ] **Authentication accuracy**: >95% with <1% FAR/FRR
- [ ] **Stability**: <5% accuracy degradation over 90 days
- [ ] **Minimal samples**: <50 enrollment samples, <5 auth samples
- [ ] **Attack resistance**: All attacks fail >99.9% of the time

### Academic metrics
- [ ] **Paper submitted** to top-tier venue
- [ ] **Reproducibility**: Public code + anonymized data
- [ ] **Novel contribution**: First practical quantum-noise-based auth

### Metrics that DON'T matter for research
- ⬜ API response time (as long as it's <10s)
- ⬜ Database query optimization (flat files are fine)
- ⬜ Frontend load time (it's a local demo)
- ⬜ Scalability (not building a service)
- ⬜ Uptime (it's a research prototype)

---

## Immediate next action (this week)

1. **Read `VISION.md`** to understand the reframed project goals
2. **Start data collection**: Enroll 10 devices as a pilot
3. **Run statistical analysis**: Do the 10 devices show different noise patterns?
4. **If yes**: Scale to 50+ devices
5. **If no**: Refine feature extraction (try different features)

**Stop working on**: React component styling, API authentication, database design

**Start working on**: Data collection, statistical analysis, model training

---

## Questions to answer (in order of priority)

1. **Can quantum noise distinguish devices?** (Week 1-3)
   - Are intra-device similarities significantly higher than inter-device?
   - What features are most discriminative?

2. **Is it stable over time?** (Week 4-10, parallel)
   - Does the same device authenticate successfully after 30/60/90 days?
   - What causes drift? (temperature, battery, etc.)

3. **Is it secure against attacks?** (Week 4-6)
   - Can replay attacks succeed?
   - Can synthetic noise fool the system?
   - Can cloned hardware authenticate?

4. **How minimal can we make it?** (Week 6-7)
   - What's the minimum number of enrollment samples?
   - What's the minimum number of auth samples?
   - Can we use fewer noise sources?

5. **How does it compare to existing methods?** (Week 8)
   - QNA-Auth vs. password
   - QNA-Auth vs. biometrics
   - QNA-Auth vs. hardware tokens

**Answering these 5 questions is the entire project.** The web UI is just a tool to help answer them.

---

## Summary

**What QNA-Auth is**: A research contribution showing quantum noise can fingerprint devices

**What QNA-Auth is NOT**: A production-ready web authentication service

**What to build**: Data collection pipeline + ML models + statistical analysis + paper

**What NOT to build**: Polished React UI + scalable API + production database

**Success looks like**: Accepted paper at a top security conference

**Failure looks like**: Beautiful web UI but no proof that the core idea works
