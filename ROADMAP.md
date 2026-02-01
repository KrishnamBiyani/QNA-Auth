# QNA-Auth: Roadmap to Research-Level Capstone

A focused outline to elevate the project from a solid capstone to a **research-level capstone** (clear problem, method, evaluation, and limitations).

---

## 1. Data & Persistence

**→ Detailed dataset guide:** [docs/DATASET.md](docs/DATASET.md) (what to collect, scale, layout, splits, versioning, and how it feeds evaluation).

| Priority | Task | Why It Matters |
|----------|------|----------------|
| **High** | **Add a real database** for devices, embeddings metadata, and audit logs (e.g. SQLite + SQLAlchemy, or PostgreSQL). | Reproducibility, multi-user, no in-memory state; reviewers expect “where is the data?” |
| **High** | **Store embeddings in DB or encrypted files** (not plain `.pt` on disk). Include schema version and feature-vector version. | Security, traceability, and “same pipeline at train and serve.” |
| **Medium** | **Dataset versioning** – versioned training/enrollment datasets (paths, hashes, or DVC). | So you can say “model X was trained on dataset v1.2” and rerun experiments. |
| **Medium** | **Config-driven paths** – all paths (model, checkpoints, DB, storage) from `config` / env, no hardcoded `./auth/...`. | Reproducibility across machines and for evaluation scripts. |

---

## 2. ML Model & Training

| Priority | Task | Why It Matters |
|----------|------|----------------|
| **High** | **Fixed, versioned feature pipeline** – one canonical feature list (e.g. `FEATURE_NAMES` + version) used in preprocessing, training, and server. Save it with the model. | Avoids train/serve mismatch and supports “same features everywhere.” |
| **High** | **Reproducible training** – set `torch.manual_seed`, numpy seed, dataloader worker seed; document in README. | “We can reproduce Table 2” is a basic research expectation. |
| **High** | **Training on real multi-device data** – script to collect N devices × M samples, build dataset, train, then run evaluation. | Without this, you can’t claim anything about real-world performance. |
| **Medium** | **Checkpointing** – save best model by validation metric; optionally save last N checkpoints. | Enables proper evaluation and ablation without retraining from scratch. |
| **Medium** | **Simple baselines** – e.g. cosine similarity on raw features (no NN), or a small MLP. | “Our siamese net beats baseline X” is a minimal contribution statement. |
| **Low** | **Optional: ONNX export** for inference and to mention “deployable” in the report. | Nice extra, not required for a research-level capstone. |

---

## 3. Evaluation (What Makes It “Research-Level”)

| Priority | Task | Why It Matters |
|----------|------|----------------|
| **High** | **Formal metrics** – FAR (false accept rate), FRR (false reject rate), EER (equal error rate), ROC curve, and optionally DET. | Standard way to say “our system has X% EER under Y setup.” |
| **High** | **Threshold sweep** – vary similarity threshold, plot FAR/FRR vs threshold, report EER and chosen operating point. | Shows you didn’t pick the threshold arbitrarily. |
| **High** | **Cross-device / cross-session** – e.g. train on session 1, test on session 2; or leave-one-device-out. | Demonstrates generalization, not overfitting to one capture. |
| **Medium** | **Ablations** – accuracy with QRNG only, camera only, mic only, and combined. | “Combined sources improve EER by Z%” is a clear contribution. |
| **Medium** | **Confidence intervals** – bootstrap or multiple runs, report mean ± std or 95% CI for EER/FAR/FRR. | Shows stability and rigor. |
| **Medium** | **Failure analysis** – when does auth fail? (low samples, bad lighting, mic off, etc.) Document in report. | Shows you understand limitations. |
| **Low** | **Comparison to a simple baseline** – e.g. “our model vs raw-feature similarity” in a small table. | Strengthens the “contribution” narrative. |

---

## 4. Security & Robustness

| Priority | Task | Why It Matters |
|----------|------|----------------|
| **High** | **Remove all hardcoded secrets** – API keys, default passwords only in `config.example` or env; document in README. | Non-negotiable for any “research-level” or “production-ready” claim. |
| **High** | **API protection** – at least API keys or JWT for enroll/authenticate/delete; rate limiting on auth endpoints. | Shows you take threat model seriously. |
| **Medium** | **CORS** – restrict to your frontend origin(s); no `*` with credentials in production. | Expected in any security-conscious design. |
| **Medium** | **Challenge storage** – move challenges to DB or Redis with TTL; no in-memory dict. | Survives restarts and multi-instance; prevents replay. |
| **Low** | **Embeddings at rest** – encrypt stored embeddings (e.g. key from env); mention in report. | Good practice and easy to mention in “future work.” |

---

## 5. Reproducibility & Documentation

| Priority | Task | Why It Matters |
|----------|------|----------------|
| **High** | **Single “run evaluation” path** – e.g. `python scripts/run_evaluation.py --config configs/eval.yaml` that loads data, model, and prints/saves metrics. | Reviewers/readers can “run the numbers” themselves. |
| **High** | **README** – install (venv, Python version, optional CUDA), dataset layout, how to collect data, train, run evaluation, and start server. | Standard expectation for a capstone. |
| **High** | **Report/slides** – problem, related work (device auth / PUF / QRNG), method (pipeline diagram), experiments (tables/figures), limitations, future work. | This is what turns “code” into “research-level capstone.” |
| **Medium** | **Requirements pinned** – `requirements.txt` with versions (e.g. `torch==2.x.x`). | So `pip install -r requirements.txt` gives the same environment. |
| **Medium** | **Config for experiments** – one config (or small set) that defines data paths, model path, thresholds, and evaluation splits. | Makes ablation and “Table 2” reproducible. |

---

## 6. Frontend & UX (Lower Priority for “Research”)

| Priority | Task | Why It Matters |
|----------|------|----------------|
| **Medium** | **Clear error states** – “No devices enrolled,” “Authentication failed: similarity 0.72 (threshold 0.85),” “Camera unavailable.” | Helps demos and user studies. |
| **Low** | **Optional: use challenge–response in the UI** – if you describe the protocol in the report, having it in the UI strengthens the story. | Not required for research-level, but consistent. |

---

## 7. Suggested Order of Work

1. **Data & DB** – Add DB for devices/metadata; move embeddings to DB or versioned, config-driven paths.
2. **Feature versioning** – Lock feature list + version; use everywhere and save with model.
3. **Evaluation script** – Implement FAR/FRR/EER, threshold sweep, one clear “run evaluation” command.
4. **Real data + training** – Collect multi-device data; train with fixed seeds; document in README.
5. **Ablations** – QRNG vs camera vs mic vs combined; put results in a table in the report.
6. **Security cleanup** – No secrets in code; API auth; CORS; optional rate limiting.
7. **Reproducibility** – Single eval config, pinned requirements, README, then write the report with tables/figures from (3)–(5).

---

## 8. One-Paragraph “Research-Level” Claim

After the above, you can honestly say:

> We built an end-to-end system for device authentication using quantum and sensor noise, with a siamese network and challenge–response. We use a real database and versioned features, report FAR/FRR/EER and ablations over noise sources, and provide a reproducible evaluation pipeline and fixed seeds. Security is addressed by removing hardcoded secrets, protecting APIs, and storing challenges in a persistent store. Limitations and failure cases are documented.

---

## Quick Reference: Priority Legend

- **High** – Needed for a credible “research-level” claim.
- **Medium** – Strongly recommended; improves rigor and presentation.
- **Low** – Nice to have; mention in report or future work.
