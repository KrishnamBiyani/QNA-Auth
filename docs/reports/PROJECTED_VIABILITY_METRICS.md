# Projected Viability Metrics for Presentation Use

This file is a slide-support artifact for presentation planning.

Important reporting rule:
- `measured` values below come directly from repository artifacts.
- `projected` values are forecast estimates for a larger, better-calibrated follow-up study.
- Do not present projected values as experimentally measured results.

## Measured Baselines Already in the Repo

### A. Strongest recent artifact

Source:
- `artifacts/capstone_eval/20260422T122948Z/results.json`

Measured microphone-only Siamese result on the April 22, 2026 capstone artifact:

| Metric | Value |
|---|---:|
| Accuracy | 0.8333 |
| Precision | 0.5000 |
| Recall | 1.0000 |
| FAR | 0.2000 |
| FRR | 0.0000 |
| EER | 0.1000 |

### B. Thesis modality table

Source:
- `report_mermaid/report_metrics_tables.md`

Measured device-level / device-session headline values already used in the thesis:

| Modality | ROC-AUC | PR-AUC | EER | Accuracy | Precision | Recall | FAR | FRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Camera | 0.8579 | 0.8820 | 0.2678 | 0.7887 | 0.9079 | 0.6345 | 0.0622 | 0.3655 |
| Microphone | 0.7096 | 0.7156 | 0.3641 | 0.6457 | 0.6715 | 0.5984 | 0.3050 | 0.4016 |

## Why a Projection is Reasonable

These forecast numbers assume a follow-up experiment with:
- 25+ devices instead of the current small-scale prototype data;
- at least 2 to 3 sessions per device;
- threshold calibration on a validation split instead of ad hoc operating transfer;
- camera-dominant fusion with microphone support;
- rejection of uncertain samples rather than forcing a binary decision.

The projection is intentionally bounded:
- it does not try to beat offline PRNU literature;
- it only targets viability for a live short-sample prototype;
- it stays close to the strongest measured artifact already in the repo.

## Projected Metrics for a Future Fused Model

Use these three scenarios in the deck if you need a forecast slide:

| Scenario | Claim Type | ROC-AUC | PR-AUC | EER | Accuracy | Precision | Recall | FAR | FRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Conservative fusion | projected | 0.900 | 0.890 | 0.120 | 0.850 | 0.740 | 0.870 | 0.110 | 0.130 |
| Target fusion | projected | 0.930 | 0.920 | 0.080 | 0.890 | 0.810 | 0.900 | 0.070 | 0.090 |
| Stretch fusion | projected | 0.950 | 0.940 | 0.060 | 0.920 | 0.860 | 0.930 | 0.050 | 0.070 |

Interpretation:
- `Conservative fusion` says the idea remains viable even with only moderate gains from more data and calibration.
- `Target fusion` is the most defensible presentation forecast.
- `Stretch fusion` should be labeled as best-case future work, not near-term expectation.

## Suggested Slide-19 / Slide-20 Wording

Recommended wording:

> Measured results in the repo already show viable device verification under a bounded evaluation setup.
> Based on those artifacts, we forecast that a larger multi-session fused dataset should move QNA-Auth toward roughly 0.93 ROC-AUC and 8% EER, while still operating on short live samples rather than many offline images.

Safer wording for viva questions:

> These forecast values are not claimed as measured output. They are scenario estimates derived from our current artifact range and from the expected benefit of calibration, fusion, and additional sessions.

## Recommended Comparison Narrative with Existing Work

Use this framing:
- QNA-Auth should not be compared as if it were a classic offline PRNU system with dozens of images per device.
- A fairer comparison is against short-sample or live authentication settings.
- Under the `Target fusion` forecast, QNA-Auth would likely outperform older acoustic-only baselines while still remaining below the best offline camera-only PRNU pipelines.

That is the defensible narrative:
- better than older single-mic live fingerprints;
- weaker than mature offline PRNU;
- still viable because the operating constraints are much harder.

## Slide-Ready Table with PPT Column Names

The presentation slide uses this exact column order:

| Method | Modality | ROC-AUC | EER | Accuracy | Fusion | Replay Protection | Method |
|---|---|---:|---:|---:|---|---|---|
| Lukas et al. (2006) - PRNU | Camera only | ~0.95+ | ~4% | ~96% | None | None | Lukas et al. (2006) - PRNU |
| Goljan et al. (2009) - Large-scale PRNU | Camera only | ~0.97+ | ~2.5% | ~97% | None | None | Goljan et al. (2009) - Large-scale PRNU |
| Chen et al. (2008) - ML-PRNU | Camera only | ~0.96+ | ~2% | ~97% | None | None | Chen et al. (2008) - ML-PRNU |
| Das et al. (2014) - Acoustic fingerprint | Mic only | ~0.82 | ~9% | ~85% | None | None | Das et al. (2014) - Acoustic fingerprint |
| QNA-Auth - Camera only | Camera | 0.858 | 26.8% | 78.9% | No | HKDF nonce | QNA-Auth - Camera only |
| QNA-Auth - Mic only | Mic | 0.710 | 36.4% | 64.6% | No | HKDF nonce | QNA-Auth - Mic only |
| QNA-Auth - Fused system (conservative forecast) | Camera + Mic | 0.900 | 12.0% | 85.0% | Yes | HKDF nonce | QNA-Auth - Fused system (conservative forecast) |
| QNA-Auth - Fused system (target forecast) | Camera + Mic | 0.930 | 8.0% | 89.0% | Yes | HKDF nonce | QNA-Auth - Fused system (target forecast) |
| QNA-Auth - Fused system (stretch forecast) | Camera + Mic | 0.950 | 6.0% | 92.0% | Yes | HKDF nonce | QNA-Auth - Fused system (stretch forecast) |

Notes:
- The duplicated `Method` column is preserved intentionally because that is how the PowerPoint slide structure was specified.
- The measured QNA-Auth camera and microphone rows come from repository evaluation artifacts.
- The fused rows are forecast scenarios and should stay labeled as projected in discussion.
