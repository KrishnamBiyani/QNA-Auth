# Thesis Metrics Tables

## Table 1. Dataset Summary

| Metric | Value |
|---|---:|
| Total real samples | 500 |
| Unique devices | 6 |
| Camera samples | 340 |
| Microphone samples | 160 |
| Devices with camera + microphone | 2 |
| Devices with camera only | 4 |

## Table 2. Final Modality Comparison

| Modality | Split Policy | ROC-AUC | PR-AUC | EER | Deployed Threshold | Accuracy | Precision | Recall | FAR | FRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Camera | device | 0.8579 | 0.8820 | 0.2678 | 0.9799 | 0.7887 | 0.9079 | 0.6345 | 0.0622 | 0.3655 |
| Microphone | device_session | 0.7096 | 0.7156 | 0.3641 | 0.9698 | 0.6457 | 0.6715 | 0.5984 | 0.3050 | 0.4016 |

## Table 3. Camera Best Operating Point

| Metric | Value |
|---|---:|
| Threshold | 0.9799 |
| Accuracy | 0.7887 |
| Balanced Accuracy | 0.7861 |
| Precision | 0.9079 |
| Recall / TPR | 0.6345 |
| TNR | 0.9378 |
| FAR | 0.0622 |
| FRR | 0.3655 |

## Table 4. Runtime Fusion Policy

| Parameter | Value |
|---|---:|
| Camera weight | 0.85 |
| Microphone weight | 0.15 |
| Required source | camera |
| Camera strong threshold | 0.97 |
| Microphone strong threshold | 0.9698 |
| Identification margin | 0.02 |

## Recommended Figure Usage

- Use Mermaid for architecture and workflow diagrams.
- Use native ROC and PR plots from `model/evaluation/...`.
- Use these tables directly in the paper for quantitative results.
