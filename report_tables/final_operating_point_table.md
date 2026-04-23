# Final Operating Point Table

| Configuration | Deployed Threshold | Camera Weight | Microphone Weight | FAR | FRR | TPR | TNR | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|
| Camera only | 0.9799 | 1.0 | 0.0 | 0.0622 | 0.3655 | 0.6345 | 0.9378 | 0.9079 | 0.6345 |
| Microphone only | 0.9698 | 0.0 | 1.0 | 0.305 | 0.4016 | 0.5984 | 0.695 | 0.6715 | 0.5984 |
| Planned runtime fusion policy | camera required | 0.85 | 0.15 | N/A | N/A | N/A | N/A | N/A | N/A |

Caption: Final operating points for the camera and microphone models. The deployed runtime configuration uses camera-dominant weighted fusion (0.85 / 0.15) with camera as the required modality. Fusion FAR/FRR are marked N/A because fused performance was not benchmarked as a separate end-to-end experiment.