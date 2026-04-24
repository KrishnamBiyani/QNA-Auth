from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "Predicted report"
DATA_DIR = OUTPUT_DIR / "data"
IMAGES_DIR = OUTPUT_DIR / "images"

CAMERA_METRICS = ROOT / "model" / "evaluation" / "camera_real_aug_v2" / "metrics.json"
MIC_METRICS = ROOT / "model" / "evaluation" / "microphone_real_aug_v2" / "metrics.json"
CAPSTONE_RESULTS = ROOT / "artifacts" / "capstone_eval" / "20260422T122948Z" / "results.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def series_from_anchor_points(anchor_points: list[tuple[float, float]], x_values: np.ndarray) -> np.ndarray:
    xs = np.array([p[0] for p in anchor_points], dtype=float)
    ys = np.array([p[1] for p in anchor_points], dtype=float)
    return np.interp(x_values, xs, ys)


def build_projected_payload(camera: dict, microphone: dict, capstone: dict) -> dict:
    scenarios = {
        "conservative": {
            "roc_auc": 0.900,
            "pr_auc": 0.890,
            "eer": 0.120,
            "accuracy": 0.850,
            "balanced_accuracy": 0.850,
            "precision": 0.740,
            "recall": 0.870,
            "f1_score": 0.800,
            "far": 0.110,
            "frr": 0.130,
            "tnr": 0.890,
            "tpr": 0.870,
            "deployed_threshold": 0.962,
            "optimal_threshold": 0.948,
            "eer_threshold": 0.913,
            "num_devices": 25,
            "total_samples": 375,
            "genuine_mean": 0.776,
            "genuine_std": 0.134,
            "impostor_mean": 0.232,
            "impostor_std": 0.186,
            "replay_asr": 0.030,
            "impersonation_asr": 0.050,
            "synthetic_asr": 0.040,
        },
        "target": {
            "roc_auc": 0.930,
            "pr_auc": 0.920,
            "eer": 0.080,
            "accuracy": 0.890,
            "balanced_accuracy": 0.890,
            "precision": 0.810,
            "recall": 0.900,
            "f1_score": 0.853,
            "far": 0.070,
            "frr": 0.090,
            "tnr": 0.930,
            "tpr": 0.900,
            "deployed_threshold": 0.971,
            "optimal_threshold": 0.955,
            "eer_threshold": 0.924,
            "num_devices": 30,
            "total_samples": 480,
            "genuine_mean": 0.822,
            "genuine_std": 0.116,
            "impostor_mean": 0.188,
            "impostor_std": 0.152,
            "replay_asr": 0.020,
            "impersonation_asr": 0.035,
            "synthetic_asr": 0.025,
        },
        "stretch": {
            "roc_auc": 0.950,
            "pr_auc": 0.940,
            "eer": 0.060,
            "accuracy": 0.920,
            "balanced_accuracy": 0.920,
            "precision": 0.860,
            "recall": 0.930,
            "f1_score": 0.894,
            "far": 0.050,
            "frr": 0.070,
            "tnr": 0.950,
            "tpr": 0.930,
            "deployed_threshold": 0.978,
            "optimal_threshold": 0.964,
            "eer_threshold": 0.935,
            "num_devices": 36,
            "total_samples": 648,
            "genuine_mean": 0.862,
            "genuine_std": 0.098,
            "impostor_mean": 0.152,
            "impostor_std": 0.128,
            "replay_asr": 0.010,
            "impersonation_asr": 0.020,
            "synthetic_asr": 0.015,
        },
    }

    measured = {
        "camera_current": {
            "roc_auc": camera["roc_auc"],
            "pr_auc": camera["pr_auc"],
            "eer": camera["eer"],
            "accuracy": camera["deployed_threshold_metrics"]["accuracy"],
            "precision": camera["deployed_threshold_metrics"]["precision"],
            "recall": camera["deployed_threshold_metrics"]["recall"],
            "far": camera["deployed_threshold_metrics"]["far"],
            "frr": camera["deployed_threshold_metrics"]["frr"],
        },
        "microphone_current": {
            "roc_auc": microphone["roc_auc"],
            "pr_auc": microphone["pr_auc"],
            "eer": microphone["eer"],
            "accuracy": microphone["deployed_threshold_metrics"]["accuracy"],
            "precision": microphone["deployed_threshold_metrics"]["precision"],
            "recall": microphone["deployed_threshold_metrics"]["recall"],
            "far": microphone["deployed_threshold_metrics"]["far"],
            "frr": microphone["deployed_threshold_metrics"]["frr"],
        },
        "capstone_artifact": {
            "accuracy": capstone["methods"]["siamese"]["metrics"]["accuracy"],
            "precision": capstone["methods"]["siamese"]["metrics"]["precision"],
            "recall": capstone["methods"]["siamese"]["metrics"]["recall"],
            "f1_score": capstone["methods"]["siamese"]["metrics"]["f1_score"],
            "far": capstone["methods"]["siamese"]["metrics"]["far"],
            "frr": capstone["methods"]["siamese"]["metrics"]["frr"],
            "eer": capstone["methods"]["siamese"]["eer"],
        },
    }

    assumptions = {
        "labeling_rule": "All projected artifacts are forecast scenarios, not measured outputs.",
        "dataset_scale": "Projection assumes 25-36 devices with 2-3 sessions per device and better threshold calibration.",
        "fusion_policy": "Camera remains primary modality with microphone as supporting evidence.",
        "comparison_boundary": "Projected fused performance is expected to beat older acoustic-only live baselines, not mature offline PRNU pipelines.",
    }
    return {"measured_baselines": measured, "projected_scenarios": scenarios, "assumptions": assumptions}


def build_threshold_sweep(scenario_name: str, metrics: dict) -> list[dict]:
    if scenario_name == "conservative":
        thresholds = np.linspace(0.82, 0.995, 16)
        far_points = [(0.82, 0.30), (0.90, 0.18), (0.94, 0.13), (0.962, metrics["far"]), (0.995, 0.02)]
        frr_points = [(0.82, 0.02), (0.90, 0.05), (0.94, 0.08), (0.962, metrics["frr"]), (0.995, 0.28)]
    elif scenario_name == "target":
        thresholds = np.linspace(0.85, 0.997, 18)
        far_points = [(0.85, 0.22), (0.90, 0.14), (0.94, 0.10), (0.971, metrics["far"]), (0.997, 0.015)]
        frr_points = [(0.85, 0.01), (0.90, 0.03), (0.94, 0.05), (0.971, metrics["frr"]), (0.997, 0.22)]
    else:
        thresholds = np.linspace(0.88, 0.998, 18)
        far_points = [(0.88, 0.16), (0.92, 0.10), (0.95, 0.07), (0.978, metrics["far"]), (0.998, 0.010)]
        frr_points = [(0.88, 0.01), (0.92, 0.02), (0.95, 0.04), (0.978, metrics["frr"]), (0.998, 0.18)]

    far_values = series_from_anchor_points(far_points, thresholds)
    frr_values = series_from_anchor_points(frr_points, thresholds)
    rows: list[dict] = []
    for threshold, far, frr in zip(thresholds, far_values, frr_values):
        recall = max(0.0, 1.0 - frr)
        tnr = max(0.0, 1.0 - far)
        precision = max(0.0, min(0.99, recall / (recall + far + 1e-9)))
        accuracy = (recall + tnr) / 2.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        rows.append(
            {
                "scenario": scenario_name,
                "threshold": round(float(threshold), 6),
                "far": round(float(far), 6),
                "frr": round(float(frr), 6),
                "accuracy": round(float(accuracy), 6),
                "precision": round(float(precision), 6),
                "recall": round(float(recall), 6),
                "f1_score": round(float(f1), 6),
                "tpr": round(float(recall), 6),
                "tnr": round(float(tnr), 6),
            }
        )
    return rows


def build_roc_rows() -> list[dict]:
    rows: list[dict] = []
    anchor_map = {
        "conservative": [(0.00, 0.00), (0.02, 0.26), (0.05, 0.50), (0.08, 0.66), (0.12, 0.77), (0.18, 0.85), (0.28, 0.91), (0.45, 0.96), (1.00, 1.00)],
        "target": [(0.00, 0.00), (0.01, 0.34), (0.03, 0.57), (0.05, 0.71), (0.08, 0.83), (0.12, 0.90), (0.18, 0.95), (0.28, 0.98), (1.00, 1.00)],
        "stretch": [(0.00, 0.00), (0.01, 0.42), (0.02, 0.62), (0.04, 0.78), (0.06, 0.88), (0.10, 0.94), (0.16, 0.97), (0.25, 0.99), (1.00, 1.00)],
    }
    for scenario, points in anchor_map.items():
        for fpr, tpr in points:
            rows.append({"scenario": scenario, "fpr": fpr, "tpr": tpr})
    return rows


def build_pr_rows() -> list[dict]:
    rows: list[dict] = []
    anchor_map = {
        "conservative": [(0.00, 1.00), (0.15, 0.98), (0.35, 0.94), (0.55, 0.88), (0.70, 0.80), (0.82, 0.72), (0.92, 0.60), (1.00, 0.48)],
        "target": [(0.00, 1.00), (0.20, 0.99), (0.40, 0.96), (0.60, 0.90), (0.75, 0.84), (0.86, 0.77), (0.94, 0.69), (1.00, 0.58)],
        "stretch": [(0.00, 1.00), (0.20, 0.995), (0.45, 0.975), (0.65, 0.93), (0.80, 0.88), (0.90, 0.82), (0.96, 0.76), (1.00, 0.66)],
    }
    for scenario, points in anchor_map.items():
        for recall, precision in points:
            rows.append({"scenario": scenario, "recall": recall, "precision": precision})
    return rows


def build_existing_work_rows(payload: dict) -> list[dict]:
    target = payload["projected_scenarios"]["target"]
    conservative = payload["projected_scenarios"]["conservative"]
    stretch = payload["projected_scenarios"]["stretch"]
    measured = payload["measured_baselines"]
    return [
        {"method": "Lukas et al. (2006) - PRNU", "modality": "Camera only", "roc_auc": 0.950, "eer": 0.040, "accuracy": 0.960, "claim_type": "literature"},
        {"method": "Goljan et al. (2009) - Large-scale PRNU", "modality": "Camera only", "roc_auc": 0.970, "eer": 0.025, "accuracy": 0.970, "claim_type": "literature"},
        {"method": "Chen et al. (2008) - ML-PRNU", "modality": "Camera only", "roc_auc": 0.960, "eer": 0.020, "accuracy": 0.970, "claim_type": "literature"},
        {"method": "Das et al. (2014) - Acoustic fingerprint", "modality": "Mic only", "roc_auc": 0.820, "eer": 0.090, "accuracy": 0.850, "claim_type": "literature"},
        {"method": "QNA-Auth - Camera only", "modality": "Camera", "roc_auc": measured["camera_current"]["roc_auc"], "eer": measured["camera_current"]["eer"], "accuracy": measured["camera_current"]["accuracy"], "claim_type": "measured"},
        {"method": "QNA-Auth - Mic only", "modality": "Mic", "roc_auc": measured["microphone_current"]["roc_auc"], "eer": measured["microphone_current"]["eer"], "accuracy": measured["microphone_current"]["accuracy"], "claim_type": "measured"},
        {"method": "QNA-Auth - Fused conservative", "modality": "Camera + Mic", "roc_auc": conservative["roc_auc"], "eer": conservative["eer"], "accuracy": conservative["accuracy"], "claim_type": "projected"},
        {"method": "QNA-Auth - Fused target", "modality": "Camera + Mic", "roc_auc": target["roc_auc"], "eer": target["eer"], "accuracy": target["accuracy"], "claim_type": "projected"},
        {"method": "QNA-Auth - Fused stretch", "modality": "Camera + Mic", "roc_auc": stretch["roc_auc"], "eer": stretch["eer"], "accuracy": stretch["accuracy"], "claim_type": "projected"},
    ]


def build_confidence_intervals(payload: dict) -> dict:
    intervals = {}
    for scenario, metrics in payload["projected_scenarios"].items():
        intervals[scenario] = {
            "roc_auc": {
                "mean": metrics["roc_auc"],
                "ci_low": round(metrics["roc_auc"] - 0.020, 3),
                "ci_high": round(min(0.999, metrics["roc_auc"] + 0.018), 3),
            },
            "eer": {
                "mean": metrics["eer"],
                "ci_low": round(max(0.0, metrics["eer"] - 0.020), 3),
                "ci_high": round(metrics["eer"] + 0.025, 3),
            },
            "accuracy": {
                "mean": metrics["accuracy"],
                "ci_low": round(max(0.0, metrics["accuracy"] - 0.030), 3),
                "ci_high": round(min(0.999, metrics["accuracy"] + 0.025), 3),
            },
        }
    return intervals


def build_attack_rows(payload: dict) -> list[dict]:
    rows = []
    for scenario, metrics in payload["projected_scenarios"].items():
        rows.extend(
            [
                {"scenario": scenario, "attack": "replay_asr", "value": metrics["replay_asr"]},
                {"scenario": scenario, "attack": "impersonation_asr", "value": metrics["impersonation_asr"]},
                {"scenario": scenario, "attack": "synthetic_statistics_asr", "value": metrics["synthetic_asr"]},
            ]
        )
    return rows


def build_summary_rows(payload: dict) -> list[dict]:
    measured = payload["measured_baselines"]
    return [
        {"series": "camera_measured", **measured["camera_current"]},
        {"series": "microphone_measured", **measured["microphone_current"]},
        {"series": "capstone_measured", **measured["capstone_artifact"]},
        {"series": "fusion_conservative_projected", **payload["projected_scenarios"]["conservative"]},
        {"series": "fusion_target_projected", **payload["projected_scenarios"]["target"]},
        {"series": "fusion_stretch_projected", **payload["projected_scenarios"]["stretch"]},
    ]


def save_payloads(payload: dict) -> None:
    write_json(DATA_DIR / "projected_metrics.json", payload)
    write_csv(DATA_DIR / "projected_summary_metrics.csv", build_summary_rows(payload))
    write_csv(DATA_DIR / "results_comparison_existing_work.csv", build_existing_work_rows(payload))
    write_json(DATA_DIR / "confidence_intervals.json", build_confidence_intervals(payload))
    write_csv(DATA_DIR / "attack_projection.csv", build_attack_rows(payload))
    write_json(DATA_DIR / "attack_projection.json", {"rows": build_attack_rows(payload)})

    all_sweep_rows: list[dict] = []
    for scenario, metrics in payload["projected_scenarios"].items():
        rows = build_threshold_sweep(scenario, metrics)
        write_csv(DATA_DIR / f"threshold_sweep_{scenario}.csv", rows)
        all_sweep_rows.extend(rows)
    write_csv(DATA_DIR / "threshold_sweep_all.csv", all_sweep_rows)

    roc_rows = build_roc_rows()
    pr_rows = build_pr_rows()
    write_csv(DATA_DIR / "projected_roc_curves.csv", roc_rows)
    write_csv(DATA_DIR / "projected_pr_curves.csv", pr_rows)


def add_projection_note(ax: plt.Axes, text: str = "Projected data, not measured output") -> None:
    ax.text(
        0.99,
        0.02,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#cccccc"},
    )


def plot_roc(payload: dict) -> None:
    roc_rows = build_roc_rows()
    plt.figure(figsize=(8.8, 6.4))
    colors = {"conservative": "#3b82f6", "target": "#10b981", "stretch": "#f59e0b"}
    for scenario in ["conservative", "target", "stretch"]:
        rows = [row for row in roc_rows if row["scenario"] == scenario]
        plt.plot([r["fpr"] for r in rows], [r["tpr"] for r in rows], linewidth=2.5, color=colors[scenario], label=f"{scenario.title()} (AUC ~ {payload['projected_scenarios'][scenario]['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="#9ca3af", linewidth=1.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("True Positive Rate (1 - FRR)")
    plt.title("Projected ROC Curves for Future Fused QNA-Auth")
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend()
    add_projection_note(plt.gca())
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "projected_roc_curves.png", dpi=240)
    plt.close()


def plot_pr(payload: dict) -> None:
    pr_rows = build_pr_rows()
    plt.figure(figsize=(8.8, 6.4))
    colors = {"conservative": "#3b82f6", "target": "#10b981", "stretch": "#f59e0b"}
    for scenario in ["conservative", "target", "stretch"]:
        rows = [row for row in pr_rows if row["scenario"] == scenario]
        plt.plot([r["recall"] for r in rows], [r["precision"] for r in rows], linewidth=2.5, color=colors[scenario], label=f"{scenario.title()} (PR-AUC ~ {payload['projected_scenarios'][scenario]['pr_auc']:.3f})")
    plt.xlim(0, 1)
    plt.ylim(0.4, 1.02)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Projected Precision-Recall Curves for Future Fused QNA-Auth")
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend()
    add_projection_note(plt.gca())
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "projected_pr_curves.png", dpi=240)
    plt.close()


def plot_threshold_sweep(payload: dict) -> None:
    plt.figure(figsize=(9.4, 6.6))
    colors = {"conservative": "#3b82f6", "target": "#10b981", "stretch": "#f59e0b"}
    for scenario in ["conservative", "target", "stretch"]:
        rows = build_threshold_sweep(scenario, payload["projected_scenarios"][scenario])
        thresholds = [r["threshold"] for r in rows]
        plt.plot(thresholds, [r["far"] for r in rows], color=colors[scenario], linestyle="-", linewidth=2.2, label=f"{scenario.title()} FAR")
        plt.plot(thresholds, [r["frr"] for r in rows], color=colors[scenario], linestyle="--", linewidth=2.2, label=f"{scenario.title()} FRR")
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title("Projected Threshold Sweep: FAR and FRR")
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend(ncol=2, fontsize=9)
    add_projection_note(plt.gca())
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "projected_threshold_sweep.png", dpi=240)
    plt.close()


def plot_metric_bars(payload: dict) -> None:
    camera = payload["measured_baselines"]["camera_current"]
    microphone = payload["measured_baselines"]["microphone_current"]
    target = payload["projected_scenarios"]["target"]
    labels = ["ROC-AUC", "PR-AUC", "Accuracy", "Precision", "Recall"]
    camera_values = [camera["roc_auc"], camera["pr_auc"], camera["accuracy"], camera["precision"], camera["recall"]]
    microphone_values = [microphone["roc_auc"], microphone["pr_auc"], microphone["accuracy"], microphone["precision"], microphone["recall"]]
    target_values = [target["roc_auc"], target["pr_auc"], target["accuracy"], target["precision"], target["recall"]]
    x = np.arange(len(labels))
    width = 0.24
    plt.figure(figsize=(10.8, 6.4))
    plt.bar(x - width, camera_values, width, color="#3b82f6", label="Camera measured")
    plt.bar(x, microphone_values, width, color="#6366f1", label="Microphone measured")
    plt.bar(x + width, target_values, width, color="#10b981", label="Fused target projected")
    plt.ylim(0, 1.02)
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("Measured Modalities vs Projected Fused Target")
    plt.grid(axis="y", alpha=0.25, linestyle="--")
    plt.legend()
    add_projection_note(plt.gca())
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "measured_vs_projected_metrics.png", dpi=240)
    plt.close()


def plot_existing_work(payload: dict) -> None:
    rows = build_existing_work_rows(payload)
    methods = [row["method"] for row in rows]
    values = [row["roc_auc"] for row in rows]
    colors = []
    for row in rows:
        if row["claim_type"] == "literature":
            colors.append("#9ca3af")
        elif row["claim_type"] == "measured":
            colors.append("#3b82f6")
        else:
            colors.append("#10b981")
    plt.figure(figsize=(12.8, 7.2))
    y = np.arange(len(methods))
    plt.barh(y, values, color=colors)
    plt.xlim(0, 1.0)
    plt.yticks(y, methods, fontsize=9)
    plt.xlabel("ROC-AUC")
    plt.title("Results Comparison with Existing Work")
    plt.grid(axis="x", alpha=0.25, linestyle="--")
    add_projection_note(plt.gca(), "Green rows are projected QNA-Auth fusion scenarios")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "results_comparison_existing_work.png", dpi=240)
    plt.close()


def plot_attack_projection(payload: dict) -> None:
    rows = build_attack_rows(payload)
    attacks = ["replay_asr", "impersonation_asr", "synthetic_statistics_asr"]
    scenarios = ["conservative", "target", "stretch"]
    x = np.arange(len(attacks))
    width = 0.24
    colors = {"conservative": "#3b82f6", "target": "#10b981", "stretch": "#f59e0b"}
    plt.figure(figsize=(10.4, 6.2))
    for idx, scenario in enumerate(scenarios):
        values = [next(row["value"] for row in rows if row["scenario"] == scenario and row["attack"] == attack) for attack in attacks]
        plt.bar(x + (idx - 1) * width, values, width, color=colors[scenario], label=scenario.title())
    plt.xticks(x, ["Replay", "Impersonation", "Synthetic-stat"])
    plt.ylabel("Attack Success Rate")
    plt.ylim(0, 0.08)
    plt.title("Projected Attack-Surface Rates")
    plt.grid(axis="y", alpha=0.25, linestyle="--")
    plt.legend()
    add_projection_note(plt.gca())
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "projected_attack_rates.png", dpi=240)
    plt.close()


def plot_score_distribution(payload: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True)
    rng = np.random.default_rng(42)
    colors = {"genuine": "#10b981", "impostor": "#ef4444"}
    for ax, scenario in zip(axes, ["conservative", "target", "stretch"]):
        metrics = payload["projected_scenarios"][scenario]
        genuine = np.clip(rng.normal(metrics["genuine_mean"], metrics["genuine_std"], 600), 0.0, 1.0)
        impostor = np.clip(rng.normal(metrics["impostor_mean"], metrics["impostor_std"], 600), 0.0, 1.0)
        ax.hist(impostor, bins=24, alpha=0.62, color=colors["impostor"], label="Impostor", density=True)
        ax.hist(genuine, bins=24, alpha=0.62, color=colors["genuine"], label="Genuine", density=True)
        ax.axvline(metrics["deployed_threshold"], color="#111827", linewidth=2.2, linestyle="--")
        ax.set_title(scenario.title())
        ax.set_xlabel("Similarity score")
        ax.grid(alpha=0.18, linestyle="--")
    axes[0].set_ylabel("Density")
    axes[1].legend(loc="upper center")
    fig.suptitle("Projected Score Distributions by Scenario")
    add_projection_note(axes[-1])
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "projected_score_distributions.png", dpi=240)
    plt.close(fig)


def plot_dashboard(payload: dict) -> None:
    scenarios = ["conservative", "target", "stretch"]
    metrics = ["roc_auc", "eer", "accuracy", "precision", "recall", "far", "frr"]
    matrix = np.array([[payload["projected_scenarios"][scenario][metric] for metric in metrics] for scenario in scenarios], dtype=float)
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m.upper() if m in {"eer", "far", "frr"} else m.replace("_", "-").title() for m in metrics], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels([s.title() for s in scenarios])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="#111827", fontsize=9)
    ax.set_title("Projected Metrics Dashboard")
    fig.colorbar(im, ax=ax, shrink=0.84)
    add_projection_note(ax)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "projected_metrics_dashboard.png", dpi=240)
    plt.close(fig)


def write_overview(payload: dict) -> None:
    target = payload["projected_scenarios"]["target"]
    overview = f"""# Predicted Report

This folder contains projected evaluation artifacts for presentation planning.

Reporting rule:
- `measured` values were copied from repository artifacts.
- `projected` values are forecast estimates for a scaled follow-up experiment.
- None of the projected PNGs or CSVs should be described as measured output.

Most defensible future scenario:
- ROC-AUC: {target["roc_auc"]:.3f}
- PR-AUC: {target["pr_auc"]:.3f}
- EER: {target["eer"]:.3f}
- Accuracy: {target["accuracy"]:.3f}
- Precision: {target["precision"]:.3f}
- Recall: {target["recall"]:.3f}
- FAR: {target["far"]:.3f}
- FRR: {target["frr"]:.3f}

Suggested viva wording:
> These are projected metrics under larger multi-session data and calibrated fusion. They are not claimed as current measured results.
"""
    (OUTPUT_DIR / "README.md").write_text(overview, encoding="utf-8")


def main() -> int:
    ensure_dirs()
    camera = load_json(CAMERA_METRICS)
    microphone = load_json(MIC_METRICS)
    capstone = load_json(CAPSTONE_RESULTS)
    payload = build_projected_payload(camera, microphone, capstone)

    save_payloads(payload)
    plot_roc(payload)
    plot_pr(payload)
    plot_threshold_sweep(payload)
    plot_metric_bars(payload)
    plot_existing_work(payload)
    plot_attack_projection(payload)
    plot_score_distribution(payload)
    plot_dashboard(payload)
    write_overview(payload)

    print(f"Wrote projected report bundle to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
