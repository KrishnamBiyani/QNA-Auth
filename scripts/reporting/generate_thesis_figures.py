from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "report_images"
CAMERA_METRICS = ROOT / "model" / "evaluation" / "camera_real_aug_v2" / "metrics.json"
MIC_METRICS = ROOT / "model" / "evaluation" / "microphone_real_aug_v2" / "metrics.json"
MANIFEST_PATH = ROOT / "dataset" / "samples" / "manifest.v1.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "segoeuib.ttf" if bold else "segoeui.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_box(draw: ImageDraw.ImageDraw, xy, fill, outline, radius=26):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=3)


def _draw_arrow(draw: ImageDraw.ImageDraw, start, end, fill, width=6):
    draw.line([start, end], fill=fill, width=width)
    ex, ey = end
    sx, sy = start
    dx = ex - sx
    dy = ey - sy
    angle = np.arctan2(dy, dx)
    left = (ex - 16 * np.cos(angle - np.pi / 6), ey - 16 * np.sin(angle - np.pi / 6))
    right = (ex - 16 * np.cos(angle + np.pi / 6), ey - 16 * np.sin(angle + np.pi / 6))
    draw.polygon([end, left, right], fill=fill)


def _multiline_center(draw: ImageDraw.ImageDraw, box, text, font, fill):
    left, top, right, bottom = box
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=6, align="center")
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = left + (right - left - tw) / 2
    y = top + (bottom - top - th) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=6, align="center")


def make_architecture_diagram(output: Path) -> None:
    img = Image.new("RGB", (1800, 1100), "#07111d")
    draw = ImageDraw.Draw(img)
    title_font = _font(54, bold=True)
    box_title = _font(30, bold=True)
    box_body = _font(22)
    accent = "#65e3b4"
    blue = "#5ea4ff"
    fg = "#eef5ff"
    line = "#24496f"

    draw.text((70, 50), "QNA-Auth System Architecture", font=title_font, fill=fg)
    draw.text((70, 118), "Camera-first multimodal device authentication over a FastAPI service", font=_font(26), fill="#9cb1d2")

    boxes = {
        "capture": (70, 210, 490, 430),
        "preprocess": (560, 210, 980, 430),
        "model": (1050, 210, 1470, 430),
        "decision": (560, 530, 980, 760),
        "storage": (1050, 530, 1470, 760),
        "client": (70, 530, 490, 760),
    }
    fills = {
        "capture": "#10243a",
        "preprocess": "#122740",
        "model": "#10243a",
        "decision": "#173049",
        "storage": "#10243a",
        "client": "#173049",
    }
    for key, box in boxes.items():
        _draw_box(draw, box, fills[key], line)

    _multiline_center(draw, boxes["capture"], "Physical Capture\n\nCamera dark-frame noise\nMicrophone ambient noise", box_title, fg)
    _multiline_center(draw, boxes["preprocess"], "Preprocessing + Features\n\nNormalization\nStatistics\nFFT / autocorrelation\nComplexity features", box_title, fg)
    _multiline_center(draw, boxes["model"], "Siamese / Embedding Model\n\n128-D L2-normalized embeddings\nTriplet training\nPer-source inference", box_title, fg)
    _multiline_center(draw, boxes["client"], "Client Collection Paths\n\nLaptop collector script\nPhone LAN collector page\nSession-based sampling", box_title, fg)
    _multiline_center(draw, boxes["decision"], "Authentication Logic\n\nWeighted fusion\nCamera 0.85\nMicrophone 0.15\nMargin + threshold checks", box_title, fg)
    _multiline_center(draw, boxes["storage"], "Persistence Layer\n\nDevice templates\nSource templates\nThreshold metadata\nCollected dataset sessions", box_title, fg)

    _draw_arrow(draw, (490, 320), (560, 320), blue)
    _draw_arrow(draw, (980, 320), (1050, 320), blue)
    _draw_arrow(draw, (1260, 430), (1260, 530), accent)
    _draw_arrow(draw, (770, 430), (770, 530), accent)
    _draw_arrow(draw, (490, 645), (560, 645), blue)
    _draw_arrow(draw, (980, 645), (1050, 645), blue)
    _draw_arrow(draw, (1260, 530), (980, 645), accent)

    draw.text((70, 930), "Key design choice: camera is the required primary modality; microphone contributes supportive evidence but does not veto the decision.", font=_font(26), fill=accent)
    img.save(output)


def make_flow_diagram(output: Path) -> None:
    img = Image.new("RGB", (1800, 900), "#08111b")
    draw = ImageDraw.Draw(img)
    fg = "#eef5ff"
    muted = "#9db2d3"
    green = "#65e3b4"
    blue = "#6aa7ff"
    line = "#2a4b6d"
    title_font = _font(50, bold=True)
    step_font = _font(24, bold=True)
    body_font = _font(20)

    draw.text((70, 48), "Enrollment and Verification Flow", font=title_font, fill=fg)
    draw.text((70, 110), "End-to-end collection, feature generation, template storage, and weighted verification", font=_font(24), fill=muted)

    steps = [
        ("1. Collect session", "Capture camera and optional microphone\nsamples from laptop or phone"),
        ("2. Extract features", "Convert raw arrays into 33-D fixed\nfeature vectors"),
        ("3. Create templates", "Build per-source template banks and\ncombined embeddings"),
        ("4. Store profile", "Persist embeddings, metadata, and\nsource-specific thresholds"),
        ("5. Verify live probe", "Score probe against templates with\ncamera-heavy weighting"),
        ("6. Decide", "Accept / uncertain / reject with\nmargin checks and audit details"),
    ]
    x = 70
    y = 220
    w = 250
    h = 180
    gap = 30
    for index, (title, body) in enumerate(steps):
        box = (x + index * (w + gap), y, x + index * (w + gap) + w, y + h)
        _draw_box(draw, box, "#102238" if index % 2 == 0 else "#15314b", line)
        _multiline_center(draw, (box[0] + 12, box[1] + 12, box[2] - 12, box[1] + 68), title, step_font, fg)
        _multiline_center(draw, (box[0] + 16, box[1] + 74, box[2] - 16, box[3] - 18), body, body_font, "#d7e4fb")
        if index < len(steps) - 1:
            _draw_arrow(draw, (box[2], y + h // 2), (box[2] + gap, y + h // 2), blue)

    callout = (120, 520, 1680, 780)
    _draw_box(draw, callout, "#0f2235", line)
    draw.text((160, 565), "Operational policy used for the thesis build", font=_font(32, bold=True), fill=green)
    bullets = [
        "Camera is the primary modality and the only required source during final accept/reject.",
        "Microphone remains auxiliary: it adds evidence when clean but does not override a strong camera match.",
        "Thresholds are locked from held-out evaluation reports rather than tuned on the same deployment probes.",
    ]
    yy = 625
    for bullet in bullets:
        draw.text((190, yy), f"- {bullet}", font=_font(24), fill=fg)
        yy += 52
    img.save(output)


def make_modality_comparison(camera: dict, mic: dict, output: Path) -> None:
    labels = ["ROC-AUC", "PR-AUC", "1 - EER", "Accuracy", "Recall"]
    cam = [
        camera["roc_auc"],
        camera["pr_auc"],
        1.0 - camera["eer"],
        camera["deployed_threshold_metrics"]["accuracy"],
        camera["deployed_threshold_metrics"]["recall"],
    ]
    mic_vals = [
        mic["roc_auc"],
        mic["pr_auc"],
        1.0 - mic["eer"],
        mic["deployed_threshold_metrics"]["accuracy"],
        mic["deployed_threshold_metrics"]["recall"],
    ]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(11, 6.5))
    plt.bar(x - width / 2, cam, width, label="Camera", color="#4ca7ff")
    plt.bar(x + width / 2, mic_vals, width, label="Microphone", color="#7be0b3")
    plt.ylim(0, 1.0)
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("Modality Comparison on Held-Out Evaluation")
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.legend()
    for i, value in enumerate(cam):
        plt.text(i - width / 2, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)
    for i, value in enumerate(mic_vals):
        plt.text(i + width / 2, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()


def make_operating_point_chart(camera: dict, mic: dict, output: Path) -> None:
    labels = ["FAR", "FRR", "Precision", "Recall"]
    cam = [
        camera["deployed_threshold_metrics"]["far"],
        camera["deployed_threshold_metrics"]["frr"],
        camera["deployed_threshold_metrics"]["precision"],
        camera["deployed_threshold_metrics"]["recall"],
    ]
    mic_vals = [
        mic["deployed_threshold_metrics"]["far"],
        mic["deployed_threshold_metrics"]["frr"],
        mic["deployed_threshold_metrics"]["precision"],
        mic["deployed_threshold_metrics"]["recall"],
    ]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(10.5, 6.2))
    plt.bar(x - width / 2, cam, width, label="Camera deployed threshold", color="#0d6efd")
    plt.bar(x + width / 2, mic_vals, width, label="Microphone deployed threshold", color="#20c997")
    plt.ylim(0, 1.05)
    plt.xticks(x, labels)
    plt.ylabel("Rate / score")
    plt.title("Deployed Operating Point Comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()


def make_training_curves(camera: dict, mic: dict, output: Path) -> None:
    plt.figure(figsize=(11, 6.8))
    cam_train = camera["history"]["train_loss"]
    mic_train = mic["history"]["train_loss"]
    plt.plot(range(1, len(cam_train) + 1), cam_train, label="Camera train loss", color="#4ca7ff", linewidth=2.2)
    plt.plot(range(1, len(mic_train) + 1), mic_train, label="Microphone train loss", color="#65e3b4", linewidth=2.2)
    mic_val = mic["history"].get("val_loss", [])
    if mic_val:
        plt.plot(range(1, len(mic_val) + 1), mic_val, label="Microphone val loss", color="#ffb84d", linewidth=2.0, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Dynamics by Modality")
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()


def make_dataset_summary(manifest: dict, output: Path) -> None:
    devices = list(manifest["devices"].keys())
    camera_counts = []
    mic_counts = []
    for device_id in devices:
        total = manifest["devices"][device_id]["samples"]
        sources = set(manifest["devices"][device_id]["sources"])
        if sources == {"camera"}:
            camera_counts.append(total)
            mic_counts.append(0)
        else:
            # Samples are balanced per source in this dataset construction.
            camera_counts.append(total // 2)
            mic_counts.append(total // 2)

    x = np.arange(len(devices))
    plt.figure(figsize=(12, 6.6))
    plt.bar(x, camera_counts, label="Camera samples", color="#4ca7ff")
    plt.bar(x, mic_counts, bottom=camera_counts, label="Microphone samples", color="#65e3b4")
    plt.xticks(x, [device[:8] for device in devices], rotation=0)
    plt.ylabel("Samples")
    plt.title("Dataset Composition by Device")
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()


def make_score_summary(camera: dict, mic: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), sharey=True)
    for ax, report, title, colors in [
        (axes[0], camera, "Camera score summary", ("#4ca7ff", "#003b7a")),
        (axes[1], mic, "Microphone score summary", ("#65e3b4", "#00695c")),
    ]:
        labels = ["Genuine", "Impostor"]
        means = [report["score_summary"]["genuine"]["mean"], report["score_summary"]["impostor"]["mean"]]
        stds = [report["score_summary"]["genuine"]["std"], report["score_summary"]["impostor"]["std"]]
        ax.bar(labels, means, yerr=stds, color=[colors[0], colors[1]], alpha=0.88, capsize=8)
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.suptitle("Score Distribution Summary (mean ± std)")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    camera = _load_json(CAMERA_METRICS)
    mic = _load_json(MIC_METRICS)
    manifest = _load_json(MANIFEST_PATH)

    make_architecture_diagram(OUTPUT_DIR / "architecture_system_overview.png")
    make_flow_diagram(OUTPUT_DIR / "enrollment_authentication_flow.png")
    make_modality_comparison(camera, mic, OUTPUT_DIR / "modality_performance_comparison.png")
    make_operating_point_chart(camera, mic, OUTPUT_DIR / "deployed_operating_point_comparison.png")
    make_training_curves(camera, mic, OUTPUT_DIR / "training_dynamics.png")
    make_dataset_summary(manifest, OUTPUT_DIR / "dataset_composition.png")
    make_score_summary(camera, mic, OUTPUT_DIR / "score_distribution_summary.png")

    print(f"Wrote thesis figures to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
