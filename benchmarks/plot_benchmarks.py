"""
Generate benchmark charts from the CSV result tables.

Reads the CSVs in this directory and writes PNG charts to ../docs/media/.
Run from anywhere:  python benchmarks/plot_benchmarks.py
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = Path(__file__).resolve().parent
MEDIA = HERE.parent / "docs" / "media"
MEDIA.mkdir(parents=True, exist_ok=True)

# Consistent colour per deployment environment across all charts.
COLORS = {
    "Jetson Native": "#2a9d8f",
    "Jetson Docker": "#e76f51",
    "Laptop CPU": "#6c757d",
    "Desktop GPU": "#264653",
    "SSH offload (remote GPU)": "#e9c46a",
}


def _bar_colors(labels):
    return [COLORS.get(lbl, "#888888") for lbl in labels]


def plot_yolov9c_inference():
    """Same model (YOLOv9-C) across environments: inference time, log scale."""
    df = pd.read_csv(HERE / "offline_validation_yolov9c.csv")
    df = df[df["resolution"] == "640x640"].copy()
    labels = df["deployment"].tolist()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, df["inference_ms"], color=_bar_colors(labels))
    ax.set_yscale("log")
    ax.set_ylabel("Inference time per frame (ms, log scale)")
    ax.set_title("YOLOv9-C offline inference across deployment environments\n(COCO128, 640\u00d7640)")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    for b, v in zip(bars, df["inference_ms"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(MEDIA / "yolov9c_inference_by_environment.png", dpi=140)
    plt.close(fig)


def plot_speed_vs_accuracy():
    """Speed vs accuracy trade-off for the best-fit model on each environment."""
    df = pd.read_csv(HERE / "offline_validation_best_fit.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in df.iterrows():
        ax.scatter(row["inference_ms"], row["map_50_95"],
                   color=COLORS.get(row["deployment"], "#888888"), s=90, zorder=3)
        ax.annotate(f'{row["model"]} ({row["format"]})',
                    (row["inference_ms"], row["map_50_95"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.set_xlabel("Inference time per frame (ms)")
    ax.set_ylabel("mAP@0.5:0.95")
    ax.set_title("Speed vs. accuracy: best-fit model per environment\n(COCO128 offline validation)")
    ax.grid(alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(MEDIA / "speed_vs_accuracy.png", dpi=140)
    plt.close(fig)


def plot_realtime_fps():
    """Real-time webcam FPS across configurations."""
    df = pd.read_csv(HERE / "realtime_inference.csv")
    df = df.sort_values("real_fps")
    labels = [f'{m}\n{d} ({r})' for m, d, r in
              zip(df["model"], df["deployment"], df["resolution"])]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, df["real_fps"], color=_bar_colors(df["deployment"].tolist()))
    ax.set_xlabel("Real-time throughput (FPS)")
    ax.set_title("Real-time webcam inference across deployments")
    for b, v in zip(bars, df["real_fps"]):
        ax.text(v, b.get_y() + b.get_height() / 2, f" {v}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(MEDIA / "realtime_fps.png", dpi=140)
    plt.close(fig)


def main():
    plot_yolov9c_inference()
    plot_speed_vs_accuracy()
    plot_realtime_fps()
    print(f"Charts written to {MEDIA}")


if __name__ == "__main__":
    main()
