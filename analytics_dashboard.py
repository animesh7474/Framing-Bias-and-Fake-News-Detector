"""
analytics_dashboard.py — Statistical analytics and chart generation.
Domain: Big Data Analytics
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from collections import Counter
from config import DATASET_PATH, REPORTS_DIR, FRAME_LABELS
from logger import get_logger

log = get_logger("analytics")


def save(fig, name):
    path = os.path.join(REPORTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#12141a")
    plt.close(fig)
    log.info(f"Saved chart -> {path}")
    return path


def run_analytics(df: pd.DataFrame = None):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    if df is None:
        df = pd.read_csv(DATASET_PATH)

    palette = {
        "Economic":    "#f59e0b",
        "Political":   "#3b82f6",
        "Social":      "#10b981",
        "Security":    "#ef4444",
        "Environment": "#22c55e",
    }
    style = {
        "axes.facecolor":   "#1a1d26",
        "figure.facecolor": "#12141a",
        "text.color":       "#e8eaf0",
        "axes.labelcolor":  "#e8eaf0",
        "xtick.color":      "#6b7280",
        "ytick.color":      "#6b7280",
        "axes.edgecolor":   "#252836",
        "grid.color":       "#252836",
    }
    plt.rcParams.update(style)

    # 1. Label distribution bar chart
    counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=[palette.get(l, "#6c63ff") for l in counts.index],
                  edgecolor="none", width=0.6)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(int(bar.get_height())), ha="center", va="bottom",
                color="#e8eaf0", fontsize=11, fontweight="bold")
    ax.set_title("Frame Label Distribution (Dataset)", fontsize=14, color="#e8eaf0", pad=12)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_xlabel("Frame Category", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "label_distribution.png")

    # 2. Text length distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    for label in FRAME_LABELS:
        subset = df[df["label"] == label]["text_length"] if "text_length" in df.columns else \
                 df[df["label"] == label]["text"].str.split().str.len()
        ax.hist(subset, bins=20, alpha=0.6, label=label, color=palette.get(label))
    ax.set_title("Text Length Distribution by Frame", fontsize=14, color="#e8eaf0", pad=12)
    ax.set_xlabel("Word Count", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend(facecolor="#1a1d26", edgecolor="#252836", labelcolor="#e8eaf0")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "text_length_distribution.png")

    # 3. Top keywords per frame
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for ax, label in zip(axes, FRAME_LABELS):
        all_words = " ".join(df[df["label"] == label]["text"]).lower().split()
        stopwords = {"the","a","is","in","of","to","and","for","on","at","with","this","that","are","was","it","by","an","be","has","have","will"}
        words = [w for w in all_words if w not in stopwords and len(w) > 3]
        top = Counter(words).most_common(7)
        if top:
            ws, cs = zip(*top)
            ax.barh(ws[::-1], cs[::-1], color=palette.get(label, "#6c63ff"), edgecolor="none")
        ax.set_title(label, color=palette.get(label), fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors="#e8eaf0")
    fig.suptitle("Top Keywords per Frame Category", fontsize=14, color="#e8eaf0", y=1.02)
    save(fig, "top_keywords_per_frame.png")

    paths = {
        "label_distribution": os.path.join(REPORTS_DIR, "label_distribution.png"),
        "text_length":        os.path.join(REPORTS_DIR, "text_length_distribution.png"),
        "top_keywords":       os.path.join(REPORTS_DIR, "top_keywords_per_frame.png"),
    }
    log.info("Analytics complete. Charts saved to reports/")
    return paths


if __name__ == "__main__":
    run_analytics()
    print("Analytics charts saved to reports/")
