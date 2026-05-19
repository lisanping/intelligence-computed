"""
L30 - AI Coding - Evaluator richness bar chart (V4)
=============================================================
Goal   : Visualize why coding has the richest Evaluator signals
         compared to other domains (writing, QA, math, translation)
Figures:
  figures/evaluator_richness.png  -- 5-domain Evaluator signal comparison (V4)
Run    : python demo/coding_eval_viz.py
Deps   : pip install matplotlib numpy
Seed   : 1337
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(encoding="utf-8")

np.random.seed(1337)
FIG_DIR = Path(__file__).parent / "figures"

def plot_evaluator_richness(save: bool = True):
    domains = ["Coding", "Math", "Translation", "QA", "Creative\nWriting"]
    signals = {
        "Syntax/Compile": [95, 90, 20, 10, 5],
        "Execution/Tests": [90, 70, 10, 5, 0],
        "Static Analysis": [85, 30, 50, 20, 10],
        "Human Pref.":     [70, 60, 80, 75, 90],
        "Auto Metrics":    [80, 85, 75, 60, 40],
    }
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#f39c12", "#e74c3c"]

    x = np.arange(len(domains))
    n = len(signals)
    w = 0.15

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (label, vals) in enumerate(signals.items()):
        offset = (i - n / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=label, color=colors[i],
                       alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=11)
    ax.set_ylabel("Signal Availability (%)", fontsize=12)
    ax.set_title("Evaluator Signal Richness by Domain",
                 fontsize=14, weight="bold")
    ax.legend(fontsize=9, ncol=3, loc="upper right")
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", alpha=0.3)
    ax.annotate("Coding: richest\nevaluator signals", xy=(0, 95),
                xytext=(1.5, 100), fontsize=10, color="green", weight="bold",
                arrowprops=dict(arrowstyle="->", color="green", lw=2))
    plt.tight_layout()

    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / "evaluator_richness.png", dpi=150, bbox_inches="tight")
        print(f"  saved -> {FIG_DIR / 'evaluator_richness.png'}")
    plt.close(fig)


if __name__ == "__main__":
    print("L30 AI Coding -- generating visualization ...")
    plot_evaluator_richness()
    print("Done.")
