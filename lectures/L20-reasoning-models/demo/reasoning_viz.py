"""
L26 · Reasoning Models — 可视化合集
=============================================================
Goal   : 生成推理模型讲座的 2 张核心 matplotlib 可视化
Figures:
  figures/gsm8k_cot.png          — CoT 涌现：模型规模 × 提示方式 (V3)
  figures/scaling_curves.png     — Train-Time vs Test-Time Scaling (V5)
Run    : python demo/reasoning_viz.py [--fig cot|scaling|all]
Deps   : pip install matplotlib numpy
Seed   : 1337
"""
import argparse
import numpy as np
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[4] / "design" / "meta"))
import cjk_font  # noqa: F401  - applies CJK font config
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(1337)
FIG_DIR = Path(__file__).parent / "figures"


# ── V3: CoT emergence on GSM8K ──────────────────────────────
def plot_gsm8k_cot(save: bool = True):
    """GSM8K accuracy: model size × standard vs CoT prompt."""
    sizes = ["8B", "62B", "540B"]
    standard = [17.9, 33.0, 56.5]
    cot = [3.1, 33.7, 74.4]  # Wei et al. 2022, PaLM on GSM8K

    x = np.arange(len(sizes))
    w = 0.32

    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - w / 2, standard, w, label="Standard Prompt",
                color="steelblue", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w / 2, cot, w, label="Chain-of-Thought",
                color="darkorange", alpha=0.85, edgecolor="white")

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                    f"{bar.get_height():.1f}%", ha="center", fontsize=10, weight="bold")

    # Emergence arrow
    ax.annotate("CoT emergence!\n(+17.9pp)", xy=(2 + w / 2, 74.4),
                xytext=(2.5, 55), fontsize=11, color="darkorange", weight="bold",
                arrowprops=dict(arrowstyle="->", color="darkorange", lw=2))
    ax.annotate("CoT hurts\nsmall models", xy=(0 + w / 2, 3.1),
                xytext=(0.5, 18), fontsize=10, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    ax.set_xlabel("Model Size (PaLM)", fontsize=13)
    ax.set_ylabel("GSM8K Accuracy (%)", fontsize=13)
    ax.set_title("Chain-of-Thought: An Emergent Ability of Scale",
                 fontsize=14, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 90)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "gsm8k_cot.png", save)


# ── V5: Train-time vs Test-time scaling ──────────────────────
def plot_scaling_curves(save: bool = True):
    """Train-time vs test-time compute scaling (conceptual)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Train-time scaling (log-linear, diminishing)
    train_flops = np.array([1e22, 3e22, 1e23, 3e23, 1e24, 3e24, 1e25])
    train_score = 42 + 18 * np.log10(train_flops / 1e22) / np.log10(1e25 / 1e22)
    train_score = np.clip(train_score + np.array([0, 1, 0, -0.5, 0, -1, -1.5]), 40, 62)

    # Test-time scaling (same base model, more inference compute)
    test_tokens = np.array([1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5])
    test_score = 48 + 15 * np.log10(test_tokens / 1e2) / np.log10(1e5 / 1e2)

    ax.plot(np.arange(len(train_flops)), train_score, "b-o", lw=2.5, ms=8,
            label="Train-Time Scaling (bigger model)", zorder=3)
    ax.plot(np.arange(len(test_tokens)), test_score, color="darkorange",
            ls="-", marker="s", lw=2.5, ms=8,
            label="Test-Time Scaling (more inference)", zorder=3)

    ax.set_xticks(np.arange(len(train_flops)))
    ax.set_xticklabels(["1x", "3x", "10x", "30x", "100x", "300x", "1000x"],
                        fontsize=10)
    ax.set_xlabel("Relative Compute Budget", fontsize=13)
    ax.set_ylabel("Benchmark Score", fontsize=13)
    ax.set_title("Two Scaling Paradigms: Train-Time vs Test-Time Compute",
                 fontsize=14, weight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Diminishing returns annotation
    ax.annotate("diminishing returns\n($100M → $1B)", xy=(5, train_score[5]),
                xytext=(4, 55), fontsize=10, color="blue",
                arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))

    # Test-time advantage
    ax.annotate("same model,\nmore thinking", xy=(5, test_score[5]),
                xytext=(3, 61), fontsize=10, color="darkorange",
                arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.5))

    # Cost comparison inset
    inset_text = (
        "Cost per problem (est.)\n"
        "─────────────────\n"
        "GPT-4o:  ~$0.005\n"
        "o3:      ~$0.5–2.0\n"
        "Ratio:   ~100–400×"
    )
    ax.text(0.02, 0.97, inset_text, transform=ax.transAxes, fontsize=9,
            va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    plt.tight_layout()
    _save(fig, "scaling_curves.png", save)


# ── helpers ──────────────────────────────────────────────────
def _save(fig, name, save):
    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
        print(f"  saved → {FIG_DIR / name}")
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="L20 Reasoning Models visualizations")
    p.add_argument("--fig", choices=["cot", "scaling", "all"],
                   default="all", help="which figure to generate (default: all)")
    args = p.parse_args()

    dispatch = dict(cot=plot_gsm8k_cot, scaling=plot_scaling_curves)
    if args.fig == "all":
        print("L20 Reasoning Models — generating all figures …")
        for name, fn in dispatch.items():
            print(f"[{name}]")
            fn()
        print("Done ✓")
    else:
        dispatch[args.fig]()


if __name__ == "__main__":
    main()
