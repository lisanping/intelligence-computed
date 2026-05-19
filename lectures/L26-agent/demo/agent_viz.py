"""
L32 · Agent — Reflexion 消融对比图
=============================================================
Goal   : 生成 Reflexion 效果对比柱状图 (V8)
Figures:
  figures/reflexion_ablation.png  — GPT-4 单次 vs Reflexion vs 无记忆循环
Run    : python demo/agent_viz.py
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


def plot_reflexion_ablation(save: bool = True):
    """V8 — Reflexion ablation: single-shot vs reflexion vs loop-no-memory."""
    configs = ["GPT-4\nSingle-shot", "GPT-4 +\nLoop (no memory)", "GPT-4 +\nReflexion (3 rounds)"]
    scores = [80.1, 83.4, 91.0]
    colors = ["steelblue", "lightcoral", "seagreen"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(configs, scores, color=colors, alpha=0.85, edgecolor="white", width=0.55)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{score:.1f}%", ha="center", fontsize=12, weight="bold")

    # Highlight the gap
    ax.annotate("", xy=(2, 91), xytext=(0, 80.1),
                arrowprops=dict(arrowstyle="->", color="green", lw=2,
                                connectionstyle="arc3,rad=0.2"))
    ax.text(1.5, 85, "+10.9pp", fontsize=12, color="green", weight="bold")

    ax.annotate("memory injection\nis the key", xy=(1.5, 83.4),
                xytext=(1.8, 78), fontsize=10, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    ax.set_ylabel("HumanEval Pass@1 (%)", fontsize=13)
    ax.set_title("Reflexion Ablation: Memory Injection Matters",
                 fontsize=14, weight="bold")
    ax.set_ylim(70, 96)
    ax.grid(True, axis="y", alpha=0.3)
    ax.text(0.98, 0.02, "Source: Shinn et al. 2023",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            color="gray")
    plt.tight_layout()

    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / "reflexion_ablation.png", dpi=150, bbox_inches="tight")
        print(f"  saved → {FIG_DIR / 'reflexion_ablation.png'}")
    plt.close(fig)


def main():
    print("L26 Agent — generating visualization …")
    plot_reflexion_ablation()
    print("Done ✓")


if __name__ == "__main__":
    main()
