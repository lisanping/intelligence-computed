"""
L21 · Pretraining — 可视化合集
=============================================================
Goal   : 生成预训练讲座的 5 张核心 matplotlib 可视化
Figures:
  figures/data_mix.png           — 数据混合配方对比 (V3)
  figures/data_funnel.png        — 数据清洗漏斗 (V4)
  figures/pipeline_bubble.png    — 流水线气泡对比 (V8)
  figures/training_cost.png      — 训练成本阶梯图 (V11)
  figures/memory_budget.png      — GPU 显存预算拆解
Run    : python demo/pretraining_viz.py [--fig mix|funnel|bubble|cost|mem|all]
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


# ── V3: Data mix stacked bar ────────────────────────────────
def plot_data_mix(save: bool = True):
    """Data composition across LLaMA 1, LLaMA 3, DeepSeek-V3."""
    models = ["LLaMA 1\n(1.4T)", "LLaMA 3\n(15T)", "DeepSeek-V3\n(14.8T)"]
    cats = ["Common Crawl", "Code", "Books", "ArXiv/Academic", "Wikipedia", "Other"]
    data = np.array([
        [67, 4.5, 4.5, 2.5, 4.5, 17],    # LLaMA 1
        [82, 5, 3, 2.5, 2.5, 5],           # LLaMA 3
        [75, 10, 2, 3, 2, 8],              # DeepSeek-V3 (estimated)
    ])
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#b07aa1"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bottom = np.zeros(len(models))
    for i, (cat, col) in enumerate(zip(cats, colors)):
        bars = ax.bar(models, data[:, i], bottom=bottom, label=cat,
                       color=col, alpha=0.85, edgecolor="white")
        for j, (b, v) in enumerate(zip(bars, data[:, i])):
            if v >= 4:
                ax.text(b.get_x() + b.get_width() / 2,
                        bottom[j] + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=8, weight="bold")
        bottom += data[:, i]

    ax.set_ylabel("Data Mix (%)", fontsize=12)
    ax.set_title("Pretraining Data Composition", fontsize=14, weight="bold")
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "data_mix.png", save)


# ── V4: Data cleaning funnel ────────────────────────────────
def plot_data_funnel(save: bool = True):
    """Data cleaning pipeline: 100TB raw → 3TB clean (97% discarded)."""
    stages = ["Raw Crawl", "URL Filter", "Language\nDetect", "HTML\nParse",
              "Dedup", "Quality\nScore", "Safety\nFilter"]
    sizes = [100, 70, 55, 40, 22, 8, 3]  # TB

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(stages)))
    bars = ax.barh(range(len(stages)), sizes, color=colors, edgecolor="white",
                   height=0.6)
    for i, (bar, sz) in enumerate(zip(bars, sizes)):
        pct = sz / sizes[0] * 100
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{sz} TB ({pct:.0f}%)", va="center", fontsize=10, weight="bold")
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Data Size (TB)", fontsize=12)
    ax.set_title("Data Cleaning Pipeline: 97% Discarded",
                 fontsize=14, weight="bold")
    ax.set_xlim(0, 115)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    _save(fig, "data_funnel.png", save)


# ── V8: Pipeline bubble (naive vs 1F1B) ─────────────────────
def plot_pipeline_bubble(save: bool = True):
    """Pipeline parallelism: naive vs 1F1B schedule on 4 GPUs."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    # Naive: GPU fills sequentially, big bubble
    for i, ax in enumerate(axes):
        n_gpu, n_micro = 4, 8
        for g in range(n_gpu):
            if i == 0:  # Naive
                for m in range(n_micro):
                    ax.barh(g, 1, left=g + m * n_gpu, height=0.5,
                            color=f"C{m % 4}", alpha=0.7, edgecolor="white")
            else:  # 1F1B
                for m in range(n_micro):
                    fwd_start = g + m
                    ax.barh(g, 0.8, left=fwd_start, height=0.5,
                            color=f"C{m % 4}", alpha=0.7, edgecolor="white")
        ax.set_yticks(range(n_gpu))
        ax.set_yticklabels([f"GPU {g}" for g in range(n_gpu)], fontsize=10)
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)

    axes[0].set_title("Naive Pipeline: ~50% Bubble", fontsize=12, weight="bold")
    axes[1].set_title("1F1B Schedule: ~12% Bubble", fontsize=12, weight="bold")
    axes[1].set_xlabel("Time Steps", fontsize=11)
    fig.suptitle("Pipeline Parallelism: Reducing GPU Idle Time",
                 fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "pipeline_bubble.png", save)


# ── V11: Training cost staircase ─────────────────────────────
def plot_training_cost(save: bool = True):
    """Training cost evolution: GPT-2 → DeepSeek-V3."""
    models = ["GPT-2\n(2019)", "GPT-3\n(2020)", "PaLM\n(2022)",
              "LLaMA 3\n(2024)", "GPT-4\n(2023)", "DeepSeek\n-V3 (2024)"]
    costs = [0.05, 4.6, 12, 30, 80, 5.5]  # $M
    colors = ["steelblue"] * 5 + ["crimson"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(models, costs, color=colors, alpha=0.85, edgecolor="white",
                  width=0.55)
    for bar, cost in zip(bars, costs):
        label = f"${cost:.1f}M" if cost >= 1 else f"${cost*1000:.0f}K"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                label, ha="center", fontsize=10, weight="bold")

    ax.annotate("14.5x cheaper\nthan GPT-4!", xy=(5, 5.5),
                xytext=(4.2, 50), fontsize=11, color="crimson", weight="bold",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=2))
    ax.set_ylabel("Training Cost ($M)", fontsize=12)
    ax.set_title("Pretraining Cost: The DeepSeek Surprise",
                 fontsize=14, weight="bold")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "training_cost.png", save)


# ── GPU memory budget ────────────────────────────────────────
def plot_memory_budget(save: bool = True):
    """GPU memory breakdown for training a 70B model."""
    cats = ["Params\n(fp16)", "Gradients\n(fp16)", "Optimizer\nStates (fp32)", "Activations\n(est.)"]
    sizes = [140, 140, 560, 200]  # GB
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cats, sizes, color=colors, alpha=0.85, edgecolor="white", width=0.55)
    for bar, sz in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                f"{sz} GB", ha="center", fontsize=11, weight="bold")

    total = sum(sizes)
    ax.axhline(80, color="red", ls="--", alpha=0.6, lw=1.5)
    ax.text(3.5, 85, "A100 80GB", color="red", fontsize=10, ha="right")
    ax.set_ylabel("Memory (GB)", fontsize=12)
    ax.set_title(f"70B Model Training: {total:,} GB Total (need {total//80}+ A100s)",
                 fontsize=13, weight="bold")
    ax.set_ylim(0, 650)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "memory_budget.png", save)


def _save(fig, name, save):
    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
        print(f"  saved → {FIG_DIR / name}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="L15 Pretraining visualizations")
    p.add_argument("--fig", choices=["mix", "funnel", "bubble", "cost", "mem", "all"],
                   default="all", help="which figure to generate")
    args = p.parse_args()
    dispatch = dict(mix=plot_data_mix, funnel=plot_data_funnel,
                    bubble=plot_pipeline_bubble, cost=plot_training_cost,
                    mem=plot_memory_budget)
    if args.fig == "all":
        print("L15 Pretraining — generating all figures …")
        for name, fn in dispatch.items():
            print(f"[{name}]")
            fn()
        print("Done.")
    else:
        dispatch[args.fig]()


if __name__ == "__main__":
    main()
