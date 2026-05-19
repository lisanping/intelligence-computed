"""
L28 · Context Engineering — 可视化合集
=============================================================
Goal   : 生成上下文工程讲座的 3 张核心 matplotlib 可视化
Figures:
  figures/needle_in_haystack.png   — Needle-in-a-Haystack 热力图 (V5a)
  figures/lost_in_middle.png       — Lost-in-the-Middle U 曲线 (V5b)
  figures/token_cost_curves.png    — Naive vs Optimized 成本曲线 (V9)
Run    : python demo/context_viz.py [--fig needle|lost|cost|all]
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


# ── V5a: Needle-in-a-Haystack heatmap ───────────────────────
def plot_needle_heatmap(save: bool = True):
    """Retrieval accuracy by context length × insertion position."""
    ctx_lengths = [1, 2, 4, 8, 16, 32, 64, 128]  # K tokens
    positions = np.linspace(0, 1, 10)  # 0=top, 1=bottom
    acc = np.zeros((len(positions), len(ctx_lengths)))

    for j, cl in enumerate(ctx_lengths):
        for i, pos in enumerate(positions):
            # Simulated: accuracy drops for longer context + middle positions
            length_penalty = max(0, (np.log2(cl) - 2) * 5)
            middle_penalty = 25 * np.exp(-8 * (pos - 0.5) ** 2) * (cl > 4)
            base = 98 - length_penalty - middle_penalty
            acc[i, j] = np.clip(base + np.random.normal(0, 2), 40, 100)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(acc, aspect="auto", cmap="RdYlGn", vmin=40, vmax=100,
                   interpolation="bilinear")
    ax.set_xticks(range(len(ctx_lengths)))
    ax.set_xticklabels([f"{k}K" for k in ctx_lengths], fontsize=10)
    ax.set_yticks(range(len(positions)))
    ax.set_yticklabels([f"{int(p*100)}%" for p in positions], fontsize=9)
    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel("Needle Position (top → bottom)", fontsize=12)
    ax.set_title("Needle-in-a-Haystack: Retrieval Accuracy (%)",
                 fontsize=14, weight="bold")
    fig.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)
    ax.text(6, 4.5, "Lost in\nthe Middle", fontsize=11, color="white",
            weight="bold", ha="center",
            bbox=dict(boxstyle="round", facecolor="red", alpha=0.6))
    plt.tight_layout()
    _save(fig, "needle_in_haystack.png", save)


# ── V5b: Lost-in-the-Middle U-curve ─────────────────────────
def plot_lost_in_middle(save: bool = True):
    """Liu et al. 2023 — accuracy vs document position."""
    n_docs = np.arange(1, 21)
    # U-shaped: high at edges, low in middle
    center = 10.5
    acc = 80 - 28 * np.exp(-0.15 * (n_docs - center) ** 2)
    acc += np.random.normal(0, 1.5, len(n_docs))
    acc[0] = 80.2; acc[-1] = 78.1; acc[9] = 52.3; acc[10] = 53.8

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(n_docs, acc, "o-", color="steelblue", lw=2, ms=6)
    ax.axhline(np.mean(acc), color="gray", ls="--", alpha=0.4, label="mean")
    ax.fill_between(n_docs, acc, alpha=0.1, color="steelblue")

    ax.annotate(f"start: {acc[0]:.0f}%", xy=(1, acc[0]), fontsize=10,
                xytext=(3, 85), arrowprops=dict(arrowstyle="->", color="green"),
                color="green", weight="bold")
    ax.annotate(f"middle: {acc[9]:.0f}%", xy=(10, acc[9]), fontsize=10,
                xytext=(12, 48), arrowprops=dict(arrowstyle="->", color="red"),
                color="red", weight="bold")
    ax.annotate(f"end: {acc[-1]:.0f}%", xy=(20, acc[-1]), fontsize=10,
                xytext=(17, 85), arrowprops=dict(arrowstyle="->", color="green"),
                color="green", weight="bold")

    ax.set_xlabel("Relevant Document Position (1=first, 20=last)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Lost in the Middle (Liu et al. 2023)", fontsize=14, weight="bold")
    ax.set_ylim(40, 92); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "lost_in_middle.png", save)


# ── V9: Token cost curves — naive vs optimized ──────────────
def plot_cost_curves(save: bool = True):
    """Naive (linear) vs optimized (flat) token consumption over 50 turns."""
    turns = np.arange(1, 51)
    sys_tokens = 800  # system prompt
    avg_turn_tokens = 400  # avg user+assistant per turn

    # Naive: accumulates all history
    naive = sys_tokens + turns * avg_turn_tokens  # linear growth

    # Optimized: compress at 20 turns, keep last 10, summary ~2K
    optimized = np.zeros_like(turns, dtype=float)
    for t in turns:
        recent = min(t, 10) * avg_turn_tokens
        summary = 2000 if t > 20 else 0
        optimized[t - 1] = sys_tokens + summary + recent
    optimized = np.clip(optimized, 0, None)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(turns, naive, "b-", lw=2.5, label="Naive (full history)")
    ax.plot(turns, optimized, "g-", lw=2.5, label="Optimized (compress + truncate)")
    ax.fill_between(turns, optimized, naive, alpha=0.12, color="red")

    # Savings annotation
    mid = 35
    gap = naive[mid] - optimized[mid]
    ax.annotate(f"~{naive[mid]/optimized[mid]:.0f}× savings\n({gap:,.0f} tokens)",
                xy=(mid + 1, (naive[mid] + optimized[mid]) / 2),
                fontsize=11, color="red", weight="bold", ha="center",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    ax.set_xlabel("Conversation Turn", fontsize=13)
    ax.set_ylabel("Tokens per Request", fontsize=13)
    ax.set_title("Context Cost: Naive vs Optimized Strategy",
                 fontsize=14, weight="bold")
    ax.legend(fontsize=12)
    ax.set_xlim(1, 50); ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.02,
            "Compression trigger: 20 turns\nKeep last 10 + summary (~2K tokens)",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8))
    plt.tight_layout()
    _save(fig, "token_cost_curves.png", save)


def _save(fig, name, save):
    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
        print(f"  saved → {FIG_DIR / name}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="L22 Context Engineering visualizations")
    p.add_argument("--fig", choices=["needle", "lost", "cost", "all"],
                   default="all", help="which figure to generate")
    args = p.parse_args()

    dispatch = dict(needle=plot_needle_heatmap, lost=plot_lost_in_middle,
                    cost=plot_cost_curves)
    if args.fig == "all":
        print("L22 Context Engineering — generating all figures …")
        for name, fn in dispatch.items():
            print(f"[{name}]")
            fn()
        print("Done ✓")
    else:
        dispatch[args.fig]()


if __name__ == "__main__":
    main()
