"""
L24 · Alignment — 可视化合集
=============================================================
Goal   : 生成对齐讲座的 4 张核心 matplotlib 可视化
Figures:
  figures/rm_loss_curve.png      — 奖励模型 Bradley-Terry 损失 (V4)
  figures/ppo_clipping.png       — PPO 裁剪目标函数 (V5)
  figures/distribution_shift.png — Pretrain→SFT→RLHF 分布漂移 (V10)
  figures/alignment_tax.png      — 对齐税双轴柱状图 (V9)
Run    : python demo/alignment_viz.py [--fig rm|ppo|dist|tax|all]
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


# ── V4: Reward Model loss ────────────────────────────────────
def plot_rm_loss(save: bool = True):
    """RM Bradley-Terry loss: -log σ(r_w - r_l)"""
    delta = np.linspace(-6, 6, 500)
    loss = -np.log(1 / (1 + np.exp(-delta)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(delta, loss, "b-", linewidth=2.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(np.log(2), color="red", linestyle=":", alpha=0.5,
               label=f"random baseline (ln2≈{np.log(2):.2f})")
    ax.fill_between(delta, loss, alpha=0.08, color="blue")
    ax.set_xlabel(r"$r_w - r_l$  (reward gap)", fontsize=13)
    ax.set_ylabel(r"$-\log\,\sigma(\Delta r)$", fontsize=13)
    ax.set_title("Reward Model: Bradley-Terry Loss", fontsize=14, weight="bold")
    ax.legend(fontsize=11)
    ax.annotate("RM confident\n(correct ranking)", xy=(4, 0.05), fontsize=10,
                ha="center", bbox=dict(boxstyle="round,pad=0.3",
                facecolor="lightgreen", alpha=0.7))
    ax.annotate("RM confused\n(wrong ranking)", xy=(-4, 4.5), fontsize=10,
                ha="center", bbox=dict(boxstyle="round,pad=0.3",
                facecolor="lightsalmon", alpha=0.7))
    ax.set_xlim(-6, 6); ax.set_ylim(0, 6.5); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "rm_loss_curve.png", save)


# ── V5: PPO clipping ────────────────────────────────────────
def plot_ppo_clipping(save: bool = True, eps: float = 0.2):
    """PPO clipped surrogate objective for A>0 and A<0."""
    ratio = np.linspace(0.0, 2.0, 500)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (a_val, title) in enumerate([
        (1.0, r"Advantage $\hat{A}_t > 0$ (encourage)"),
        (-1.0, r"Advantage $\hat{A}_t < 0$ (discourage)"),
    ]):
        ax = axes[i]
        unclipped = ratio * a_val
        clipped = np.clip(ratio, 1 - eps, 1 + eps) * a_val
        objective = np.minimum(unclipped, clipped)

        ax.plot(ratio, unclipped, "b--", alpha=0.6, lw=1.5,
                label=r"$r_t(\theta)\hat{A}_t$")
        ax.plot(ratio, clipped, "r--", alpha=0.6, lw=1.5,
                label=r"$\mathrm{clip}(r_t,1\pm\epsilon)\hat{A}_t$")
        ax.plot(ratio, objective, "k-", lw=2.5, label=r"$L^{CLIP}$ (min)")
        ax.axvline(1.0, color="gray", ls=":", alpha=0.5)
        ax.axvline(1 - eps, color="orange", ls=":", alpha=0.4)
        ax.axvline(1 + eps, color="orange", ls=":", alpha=0.4)
        ax.axhline(0, color="gray", alpha=0.3)
        mask = (ratio >= 1 - eps) & (ratio <= 1 + eps)
        ax.fill_between(ratio[mask], objective[mask], alpha=0.15, color="green")
        ax.set_xlabel(r"Probability ratio $r_t(\theta)$", fontsize=12)
        ax.set_ylabel("Objective", fontsize=12)
        ax.set_title(title, fontsize=13, weight="bold")
        ax.legend(fontsize=10, loc="upper left" if a_val > 0 else "lower left")
        ax.set_xlim(0, 2); ax.grid(True, alpha=0.3)

    fig.suptitle(f"PPO Clipped Surrogate Objective (ε={eps})",
                 fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "ppo_clipping.png", save)


# ── V10: Distribution shift ─────────────────────────────────
def plot_distribution_shift(save: bool = True):
    """Pretrain→SFT→RLHF KDE distribution shift."""
    x = np.linspace(-4, 8, 1000)
    pretrain = (0.3 * _gauss(x, 1, 1.8) + 0.15 * _gauss(x, -1, 1.2))
    sft = 0.6 * _gauss(x, 3.0, 1.2)
    rlhf = 0.8 * _gauss(x, 4.5, 0.8)

    fig, ax = plt.subplots(figsize=(10, 5))
    for y, c, lbl in [(pretrain, "gray", "Pretrained LM"),
                       (sft, "blue", "After SFT"),
                       (rlhf, "green", "After RLHF")]:
        ax.fill_between(x, y, alpha=0.2, color=c)
        ax.plot(x, y, color=c, lw=2, label=lbl)

    ax.annotate("", xy=(3.0, 0.22), xytext=(1.0, 0.10),
                arrowprops=dict(arrowstyle="->", color="blue", lw=2))
    ax.annotate("SFT narrows", xy=(2.0, 0.17), fontsize=10, color="blue")
    ax.annotate("", xy=(4.5, 0.35), xytext=(3.0, 0.22),
                arrowprops=dict(arrowstyle="->", color="green", lw=2))
    ax.annotate("RLHF sharpens", xy=(3.5, 0.30), fontsize=10, color="green")
    ax.set_xlabel("Response Quality →", fontsize=13)
    ax.set_ylabel("Probability Density", fontsize=13)
    ax.set_title("Distribution Shift: Pretrain → SFT → RLHF",
                 fontsize=14, weight="bold")
    ax.legend(fontsize=12, loc="upper left")
    ax.set_xlim(-4, 8); ax.set_ylim(0, 0.45); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "distribution_shift.png", save)


# ── V9: Alignment tax ───────────────────────────────────────
def plot_alignment_tax(save: bool = True):
    """Alignment tax: capability vs safety dual-axis bar chart."""
    cats = ["Math", "Code", "Factual QA", "Creative\nWriting", "Safety\nRefusal"]
    base = [82, 78, 75, 70, 15]
    aligned = [79, 76, 80, 85, 92]
    x = np.arange(len(cats))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    b1 = ax.bar(x - w / 2, base, w, label="Base Model",
                color="steelblue", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w / 2, aligned, w, label="Aligned Model",
                color="seagreen", alpha=0.85, edgecolor="white")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(int(bar.get_height())), ha="center", fontsize=10, weight="bold")
    for i in range(len(cats)):
        diff = aligned[i] - base[i]
        color = "green" if diff >= 0 else "red"
        ax.text(x[i], max(base[i], aligned[i]) + 6,
                f"{'+' if diff >= 0 else ''}{diff}",
                ha="center", fontsize=9, color=color, weight="bold")

    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Alignment Tax: Capability vs Safety Trade-off",
                 fontsize=14, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=11)
    ax.legend(fontsize=12); ax.set_ylim(0, 110)
    ax.grid(True, axis="y", alpha=0.3)
    ax.text(0.98, 0.02, "Alignment Tax = capability drop\nfor safety gain",
            transform=ax.transAxes, fontsize=10, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    plt.tight_layout()
    _save(fig, "alignment_tax.png", save)


# ── helpers ──────────────────────────────────────────────────
def _gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _save(fig, name, save):
    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
        print(f"  saved → {FIG_DIR / name}")
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="L18 Alignment visualizations")
    p.add_argument("--fig", choices=["rm", "ppo", "dist", "tax", "all"],
                   default="all", help="which figure to generate (default: all)")
    args = p.parse_args()

    dispatch = dict(rm=plot_rm_loss, ppo=plot_ppo_clipping,
                    dist=plot_distribution_shift, tax=plot_alignment_tax)
    if args.fig == "all":
        print("L18 Alignment — generating all figures …")
        for name, fn in dispatch.items():
            print(f"[{name}]")
            fn()
        print("Done ✓")
    else:
        dispatch[args.fig]()


if __name__ == "__main__":
    main()
