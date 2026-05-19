"""L18 · 对齐 — 补充对齐图表（精确填补 outline 引用的 4 个文件名）

outline 引用但当前 demo/figures/ 里只有不同命名的旧版本，需要新版本：
  - rm_accuracy_curve.png       (Reward Model 训练 accuracy 而非 loss)
  - ppo_clipping_diagram.png    (PPO clipping objective 的几何意义示意图)
  - alignment_tax_detailed.png  (对齐税在 7 个 benchmark 上的细节柱状图)
  - alignment_tax_radar.png     (对齐税雷达图：5 维能力 vs 5 维行为指标)

依赖：numpy, matplotlib
运行：python generate_alignment_figures.py
"""
from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "design" / "meta"))
import cjk_font  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(1337)


# ─────────────────────────────────────────────────────────────────────
# 1. rm_accuracy_curve.png — Reward Model 训练 accuracy
# ─────────────────────────────────────────────────────────────────────
def rm_accuracy_curve():
    epochs = np.arange(0, 31)
    # 模拟三种数据规模下的 RM accuracy 曲线
    sizes = [10_000, 50_000, 200_000]
    colors = ["#C0392B", "#E67E22", "#27AE60"]

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for sz, col in zip(sizes, colors):
        # 大数据集收敛更快、最终更高
        target = 0.62 + 0.08 * np.log10(sz / 10_000)
        acc = target * (1 - np.exp(-0.18 * epochs))
        acc += 0.005 * np.random.randn(len(epochs))
        acc[0] = 0.5  # 初始随机
        ax.plot(epochs, acc, color=col, lw=2.5, marker="o", markersize=5,
                label=f"{sz:,} 偏好对")

    ax.axhline(0.5, ls="--", color="#888", lw=0.8)
    ax.text(28, 0.52, "随机基线 50%", fontsize=9, color="#888", ha="right")
    ax.axhline(0.75, ls=":", color="#444", lw=0.8)
    ax.text(28, 0.76, "InstructGPT 论文报告 ~75%", fontsize=9, color="#444",
            ha="right")
    ax.set_xlabel("训练 epoch", fontsize=11)
    ax.set_ylabel("Reward Model 验证集 accuracy", fontsize=11)
    ax.set_title("RM 训练 accuracy 曲线 — 偏好对数量是关键瓶颈",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0.45, 0.85)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)
    out = OUT / "rm_accuracy_curve.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────
# 2. ppo_clipping_diagram.png — PPO Clipping objective 几何意义
# ─────────────────────────────────────────────────────────────────────
def ppo_clipping_diagram():
    # x 轴：策略概率比 r = π_new / π_old
    r = np.linspace(0, 2, 500)
    epsilon = 0.2
    # 两种 advantage 情况
    A_pos = 1.0   # 正 advantage
    A_neg = -1.0  # 负 advantage

    # 未裁剪 objective: r * A
    raw_pos = r * A_pos
    raw_neg = r * A_neg
    # 裁剪 objective: clip(r, 1-ε, 1+ε) * A
    clipped = np.clip(r, 1 - epsilon, 1 + epsilon)
    clip_pos = clipped * A_pos
    clip_neg = clipped * A_neg
    # 最终 PPO objective: min(raw, clipped) when A>0, max when A<0
    L_pos = np.minimum(raw_pos, clip_pos)
    L_neg = np.maximum(raw_neg, clip_neg)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    # 正 A
    ax = axes[0]
    ax.plot(r, raw_pos, "k--", lw=1.0, alpha=0.5, label="未裁剪 r·A")
    ax.plot(r, L_pos, color="#27AE60", lw=2.8, label="PPO objective")
    ax.axvspan(1 - epsilon, 1 + epsilon, alpha=0.10, color="#27AE60",
               label=f"信赖域 [1±ε], ε={epsilon}")
    ax.axvline(1, color="#888", ls=":", lw=0.8)
    ax.set_title(f"A > 0  (好动作 — 想增大 r)\n超过 1+ε 后 objective 被卡住",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("策略概率比  r = π_new(a|s) / π_old(a|s)", fontsize=10)
    ax.set_ylabel("Surrogate Objective", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    # 负 A
    ax = axes[1]
    ax.plot(r, raw_neg, "k--", lw=1.0, alpha=0.5, label="未裁剪 r·A")
    ax.plot(r, L_neg, color="#C0392B", lw=2.8, label="PPO objective")
    ax.axvspan(1 - epsilon, 1 + epsilon, alpha=0.10, color="#C0392B",
               label=f"信赖域 [1±ε], ε={epsilon}")
    ax.axvline(1, color="#888", ls=":", lw=0.8)
    ax.set_title(f"A < 0  (坏动作 — 想减小 r)\n低于 1-ε 后 objective 被卡住",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("策略概率比  r = π_new(a|s) / π_old(a|s)", fontsize=10)
    ax.set_ylabel("Surrogate Objective", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    fig.suptitle("PPO Clipping — 限制单步策略改变幅度，防止过度优化",
                 fontsize=13, fontweight="bold")
    out = OUT / "ppo_clipping_diagram.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────
# 3. alignment_tax_detailed.png — 7 个 benchmark 上的细节柱状图
# ─────────────────────────────────────────────────────────────────────
def alignment_tax_detailed():
    # 数据 (示意)：对齐前 vs 对齐后在 7 个能力 benchmark 上的得分
    benchmarks = ["MMLU", "HellaSwag", "ARC", "GSM8K", "HumanEval",
                   "BBH", "TruthfulQA"]
    base_score   = [69.5, 84.0, 73.0, 50.0, 38.0, 52.0, 40.0]
    aligned_score = [68.5, 82.5, 71.5, 53.0, 37.0, 51.0, 60.0]

    delta = [a - b for a, b in zip(aligned_score, base_score)]
    colors = ["#C0392B" if d < 0 else "#27AE60" for d in delta]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), constrained_layout=True,
                             sharex=True)
    # 上：原始得分对比
    x = np.arange(len(benchmarks))
    w = 0.35
    axes[0].bar(x - w/2, base_score, w, label="对齐前 (Base)",
                 color="#888")
    axes[0].bar(x + w/2, aligned_score, w, label="对齐后 (RLHF)",
                 color="#2980B9")
    axes[0].set_ylabel("Benchmark 得分", fontsize=11)
    axes[0].set_title("对齐前后能力对比 (示意数据)",
                       fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, 95)
    # 下：差值柱
    axes[1].bar(x, delta, color=colors)
    for i, d in enumerate(delta):
        axes[1].text(i, d + (0.5 if d > 0 else -1.5), f"{d:+.1f}",
                      ha="center", fontsize=10,
                      color="#27AE60" if d > 0 else "#C0392B",
                      fontweight="bold")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(benchmarks, fontsize=10, rotation=15)
    axes[1].set_ylabel("Δ = 对齐后 − 对齐前", fontsize=11)
    axes[1].set_title("对齐税 — 大部分能力小幅下降，TruthfulQA / GSM8K 反而上升",
                       fontsize=12, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_ylim(-4, 22)

    out = OUT / "alignment_tax_detailed.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────
# 4. alignment_tax_radar.png — 5×5 雷达图（能力 vs 行为）
# ─────────────────────────────────────────────────────────────────────
def alignment_tax_radar():
    # 5 维能力 / 5 维行为
    capabilities = ["数学", "代码", "推理", "知识", "创造"]
    behaviors    = ["有用", "无害", "诚实", "可控", "拒答合理"]

    # 数据：base / aligned
    cap_base    = [55, 50, 60, 75, 70]
    cap_aligned = [58, 49, 62, 73, 60]
    beh_base    = [50, 35, 45, 30, 30]
    beh_aligned = [88, 92, 85, 90, 88]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5),
                             subplot_kw=dict(polar=True),
                             constrained_layout=True)
    # 能力雷达
    ax = axes[0]
    angles = np.linspace(0, 2 * np.pi, len(capabilities), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    cb = cap_base + [cap_base[0]]
    ca = cap_aligned + [cap_aligned[0]]
    ax.plot(angles, cb, color="#888", lw=2, label="Base")
    ax.fill(angles, cb, color="#888", alpha=0.15)
    ax.plot(angles, ca, color="#2980B9", lw=2, label="Aligned")
    ax.fill(angles, ca, color="#2980B9", alpha=0.20)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(capabilities, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title("能力维度 — 对齐对原始能力影响小",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(fontsize=10, loc="upper right", bbox_to_anchor=(1.25, 1.1))
    # 行为雷达
    ax = axes[1]
    angles = np.linspace(0, 2 * np.pi, len(behaviors), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    bb = beh_base + [beh_base[0]]
    ba = beh_aligned + [beh_aligned[0]]
    ax.plot(angles, bb, color="#888", lw=2, label="Base")
    ax.fill(angles, bb, color="#888", alpha=0.15)
    ax.plot(angles, ba, color="#27AE60", lw=2, label="Aligned")
    ax.fill(angles, ba, color="#27AE60", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(behaviors, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title("行为维度 — 对齐显著提升 HHH 表现",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(fontsize=10, loc="upper right", bbox_to_anchor=(1.25, 1.1))

    fig.suptitle("对齐税雷达图 — 能力小损失 (~3%)，行为大提升 (~50pp)",
                 fontsize=14, fontweight="bold")
    out = OUT / "alignment_tax_radar.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


if __name__ == "__main__":
    print("=== L18 alignment supplementary figures ===")
    rm_accuracy_curve()
    ppo_clipping_diagram()
    alignment_tax_detailed()
    alignment_tax_radar()
    print("Done.")
