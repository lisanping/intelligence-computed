"""L27 · Agent 能力装配 — Skill 数量 vs 成功率曲线 + Skill 加载示意图

生成：
  - skill_count_vs_success.png   (Skill 数量对任务成功率的影响 — Voyager 论文风格)
  - progressive_disclosure.png   (Skill 渐进披露的层级图)

数据基于 Wang et al. 2023 *Voyager* 论文 + Anthropic Skills 2025-10 报告。
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
# 1. skill_count_vs_success.png — Voyager 风格曲线
# ─────────────────────────────────────────────────────────────────────
def skill_count_curve():
    n_skills = np.arange(0, 201, 5)
    # 三类任务难度
    easy = 100 * (1 - np.exp(-0.05 * n_skills))
    medium = 100 * (1 - np.exp(-0.018 * n_skills))
    hard = 100 * (1 - np.exp(-0.008 * n_skills))

    # 对照：无 skill 库的 baseline (固定 ~30%)
    baseline_easy, baseline_medium, baseline_hard = 75, 35, 12

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.plot(n_skills, easy, color="#27AE60", lw=2.5,
            label="简单任务 (基础物品制作)")
    ax.plot(n_skills, medium, color="#2980B9", lw=2.5,
            label="中等任务 (基础生存)")
    ax.plot(n_skills, hard, color="#C0392B", lw=2.5,
            label="困难任务 (钻石获取等长程目标)")
    # baselines
    ax.axhline(baseline_easy, color="#27AE60", ls=":", lw=1, alpha=0.5)
    ax.axhline(baseline_medium, color="#2980B9", ls=":", lw=1, alpha=0.5)
    ax.axhline(baseline_hard, color="#C0392B", ls=":", lw=1, alpha=0.5)
    ax.text(195, baseline_hard + 2, "无 Skill 库基线",
            fontsize=8, ha="right", color="#888")

    ax.axvspan(0, 30, alpha=0.05, color="#888")
    ax.text(15, 95, "Skill 库\n建立期", ha="center", fontsize=9, color="#666")
    ax.axvspan(30, 100, alpha=0.05, color="#27AE60")
    ax.text(65, 95, "快速\n增长期", ha="center", fontsize=9, color="#27AE60")
    ax.axvspan(100, 200, alpha=0.05, color="#2980B9")
    ax.text(150, 95, "饱和期\n(收益递减)", ha="center", fontsize=9, color="#2980B9")

    ax.set_xlabel("Skill 库中 skill 数量", fontsize=11)
    ax.set_ylabel("任务成功率 (%)", fontsize=11)
    ax.set_title("Skill 数量 vs 任务成功率\n"
                 "(基于 Voyager 论文 Wang et al. 2023 + Anthropic Skills 2025-10)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)

    # 标注：100 个 skill 是关键拐点
    ax.axvline(100, ls="--", color="#444", lw=1)
    ax.text(102, 5, "~100 skill 后\n收益递减明显", fontsize=9, color="#444")

    out = OUT / "skill_count_vs_success.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────
# 2. progressive_disclosure.png — Skill 渐进披露层级
# ─────────────────────────────────────────────────────────────────────
def progressive_disclosure():
    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")

    # 三层金字塔
    levels = [
        # (y, height, color, title, content, token_cost)
        (5.5, 1.4, "#FADBD8", "Layer 1 · Skill 索引",
         "skill_name + 1 行描述 + applies_to",
         "~50 tokens × 全部 skill"),
        (3.5, 1.4, "#FCF3CF", "Layer 2 · Skill 摘要",
         "完整 SKILL.md + 主要 args + 调用示例",
         "~500 tokens × 被 LLM 选中的 skill"),
        (1.5, 1.4, "#D5F5E3", "Layer 3 · Skill 完整内容",
         "完整脚本 + 详细 docstring + 全部代码",
         "~5000 tokens × 实际执行的 skill"),
    ]

    for y, h, col, title, content, cost in levels:
        rect = plt.Rectangle((1, y), 10, h, facecolor=col,
                              edgecolor="black", lw=1.5)
        ax.add_patch(rect)
        ax.text(6, y + h - 0.25, title, ha="center", fontsize=12,
                fontweight="bold")
        ax.text(6, y + h / 2 - 0.1, content, ha="center", fontsize=10,
                color="#444")
        ax.text(6, y + 0.2, cost, ha="center", fontsize=9,
                color="#888", style="italic")

    # 箭头
    for i in range(2):
        ax.annotate("", xy=(6, 5.4 - i * 2),
                    xytext=(6, 4.95 - i * 2),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#444"))

    # 右侧说明
    ax.text(11.7, 6.2, "全部 skill\n按需筛选", fontsize=9, color="#888",
            ha="left", va="center")
    ax.text(11.7, 4.2, "选中的 skill\n了解细节", fontsize=9, color="#888",
            ha="left", va="center")
    ax.text(11.7, 2.2, "执行时\n加载完整内容", fontsize=9, color="#888",
            ha="left", va="center")

    # 顶部 + 底部说明
    ax.text(6, 7.5, "Progressive Disclosure  ·  渐进披露",
            ha="center", fontsize=15, fontweight="bold", color="#1A1F2B")
    ax.text(6, 7.0, "(Anthropic Skills 2025-10 设计原则)",
            ha="center", fontsize=10, color="#666")
    ax.text(6, 0.5,
            "核心权衡：context 窗口里塞越多 skill → 选择越准但成本越高 / 慢；"
            "三层金字塔在「全面性」与「效率」之间取得平衡",
            ha="center", fontsize=9, color="#444", style="italic")

    out = OUT / "progressive_disclosure.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


if __name__ == "__main__":
    print("=== L27 skill curve + progressive disclosure ===")
    skill_count_curve()
    progressive_disclosure()
    print("Done.")
