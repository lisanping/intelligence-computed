"""
第 35 讲 · Demo: 五个预测记分卡 + 怀疑论者论点可视化

生成两张图：
1. 五个可被验证的具体预测——时间线 + 置信度 + 验证状态卡片
2. 四位怀疑论者攻击面雷达图

Usage:
    python prediction_scorecard.py
    python prediction_scorecard.py --ablate no_radar  # 只生成预测卡片
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SEED = 1337
np.random.seed(SEED)

# ── 目录 ──────────────────────────────────────────────
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── 数据 ──────────────────────────────────────────────
PREDICTIONS = [
    {
        "id": 1,
        "title": "Transformer 不会被取代\n但会被增强",
        "deadline": "2029",
        "confidence": 85,
        "status": "pending",
        "color": "#4C72B0",
    },
    {
        "id": 2,
        "title": "推理能力将超越\n大部分人类专家任务",
        "deadline": "2028",
        "confidence": 75,
        "status": "pending",
        "color": "#55A868",
    },
    {
        "id": 3,
        "title": "ARC-AGI 被破解\n但不靠纯 Scaling",
        "deadline": "2028",
        "confidence": 60,
        "status": "pending",
        "color": "#C44E52",
    },
    {
        "id": 4,
        "title": "多模态模型获得\n物理直觉",
        "deadline": "2030",
        "confidence": 55,
        "status": "pending",
        "color": "#8172B2",
    },
    {
        "id": 5,
        "title": "AI 安全成为\n工程标配",
        "deadline": "2029",
        "confidence": 80,
        "status": "pending",
        "color": "#CCB974",
    },
]

CRITICS = {
    "labels": [
        "符号推理\n(Marcus)",
        "学习效率\n(LeCun)",
        "抽象泛化\n(Chollet)",
        "物理 Grounding\n(具身派)",
        "可靠性\n(综合)",
    ],
    "attack_strength": [0.9, 0.85, 0.95, 0.7, 0.8],
    "current_defense": [0.5, 0.4, 0.3, 0.35, 0.6],
}


def plot_prediction_scorecard():
    """生成五个预测的时间线记分卡。"""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(2026, 2031)
    ax.set_ylim(-0.5, len(PREDICTIONS) - 0.5)
    ax.invert_yaxis()

    # 时间线背景
    for year in range(2026, 2031):
        ax.axvline(year, color="#E0E0E0", linewidth=0.8, zorder=0)

    for i, p in enumerate(PREDICTIONS):
        deadline = int(p["deadline"])
        conf = p["confidence"]

        # 置信度条
        bar_start = 2026
        bar_width = (deadline - 2026) * (conf / 100)
        ax.barh(i, bar_width, left=bar_start, height=0.55,
                color=p["color"], alpha=0.7, zorder=2)

        # 截止线
        ax.plot([deadline, deadline], [i - 0.3, i + 0.3],
                color=p["color"], linewidth=3, zorder=3)

        # 标题
        ax.text(2025.9, i, f"P{p['id']}: {p['title']}",
                va="center", ha="right", fontsize=9, fontweight="bold",
                fontfamily="sans-serif")

        # 置信度标签
        ax.text(deadline + 0.05, i, f"{conf}%",
                va="center", ha="left", fontsize=10, fontweight="bold",
                color=p["color"])

    ax.set_xlabel("验证截止年份", fontsize=12)
    ax.set_xlim(2025.0, 2030.8)
    ax.set_xticks(range(2026, 2031))
    ax.set_yticks([])
    ax.set_title("第 35 讲 · 五个可被验证的具体预测",
                 fontsize=14, fontweight="bold", pad=15)

    # 图例
    legend_elements = [
        mpatches.Patch(color="#999999", alpha=0.5, label="置信度 × 时间跨度"),
        plt.Line2D([0], [0], color="#333333", linewidth=3, label="验证截止线"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "prediction_scorecard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")


def plot_critic_radar():
    """生成四位怀疑论者攻击面的雷达图。"""
    labels = CRITICS["labels"]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    attack = CRITICS["attack_strength"] + CRITICS["attack_strength"][:1]
    defense = CRITICS["current_defense"] + CRITICS["current_defense"][:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color="#666")

    ax.fill(angles, attack, alpha=0.15, color="#C44E52")
    ax.plot(angles, attack, linewidth=2, color="#C44E52", label="批判强度")

    ax.fill(angles, defense, alpha=0.15, color="#4C72B0")
    ax.plot(angles, defense, linewidth=2, color="#4C72B0", label="当前应对水平")

    ax.legend(loc="lower right", bbox_to_anchor=(1.15, -0.05), fontsize=10)
    ax.set_title("怀疑论者攻击面 vs 当前 AI 应对能力",
                 fontsize=13, fontweight="bold", pad=20)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "critic_radar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")


def main():
    parser = argparse.ArgumentParser(description="L35 预测记分卡可视化")
    parser.add_argument("--ablate", type=str, default=None,
                        choices=["no_radar"],
                        help="消融实验：no_radar = 只生成预测卡片")
    args = parser.parse_args()

    # 中文字体支持
    for font in ["Microsoft YaHei", "SimHei", "PingFang SC",
                  "Noto Sans CJK SC", "WenQuanYi Micro Hei"]:
        try:
            matplotlib.font_manager.findfont(font, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [font] + plt.rcParams["font.sans-serif"]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    print("=" * 60)
    print("第 35 讲 · 五个预测记分卡 + 怀疑论者雷达图")
    print("=" * 60)

    plot_prediction_scorecard()

    if args.ablate != "no_radar":
        plot_critic_radar()

    print("\nDone. Figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
