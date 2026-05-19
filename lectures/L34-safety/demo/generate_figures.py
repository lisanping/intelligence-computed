"""Generate L34 missing diagnostic figures.

Outputs (to demo/figures/):
  - red_team_flow.png         5-step red-team workflow diagram
  - sae_visualization.png     Sparse Autoencoder activation overlay
  - feature_circuits.png      Feature → circuit causal chain
  - golden_gate_example.png   Anthropic's "Golden Gate Claude" — feature steering

Pure matplotlib — no model required.

Run:  python generate_figures.py
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'PingFang SC',
    'Microsoft JhengHei', 'Segoe UI Emoji', 'DejaVu Sans',
]
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
def red_team_flow() -> None:
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4)
    ax.set_axis_off()

    steps = [
        ("1. Threat\nModel", "定义攻击面\n(jailbreak / inj /\nleak / harm)"),
        ("2. Attack\nLibrary", "组装 50-500\n个分类化的\nadversarial prompts"),
        ("3. Run\nEvaluation", "对目标模型批量\n执行；记录响应\n与 metadata"),
        ("4. Auto\nJudge", "Llamaguard / GPT\n打分；refusal 检测\n+ harm 检测"),
        ("5. Triage &\nReport", "人工复核 top-1%\n失败案例；提交\nbug bounty"),
    ]
    palette = ["#D62728", "#FF7043", "#FFB300", "#4CAF50", "#1976D2"]
    for i, ((title, body), color) in enumerate(zip(steps, palette)):
        x = 0.4 + i * 2.45
        b = FancyBboxPatch((x, 1.2), 2.1, 1.8, boxstyle="round,pad=0.05",
                           facecolor=color, edgecolor="black", lw=1.0,
                           alpha=0.9)
        ax.add_patch(b)
        ax.text(x + 1.05, 2.5, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
        ax.text(x + 1.05, 1.7, body, ha="center", va="center",
                fontsize=9, color="white")
        if i < 4:
            ax.add_patch(FancyArrowPatch((x + 2.1, 2.1),
                                         (x + 2.45, 2.1),
                                         arrowstyle="-|>",
                                         color="black",
                                         mutation_scale=15, lw=1.4))

    ax.text(6.5, 3.65,
            "AI Red Team — 五步标准化流程  ·  与 OWASP LLM Top-10 对齐",
            ha="center", fontsize=13, fontweight="bold")
    ax.text(6.5, 0.7,
            "→ 闭环：每次发现 → mitigation 加入训练 / guardrail → 重新红队",
            ha="center", fontsize=10, color="#444444", style="italic")
    out = os.path.join(OUT, "red_team_flow.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ──────────────────────────────────────────────────────────────────────
def sae_visualization() -> None:
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Left: dense activations (polysemantic — many features fire)
    dense = rng.normal(0.3, 0.25, 100)
    dense = np.maximum(dense, 0)
    axes[0].bar(range(100), dense, color="#888888", width=0.9)
    axes[0].set_title("BEFORE：原始 hidden state\n几乎所有维度都激活（多义）",
                      fontsize=11)
    axes[0].set_xlabel("hidden dim (100)")
    axes[0].set_ylabel("activation")
    axes[0].set_ylim(0, 1.2)

    # Middle: SAE block diagram
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_axis_off()
    ax.text(5, 5.5, "Sparse Autoencoder (SAE)\n(Anthropic 2024)",
            ha="center", fontsize=12, fontweight="bold")
    # encoder
    enc = FancyBboxPatch((0.5, 2.5), 2, 1.2, boxstyle="round,pad=0.05",
                         facecolor="#E8F0FF", edgecolor="#4C72B0", lw=1.4)
    ax.add_patch(enc)
    ax.text(1.5, 3.1, "Encoder W_e\n(d → 8d)", ha="center",
            fontsize=10, fontweight="bold")
    # ReLU + L1
    relu = FancyBboxPatch((3.5, 2.5), 2, 1.2, boxstyle="round,pad=0.05",
                          facecolor="#FFFDE7", edgecolor="#F9A825", lw=1.4)
    ax.add_patch(relu)
    ax.text(4.5, 3.1, "ReLU + L1\n(稀疏约束)", ha="center", fontsize=10,
            fontweight="bold")
    # decoder
    dec = FancyBboxPatch((6.5, 2.5), 2, 1.2, boxstyle="round,pad=0.05",
                         facecolor="#E8F5E9", edgecolor="#2CA02C", lw=1.4)
    ax.add_patch(dec)
    ax.text(7.5, 3.1, "Decoder W_d\n(8d → d)", ha="center", fontsize=10,
            fontweight="bold")
    for x1, x2 in [(2.5, 3.5), (5.5, 6.5)]:
        ax.add_patch(FancyArrowPatch((x1, 3.1), (x2, 3.1),
                                     arrowstyle="-|>", color="black",
                                     mutation_scale=15, lw=1.2))
    ax.text(5, 1.5, "L = ‖x − Wd·ReLU(We·x)‖² + λ‖h‖₁",
            ha="center", fontsize=11, family="monospace")
    ax.text(5, 0.7, "λ 控制稀疏性  ·  典型 8-32× 过完备字典",
            ha="center", fontsize=9, color="#555555", style="italic")

    # Right: sparse features (one-hot-ish)
    sparse = np.zeros(100)
    active_idx = [12, 31, 67]
    sparse[active_idx] = [0.9, 0.6, 0.45]
    colors = ["#888888"] * 100
    labels = {12: "feature 12\n'Golden Gate'",
              31: "feature 31\n'Bridge structure'",
              67: "feature 67\n'San Francisco'"}
    for idx in active_idx:
        colors[idx] = "#D62728"
    axes[2].bar(range(100), sparse, color=colors, width=0.9)
    for idx, label in labels.items():
        axes[2].annotate(label, xy=(idx, sparse[idx]),
                         xytext=(idx, sparse[idx] + 0.18),
                         ha="center", fontsize=9, color="#D62728",
                         arrowprops=dict(arrowstyle="-", color="#D62728"))
    axes[2].set_title("AFTER：SAE 编码后\n只有少数 monosemantic 特征激活",
                      fontsize=11)
    axes[2].set_xlabel("SAE feature index (8d total)")
    axes[2].set_ylabel("activation")
    axes[2].set_ylim(0, 1.2)

    fig.suptitle(
        "Sparse Autoencoder：把多义 dense 表示拆成可解释的稀疏特征",
        fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUT, "sae_visualization.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ──────────────────────────────────────────────────────────────────────
def feature_circuits() -> None:
    """Feature A → Feature B → Feature C causal chain."""
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.set_axis_off()

    nodes = [
        (1.5, 5.0, "L4 'France'\n(geography)", "#4C72B0"),
        (1.5, 3.0, "L4 'Capital'\n(role)", "#4C72B0"),
        (5.5, 4.0, "L8 'France-Capital'\n(combined)", "#9C27B0"),
        (9.5, 4.0, "L11 'Paris'\n(answer token)", "#D62728"),
        (1.5, 1.0, "L4 'Italy'\n(distractor)", "#888888"),
        (5.5, 1.5, "L8 'Italy-Capital'\n(suppressed)", "#888888"),
    ]
    for x, y, lab, c in nodes:
        ax.add_patch(Circle((x, y), 0.6, facecolor=c, edgecolor="black",
                            lw=1, alpha=0.85))
        ax.text(x, y, lab, ha="center", va="center", fontsize=8.5,
                color="white", fontweight="bold")

    edges = [
        (1.5, 5.0, 5.5, 4.0, "+", "#2CA02C"),
        (1.5, 3.0, 5.5, 4.0, "+", "#2CA02C"),
        (5.5, 4.0, 9.5, 4.0, "+", "#2CA02C"),
        (1.5, 1.0, 5.5, 1.5, "+", "#888888"),
        (5.5, 1.5, 5.5, 4.0, "−", "#D62728"),
    ]
    for x1, y1, x2, y2, sign, color in edges:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>",
                                     color=color, lw=1.6,
                                     mutation_scale=15,
                                     connectionstyle="arc3,rad=-0.1"))
        # mid label
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.15, my + 0.15, sign, fontsize=14,
                color=color, fontweight="bold")

    ax.text(6.5, 5.8,
            "Feature Circuit  ·  从特征到回路：'What is the capital of France' 的内部计算",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(6.5, 0.3,
            "→ 因果可验证：消融 (ablate) L4-France 节点 → L11-Paris 不再激活；\n"
            "   消融 L8-Italy-Capital 节点 → 模型可能错答 Rome（因为抑制被解除）",
            ha="center", fontsize=10, color="#444444", style="italic")
    out = os.path.join(OUT, "feature_circuits.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ──────────────────────────────────────────────────────────────────────
def golden_gate_example() -> None:
    """Mock the Anthropic 2024-05 'Golden Gate Claude' demo."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))

    examples_left = [
        ("Q: What's a fun thing to do in Paris?",
         "A: Visit the Louvre, stroll along the Seine,\n"
         "and enjoy a café in Le Marais.")
    ]
    examples_right = [
        ("Q: What's a fun thing to do in Paris?",
         "A: I'd recommend driving across the Golden Gate Bridge!\n"
         "Its iconic red towers are a must-see, and the view\n"
         "from the bridge is breathtaking.")
    ]

    def render(ax, title, examples, accent):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_axis_off()
        ax.text(5, 9.5, title, ha="center", fontsize=13,
                fontweight="bold", color=accent)
        y = 8.0
        for q, a in examples:
            ax.text(0.5, y, "USER  " + q, fontsize=11, color="black",
                    fontweight="bold")
            y -= 0.6
            box = FancyBboxPatch((0.5, y - 2.5), 9, 2.3,
                                 boxstyle="round,pad=0.1",
                                 facecolor="#F5F5F5",
                                 edgecolor=accent, lw=1.5)
            ax.add_patch(box)
            ax.text(0.7, y - 0.3, "CLAUDE  " + a, fontsize=10.5,
                    color="#333333", verticalalignment="top")
            y -= 3.3

    render(axes[0], "Normal Claude", examples_left, "#4C72B0")
    render(axes[1],
           "Steered Claude  (feature #34M/31164842 'Golden Gate' = +20σ)",
           examples_right, "#D62728")

    fig.suptitle(
        "Golden Gate Claude (Anthropic 2024-05)  ·  通过 SAE 特征强制激活操控模型行为",
        fontsize=13)
    fig.text(0.5, 0.02,
             "→ 证明：mech interp 不只是观察 — 还能干预；为对齐提供新工具，"
             "也为安全提出新挑战（恶意操控）",
             ha="center", fontsize=10, style="italic", color="#444444")
    fig.tight_layout()
    out = os.path.join(OUT, "golden_gate_example.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    red_team_flow()
    sae_visualization()
    feature_circuits()
    golden_gate_example()


if __name__ == "__main__":
    main()
