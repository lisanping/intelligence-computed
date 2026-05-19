"""ControlNet concept diagram + commercial-landing data chart.

Generates two figures:
  - controlnet_diagram.png       Architecture: SD UNet + zero-conv + condition
  - generative_landing_2026.png  Commercial-landing bar chart (ARR / users)

Pure matplotlib — no diffusion model needed.

Run:  python controlnet_and_landing.py
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# 1.  ControlNet architecture diagram
# ──────────────────────────────────────────────────────────────────────
def controlnet_diagram() -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.set_axis_off()

    def box(x, y, w, h, label, color="#E8F0FF", edge="#4C72B0",
            fontsize=10, weight="normal"):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                           facecolor=color, edgecolor=edge, lw=1.4)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, fontweight=weight)

    def arrow(x1, y1, x2, y2, color="#444444", style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle=style, color=color,
                                     mutation_scale=15, lw=1.2))

    # Title row
    ax.text(6, 6.6, "ControlNet (Zhang & Agrawala 2023) — 给 Stable Diffusion 加'控制条件'",
            ha="center", fontsize=13, fontweight="bold")

    # Frozen original UNet (left column)
    box(0.5, 4.5, 2.6, 0.9, "Stable Diffusion\nUNet (FROZEN)",
        color="#FFEBEE", edge="#D62728", weight="bold")
    box(0.5, 3.2, 2.6, 0.9, "Encoder blocks\n(冻结)")
    box(0.5, 1.9, 2.6, 0.9, "Mid block\n(冻结)")
    box(0.5, 0.6, 2.6, 0.9, "Decoder blocks\n(冻结)")

    # Trainable copy (middle column)
    box(4.0, 4.5, 2.6, 0.9, "Trainable Copy\n(初始化=冻结副本)",
        color="#E8F5E9", edge="#2CA02C", weight="bold")
    box(4.0, 3.2, 2.6, 0.9, "Encoder copy")
    box(4.0, 1.9, 2.6, 0.9, "Mid copy")

    # Zero conv (between)
    for y in [4.95, 3.65, 2.35]:
        box(7.0, y - 0.3, 1.0, 0.6, "0-conv",
            color="#FFFDE7", edge="#F9A825", fontsize=8)

    # Conditional input (left)
    box(0.5, 5.8, 3.5, 0.6, "Condition: 边缘图 / 深度图 / 姿态 / Canny / Scribble",
        color="#E1F5FE", edge="#0277BD", fontsize=10, weight="bold")
    arrow(2.2, 5.8, 5.3, 5.4)

    # Original input
    ax.text(1.8, 4.1, "x_t (noisy latent)", ha="center", fontsize=9,
            style="italic")

    # Arrows: condition → trainable copy → zero conv → frozen UNet
    arrow(5.3, 4.5, 5.3, 4.1)
    arrow(5.3, 3.2, 5.3, 2.8)
    arrow(5.3, 1.9, 5.3, 1.5)
    for y in [4.95, 3.65, 2.35]:
        arrow(6.6, y, 7.0, y)
        arrow(8.0, y, 9.0, y)
    # zero conv merge into frozen
    arrow(9.0, 4.95, 3.1, 4.95)
    arrow(9.0, 3.65, 3.1, 3.65)
    arrow(9.0, 2.35, 3.1, 2.35)

    # Output
    box(9.5, 0.6, 2.0, 0.9, "Predicted noise ε_θ\n(去噪输出)",
        color="#F3E5F5", edge="#7B1FA2", fontsize=10, weight="bold")
    arrow(3.1, 1.05, 9.5, 1.05)

    # Caption
    ax.text(6, 0.05,
            "[KEY] 关键创意：Zero conv (初始权重为 0) 保证训练初期不破坏冻结模型；\n"
            "   只用少量数据（~50K 条）即可训练出强可控生成。",
            ha="center", fontsize=10, color="#444444", style="italic")

    out = os.path.join(OUT, "controlnet_diagram.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ──────────────────────────────────────────────────────────────────────
# 2.  Commercial-landing chart
# ──────────────────────────────────────────────────────────────────────
def commercial_chart() -> None:
    # Public reporting + analyst estimates as of 2026-Q1
    rows = [
        ("Midjourney",        200, 2_000_000,  "图像 SaaS"),
        ("Suno",              120, 12_000_000, "音乐生成"),
        ("ElevenLabs",        180,  3_000_000, "语音 TTS"),
        ("Runway ML",         100,  1_500_000, "视频生成"),
        ("Pika Labs",          45,  4_500_000, "视频生成"),
        ("HeyGen",             80,    800_000, "数字人"),
        ("Synthesia",         110,  1_200_000, "数字人"),
        ("Adobe Firefly",     "整合", 30_000_000, "创作工业"),
        ("Canva Magic",       "整合", 25_000_000, "创作工业"),
        ("Notion AI",          50,  4_000_000, "知识工作"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Chart A: ARR
    arr_rows = [(n, a, c) for n, a, _, c in rows
                if isinstance(a, (int, float))]
    arr_rows.sort(key=lambda r: r[1], reverse=True)
    cmap = {"图像 SaaS": "#4C72B0", "音乐生成": "#9C27B0",
            "语音 TTS": "#FF7043", "视频生成": "#D62728",
            "数字人": "#2CA02C", "创作工业": "#666666",
            "知识工作": "#FFD600"}
    names = [r[0] for r in arr_rows]
    vals = [r[1] for r in arr_rows]
    cols = [cmap[r[2]] for r in arr_rows]
    bars = axes[0].barh(names, vals, color=cols, edgecolor="black", lw=0.4)
    for b, v in zip(bars, vals):
        axes[0].text(v + 3, b.get_y() + b.get_height() / 2,
                     f"${v}M", va="center", fontsize=10)
    axes[0].set_xlabel("年化经常性收入 ARR (USD millions)")
    axes[0].set_title("生成式 AI 商业落地  ·  ARR (2026-Q1)")
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.3)

    # Chart B: Active users
    user_rows = [(n, u // 1000, c) for n, _, u, c in rows]
    user_rows.sort(key=lambda r: r[1], reverse=True)
    names2 = [r[0] for r in user_rows]
    vals2 = [r[1] for r in user_rows]
    cols2 = [cmap[r[2]] for r in user_rows]
    bars2 = axes[1].barh(names2, vals2, color=cols2,
                         edgecolor="black", lw=0.4)
    for b, v in zip(bars2, vals2):
        label = f"{v/1000:.1f}M" if v >= 1000 else f"{v}K"
        axes[1].text(v + max(vals2) * 0.01,
                     b.get_y() + b.get_height() / 2,
                     label, va="center", fontsize=10)
    axes[1].set_xlabel("月活用户 MAU (thousands)")
    axes[1].set_title("活跃用户规模  ·  MAU (2026-Q1)")
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.3)

    # legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=k) for k, c in cmap.items()]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "生成式 AI — 创作工业的 2026 年快照（季度刷新）",
        fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUT, "generative_landing_2026.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    controlnet_diagram()
    commercial_chart()


if __name__ == "__main__":
    main()
