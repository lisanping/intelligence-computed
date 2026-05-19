"""Generate the missing `tradeoff.png` referenced by L22 slides_outline.md.

Visualizes the Long-Context vs RAG trade-off across four axes:
  cost / latency / freshness / consistency

Run:  python generate_tradeoff.py
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

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)


def main() -> None:
    axes_labels = ["成本\n(per query)", "延迟\n(TTFT)",
                   "知识新鲜度\n(可热更)", "答案一致性\n(同输入同输出)",
                   "可解释性\n(citation)", "上下文连贯性"]
    # 0..1 scores (higher = better)
    rag_scores = [0.85, 0.75, 0.95, 0.70, 0.90, 0.55]
    long_ctx_scores = [0.30, 0.35, 0.45, 0.55, 0.40, 0.90]

    angles = np.linspace(0, 2 * np.pi, len(axes_labels),
                         endpoint=False).tolist()
    angles += angles[:1]
    rag_scores += rag_scores[:1]
    long_ctx_scores += long_ctx_scores[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.5),
                           subplot_kw={"projection": "polar"})
    ax.plot(angles, rag_scores, "o-", color="#4C72B0", lw=2,
            label="RAG（检索 + 生成）")
    ax.fill(angles, rag_scores, color="#4C72B0", alpha=0.18)
    ax.plot(angles, long_ctx_scores, "o-", color="#D62728", lw=2,
            label="Long-Context（全装进 prompt）")
    ax.fill(angles, long_ctx_scores, color="#D62728", alpha=0.18)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["差", "一般", "好", "极好"], fontsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10), fontsize=10)
    ax.set_title("RAG vs 长上下文：六维取舍\n（数值为生产经验估计，非 benchmark）",
                 fontsize=12, pad=22)
    fig.tight_layout()
    out = os.path.join(OUT, "tradeoff.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
