"""L13 · BERT vs GPT — Attention mask 可视化 + 补充图表

生成：
  - attention_mask_comparison.png  (Causal vs Bidirectional vs Prefix-LM 三种 mask)
  - mlm_vs_clm_objective.png       (训练目标：完形填空 vs 下一 token 预测)
  - bert_vs_gpt_finetune.png       (微调范式对比柱状图)

依赖：numpy, matplotlib
运行：python visualize_attention_masks.py
"""
from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "design" / "meta"))
import cjk_font  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(1337)


# ─────────────────────────────────────────────────────────────────────
# 1. attention_mask_comparison.png — 三种 mask 的可见性矩阵
# ─────────────────────────────────────────────────────────────────────
def attention_masks():
    seq_len = 8
    tokens = ["[CLS]", "the", "cat", "sat", "on", "the", "mat", "."]
    # Causal (GPT)
    causal = np.tril(np.ones((seq_len, seq_len)))
    # Bidirectional (BERT)
    bidir = np.ones((seq_len, seq_len))
    # Prefix-LM (UniLM / T5 encoder-decoder hybrid)
    # 前 4 个 token 双向；后 4 个 causal
    prefix_len = 4
    prefix = np.zeros((seq_len, seq_len))
    prefix[:prefix_len, :prefix_len] = 1
    prefix[prefix_len:, :] = np.tril(np.ones((seq_len - prefix_len, seq_len)),
                                       k=prefix_len)
    prefix[prefix_len:, :prefix_len] = 1

    cmap = LinearSegmentedColormap.from_list(
        "att", ["#F4F1E8", "#2D5F8A"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    masks = [
        ("Causal (GPT / 解码器)", causal, "下三角 — 只能看过去"),
        ("Bidirectional (BERT / 编码器)", bidir, "全连接 — 双向上下文"),
        ("Prefix-LM (UniLM / 混合)", prefix, "前缀双向 + 后缀 causal"),
    ]
    for ax, (name, m, sub) in zip(axes, masks):
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(tokens, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(tokens, fontsize=9)
        ax.set_title(f"{name}\n{sub}", fontsize=11, fontweight="bold")
        # Grid
        for i in range(seq_len + 1):
            ax.axhline(i - 0.5, color="white", lw=0.5)
            ax.axvline(i - 0.5, color="white", lw=0.5)
        ax.set_xlabel("可被关注的 key 位置", fontsize=9)
        ax.set_ylabel("当前 query 位置", fontsize=9)
    fig.suptitle("三种 Attention Mask — 这是 BERT vs GPT 的最关键架构差异",
                 fontsize=13, fontweight="bold")
    out = OUT / "attention_mask_comparison.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────
# 2. mlm_vs_clm_objective.png — 训练目标对比
# ─────────────────────────────────────────────────────────────────────
def mlm_vs_clm():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

    # MLM (BERT)
    ax = axes[0]
    sentence = ["The", "cat", "[MASK]", "on", "the", "[MASK]"]
    for i, w in enumerate(sentence):
        col = "#C0392B" if w == "[MASK]" else "#1A1F2B"
        ax.text(i, 0.5, w, fontsize=15, ha="center", va="center",
                color=col, fontweight="bold" if w == "[MASK]" else "normal")
    # 双向箭头
    ax.annotate("", xy=(2, 0.3), xytext=(0, 0.3),
                arrowprops=dict(arrowstyle="<->", color="#2980B9", lw=1.5))
    ax.annotate("", xy=(2, 0.3), xytext=(5, 0.3),
                arrowprops=dict(arrowstyle="<->", color="#2980B9", lw=1.5))
    ax.text(2, 0.15, "双向上下文 → 预测被 mask 的词",
            ha="center", fontsize=10, color="#2980B9")
    ax.text(2, 0.85, "MLM (Masked Language Modeling)\nBERT 训练目标",
            ha="center", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.5, 5.5); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("BERT 看到全部 → 填空", fontsize=11)

    # CLM (GPT)
    ax = axes[1]
    sentence = ["The", "cat", "sat", "on", "the", "?"]
    for i, w in enumerate(sentence):
        col = "#27AE60" if w == "?" else "#1A1F2B"
        ax.text(i, 0.5, w, fontsize=15, ha="center", va="center",
                color=col, fontweight="bold" if w == "?" else "normal")
    # 单向箭头
    for j in range(5):
        ax.annotate("", xy=(5, 0.3), xytext=(j, 0.3),
                    arrowprops=dict(arrowstyle="->", color="#27AE60",
                                   lw=1.0, alpha=0.5))
    ax.text(2.5, 0.15, "只看左侧上下文 → 预测下一个词",
            ha="center", fontsize=10, color="#27AE60")
    ax.text(2.5, 0.85, "CLM (Causal Language Modeling)\nGPT 训练目标",
            ha="center", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.5, 5.5); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("GPT 只看过去 → 续写", fontsize=11)

    fig.suptitle("两个训练目标 → 两条产品路线（理解 vs 生成）",
                 fontsize=13, fontweight="bold")
    out = OUT / "mlm_vs_clm_objective.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────
# 3. bert_vs_gpt_finetune.png — 微调范式对比
# ─────────────────────────────────────────────────────────────────────
def finetune_paradigm():
    tasks = ["分类\n(SST-2)", "NER", "问答\n(SQuAD)", "摘要", "翻译", "对话生成"]
    bert_score = [93, 92, 88, 70, 65, 60]
    gpt_score  = [85, 78, 75, 88, 82, 92]

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    bars1 = ax.bar(x - width/2, bert_score, width, label="BERT (encoder)",
                    color="#2980B9")
    bars2 = ax.bar(x + width/2, gpt_score, width, label="GPT (decoder)",
                    color="#27AE60")

    for bars in [bars1, bars2]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 1, f"{int(h)}",
                    ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=10)
    ax.set_ylabel("典型任务分数 (示意值)", fontsize=11)
    ax.set_title("BERT vs GPT 在不同任务上的相对优势 (2018-2020 时代)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 110)

    # 标注
    ax.axvline(2.5, ls="--", color="#888", lw=1.0)
    ax.text(1, 105, "BERT 占优\n(理解类任务)",
            ha="center", fontsize=10, color="#2980B9", fontweight="bold")
    ax.text(4, 105, "GPT 占优\n(生成类任务)",
            ha="center", fontsize=10, color="#27AE60", fontweight="bold")

    out = OUT / "bert_vs_gpt_finetune.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


if __name__ == "__main__":
    print("=== L13 attention mask + finetune comparison ===")
    attention_masks()
    mlm_vs_clm()
    finetune_paradigm()
    print("Done.")
