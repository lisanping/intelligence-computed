"""
第 12 讲 · V4 六步注意力分解图
============================

输出一张 `attention_six_steps.png`，对应讲稿 [08:00 – 14:00] 的六步：
    1. 输入 embedding X
    2. 投影出 Q/K/V
    3. Q @ K^T 相似度矩阵
    4. 除以 sqrt(d_k)
    5. 因果掩码 + softmax
    6. 权重 @ V → 新表示

不调模型、不依赖训练结果——全用手造的小例子（T=4, d=6, d_head=3）。
课件需要时，可将各子图独立导出作为分步动画帧。
"""
from __future__ import annotations

import math
from pathlib import Path

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[4] / "design" / "meta"))
import cjk_font  # noqa: F401  - applies CJK font config
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

np.random.seed(1337)

TOKENS = ["The", "cat", "sat", "on"]
T = len(TOKENS)
D = 6
DH = 3

X = np.random.randn(T, D) * 0.5
Wq = np.random.randn(D, DH) * 0.4
Wk = np.random.randn(D, DH) * 0.4
Wv = np.random.randn(D, DH) * 0.4

Q = X @ Wq
K = X @ Wk
V = X @ Wv

raw_scores = Q @ K.T
scaled = raw_scores / math.sqrt(DH)
mask = np.triu(np.ones((T, T), dtype=bool), k=1)
masked = np.where(mask, -np.inf, scaled)


def _softmax(a: np.ndarray) -> np.ndarray:
    a = a - np.nanmax(np.where(np.isinf(a), -1e9, a), axis=-1, keepdims=True)
    e = np.exp(a)
    e[np.isinf(a)] = 0.0
    return e / e.sum(axis=-1, keepdims=True)


attn = _softmax(masked)
out = attn @ V


def heatmap(ax, mat, title, *, labels_x=None, labels_y=None, cmap="RdBu_r",
            fmt="{:+.2f}", annotate=True, vlim=None):
    if vlim is None:
        v = np.nanmax(np.abs(np.where(np.isinf(mat), 0, mat)))
        vlim = (-v, v)
    disp = np.where(np.isinf(mat), np.nan, mat)
    im = ax.imshow(disp, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
    ax.set_title(title, fontsize=10)
    if labels_x is not None:
        ax.set_xticks(range(len(labels_x)))
        ax.set_xticklabels(labels_x, fontsize=8)
    else:
        ax.set_xticks([])
    if labels_y is not None:
        ax.set_yticks(range(len(labels_y)))
        ax.set_yticklabels(labels_y, fontsize=8)
    else:
        ax.set_yticks([])
    if annotate:
        for (i, j), v_ij in np.ndenumerate(mat):
            if np.isinf(v_ij):
                ax.text(j, i, "-inf", ha="center", va="center", fontsize=7, color="white")
            else:
                ax.text(j, i, fmt.format(v_ij), ha="center", va="center", fontsize=7)
    return im


fig, axes = plt.subplots(2, 3, figsize=(13, 7))

heatmap(axes[0, 0], X, "1. Input X  (T x D)", labels_y=TOKENS)
heatmap(axes[0, 1], np.hstack([Q, K, V]),
        "2. Project -> Q | K | V  (T x 3 Dh)",
        labels_y=TOKENS,
        labels_x=[f"Q{i}" for i in range(DH)] + [f"K{i}" for i in range(DH)] + [f"V{i}" for i in range(DH)])
heatmap(axes[0, 2], raw_scores,
        "3. Scores = Q @ K^T",
        labels_x=TOKENS, labels_y=TOKENS)

heatmap(axes[1, 0], scaled,
        f"4. Scaled / sqrt(d_k={DH})",
        labels_x=TOKENS, labels_y=TOKENS)
heatmap(axes[1, 1], attn,
        "5. softmax(mask + scaled)   rows sum to 1",
        labels_x=TOKENS, labels_y=TOKENS, cmap="Blues", vlim=(0, 1), fmt="{:.2f}")
heatmap(axes[1, 2], out,
        "6. Output = Attn @ V",
        labels_y=TOKENS)

plt.suptitle("Self-Attention in 6 steps  (T=4, D=6, d_head=3, causal mask on)", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = OUT / "attention_six_steps.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"saved {out_path}")