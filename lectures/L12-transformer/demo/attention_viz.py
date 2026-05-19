"""
第 12 讲 · 可视化辅助脚本
=======================

生成课件需要的几张静态图：
    1. causal_mask.png       —— 因果掩码三角矩阵（V6）
    2. positional_encoding.png —— 正弦位置编码热力图（V7）
    3. attention_heatmap.png —— 一个训练好的 TinyGPT 某层某头的注意力热力图（V5 简化版）

用法：
    python attention_viz.py                       # 生成前两张
    python attention_viz.py --with-model PATH     # 加上第三张（需要先跑 nanogpt_from_scratch.py 保存 ckpt）
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[4] / "design" / "meta"))
import cjk_font  # noqa: F401  - applies CJK font config
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)


def plot_causal_mask(T: int = 16) -> None:
    mask = np.triu(np.ones((T, T)), k=1)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mask, cmap="Reds", vmin=0, vmax=1)
    ax.set_title(f"Causal Mask (T={T})\nred = -inf  (future positions, blocked)")
    ax.set_xlabel("Key position (attended to)")
    ax.set_ylabel("Query position (current)")
    plt.tight_layout()
    out = OUT / "causal_mask.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved {out}")


def plot_positional_encoding(T: int = 50, D: int = 128) -> None:
    pos = np.arange(T)[:, None]
    i = np.arange(D)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / D)
    pe = np.zeros((T, D))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pe, aspect="auto", cmap="RdBu")
    ax.set_title(f"Sinusoidal Positional Encoding (T={T}, D={D})")
    ax.set_xlabel("embedding dim d")
    ax.set_ylabel("position t")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    out = OUT / "positional_encoding.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved {out}")


def plot_attention_heatmap(ckpt_path: str) -> None:
    import torch

    import sys
    sys.path.insert(0, str(HERE))
    from nanogpt_from_scratch import TinyGPT, BLOCK_SIZE  # noqa: E402

    state = torch.load(ckpt_path, map_location="cpu")
    model = TinyGPT(vocab_size=state["vocab_size"])
    model.load_state_dict(state["model"])
    model.eval()

    # 钩住第 0 层 attention 的 softmax 输出
    captured: dict = {}
    original_forward = model.blocks[0].attn.forward

    def hooked(x):
        B, T, D = x.shape
        H, Dh = model.blocks[0].attn.n_head, model.blocks[0].attn.d_head
        q, k, v = model.blocks[0].attn.qkv_proj(x).split(D, dim=-1)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(Dh)
        scores = scores.masked_fill(model.blocks[0].attn.causal_mask[:T, :T], float("-inf"))
        captured["attn"] = torch.softmax(scores, dim=-1).detach().cpu().numpy()
        return original_forward(x)

    model.blocks[0].attn.forward = hooked
    text = "ROMEO: But soft, what light through yonder window breaks?"
    s2i = state["s2i"]
    idx = torch.tensor([[s2i.get(c, 0) for c in text[:BLOCK_SIZE]]])
    with torch.no_grad():
        model(idx)

    attn = captured["attn"][0]  # (H, T, T)
    H = attn.shape[0]
    fig, axes = plt.subplots(1, H, figsize=(3 * H, 3), sharey=True)
    for h, ax in enumerate(axes):
        ax.imshow(attn[h], cmap="viridis", aspect="auto")
        ax.set_title(f"Head {h}")
    fig.suptitle("Layer 0 attention patterns (prompt: ROMEO...)")
    plt.tight_layout()
    out = OUT / "attention_heatmap.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-model", type=str, default=None,
                        help="TinyGPT checkpoint path（可选，用于生成真实注意力热力图）")
    args = parser.parse_args()

    plot_causal_mask()
    plot_positional_encoding()
    if args.with_model:
        plot_attention_heatmap(args.with_model)


if __name__ == "__main__":
    main()