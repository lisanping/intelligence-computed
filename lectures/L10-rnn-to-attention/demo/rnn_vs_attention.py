"""
第 10 讲 · 动手环节 · RNN vs Attention 对比实验
==============================================

目标：在字符级序列预测任务上对比 Vanilla RNN、LSTM 和 LSTM+Attention，
     展示 Attention 在长距离依赖上的优势，并可视化注意力权重。

运行：
    python rnn_vs_attention.py                        # 基线（LSTM + Attention）
    python rnn_vs_attention.py --ablate no_attention   # LSTM Seq2Seq，无 Attention
    python rnn_vs_attention.py --ablate no_lstm        # Vanilla RNN（无门控）
    python rnn_vs_attention.py --ablate short_sequences # 短序列（长距离依赖消失）

所有实验使用 seed=1337。
依赖：torch matplotlib numpy
"""
from __future__ import annotations

import argparse
import math
import os

# torch must be imported before matplotlib on Windows to avoid OpenMP/DLL
# conflicts (libiomp5md.dll vs libomp.dll, plus shm.dll load order).
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[4] / "design" / "meta"))
import cjk_font  # noqa: F401  - applies CJK font config
import matplotlib.pyplot as plt
import numpy as np

# ── 固定随机种子 ─────────────────────────────────────────────
SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── 命令行参数 ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="L10 RNN vs Attention 对比实验")
parser.add_argument(
    "--ablate", type=str, default=None,
    choices=["no_attention", "no_lstm", "short_sequences"],
    help="消融实验开关",
)
parser.add_argument("--epochs", type=int, default=40, help="训练 epoch 数")
parser.add_argument("--batch_size", type=int, default=64, help="batch 大小")
parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
parser.add_argument("--hidden_dim", type=int, default=64, help="隐藏层维度")
parser.add_argument("--seq_len", type=int, default=60, help="序列长度")
parser.add_argument("--save_figures", action="store_true", default=True)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# ═══════════════════════════════════════════════════════════
# 1. 合成数据：带长距离依赖的字符序列
# ═══════════════════════════════════════════════════════════
# 任务设计：序列中前几个字符是"标记"（A-D），后面填充随机字符，
# 最后需要"回忆"开头标记的顺序。这迫使模型维持长距离记忆。
VOCAB = list("ABCDabcdefghijklmnop.")
CHAR2IDX = {c: i for i, c in enumerate(VOCAB)}
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}
VOCAB_SIZE = len(VOCAB)
MARKERS = list("ABCD")  # 需要记忆的标记字符
FILLERS = list("abcdefghijklmnop.")  # 填充字符


def generate_data(n_samples: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """生成带长距离依赖的序列对。

    输入序列：[marker1, marker2, filler, filler, ..., filler]
    目标：在序列末尾预测 marker1 和 marker2 的索引。
    编码为序列到序列：输入 = 上述序列，目标 = 相同序列右移一位，
    最后两位替换为 marker1, marker2（回忆任务）。
    """
    inputs = torch.zeros(n_samples, seq_len, dtype=torch.long)
    targets = torch.zeros(n_samples, seq_len, dtype=torch.long)

    for i in range(n_samples):
        # 随机选两个标记
        m1, m2 = np.random.choice(len(MARKERS), size=2, replace=True)
        # 构建序列：[marker1, marker2, fillers..., marker1, marker2]
        seq = [CHAR2IDX[MARKERS[m1]], CHAR2IDX[MARKERS[m2]]]
        for _ in range(seq_len - 4):
            seq.append(CHAR2IDX[np.random.choice(FILLERS)])
        # 最后两位是回忆目标
        seq.append(CHAR2IDX[MARKERS[m1]])
        seq.append(CHAR2IDX[MARKERS[m2]])

        inputs[i] = torch.tensor(seq)
        # 目标：下一个字符预测（右移一位）
        targets[i, :-1] = inputs[i, 1:]
        targets[i, -1] = inputs[i, 0]  # 循环

    return inputs, targets


# ═══════════════════════════════════════════════════════════
# 2. 模型定义
# ═══════════════════════════════════════════════════════════
class BahdanauAttention(nn.Module):
    """加性 Attention（Bahdanau style）。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, decoder_state: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        decoder_state: (batch, hidden_dim)
        encoder_outputs: (batch, src_len, hidden_dim)
        Returns: context (batch, hidden_dim), weights (batch, src_len)
        """
        # (batch, 1, hidden_dim)
        query = self.W1(decoder_state).unsqueeze(1)
        # (batch, src_len, hidden_dim)
        keys = self.W2(encoder_outputs)
        # (batch, src_len, 1) -> (batch, src_len)
        scores = self.v(torch.tanh(query + keys)).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        # (batch, hidden_dim)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class Seq2SeqModel(nn.Module):
    """Encoder-Decoder with optional Attention and optional LSTM/RNN switch."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        use_attention: bool = True,
        use_lstm: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        self.use_lstm = use_lstm

        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        rnn_cls = nn.LSTM if use_lstm else nn.RNN
        self.encoder = rnn_cls(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = rnn_cls(hidden_dim, hidden_dim, batch_first=True)

        if use_attention:
            self.attention = BahdanauAttention(hidden_dim)
            self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)
        else:
            self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # 存储最近一次前向的注意力权重（用于可视化）
        self.last_attention_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len) — 输入序列
        Returns: logits (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        embedded = self.embedding(x)  # (batch, seq_len, hidden_dim)

        # Encoder
        encoder_outputs, encoder_state = self.encoder(embedded)

        # Decoder（teacher forcing：用输入本身作为 decoder 输入）
        decoder_outputs, _ = self.decoder(embedded, encoder_state)

        if self.use_attention:
            all_weights = []
            contexts = []
            for t in range(seq_len):
                dec_t = decoder_outputs[:, t, :]  # (batch, hidden_dim)
                ctx, w = self.attention(dec_t, encoder_outputs)
                contexts.append(ctx)
                all_weights.append(w)

            contexts = torch.stack(contexts, dim=1)  # (batch, seq_len, hidden_dim)
            self.last_attention_weights = torch.stack(all_weights, dim=1)  # (batch, seq_len, src_len)

            combined = torch.cat([decoder_outputs, contexts], dim=-1)
            logits = self.fc_out(combined)
        else:
            self.last_attention_weights = None
            logits = self.fc_out(decoder_outputs)

        return logits


# ═══════════════════════════════════════════════════════════
# 3. 训练 & 评估
# ═══════════════════════════════════════════════════════════
def train_one_epoch(
    model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, optimizer
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for i in range(0, len(inputs), args.batch_size):
        batch_x = inputs[i : i + args.batch_size].to(DEVICE)
        batch_y = targets[i : i + args.batch_size].to(DEVICE)

        optimizer.zero_grad()
        logits = model(batch_x)  # (batch, seq_len, vocab_size)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), batch_y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_recall(
    model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor
) -> float:
    """评估模型在最后两个位置（回忆标记）的准确率。"""
    model.eval()
    correct = 0
    total = 0
    for i in range(0, len(inputs), args.batch_size):
        batch_x = inputs[i : i + args.batch_size].to(DEVICE)
        batch_y = targets[i : i + args.batch_size].to(DEVICE)

        logits = model(batch_x)
        preds = logits.argmax(dim=-1)

        # 只看最后两个位置（回忆任务的核心）
        recall_preds = preds[:, -3:-1]
        recall_targets = batch_y[:, -3:-1]
        correct += (recall_preds == recall_targets).sum().item()
        total += recall_targets.numel()

    return correct / max(total, 1)


# ═══════════════════════════════════════════════════════════
# 4. 可视化
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def visualize_attention(model: nn.Module, inputs: torch.Tensor, save_dir: str):
    """可视化注意力权重热力图。"""
    if not model.use_attention:
        print("[可视化] 无 Attention 模块，跳过热力图。")
        return

    model.eval()
    sample = inputs[0:1].to(DEVICE)
    _ = model(sample)

    if model.last_attention_weights is None:
        return

    weights = model.last_attention_weights[0].cpu().numpy()  # (seq_len, src_len)

    # 解码字符标签
    chars = [IDX2CHAR[idx.item()] for idx in inputs[0]]

    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(weights, cmap="YlOrRd", aspect="auto")

    # 标注轴
    step = max(1, len(chars) // 30)
    tick_pos = list(range(0, len(chars), step))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([chars[i] for i in tick_pos], fontsize=7, rotation=90)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([chars[i] for i in tick_pos], fontsize=7)

    ax.set_xlabel("Encoder (source position)", fontsize=11)
    ax.set_ylabel("Decoder (target position)", fontsize=11)
    ax.set_title("Attention Weights — Decoder 'looking back' at Encoder", fontsize=13)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 高亮回忆区域
    seq_len = len(chars)
    rect = plt.Rectangle(
        (-0.5, seq_len - 3.5), 2.5, 2.5,
        linewidth=2, edgecolor="blue", facecolor="none", linestyle="--"
    )
    ax.add_patch(rect)
    ax.annotate(
        "Recall zone:\nshould attend\nto positions 0-1",
        xy=(1, seq_len - 2), fontsize=8, color="blue",
        ha="center", va="center",
        xytext=(8, seq_len - 8),
        arrowprops=dict(arrowstyle="->", color="blue"),
    )

    path = os.path.join(save_dir, "attention_heatmap.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] Attention 热力图已保存到 {path}")


def visualize_comparison(results: dict, save_dir: str):
    """绘制不同模型 / 序列长度下的回忆准确率对比图。"""
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── 左图：训练损失曲线 ──────────────────────────────
    for name, data in results.items():
        ax1.plot(data["losses"], label=name, linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Training Loss", fontsize=11)
    ax1.set_title("Training Loss Over Time", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ── 右图：不同长度的回忆准确率 ──────────────────────
    lengths = sorted(results[list(results.keys())[0]]["length_accs"].keys())
    bar_width = 0.25
    x = np.arange(len(lengths))

    for i, (name, data) in enumerate(results.items()):
        accs = [data["length_accs"][l] for l in lengths]
        ax2.bar(x + i * bar_width, accs, bar_width, label=name, alpha=0.85)

    ax2.set_xlabel("Sequence Length", fontsize=11)
    ax2.set_ylabel("Recall Accuracy", fontsize=11)
    ax2.set_title("Recall Accuracy vs Sequence Length", fontsize=13)
    ax2.set_xticks(x + bar_width)
    ax2.set_xticklabels(lengths)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3, axis="y")

    path = os.path.join(save_dir, "rnn_vs_attention_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 对比图已保存到 {path}")


# ═══════════════════════════════════════════════════════════
# 5. 主流程
# ═══════════════════════════════════════════════════════════
def train_and_evaluate(
    name: str, use_attention: bool, use_lstm: bool, seq_len: int
) -> dict:
    """训练一个模型并返回结果。"""
    print(f"\n{'─'*50}")
    print(f"训练模型: {name} (seq_len={seq_len})")
    print(f"{'─'*50}")

    model = Seq2SeqModel(
        VOCAB_SIZE, args.hidden_dim,
        use_attention=use_attention, use_lstm=use_lstm,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 生成训练数据
    train_x, train_y = generate_data(2000, seq_len)
    test_x, test_y = generate_data(500, seq_len)

    losses = []
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_x, train_y, optimizer)
        losses.append(loss)
        if (epoch + 1) % 10 == 0:
            acc = evaluate_recall(model, test_x, test_y)
            print(f"  Epoch {epoch+1:3d} | Loss: {loss:.4f} | Recall Acc: {acc:.2%}")

    # 不同长度的评估
    length_accs = {}
    for test_len in [15, 30, 60, 90]:
        tx, ty = generate_data(500, test_len)
        length_accs[test_len] = evaluate_recall(model, tx, ty)

    final_acc = evaluate_recall(model, test_x, test_y)
    print(f"  最终 Recall 准确率: {final_acc:.2%}")

    return {
        "model": model,
        "losses": losses,
        "final_acc": final_acc,
        "length_accs": length_accs,
        "train_x": train_x,
    }


def main():
    ablate_name = args.ablate or "baseline"
    seq_len = 20 if args.ablate == "short_sequences" else args.seq_len

    print("=" * 60)
    print(f"L10 RNN vs Attention · 配置: {ablate_name}")
    print(f"序列长度: {seq_len} | 设备: {DEVICE}")
    print("=" * 60)

    if args.ablate == "no_attention":
        # 单模型：LSTM 无 Attention
        result = train_and_evaluate("LSTM (no Attention)", False, True, seq_len)
        if args.save_figures:
            visualize_comparison(
                {"LSTM (no Attention)": result}, FIGURES_DIR
            )

    elif args.ablate == "no_lstm":
        # 单模型：Vanilla RNN
        result = train_and_evaluate("Vanilla RNN", False, False, seq_len)
        if args.save_figures:
            visualize_comparison(
                {"Vanilla RNN": result}, FIGURES_DIR
            )

    elif args.ablate == "short_sequences":
        # 三模型对比，但用短序列
        results = {}
        results["Vanilla RNN"] = train_and_evaluate("Vanilla RNN", False, False, seq_len)
        results["LSTM (no Attn)"] = train_and_evaluate("LSTM (no Attn)", False, True, seq_len)
        results["LSTM + Attention"] = train_and_evaluate("LSTM + Attention", True, True, seq_len)

        print(f"\n{'='*60}")
        print("短序列对比结果 (seq_len=20):")
        for name, data in results.items():
            print(f"  {name}: {data['final_acc']:.2%}")
        print("→ 短序列时三个模型差距很小，Attention 优势不明显")

        if args.save_figures:
            visualize_comparison(results, FIGURES_DIR)

    else:
        # 基线：三模型完整对比
        results = {}
        results["Vanilla RNN"] = train_and_evaluate("Vanilla RNN", False, False, seq_len)
        results["LSTM (no Attn)"] = train_and_evaluate("LSTM (no Attn)", False, True, seq_len)
        results["LSTM + Attention"] = train_and_evaluate("LSTM + Attention", True, True, seq_len)

        print(f"\n{'='*60}")
        print("三模型对比结果:")
        print(f"{'模型':<25} {'Recall Acc':>12}")
        print("-" * 40)
        for name, data in results.items():
            print(f"  {name:<23} {data['final_acc']:>10.2%}")

        print(f"\n不同序列长度下的 Recall 准确率:")
        print(f"{'模型':<25} {'len=15':>8} {'len=30':>8} {'len=60':>8} {'len=90':>8}")
        print("-" * 60)
        for name, data in results.items():
            accs = data["length_accs"]
            print(
                f"  {name:<23} {accs[15]:>7.2%} {accs[30]:>7.2%} "
                f"{accs[60]:>7.2%} {accs[90]:>7.2%}"
            )

        if args.save_figures:
            visualize_comparison(results, FIGURES_DIR)
            # Attention 热力图
            attn_result = results["LSTM + Attention"]
            visualize_attention(
                attn_result["model"], attn_result["train_x"], FIGURES_DIR
            )

    print(f"\n{'='*60}")
    print("完成！")
    if args.save_figures:
        print(f"图片保存在: {FIGURES_DIR}")


if __name__ == "__main__":
    main()