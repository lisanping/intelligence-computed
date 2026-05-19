"""
第 12 讲 · 高光动手环节 · 从零搭建 Tiny Transformer
================================================

目标：120 行 PyTorch 代码，把 Transformer Decoder 的每一块**手动展开**。
     和序章 `demos/shakespeare_50lines_torch.py` 的区别是：
     - 序章用 nn.MultiheadAttention 黑盒（为了压到 50 行）
     - 本讲完全展开 Q/K/V 投影、打分、缩放、掩码、softmax、聚合——这是本讲的核心

和 Karpathy nanoGPT (github.com/karpathy/nanoGPT) 的关系：
    本文件是其 `model.py` 的教学精简版（去掉 Flash Attention、KV cache、torch.compile、
    权重绑定等生产优化，保留结构），课程 MIT 许可下衍生。

运行：
    python nanogpt_from_scratch.py                    # 基线
    python nanogpt_from_scratch.py --ablate no_pe     # 去掉位置编码
    python nanogpt_from_scratch.py --ablate no_scale  # 去掉 sqrt(d_k) 缩放
    python nanogpt_from_scratch.py --ablate no_resid  # 去掉残差连接
    python nanogpt_from_scratch.py --n_head 1         # 单头对比多头
"""
from __future__ import annotations

import argparse
import math
from urllib.request import urlopen

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 超参数（保持小，让 CPU 都能在 3 分钟内看出趋势）
# ---------------------------------------------------------------------------
BLOCK_SIZE = 128        # 上下文长度 T
BATCH_SIZE = 64
N_EMBED = 192           # 嵌入维度 d
N_HEAD = 6              # 头数 H（要求 d 能被 H 整除）
N_LAYER = 4             # Transformer Block 堆叠层数 N
DROPOUT = 0.1
LR = 3e-4
MAX_ITERS = 3000
EVAL_INTERVAL = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# 模块 1 · CausalSelfAttention —— 本讲最关键的 40 行
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """
    手动展开的多头因果自注意力。逐行对应讲稿 [08:00 – 14:00] 的六步：
        1. 投影出 Q/K/V
        2. 相似度打分 Q @ K^T
        3. 缩放 / sqrt(d_k)
        4. 因果掩码
        5. softmax
        6. 加权读取 V
    """

    def __init__(self, d: int, n_head: int, block_size: int, ablate_scale: bool = False):
        super().__init__()
        assert d % n_head == 0, "嵌入维度必须能被头数整除"
        self.n_head = n_head
        self.d_head = d // n_head
        self.ablate_scale = ablate_scale

        # 步骤 1 的三个投影：一次性算出 Q, K, V；输出维度 3d，再沿最后一维切成三份
        self.qkv_proj = nn.Linear(d, 3 * d, bias=False)
        # 最后一步把多头拼回的输出再投影一下（公式里的 W_O）
        self.out_proj = nn.Linear(d, d, bias=False)
        self.attn_dropout = nn.Dropout(DROPOUT)
        self.resid_dropout = nn.Dropout(DROPOUT)

        # 步骤 4 的因果掩码：上三角 True 表示"要屏蔽掉"
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.n_head, self.d_head

        # 步骤 1 · 投影并切分成多头：(B, T, 3D) -> 3 份 (B, H, T, Dh)
        q, k, v = self.qkv_proj(x).split(D, dim=-1)
        q = q.view(B, T, H, Dh).transpose(1, 2)  # (B, H, T, Dh)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)

        # 步骤 2 · 打分：Q @ K^T  -> (B, H, T, T)
        scores = q @ k.transpose(-2, -1)

        # 步骤 3 · 缩放 / sqrt(d_k)；消融开关可以关掉以演示训练不稳
        if not self.ablate_scale:
            scores = scores / math.sqrt(Dh)

        # 步骤 4 · 因果掩码：把未来位置打成 -inf
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))

        # 步骤 5 · softmax 沿 "被关注" 的那一维（最后一维）
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # 步骤 6 · 加权读取 V，然后把多头拼回
        y = attn @ v                                 # (B, H, T, Dh)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.resid_dropout(self.out_proj(y))
        return y


# ---------------------------------------------------------------------------
# 模块 2 · FeedForward —— position-wise MLP，参数大头在这里
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 模块 3 · Transformer Block —— Pre-LN 结构（讲稿里提到为什么用 Pre-LN）
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d: int, n_head: int, block_size: int, ablate_scale: bool, ablate_resid: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_head, block_size, ablate_scale=ablate_scale)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = FeedForward(d)
        self.ablate_resid = ablate_resid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ablate_resid:
            # 消融实验：把残差连接干掉，看梯度是否能通过深度堆叠
            x = self.attn(self.ln1(x))
            x = self.ffn(self.ln2(x))
        else:
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# 模块 4 · TinyGPT —— 把积木堆起来
# ---------------------------------------------------------------------------
class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, *, ablate_pe: bool = False,
                 ablate_scale: bool = False, ablate_resid: bool = False,
                 n_head: int = N_HEAD):
        super().__init__()
        self.block_size = BLOCK_SIZE
        self.ablate_pe = ablate_pe

        self.tok_emb = nn.Embedding(vocab_size, N_EMBED)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([
            Block(N_EMBED, n_head, BLOCK_SIZE, ablate_scale, ablate_resid)
            for _ in range(N_LAYER)
        ])
        self.ln_f = nn.LayerNorm(N_EMBED)
        self.head = nn.Linear(N_EMBED, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        if self.ablate_pe:
            # 消融实验：去掉位置编码，Attention 对顺序不敏感，loss 几乎不会下降
            x = tok
        else:
            pos = self.pos_emb(torch.arange(T, device=idx.device))
            x = tok + pos
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ---------------------------------------------------------------------------
# 训练循环 · 能在 CPU 上 3 分钟看出莎士比亚味
# ---------------------------------------------------------------------------
def get_batch(data: torch.Tensor, block_size: int, batch_size: int):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablate", choices=["no_pe", "no_scale", "no_resid"], default=None,
                        help="消融实验：去掉某个组件")
    parser.add_argument("--n_head", type=int, default=N_HEAD)
    args = parser.parse_args()

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = urlopen(url).read().decode()
    chars = sorted(set(text))
    vocab_size = len(chars)
    s2i = {c: i for i, c in enumerate(chars)}
    i2s = {i: c for c, i in s2i.items()}
    data = torch.tensor([s2i[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    model = TinyGPT(
        vocab_size,
        ablate_pe=(args.ablate == "no_pe"),
        ablate_scale=(args.ablate == "no_scale"),
        ablate_resid=(args.ablate == "no_resid"),
        n_head=args.n_head,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={DEVICE}  params={n_params/1e6:.2f}M  ablation={args.ablate}  n_head={args.n_head}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    for step in range(MAX_ITERS):
        xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
            model.eval()
            with torch.no_grad():
                xb, yb = get_batch(val_data, BLOCK_SIZE, BATCH_SIZE)
                _, val_loss = model(xb, yb)
            model.train()
            print(f"step {step:5d} | train {loss.item():.3f} | val {val_loss.item():.3f}")

    model.eval()
    start = torch.tensor([[s2i["R"], s2i["O"], s2i["M"], s2i["E"], s2i["O"], s2i[":"]]], device=DEVICE)
    out = model.generate(start, max_new_tokens=400)[0].tolist()
    print("\n--- 生成样本 ---\n" + "".join(i2s[i] for i in out))


if __name__ == "__main__":
    main()