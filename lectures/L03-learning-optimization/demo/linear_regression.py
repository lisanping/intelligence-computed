"""
第 3 讲 · 动手环节 · 线性回归 from scratch
==========================================

目标：~50 行 PyTorch 代码，演示统一框架（Model + Data + Loss + Optimizer）。
     生成合成数据，用梯度下降拟合线性模型，实时绘制 loss 曲线和拟合过程。

运行：
    python linear_regression.py                      # 基线（SGD + Momentum）
    python linear_regression.py --ablate no_momentum # 去掉动量
    python linear_regression.py --ablate huge_lr     # 学习率过大 (lr=10)
    python linear_regression.py --ablate tiny_lr     # 学习率过小 (lr=0.0001)
    python linear_regression.py --ablate no_noise    # 全批梯度下降 (Batch GD)

所有实验使用 seed=1337。
"""
from __future__ import annotations

import argparse
import torch
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[4] / "design" / "meta"))
import cjk_font  # noqa: F401  - applies CJK font config
import matplotlib.pyplot as plt

# ── 固定随机种子 ─────────────────────────────────────────────
torch.manual_seed(1337)

# ── 命令行参数 ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="L03 线性回归 demo")
parser.add_argument("--ablate", type=str, default=None,
                    choices=["no_momentum", "huge_lr", "tiny_lr", "no_noise"],
                    help="消融实验开关")
args = parser.parse_args()

# ── 超参数（根据消融调整）──────────────────────────────────
LR = {"huge_lr": 10.0, "tiny_lr": 0.0001}.get(args.ablate, 0.05)
MOMENTUM = 0.0 if args.ablate == "no_momentum" else 0.9
BATCH_SIZE = None if args.ablate == "no_noise" else 32  # None = 全批
STEPS = 200

# ── 数据：y = 2x + 1 + 噪声 ───────────────────────────────
N = 100
x = torch.randn(N, 1)
y_true = 2 * x + 1 + 0.3 * torch.randn(N, 1)

# ── 模型：两个可学习参数 ───────────────────────────────────
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# ── 优化器 ────────────────────────────────────────────────
optimizer = torch.optim.SGD([w, b], lr=LR, momentum=MOMENTUM)

# ── 训练 + 实时绘图 ──────────────────────────────────────
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
losses = []

for step in range(STEPS):
    # mini-batch 或全批
    if BATCH_SIZE is not None:
        idx = torch.randint(0, N, (BATCH_SIZE,))
        xb, yb = x[idx], y_true[idx]
    else:
        xb, yb = x, y_true

    pred = w * xb + b                         # 模型
    loss = ((pred - yb) ** 2).mean()           # 损失函数 (MSE)
    optimizer.zero_grad()
    loss.backward()                            # 计算梯度
    optimizer.step()                           # 优化器更新

    losses.append(loss.item())

    # 每 10 步刷新图
    if step % 10 == 0 or step == STEPS - 1:
        ax1.clear()
        ax1.scatter(x.numpy(), y_true.numpy(), alpha=0.4, s=15, label="data")
        x_line = torch.linspace(x.min(), x.max(), 100).unsqueeze(1)
        with torch.no_grad():
            y_line = w * x_line + b
        ax1.plot(x_line.numpy(), y_line.numpy(), "r-", linewidth=2,
                 label=f"w={w.item():.2f}, b={b.item():.2f}")
        ax1.set_title(f"Fitting (step {step})")
        ax1.legend(loc="upper left")

        ax2.clear()
        ax2.plot(losses, "b-", linewidth=1.5)
        ax2.set_xlabel("step")
        ax2.set_ylabel("MSE loss")
        ax2.set_title(f"Loss = {losses[-1]:.4f}")

        plt.tight_layout()
        plt.pause(0.01)

plt.ioff()
tag = args.ablate or "baseline"
print(f"[{tag}] final w={w.item():.4f}  b={b.item():.4f}  loss={losses[-1]:.4f}")
plt.savefig(f"figures/linear_regression_{tag}.png", dpi=150)
plt.show()