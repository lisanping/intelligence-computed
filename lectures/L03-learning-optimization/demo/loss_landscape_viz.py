"""
第 3 讲 · 3D 损失曲面 + 优化器轨迹对比
======================================

目标：生成三张图——
  1. 3D 损失曲面（V1）
  2. 等高线 + 梯度下降轨迹（V2）
  3. 三种优化器轨迹对比：SGD / SGD+Momentum / Adam（V9）

运行：
    python loss_landscape_viz.py

所有实验使用 seed=1337。
"""
from __future__ import annotations

# torch must be imported before matplotlib on Windows to avoid OpenMP/DLL
# conflicts (libiomp5md.dll vs libomp.dll, plus shm.dll load order).
import torch
import numpy as np
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[4] / "design" / "meta"))
import cjk_font  # noqa: F401  - applies CJK font config
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3D projection

torch.manual_seed(1337)
np.random.seed(1337)

# ── 损失曲面定义 ──────────────────────────────────────────
# 使用一个有局部最小值的二维函数作为 "loss landscape" 教学示例
# f(w1, w2) = (w1^2 + w2 - 11)^2 + (w1 + w2^2 - 7)^2  (Himmelblau)
# 为了更直观，我们用一个自定义曲面：多个高斯谷底 + 鞍点

def loss_surface(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """A synthetic loss surface with multiple local minima."""
    z = (1.5 - w1 + w1 * w2) ** 2 + (2.25 - w1 + w1 * w2**2) ** 2 + \
        (2.625 - w1 + w1 * w2**3) ** 2
    return np.log1p(z)  # log scale for better visualization


def loss_and_grad(w: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Compute loss and gradient at a point (PyTorch autograd)."""
    w = w.detach().requires_grad_(True)
    w1, w2 = w[0], w[1]
    z = (1.5 - w1 + w1 * w2) ** 2 + (2.25 - w1 + w1 * w2**2) ** 2 + \
        (2.625 - w1 + w1 * w2**3) ** 2
    loss = torch.log1p(z)
    loss.backward()
    return loss.item(), w.grad.clone()


# ── 网格 ────────────────────────────────────────────────
W1 = np.linspace(-4.5, 4.5, 300)
W2 = np.linspace(-4.5, 4.5, 300)
W1g, W2g = np.meshgrid(W1, W2)
Z = loss_surface(W1g, W2g)

# ── 图 1：3D 损失曲面 ──────────────────────────────────
fig1 = plt.figure(figsize=(10, 7))
ax1 = fig1.add_subplot(111, projection="3d")
ax1.plot_surface(W1g, W2g, Z, cmap=cm.coolwarm, alpha=0.85,
                 rstride=5, cstride=5, edgecolor="none")
ax1.set_xlabel("$w_1$", fontsize=12)
ax1.set_ylabel("$w_2$", fontsize=12)
ax1.set_zlabel("Loss (log)", fontsize=12)
ax1.set_title("Loss Landscape (3D)", fontsize=14)
ax1.view_init(elev=35, azim=-60)
fig1.tight_layout()
fig1.savefig("figures/loss_surface_3d.png", dpi=150)
print("Saved figures/loss_surface_3d.png")


# ── 优化器轨迹模拟 ──────────────────────────────────────
def run_optimizer(name: str, lr: float, steps: int = 150,
                  start: tuple[float, float] = (-3.5, -3.5)):
    """Run an optimizer and return the trajectory."""
    w = torch.tensor(list(start), dtype=torch.float32)
    if name == "sgd":
        opt = torch.optim.SGD([w.requires_grad_(True)], lr=lr)
    elif name == "momentum":
        opt = torch.optim.SGD([w.requires_grad_(True)], lr=lr, momentum=0.9)
    elif name == "adam":
        opt = torch.optim.Adam([w.requires_grad_(True)], lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

    traj = [w.detach().numpy().copy()]
    for _ in range(steps):
        opt.zero_grad()
        _, g = loss_and_grad(w)
        w.grad = g
        opt.step()
        traj.append(w.detach().numpy().copy())
    return np.array(traj)


traj_sgd = run_optimizer("sgd", lr=0.002, steps=200)
traj_mom = run_optimizer("momentum", lr=0.001, steps=200)
traj_adam = run_optimizer("adam", lr=0.05, steps=200)

# ── 图 2：等高线 + 单条梯度下降轨迹 ───────────────────
fig2, ax2 = plt.subplots(figsize=(8, 7))
ax2.contour(W1g, W2g, Z, levels=40, cmap=cm.coolwarm, alpha=0.7)
ax2.plot(traj_sgd[:, 0], traj_sgd[:, 1], "ko-", markersize=2,
         linewidth=1.2, label="SGD trajectory")
ax2.plot(traj_sgd[0, 0], traj_sgd[0, 1], "gs", markersize=10, label="start")
ax2.plot(traj_sgd[-1, 0], traj_sgd[-1, 1], "r*", markersize=14, label="end")
ax2.set_xlabel("$w_1$", fontsize=12)
ax2.set_ylabel("$w_2$", fontsize=12)
ax2.set_title("Gradient Descent on Loss Contours", fontsize=14)
ax2.legend()
fig2.tight_layout()
fig2.savefig("figures/gradient_descent_path.png", dpi=150)
print("Saved figures/gradient_descent_path.png")

# ── 图 3：三种优化器对比 ─────────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 7))
ax3.contour(W1g, W2g, Z, levels=40, cmap="gray", alpha=0.5)
ax3.plot(traj_sgd[:, 0], traj_sgd[:, 1], "b-", linewidth=1.5,
         alpha=0.8, label="SGD")
ax3.plot(traj_mom[:, 0], traj_mom[:, 1], "g-", linewidth=1.5,
         alpha=0.8, label="SGD + Momentum")
ax3.plot(traj_adam[:, 0], traj_adam[:, 1], "r-", linewidth=1.5,
         alpha=0.8, label="Adam")
# start / end markers
for traj, color in [(traj_sgd, "b"), (traj_mom, "g"), (traj_adam, "r")]:
    ax3.plot(traj[0, 0], traj[0, 1], "s", color=color, markersize=8)
    ax3.plot(traj[-1, 0], traj[-1, 1], "*", color=color, markersize=12)
ax3.set_xlabel("$w_1$", fontsize=12)
ax3.set_ylabel("$w_2$", fontsize=12)
ax3.set_title("Optimizer Comparison: SGD vs Momentum vs Adam", fontsize=14)
ax3.legend()
fig3.tight_layout()
fig3.savefig("figures/optimizer_comparison.png", dpi=150)
print("Saved figures/optimizer_comparison.png")

plt.show()