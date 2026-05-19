"""L03 · 学习即优化 — 补充图表生成脚本

生成 outline 引用但未存在的 3 张教学图：
  - polynomial_fitting.png   (3 阶 vs 9 阶 多项式拟合 — 欠/过拟合直观)
  - overfitting_demo.png     (训练 vs 验证 loss 双曲线，过拟合点用箭头标注)
  - double_descent_curve.png (双下降现象 — 经典 vs 现代深度学习的关键反直觉)

依赖：numpy, matplotlib
运行：python generate_supplementary_figures.py
"""
from __future__ import annotations
import sys, pathlib
# Ensure design/meta/cjk_font is on path
ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "design" / "meta"))
import cjk_font  # noqa: F401  — applies CJK fonts

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P

OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(1337)


# ─────────────────────────────────────────────────────────────────────
# 1. polynomial_fitting.png — 欠拟合 / 适当 / 过拟合 三联图
# ─────────────────────────────────────────────────────────────────────
def polynomial_fitting():
    # 真实函数：sin(2πx) + 噪声
    x_train = np.linspace(0, 1, 12)
    y_train = np.sin(2 * np.pi * x_train) + 0.15 * np.random.randn(len(x_train))
    x_dense = np.linspace(0, 1, 400)
    y_true = np.sin(2 * np.pi * x_dense)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    titles = [
        "1 阶多项式（欠拟合）",
        "3 阶多项式（适当）",
        "9 阶多项式（过拟合）",
    ]
    degrees = [1, 3, 9]
    colors = ["#C0392B", "#27AE60", "#2980B9"]
    for ax, deg, ttl, col in zip(axes, degrees, titles, colors):
        coef = np.polyfit(x_train, y_train, deg)
        y_fit = np.polyval(coef, x_dense)
        ax.plot(x_dense, y_true, "k--", lw=1.2, alpha=0.5, label="真实 sin")
        ax.scatter(x_train, y_train, color="#444", s=35, zorder=3,
                   label="训练数据")
        ax.plot(x_dense, y_fit, color=col, lw=2.2, label=f"{deg} 阶拟合")
        ax.set_title(ttl, fontsize=12, fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-1.6, 1.6)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("模型容量 vs 拟合质量 — 容量过低 / 适当 / 过高",
                 fontsize=14, fontweight="bold")
    out = OUT / "polynomial_fitting.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────
# 2. overfitting_demo.png — 训练/验证 loss 双曲线
# ─────────────────────────────────────────────────────────────────────
def overfitting_demo():
    epochs = np.arange(1, 101)
    # 训练 loss 单调下降
    train = 1.5 * np.exp(-0.05 * epochs) + 0.05 + 0.01 * np.random.randn(100)
    # 验证 loss 先下降后上升（过拟合）
    val = 1.5 * np.exp(-0.05 * epochs) + 0.20 + 0.04 * np.random.randn(100)
    val[40:] += 0.0008 * (epochs[40:] - 40) ** 1.5

    best_epoch = int(np.argmin(val)) + 1

    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    ax.plot(epochs, train, color="#27AE60", lw=2.2, label="训练 loss")
    ax.plot(epochs, val, color="#C0392B", lw=2.2, label="验证 loss")
    ax.axvline(best_epoch, ls="--", color="#888", lw=1.0)
    ax.annotate(
        f"最佳停止点 (epoch={best_epoch})\nEarly Stopping 应在此触发",
        xy=(best_epoch, val[best_epoch - 1]),
        xytext=(best_epoch + 12, val[best_epoch - 1] - 0.35),
        fontsize=10, color="#444",
        arrowprops=dict(arrowstyle="->", color="#888", lw=1.0),
    )
    ax.fill_between(epochs[best_epoch:], train[best_epoch:], val[best_epoch:],
                    color="#C0392B", alpha=0.12, label="过拟合 gap")
    ax.set_xlabel("训练 epoch", fontsize=11)
    ax.set_ylabel("loss", fontsize=11)
    ax.set_title("过拟合现象 — 验证 loss 在某点开始反弹",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    out = OUT / "overfitting_demo.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────
# 3. double_descent_curve.png — 现代 DL 的反直觉曲线
# ─────────────────────────────────────────────────────────────────────
def double_descent():
    # 经典 U 型 + 第二次下降
    cap = np.logspace(0, 3, 200)  # 模型容量（参数量级）
    # 经典区：U 形
    classical = 0.4 * np.exp(-0.0015 * (cap - 1)) + 0.2 + 0.0003 * cap
    # 在 interpolation threshold 处出现峰值
    interp_thresh = 100
    peak = np.exp(-((np.log10(cap) - np.log10(interp_thresh)) ** 2) / 0.05)
    # 第二次下降（over-parameterized 区）
    second_descent = 0.4 * np.exp(-0.003 * (cap - interp_thresh).clip(0))
    err = classical + 0.6 * peak + np.where(cap > interp_thresh,
                                             -0.3 * (1 - second_descent), 0)
    err = np.clip(err, 0.05, None)

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.semilogx(cap, err, color="#2980B9", lw=2.5, label="测试误差")
    # 标注三区
    ax.axvspan(cap[0], 50, alpha=0.08, color="#27AE60", label="经典欠参数区")
    ax.axvspan(50, 200, alpha=0.08, color="#E67E22",
               label="插值阈值（双下降峰）")
    ax.axvspan(200, cap[-1], alpha=0.08, color="#9B59B6",
               label="过参数区（现代 DL）")
    ax.axvline(interp_thresh, ls="--", color="#444", lw=1.2)
    # 文本注释
    ax.annotate("插值阈值\n(N≈样本数)",
                xy=(interp_thresh, err[np.argmin(np.abs(cap - interp_thresh))]),
                xytext=(interp_thresh * 0.6, 0.85),
                fontsize=10, ha="center",
                arrowprops=dict(arrowstyle="->", color="#444"))
    ax.annotate("第二次下降\n(深度学习现象)",
                xy=(500, err[np.argmin(np.abs(cap - 500))]),
                xytext=(500, 0.65),
                fontsize=10, ha="center", color="#9B59B6",
                arrowprops=dict(arrowstyle="->", color="#9B59B6"))
    ax.set_xlabel("模型容量（参数量，对数尺度）", fontsize=11)
    ax.set_ylabel("测试误差", fontsize=11)
    ax.set_title("双下降（Double Descent）现象 — 经典 ML vs 现代 DL 的关键反直觉",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.0)
    out = OUT / "double_descent_curve.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


if __name__ == "__main__":
    print("=== L03 supplementary figures ===")
    polynomial_fitting()
    overfitting_demo()
    double_descent()
    print("Done.")
