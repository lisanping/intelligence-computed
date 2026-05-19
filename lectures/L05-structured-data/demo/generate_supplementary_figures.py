"""L05 · 结构化数据 — 补充图表生成脚本

生成 3 张关键教学图（session memory batch 4 narrative-gap notes 标记）：
  - timeseries_decomposition.png  (时序四组件分解：趋势/季节/周期/残差)
  - anomaly_three_methods.png     (异常检测三算法对比：Z-score / IsoForest / LOF)
  - dim_curse_tsne.png            (维度灾难 — 高维 t-SNE 退化为团块)

依赖：numpy, matplotlib, scikit-learn
运行：python generate_supplementary_figures.py
"""
from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "design" / "meta"))
import cjk_font  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(1337)


# ─────────────────────────────────────────────────────────────────────
# 1. timeseries_decomposition.png — 趋势/季节/周期/残差 四组件
# ─────────────────────────────────────────────────────────────────────
def timeseries_decomposition():
    n = 365 * 2  # 2 年日级数据
    t = np.arange(n)
    trend = 0.03 * t + 50 + 0.00005 * t ** 1.5
    season = 8 * np.sin(2 * np.pi * t / 365) + 5 * np.cos(2 * np.pi * t / 365)
    cycle = 4 * np.sin(2 * np.pi * t / 90) * np.cos(2 * np.pi * t / 30)
    noise = 1.5 * np.random.randn(n)
    observed = trend + season + cycle + noise

    fig, axes = plt.subplots(5, 1, figsize=(11, 9), sharex=True,
                             constrained_layout=True)
    components = [
        ("观测序列  (Trend + Season + Cycle + Residual)", observed, "#1A1F2B"),
        ("Trend  趋势分量",                                 trend,    "#27AE60"),
        ("Season  季节分量 (年周期)",                       season,   "#2980B9"),
        ("Cycle  周期分量 (~90 天循环)",                    cycle,    "#E67E22"),
        ("Residual  残差 (随机噪声)",                       noise,    "#888888"),
    ]
    for ax, (lbl, y, c) in zip(axes, components):
        ax.plot(t, y, color=c, lw=1.2)
        ax.set_ylabel(lbl, fontsize=10)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="#bbb", lw=0.6)
    axes[-1].set_xlabel("天数 (2 年)", fontsize=11)
    fig.suptitle("时序四组件分解 — 把一个序列拆成 4 个可解释信号",
                 fontsize=14, fontweight="bold")
    out = OUT / "timeseries_decomposition.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────
# 2. anomaly_three_methods.png — Z-score / IsoForest / LOF 三算法对比
# ─────────────────────────────────────────────────────────────────────
def anomaly_three_methods():
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
    except ImportError:
        print("  WARN scikit-learn missing — skipping anomaly figure")
        return

    # 合成数据：1 个主簇 + 1 个小簇 + 一些孤立点
    rng = np.random.RandomState(42)
    main = rng.randn(200, 2) * 0.6
    sub = rng.randn(30, 2) * 0.3 + np.array([2.5, 2.5])
    outliers = rng.uniform(low=-4, high=4, size=(15, 2))
    X = np.vstack([main, sub, outliers])

    methods = []
    # Z-score (per-feature, threshold |z|>2.5)
    z = (X - X.mean(0)) / X.std(0)
    pred_z = (np.max(np.abs(z), axis=1) > 2.5).astype(int)
    methods.append(("Z-score (|z| > 2.5)", pred_z, "#C0392B"))
    # Isolation Forest
    iso = IsolationForest(contamination=0.06, random_state=42).fit(X)
    pred_iso = (iso.predict(X) == -1).astype(int)
    methods.append(("Isolation Forest", pred_iso, "#2980B9"))
    # LOF
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.06)
    pred_lof = (lof.fit_predict(X) == -1).astype(int)
    methods.append(("Local Outlier Factor", pred_lof, "#27AE60"))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, (name, pred, col) in zip(axes, methods):
        normal = X[pred == 0]
        anom = X[pred == 1]
        ax.scatter(normal[:, 0], normal[:, 1], c="#bbb", s=20,
                   alpha=0.7, label=f"正常 ({len(normal)})")
        ax.scatter(anom[:, 0], anom[:, 1], c=col, s=70,
                   marker="x", lw=2.0, label=f"异常 ({len(anom)})")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5)
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
        ax.set_aspect("equal")
    fig.suptitle("异常检测三算法对比 — 同一数据集，不同假设，不同结果",
                 fontsize=13, fontweight="bold")
    out = OUT / "anomaly_three_methods.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────
# 3. dim_curse_tsne.png — 维度灾难 t-SNE 退化对比
# ─────────────────────────────────────────────────────────────────────
def dim_curse():
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  WARN scikit-learn missing — skipping dim_curse figure")
        return
    rng = np.random.RandomState(42)
    n_per_class = 80
    n_classes = 4
    dims_to_test = [2, 10, 100, 500]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for ax, D in zip(axes, dims_to_test):
        # 生成每类一个高斯团；只在前 2 维有判别信号，其余维度是噪声
        X = rng.randn(n_per_class * n_classes, D) * 0.6
        labels = np.repeat(np.arange(n_classes), n_per_class)
        for c in range(n_classes):
            angle = 2 * np.pi * c / n_classes
            X[labels == c, 0] += 3.0 * np.cos(angle)
            X[labels == c, 1] += 3.0 * np.sin(angle)
        # t-SNE 投影到 2D
        proj = TSNE(n_components=2, perplexity=30, init="random",
                    random_state=42, max_iter=500).fit_transform(X)
        for c in range(n_classes):
            mask = labels == c
            ax.scatter(proj[mask, 0], proj[mask, 1], s=10, alpha=0.7,
                       label=f"类 {c}")
        # 估计聚类质量：簇内/簇间距离比
        intra = np.mean([np.std(proj[labels == c]) for c in range(n_classes)])
        ax.set_title(f"D = {D}\n簇内分散 ≈ {intra:.2f}",
                     fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(alpha=0.3)
    fig.suptitle("维度灾难 — 同样 4 个真实簇，维度从 2 → 500 后 t-SNE 投影逐渐退化",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.01,
             "结论：高维空间中『距离』变得无意义，所有点都『差不多远』 — "
             "这是经典 ML 处理结构化数据的根本难题",
             ha="center", fontsize=9, style="italic", color="#666")
    out = OUT / "dim_curse_tsne.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


if __name__ == "__main__":
    print("=== L05 supplementary figures ===")
    timeseries_decomposition()
    anomaly_three_methods()
    dim_curse()
    print("Done.")
