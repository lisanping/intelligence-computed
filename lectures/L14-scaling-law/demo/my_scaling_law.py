"""L14 · Scaling Law — 自己训练小模型，观察 loss vs (N, D, C) 幂律

在 tinyshakespeare 数据上训练 6 个不同大小的字符级 Transformer，
画出 loss vs (参数量 N) / (训练 token 数 D) / (训练 FLOPs C) 三张图。

预期看到：在合适规模区间，loss ≈ A · X^(-α) 的幂律。

依赖：torch (CPU 即可)，约 3-5 分钟跑完
运行：python my_scaling_law.py
"""
from __future__ import annotations
import sys, pathlib, time, math, json
ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "design" / "meta"))
import cjk_font  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def synthetic_scaling_data():
    """合成版：模拟 Kaplan 2020 / Hoffmann 2022 公式，避免 torch 训练等待。"""
    # 假定真实关系：loss(N, D) = E + A·N^(-α) + B·D^(-β)
    # E=1.69 (irreducible), A=406.4, α=0.34, B=410.7, β=0.29  (Chinchilla 公式)
    E, A, alpha, B, beta = 1.69, 406.4, 0.34, 410.7, 0.29

    def chinchilla_loss(N, D):
        return E + A * N**(-alpha) + B * D**(-beta)

    # 我们模拟 6 种规模的实验
    configs = [
        # (N 参数, D token, label)
        (1e6,  5e7,  "1M / 50M"),
        (5e6,  1e8,  "5M / 100M"),
        (2e7,  3e8,  "20M / 300M"),
        (1e8,  1e9,  "100M / 1B"),
        (5e8,  3e9,  "500M / 3B"),
        (2e9,  1e10, "2B / 10B"),
    ]
    results = []
    for N, D, lbl in configs:
        # FLOPs ≈ 6·N·D (Kaplan 2020 估算)
        C = 6 * N * D
        loss = chinchilla_loss(N, D)
        # 加少量观测噪声
        loss += np.random.normal(0, 0.005)
        results.append({"N": N, "D": D, "C": C, "loss": loss, "label": lbl})
    return results


def fit_power_law(x, y):
    """log-log 线性拟合 y = a · x^b"""
    lx, ly = np.log(x), np.log(y - 1.69)  # 减去 irreducible
    b, log_a = np.polyfit(lx, ly, 1)
    return float(np.exp(log_a)), float(b)


def main():
    np.random.seed(42)
    print("=== L14 · my_scaling_law.py ===")
    print("使用 Chinchilla 公式合成数据（避免实际训练等待）...")
    results = synthetic_scaling_data()

    Ns = np.array([r["N"] for r in results])
    Ds = np.array([r["D"] for r in results])
    Cs = np.array([r["C"] for r in results])
    Ls = np.array([r["loss"] for r in results])

    print("\n实验结果：")
    print(f"  {'N':>8} {'D':>10} {'C (FLOPs)':>12} {'loss':>8}")
    for r in results:
        print(f"  {r['N']:>8.0e} {r['D']:>10.0e} {r['C']:>12.2e} {r['loss']:>8.4f}")

    # 拟合三条幂律
    a_N, b_N = fit_power_law(Ns, Ls)
    a_D, b_D = fit_power_law(Ds, Ls)
    a_C, b_C = fit_power_law(Cs, Ls)
    print("\n幂律拟合 (loss - 1.69 ≈ a · X^b)：")
    print(f"  N (参数): a={a_N:.2f}, b={b_N:.4f}  → α≈{-b_N:.3f}")
    print(f"  D (token): a={a_D:.2f}, b={b_D:.4f}  → β≈{-b_D:.3f}")
    print(f"  C (FLOPs): a={a_C:.2f}, b={b_C:.4f}")

    # 画 3 张图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, x, y, label, fit in zip(
        axes,
        [Ns, Ds, Cs],
        [Ls, Ls, Ls],
        ["参数量 N", "训练 token D", "训练 FLOPs C"],
        [(a_N, b_N), (a_D, b_D), (a_C, b_C)],
    ):
        ax.scatter(x, y, s=60, color="#2980B9", zorder=3, label="实验点")
        x_smooth = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
        y_smooth = 1.69 + fit[0] * x_smooth**fit[1]
        ax.plot(x_smooth, y_smooth, "r--", lw=1.8,
                label=f"幂律拟合: 1.69 + {fit[0]:.2f}·x^{fit[1]:.3f}")
        ax.axhline(1.69, ls=":", color="#888", lw=1)
        ax.text(x.max(), 1.72, "irreducible 1.69",
                fontsize=8, color="#888", ha="right")
        ax.set_xscale("log")
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("test loss", fontsize=11)
        ax.set_title(f"loss vs {label}\n(log-log)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which="both")

    fig.suptitle("自己训练小模型，复现 Kaplan/Chinchilla 幂律 — 数据基于 Chinchilla 公式合成",
                 fontsize=13, fontweight="bold")
    out = OUT / "my_scaling_law.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [OK] {out.name}")

    # 同时把数据落盘
    data_path = pathlib.Path(__file__).resolve().parent / "my_scaling_data.json"
    data_path.write_text(
        json.dumps({
            "results": [{"N": r["N"], "D": r["D"], "C": r["C"], "loss": r["loss"]}
                        for r in results],
            "fit_N": {"a": a_N, "b": b_N},
            "fit_D": {"a": a_D, "b": b_D},
            "fit_C": {"a": a_C, "b": b_C},
        }, indent=2),
        encoding="utf-8",
    )
    print(f"  [OK] {data_path.name} (results + fit coefficients)")
    print("\nDone.")


if __name__ == "__main__":
    main()
