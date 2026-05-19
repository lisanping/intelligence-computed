"""L01 Demo: 认知偏见互动实验 — 锚定效应的统计证明

用随机锚定数字影响估计值，展示人类判断的系统性偏差。
运行后显示锚定效应的统计显著性分析。

用法：
    python anchoring_demo.py              # 模拟 200 人实验
    python anchoring_demo.py --n 1000     # 模拟 1000 人实验
    python anchoring_demo.py --interactive  # 互动模式：你来当被试

所有实验使用 seed=1337。
"""

import argparse
import numpy as np
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[4] / "design" / "meta"))
import cjk_font  # noqa: F401  - applies CJK font config
import matplotlib.pyplot as plt

SEED = 1337
np.random.seed(SEED)


def simulate_anchoring(n: int = 200) -> dict:
    """模拟锚定效应实验。

    问题："联合国中非洲国家的比例是多少？"
    - 高锚组：先看到随机数 65（轮盘结果），然后估计
    - 低锚组：先看到随机数 10（轮盘结果），然后估计

    真实答案：约 28%（54/193）
    经典结果（Tversky & Kahneman, 1974）：高锚组均值 45%，低锚组均值 25%
    """
    high_anchor = 65
    low_anchor = 10
    true_answer = 28

    # 模拟人类估计：受锚定值影响 + 个体噪声
    high_group = np.clip(
        high_anchor * 0.5 + true_answer * 0.3 + np.random.normal(0, 8, n // 2),
        5, 95
    )
    low_group = np.clip(
        low_anchor * 0.4 + true_answer * 0.4 + np.random.normal(0, 8, n // 2),
        5, 95
    )

    return {
        "high_anchor": high_anchor,
        "low_anchor": low_anchor,
        "true_answer": true_answer,
        "high_group": high_group,
        "low_group": low_group,
    }


def plot_results(data: dict, save_path: str = "anchoring_effect.png"):
    """可视化锚定效应实验结果。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 直方图对比
    ax = axes[0]
    ax.hist(data["high_group"], bins=20, alpha=0.6, color="#e74c3c",
            label=f'高锚组 (锚=65)\n均值={data["high_group"].mean():.1f}%')
    ax.hist(data["low_group"], bins=20, alpha=0.6, color="#3498db",
            label=f'低锚组 (锚=10)\n均值={data["low_group"].mean():.1f}%')
    ax.axvline(data["true_answer"], color="green", linestyle="--", linewidth=2,
               label=f'真实答案={data["true_answer"]}%')
    ax.set_xlabel("估计的非洲国家比例 (%)")
    ax.set_ylabel("人数")
    ax.set_title("锚定效应：随机数字如何扭曲你的判断")
    ax.legend(fontsize=9)

    # 箱线图
    ax = axes[1]
    bp = ax.boxplot([data["low_group"], data["high_group"]],
                    labels=["低锚组\n(锚=10)", "高锚组\n(锚=65)"],
                    patch_artist=True)
    bp["boxes"][0].set_facecolor("#3498db")
    bp["boxes"][1].set_facecolor("#e74c3c")
    ax.axhline(data["true_answer"], color="green", linestyle="--", linewidth=2,
               label=f"真实答案 {data['true_answer']}%")
    ax.set_ylabel("估计值 (%)")
    ax.set_title("两组估计值分布对比")
    ax.legend()

    # 效应量
    diff = data["high_group"].mean() - data["low_group"].mean()
    pooled_std = np.sqrt((data["high_group"].std()**2 + data["low_group"].std()**2) / 2)
    cohens_d = diff / pooled_std
    fig.suptitle(f"Cohen's d = {cohens_d:.2f} (效应量{'大' if cohens_d > 0.8 else '中' if cohens_d > 0.5 else '小'})",
                 fontsize=12, fontweight="bold", y=0.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  图已保存: {save_path}")
    plt.close()


def interactive_mode():
    """让用户亲自体验锚定效应。"""
    import random
    anchor = random.choice([10, 65])
    print(f"\n{'='*50}")
    print(f"  锚定效应互动实验")
    print(f"{'='*50}")
    print(f"\n  一个轮盘转出了数字: {anchor}")
    print(f"\n  问题：联合国 193 个成员国中，")
    try:
        guess = float(input("  非洲国家占比大约是多少？(%) > "))
    except (ValueError, EOFError):
        print("  输入无效，使用默认值 30%")
        guess = 30.0

    true_pct = 28.0
    print(f"\n  你的估计: {guess:.1f}%")
    print(f"  真实答案: {true_pct:.0f}% (54/193)")
    print(f"  你看到的锚: {anchor}")
    if (anchor == 65 and guess > true_pct) or (anchor == 10 and guess < true_pct):
        print(f"\n  ⚠️  你的估计偏向了锚定值——这就是锚定效应！")
    else:
        print(f"\n  ✅  你抵抗住了锚定效应（但大多数人做不到）。")
    print(f"\n  Tversky & Kahneman (1974) 的经典实验中：")
    print(f"    高锚组 (锚=65) 均值: 45%")
    print(f"    低锚组 (锚=10) 均值: 25%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="L01: 锚定效应实验模拟")
    ap.add_argument("--n", type=int, default=200, help="模拟人数")
    ap.add_argument("--interactive", action="store_true", help="互动模式")
    args = ap.parse_args()

    if args.interactive:
        interactive_mode()
    else:
        print(f"L01 Demo: 锚定效应模拟实验 (n={args.n})")
        data = simulate_anchoring(args.n)
        print(f"  高锚组 (锚=65) 均值: {data['high_group'].mean():.1f}%")
        print(f"  低锚组 (锚=10) 均值: {data['low_group'].mean():.1f}%")
        print(f"  真实答案: {data['true_answer']}%")
        print(f"  差异: {data['high_group'].mean() - data['low_group'].mean():.1f} 个百分点")
        plot_results(data)
