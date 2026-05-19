"""L33 Demo: Sim-to-Real 迁移差距可视化

模拟 sim-to-real gap：展示仿真环境中训练的策略在真实环境中的性能衰减，
以及 Domain Randomization 如何缩小这个差距。

用法：
    python sim2real_gap.py                # 生成 sim-to-real 对比图
    python sim2real_gap.py --trials 50    # 50 轮模拟

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


def simulate_policy_transfer(n_trials: int = 30) -> dict:
    """模拟策略从仿真到真实的迁移。

    三种策略：
    1. 纯仿真训练（Sim Only）—— 仿真中优秀，真实中崩溃
    2. Domain Randomization —— 仿真中略差，真实中稳健
    3. Sim + Real fine-tune —— 两边都不错
    """
    # 仿真环境性能
    sim_only_sim = np.clip(np.random.normal(0.92, 0.03, n_trials), 0, 1)
    domain_rand_sim = np.clip(np.random.normal(0.85, 0.05, n_trials), 0, 1)
    sim_finetune_sim = np.clip(np.random.normal(0.88, 0.04, n_trials), 0, 1)

    # 真实环境性能（sim-to-real gap）
    sim_only_real = np.clip(np.random.normal(0.45, 0.15, n_trials), 0, 1)
    domain_rand_real = np.clip(np.random.normal(0.78, 0.08, n_trials), 0, 1)
    sim_finetune_real = np.clip(np.random.normal(0.82, 0.06, n_trials), 0, 1)

    return {
        "sim_only": {"sim": sim_only_sim, "real": sim_only_real},
        "domain_rand": {"sim": domain_rand_sim, "real": domain_rand_real},
        "sim_finetune": {"sim": sim_finetune_sim, "real": sim_finetune_real},
    }


def plot_sim2real(data: dict, save_path: str = "sim2real_gap.png"):
    """可视化 sim-to-real gap。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    labels = ["Sim Only", "Domain\nRandomization", "Sim + Real\nFine-tune"]
    colors = ["#e74c3c", "#f39c12", "#27ae60"]

    # 左图：仿真 vs 真实性能对比
    ax = axes[0]
    x = np.arange(len(labels))
    width = 0.3

    sim_means = [data["sim_only"]["sim"].mean(),
                 data["domain_rand"]["sim"].mean(),
                 data["sim_finetune"]["sim"].mean()]
    real_means = [data["sim_only"]["real"].mean(),
                  data["domain_rand"]["real"].mean(),
                  data["sim_finetune"]["real"].mean()]

    bars1 = ax.bar(x - width/2, sim_means, width, label="仿真环境",
                   color=[c + "88" for c in colors], edgecolor=colors, linewidth=2)
    bars2 = ax.bar(x + width/2, real_means, width, label="真实环境",
                   color=colors, edgecolor="black", linewidth=1)

    # 标注 gap
    for i in range(3):
        gap = sim_means[i] - real_means[i]
        ax.annotate(f"gap: {gap:.0%}",
                    xy=(x[i] + width/2, real_means[i]),
                    xytext=(x[i] + 0.6, real_means[i] + 0.05),
                    fontsize=9, ha="left",
                    arrowprops=dict(arrowstyle="->", color="gray"))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("任务成功率")
    ax.set_ylim(0, 1.1)
    ax.set_title("Sim-to-Real Gap: 三种迁移策略对比")
    ax.legend(loc="upper right")

    # 右图：Moravec 悖论图示
    ax = axes[1]
    tasks = ["写诗", "下棋", "数学证明", "翻译", "识别人脸",
             "抓取物体", "走路", "叠衣服"]
    ai_difficulty = [0.3, 0.2, 0.4, 0.25, 0.5, 0.85, 0.9, 0.95]
    human_difficulty = [0.7, 0.6, 0.8, 0.5, 0.05, 0.05, 0.02, 0.1]

    y = np.arange(len(tasks))
    height = 0.35
    ax.barh(y - height/2, ai_difficulty, height, label="AI 难度",
            color="#e74c3c", alpha=0.8)
    ax.barh(y + height/2, human_difficulty, height, label="人类难度",
            color="#3498db", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(tasks, fontsize=10)
    ax.set_xlabel("相对难度")
    ax.set_title("Moravec 悖论: AI 难 ≠ 人类难")
    ax.legend(loc="lower right")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  图已保存: {save_path}")
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="L33: Sim-to-Real gap 可视化")
    ap.add_argument("--trials", type=int, default=30, help="模拟轮数")
    args = ap.parse_args()

    print(f"L33 Demo: Sim-to-Real 迁移差距 (trials={args.trials})")
    data = simulate_policy_transfer(args.trials)

    for name, d in data.items():
        gap = d["sim"].mean() - d["real"].mean()
        print(f"  {name:20s} sim={d['sim'].mean():.2f}  real={d['real'].mean():.2f}  gap={gap:.2f}")

    plot_sim2real(data)
