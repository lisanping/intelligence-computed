"""L17 · 强化学习 — 补充图表生成脚本

补齐 outline 引用但尚未存在的 4 张关键图：
  - bellman_gridworld.png       (4×4 GridWorld 的最优 V*(s) 热力图 + 箭头策略)
  - dqn_cartpole_training.png   (DQN 在 CartPole 上的训练曲线 — episode reward + epsilon decay)
  - dqn_training_curve.png      (DQN replay buffer + target network 关键超参的影响)
  - ppo_training_curves.png     (PPO 在多种环境上的 reward 曲线对比)

依赖：numpy, matplotlib
运行：python generate_rl_figures.py
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
# 1. bellman_gridworld.png — Value Iteration 收敛后的 V* + 策略箭头
# ─────────────────────────────────────────────────────────────────────
def bellman_gridworld():
    n = 4
    # 简化 GridWorld：(3,3) 是 +10 终点；(1,1) 是 -10 陷阱；其余 -0.1 步罚
    rewards = -0.1 * np.ones((n, n))
    rewards[3, 3] = 10.0
    rewards[1, 1] = -10.0
    is_terminal = np.zeros((n, n), dtype=bool)
    is_terminal[3, 3] = True
    is_terminal[1, 1] = True

    gamma = 0.9
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
    arrow_chars = ["↑", "↓", "←", "→"]

    V = np.zeros((n, n))
    for _ in range(200):
        V_new = np.copy(V)
        for i in range(n):
            for j in range(n):
                if is_terminal[i, j]:
                    V_new[i, j] = rewards[i, j]
                    continue
                best = -np.inf
                for di, dj in actions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n:
                        v = rewards[i, j] + gamma * V[ni, nj]
                    else:
                        v = rewards[i, j] + gamma * V[i, j]
                    best = max(best, v)
                V_new[i, j] = best
        if np.max(np.abs(V_new - V)) < 1e-6:
            break
        V = V_new

    # 提取最优策略
    policy = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if is_terminal[i, j]:
                continue
            best_v, best_a = -np.inf, 0
            for ai, (di, dj) in enumerate(actions):
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n:
                    v = V[ni, nj]
                else:
                    v = V[i, j]
                if v > best_v:
                    best_v, best_a = v, ai
            policy[i, j] = best_a

    fig, ax = plt.subplots(figsize=(7.5, 7), constrained_layout=True)
    im = ax.imshow(V, cmap="RdYlGn", vmin=-12, vmax=12)
    for i in range(n):
        for j in range(n):
            if is_terminal[i, j]:
                lbl = "+10" if rewards[i, j] > 0 else "-10"
                ax.text(j, i, lbl, ha="center", va="center",
                        fontsize=15, color="black", fontweight="bold")
            else:
                ax.text(j, i, f"V={V[i,j]:.2f}", ha="center", va="center",
                        fontsize=10, color="white" if abs(V[i,j])>3 else "black")
                ax.text(j, i + 0.32, arrow_chars[policy[i, j]],
                        ha="center", va="center", fontsize=22,
                        color="black", fontweight="bold")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_title("4×4 GridWorld — Bellman 最优值 V*(s) + 最优策略 π*(s)\n"
                 f"(γ={gamma}, +10 终点, -10 陷阱, -0.1 步罚)",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="V*(s)", shrink=0.7)
    out = OUT / "bellman_gridworld.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────
# 2. dqn_cartpole_training.png — DQN on CartPole 训练曲线
# ─────────────────────────────────────────────────────────────────────
def dqn_cartpole_training():
    # 模拟 DQN 训练过程的合成曲线
    episodes = np.arange(1, 401)
    # Reward 从 ~20 慢慢爬到 200（CartPole-v1 的上限）
    base = 200 * (1 - np.exp(-0.01 * episodes))
    reward = base + 25 * np.random.randn(len(episodes))
    reward = np.clip(reward, 5, 200)
    smooth = np.convolve(reward, np.ones(20) / 20, mode="same")

    # Epsilon-greedy 衰减
    epsilon = 1.0 * (0.995 ** episodes)
    epsilon = np.maximum(epsilon, 0.05)

    fig, ax1 = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ax1.plot(episodes, reward, color="#bbb", lw=0.6, alpha=0.5,
              label="单 episode 回报")
    ax1.plot(episodes, smooth, color="#27AE60", lw=2.5,
              label="20-ep 滑动平均")
    ax1.axhline(195, ls="--", color="#888", lw=1.0)
    ax1.text(380, 197, "解决阈值 (avg ≥ 195)", fontsize=9, ha="right",
              color="#888")
    ax1.set_xlabel("Episode", fontsize=11)
    ax1.set_ylabel("Episode 回报", color="#27AE60", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#27AE60")
    ax1.set_ylim(0, 220)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left", fontsize=10)

    ax2 = ax1.twinx()
    ax2.plot(episodes, epsilon, color="#C0392B", lw=2.0, ls=":",
              label="ε (探索率)")
    ax2.set_ylabel("ε-greedy 探索率", color="#C0392B", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#C0392B")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="upper right", fontsize=10)

    ax1.set_title("DQN on CartPole-v1 — 训练曲线（reward + 探索率）",
                   fontsize=13, fontweight="bold")

    out = OUT / "dqn_cartpole_training.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────
# 3. dqn_training_curve.png — Replay buffer / target network 消融
# ─────────────────────────────────────────────────────────────────────
def dqn_ablation():
    episodes = np.arange(1, 401)

    def make_curve(scale, noise_level, plateau_at=None):
        base = 200 * (1 - np.exp(-scale * episodes))
        if plateau_at is not None:
            base = np.minimum(base, plateau_at)
        c = base + noise_level * np.random.randn(len(episodes))
        c = np.clip(c, 5, 200)
        return np.convolve(c, np.ones(20) / 20, mode="same")

    full     = make_curve(0.012, 12)
    no_replay = make_curve(0.005, 25, plateau_at=80)
    no_target = make_curve(0.008, 20, plateau_at=150)
    no_both   = make_curve(0.003, 30, plateau_at=50)

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ax.plot(episodes, full, color="#27AE60", lw=2.2,
            label="完整 DQN (replay + target net)")
    ax.plot(episodes, no_replay, color="#E67E22", lw=2.2,
            label="无 replay buffer")
    ax.plot(episodes, no_target, color="#2980B9", lw=2.2,
            label="无 target network")
    ax.plot(episodes, no_both, color="#C0392B", lw=2.2,
            label="无 replay + 无 target ≈ 朴素 Q-learning")

    ax.axhline(195, ls="--", color="#888", lw=1.0)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("平均 episode 回报", fontsize=11)
    ax.set_title("DQN 关键技巧消融 — replay buffer + target network 缺一不可",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 220)

    out = OUT / "dqn_training_curve.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────
# 4. ppo_training_curves.png — PPO 在多种环境的训练曲线
# ─────────────────────────────────────────────────────────────────────
def ppo_curves():
    steps = np.arange(0, 1_000_001, 5000)

    def curve(scale, target, noise=0.05):
        base = target * (1 - np.exp(-scale * steps / 1e6))
        return base * (1 + noise * np.random.randn(len(steps)))

    envs = {
        "CartPole-v1":      ("#27AE60", curve(8.0, 500)),
        "LunarLander-v2":   ("#2980B9", curve(5.0, 250)),
        "BipedalWalker-v3": ("#E67E22", curve(3.0, 300)),
        "Hopper-v4 (MuJoCo)": ("#9B59B6", curve(2.0, 3000)),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (name, (col, vals)) in zip(axes.flatten(), envs.items()):
        smooth = np.convolve(vals, np.ones(15) / 15, mode="same")
        ax.plot(steps / 1000, vals, color=col, alpha=0.3, lw=0.6)
        ax.plot(steps / 1000, smooth, color=col, lw=2.5, label=name)
        ax.set_xlabel("环境步数 (千)", fontsize=10)
        ax.set_ylabel("平均回报", fontsize=10)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
    fig.suptitle("PPO (Schulman 2017) — 在 4 种经典环境上的训练曲线",
                 fontsize=14, fontweight="bold")

    out = OUT / "ppo_training_curves.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


if __name__ == "__main__":
    print("=== L17 supplementary RL figures ===")
    bellman_gridworld()
    dqn_cartpole_training()
    dqn_ablation()
    ppo_curves()
    print("Done.")
