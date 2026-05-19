"""
L17 – Reinforcement Learning: Tabular Q-Learning on a 5×5 Grid World
=====================================================================
Agent starts top-left (0,0), goal at bottom-right (4,4).
Three obstacle cells give −1 reward.  Goal gives +10.
Trains with ε-greedy exploration, then visualises the learned Q-values
and the reward curve.

Usage
-----
    python qlearning_gridworld.py
    python qlearning_gridworld.py --episodes 2000 --visualize
    python qlearning_gridworld.py --ablate   # no exploration decay
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

SEED = 1337
np.random.seed(SEED)

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

GRID_H, GRID_W = 5, 5
START = (0, 0)
GOAL = (4, 4)
OBSTACLES = {(1, 1), (2, 3), (3, 1)}
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
ACTION_NAMES = ["↑", "↓", "←", "→"]


def step(state, action_idx):
    dr, dc = ACTIONS[action_idx]
    nr, nc = state[0] + dr, state[1] + dc
    if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
        nr, nc = state
    ns = (nr, nc)
    if ns == GOAL:
        return ns, 10.0, True
    if ns in OBSTACLES:
        return ns, -1.0, False
    return ns, -0.1, False


def train(episodes: int, ablate: bool):
    Q = np.zeros((GRID_H, GRID_W, len(ACTIONS)))
    alpha, gamma = 0.1, 0.99
    eps_start, eps_end = 1.0, 0.05
    rewards_per_ep = []

    for ep in range(episodes):
        eps = eps_start if ablate else max(eps_end,
              eps_start - (eps_start - eps_end) * ep / (episodes * 0.8))
        state = START
        total_reward = 0.0
        for _ in range(200):
            r, c = state
            if np.random.rand() < eps:
                a = np.random.randint(len(ACTIONS))
            else:
                a = int(np.argmax(Q[r, c]))
            ns, reward, done = step(state, a)
            nr, nc = ns
            Q[r, c, a] += alpha * (reward + gamma * np.max(Q[nr, nc]) - Q[r, c, a])
            state = ns
            total_reward += reward
            if done:
                break
        rewards_per_ep.append(total_reward)

    return Q, rewards_per_ep


def plot_q_heatmap(Q):
    best_val = np.max(Q, axis=2)
    best_act = np.argmax(Q, axis=2)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(best_val, cmap="YlGn", origin="upper")
    for r in range(GRID_H):
        for c in range(GRID_W):
            arrow = ACTION_NAMES[best_act[r, c]]
            if (r, c) == GOAL:
                arrow = "★"
            if (r, c) in OBSTACLES:
                arrow = "■"
            ax.text(c, r, f"{arrow}\n{best_val[r, c]:.1f}",
                    ha="center", va="center", fontsize=8)
    ax.set_title("Q-value Heatmap (best action)")
    ax.set_xticks(range(GRID_W))
    ax.set_yticks(range(GRID_H))
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fname = os.path.join(FIG_DIR, "q_heatmap.png")
    fig.savefig(fname, dpi=150)
    print(f"[saved] {fname}")
    return fig


def plot_rewards(rewards):
    fig, ax = plt.subplots(figsize=(7, 3))
    window = max(1, len(rewards) // 50)
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    ax.plot(smoothed, linewidth=0.8)
    ax.set(xlabel="Episode", ylabel="Total Reward", title="Learning Curve")
    fig.tight_layout()
    fname = os.path.join(FIG_DIR, "q_rewards.png")
    fig.savefig(fname, dpi=150)
    print(f"[saved] {fname}")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Q-learning grid-world")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--ablate", action="store_true",
                        help="Disable ε-decay (constant ε=1)")
    args = parser.parse_args()

    Q, rewards = train(args.episodes, args.ablate)
    plot_q_heatmap(Q)
    plot_rewards(rewards)

    if args.visualize:
        plt.show()


if __name__ == "__main__":
    main()
