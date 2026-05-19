"""
L17 – Reinforcement Learning: DQN on CartPole-v1
=================================================
Deep Q-Network with experience replay and a target network, implemented
in PyTorch.  Trains on OpenAI Gym's CartPole-v1 and plots the reward
curve.

Usage
-----
    python dqn_cartpole.py
    python dqn_cartpole.py --episodes 500
"""

import argparse
import os
import random
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------- lazy imports (heavy libs) -------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

torch.manual_seed(SEED)

# ---------- hyper-parameters ----------------------------------------------
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 10      # episodes between target-net sync
MEMORY_SIZE = 10_000
HIDDEN = 128


# ---------- Q-network ----------------------------------------------------

class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, act_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------- replay buffer -------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (torch.FloatTensor(np.array(s)),
                torch.LongTensor(a),
                torch.FloatTensor(r),
                torch.FloatTensor(np.array(s2)),
                torch.FloatTensor(d))

    def __len__(self):
        return len(self.buf)


# ---------- agent ---------------------------------------------------------

class DQNAgent:
    def __init__(self, obs_dim: int, act_dim: int):
        self.act_dim = act_dim
        self.policy = QNet(obs_dim, act_dim)
        self.target = QNet(obs_dim, act_dim)
        self.target.load_state_dict(self.policy.state_dict())
        self.opt = optim.Adam(self.policy.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.eps = EPS_START

    def select_action(self, state):
        if random.random() < self.eps:
            return random.randrange(self.act_dim)
        with torch.no_grad():
            return int(self.policy(torch.FloatTensor(state)).argmax())

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return
        s, a, r, s2, d = self.memory.sample(BATCH_SIZE)
        q = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target(s2).max(1)[0]
            target = r + GAMMA * q_next * (1 - d)
        loss = nn.MSELoss()(q, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def sync_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    def decay_eps(self):
        self.eps = max(EPS_END, self.eps * EPS_DECAY)


# ---------- training loop -------------------------------------------------

def train(episodes: int):
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    agent = DQNAgent(obs_dim, act_dim)
    rewards = []

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=SEED + ep)
        total = 0.0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.memory.push(state, action, reward, next_state, float(done))
            agent.update()
            state = next_state
            total += reward
        agent.decay_eps()
        if ep % TARGET_UPDATE == 0:
            agent.sync_target()
        rewards.append(total)
        if ep % 50 == 0:
            avg = np.mean(rewards[-50:])
            print(f"Episode {ep:4d}  reward {total:6.1f}  avg50 {avg:6.1f}  ε {agent.eps:.3f}")

    env.close()
    return rewards


def plot_rewards(rewards):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(rewards, alpha=0.3, linewidth=0.6, label="per episode")
    window = max(1, len(rewards) // 20)
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    ax.plot(range(window - 1, len(rewards)), smoothed, linewidth=1.5, label="smoothed")
    ax.set(xlabel="Episode", ylabel="Reward", title="DQN – CartPole-v1")
    ax.legend()
    fig.tight_layout()
    fname = os.path.join(FIG_DIR, "dqn_cartpole.png")
    fig.savefig(fname, dpi=150)
    print(f"[saved] {fname}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="DQN CartPole demo")
    parser.add_argument("--episodes", type=int, default=300)
    args = parser.parse_args()

    rewards = train(args.episodes)
    plot_rewards(rewards)


if __name__ == "__main__":
    main()
