"""
第 7 讲 · Feature Map 可视化独立脚本
====================================

加载已训练的 MNIST CNN 模型，将一张测试图逐层通过网络，
可视化每一层的 Feature Map 并保存到 figures/。

运行：
    python feature_map_viz.py                           # 默认加载基线模型
    python feature_map_viz.py --model figures/mnist_cnn_baseline.pt
    python feature_map_viz.py --digit 3                 # 指定可视化的数字

前置：先运行 mnist_cnn.py 生成模型文件。
依赖：torch torchvision matplotlib numpy
"""
from __future__ import annotations

import argparse
import os

# torch must be imported before matplotlib on Windows to avoid OpenMP/DLL
# conflicts (libiomp5md.dll vs libomp.dll, plus shm.dll load order).
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[4] / "design" / "meta"))
import cjk_font  # noqa: F401  - applies CJK font config
import matplotlib.pyplot as plt
import numpy as np

# ── 固定随机种子 ─────────────────────────────────────────────
SEED = 1337
torch.manual_seed(SEED)

# ── 命令行参数 ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="L07 Feature Map 可视化")
parser.add_argument("--model", type=str, default=None, help="模型文件路径（默认 figures/mnist_cnn_baseline.pt）")
parser.add_argument("--digit", type=int, default=None, help="指定可视化的数字 (0-9)")
parser.add_argument("--n_images", type=int, default=3, help="可视化几张不同的图")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# ═══════════════════════════════════════════════════════════
# 1. 模型定义（与 mnist_cnn.py 中的基线一致）
# ═══════════════════════════════════════════════════════════
class MNISTConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return self.classifier(x)

    def get_layer_activations(self, x: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """逐层提取激活值，返回 [(层名, 激活张量), ...]"""
        activations = []
        x = self.conv1(x)
        activations.append(("Conv1 (16 filters, 3×3)", x))
        x = self.relu1(x)
        activations.append(("ReLU1", x))
        x = self.pool1(x)
        activations.append(("MaxPool1 (2×2)", x))
        x = self.conv2(x)
        activations.append(("Conv2 (32 filters, 3×3)", x))
        x = self.relu2(x)
        activations.append(("ReLU2", x))
        x = self.pool2(x)
        activations.append(("MaxPool2 (2×2)", x))
        return activations


# ═══════════════════════════════════════════════════════════
# 2. 加载模型 & 数据
# ═══════════════════════════════════════════════════════════
model_path = args.model or os.path.join(FIGURES_DIR, "mnist_cnn_baseline.pt")
if not os.path.exists(model_path):
    print(f"[错误] 找不到模型文件: {model_path}")
    print("请先运行: python mnist_cnn.py")
    exit(1)

model = MNISTConvNet().to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
model.eval()
print(f"已加载模型: {model_path}")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
test_set = datasets.MNIST("data", train=False, download=True, transform=transform)


# ═══════════════════════════════════════════════════════════
# 3. 选择测试图像
# ═══════════════════════════════════════════════════════════
def find_images(dataset, digit: int | None, n: int) -> list[tuple[torch.Tensor, int]]:
    """从测试集找 n 张指定数字的图片（digit=None 则随机选）。"""
    results = []
    indices = torch.randperm(len(dataset)).tolist()
    for idx in indices:
        img, label = dataset[idx]
        if digit is not None and label != digit:
            continue
        results.append((img, label))
        if len(results) >= n:
            break
    return results


images = find_images(test_set, args.digit, args.n_images)
if not images:
    print(f"[错误] 找不到数字 {args.digit} 的测试图片")
    exit(1)


# ═══════════════════════════════════════════════════════════
# 4. 可视化
# ═══════════════════════════════════════════════════════════
os.makedirs(FIGURES_DIR, exist_ok=True)

for img_idx, (img, label) in enumerate(images):
    x = img.unsqueeze(0).to(DEVICE)  # (1, 1, 28, 28)

    with torch.no_grad():
        activations = model.get_layer_activations(x)

    n_layers = len(activations)
    fig, axes = plt.subplots(n_layers + 1, 1, figsize=(14, 2.5 * (n_layers + 1)))

    # 原始输入
    axes[0].imshow(img.squeeze().numpy(), cmap="gray")
    axes[0].set_title(f"Input: digit '{label}' (28×28)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    for i, (name, act) in enumerate(activations):
        act_np = act.cpu().squeeze(0).numpy()  # (C, H, W)
        n_show = min(act_np.shape[0], 16)
        # 每个通道归一化到 [0, 1] 再拼接
        channels = []
        for c in range(n_show):
            ch = act_np[c]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max - ch_min > 1e-6:
                ch = (ch - ch_min) / (ch_max - ch_min)
            channels.append(ch)
        # 用小间隔分隔通道
        sep = np.ones((act_np.shape[1], 1)) * 0.5
        row = channels[0]
        for c in range(1, n_show):
            row = np.concatenate([row, sep, channels[c]], axis=1)

        axes[i + 1].imshow(row, cmap="viridis", vmin=0, vmax=1)
        h, w = act_np.shape[1], act_np.shape[2]
        axes[i + 1].set_title(
            f"Layer: {name}  |  shape: ({act_np.shape[0]}, {h}, {w})  |  showing {n_show} channels",
            fontsize=10,
        )
        axes[i + 1].axis("off")

    plt.suptitle(f"Feature Map Visualization — Digit '{label}'", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, f"feature_maps_digit{label}_{img_idx}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[已保存] {path}")

print(f"\n共生成 {len(images)} 张 Feature Map 可视化。")