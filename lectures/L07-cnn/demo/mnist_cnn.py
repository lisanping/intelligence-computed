"""
第 7 讲 · 动手环节 · PyTorch MNIST CNN 训练 + Feature Map 可视化
================================================================

目标：用一个小 CNN 训练 MNIST，打印逐 epoch 准确率，保存 Feature Map 可视化。
     内置消融开关验证每个组件的贡献。

运行：
    python mnist_cnn.py                        # 基线
    python mnist_cnn.py --ablate no_pool       # 去掉池化层
    python mnist_cnn.py --ablate fc_only       # 全连接网络（无卷积）
    python mnist_cnn.py --ablate single_conv   # 只用一层卷积
    python mnist_cnn.py --ablate no_relu       # 去掉 ReLU 激活

所有实验使用 seed=1337。
依赖：torch torchvision matplotlib numpy
"""
from __future__ import annotations

import argparse
import os

# torch must be imported before matplotlib on Windows to avoid OpenMP/DLL
# conflicts (libiomp5md.dll vs libomp.dll, plus shm.dll load order).
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[4] / "design" / "meta"))
import cjk_font  # noqa: F401  - applies CJK font config
import matplotlib.pyplot as plt
import numpy as np

# ── 固定随机种子 ─────────────────────────────────────────────
SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── 命令行参数 ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="L07 MNIST CNN 训练 + Feature Map 可视化")
parser.add_argument(
    "--ablate", type=str, default=None,
    choices=["no_pool", "fc_only", "single_conv", "no_relu"],
    help="消融实验开关",
)
parser.add_argument("--epochs", type=int, default=5, help="训练 epoch 数")
parser.add_argument("--batch_size", type=int, default=64, help="batch 大小")
parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
parser.add_argument("--save_figures", action="store_true", default=True, help="保存可视化到 figures/")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# ═══════════════════════════════════════════════════════════
# 1. 数据加载（MNIST 自动下载）
# ═══════════════════════════════════════════════════════════
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_set = datasets.MNIST("data", train=True, download=True, transform=transform)
test_set = datasets.MNIST("data", train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)


# ═══════════════════════════════════════════════════════════
# 2. 模型定义
# ═══════════════════════════════════════════════════════════
class MNISTConvNet(nn.Module):
    """基线 CNN：Conv→ReLU→Pool→Conv→ReLU→Pool→FC→FC"""

    def __init__(self, ablate: str | None = None):
        super().__init__()
        self.ablate = ablate
        act = nn.Identity if ablate == "no_relu" else nn.ReLU

        if ablate == "fc_only":
            # 消融：纯全连接，无卷积
            self.features = nn.Flatten()
            self.classifier = nn.Sequential(
                nn.Linear(28 * 28, 256), act(), nn.Linear(256, 128), act(), nn.Linear(128, 10),
            )
        elif ablate == "single_conv":
            # 消融：只用一层卷积
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), act(), nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(32 * 14 * 14, 128), act(), nn.Linear(128, 10),
            )
        elif ablate == "no_pool":
            # 消融：去掉池化
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), act(),  # (16, 28, 28)
                nn.Conv2d(16, 32, 3, padding=1), act(),  # (32, 28, 28)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(32 * 28 * 28, 128), act(), nn.Linear(128, 10),
            )
        else:
            # 基线 / no_relu
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), act(), nn.MaxPool2d(2),   # (16, 14, 14)
                nn.Conv2d(16, 32, 3, padding=1), act(), nn.MaxPool2d(2),  # (32, 7, 7)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ═══════════════════════════════════════════════════════════
# 3. 训练 & 评估
# ═══════════════════════════════════════════════════════════
def train_one_epoch(model: nn.Module, loader, optimizer, epoch: int):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(data), target)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model: nn.Module, loader) -> float:
    model.eval()
    correct = 0
    total = 0
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        pred = model(data).argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    return correct / total


# ═══════════════════════════════════════════════════════════
# 4. Feature Map 可视化
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def visualize_feature_maps(model: nn.Module, test_loader, save_dir: str):
    """提取并可视化 CNN 各层的 Feature Map。"""
    model.eval()
    data, _ = next(iter(test_loader))
    img = data[0:1].to(DEVICE)  # 取第一张测试图 (1, 1, 28, 28)

    if model.ablate == "fc_only":
        print("[可视化] fc_only 模式无卷积层，跳过 Feature Map 可视化。")
        return

    # 逐层提取
    activations = []
    layer_names = []
    x = img
    for name, layer in model.features.named_children():
        x = layer(x)
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.ReLU)):
            activations.append(x.cpu().squeeze(0))  # (C, H, W)
            layer_names.append(f"{name}: {layer.__class__.__name__}")

    if not activations:
        return

    os.makedirs(save_dir, exist_ok=True)

    # 绘制：原始输入 + 每层 Feature Map
    n_layers = len(activations)
    fig, axes = plt.subplots(n_layers + 1, 1, figsize=(12, 3 * (n_layers + 1)))
    if n_layers == 0:
        return

    # 原始输入
    axes[0].imshow(img.cpu().squeeze(), cmap="gray")
    axes[0].set_title("Input Image", fontsize=12)
    axes[0].axis("off")

    for i, (act, name) in enumerate(zip(activations, layer_names)):
        n_channels = min(act.shape[0], 16)  # 最多显示 16 个通道
        grid = act[:n_channels].numpy()
        # 拼成一行
        row = np.concatenate([grid[c] for c in range(n_channels)], axis=1)
        axes[i + 1].imshow(row, cmap="viridis")
        axes[i + 1].set_title(f"Layer {name} ({act.shape[0]} channels, showing {n_channels})", fontsize=10)
        axes[i + 1].axis("off")

    plt.tight_layout()
    path = os.path.join(save_dir, "feature_maps.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] Feature Map 已保存到 {path}")


# ═══════════════════════════════════════════════════════════
# 5. 主流程
# ═══════════════════════════════════════════════════════════
def main():
    ablate_name = args.ablate or "baseline"
    print(f"{'='*50}")
    print(f"L07 MNIST CNN · 配置: {ablate_name}")
    print(f"{'='*50}")

    model = MNISTConvNet(ablate=args.ablate).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"参数量: {param_count:,}")
    print(f"设备: {DEVICE}")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, epoch)
        acc = evaluate(model, test_loader)
        print(f"  Epoch {epoch}/{args.epochs}  测试准确率: {acc:.4f} ({acc*100:.2f}%)")

    print()
    final_acc = evaluate(model, test_loader)
    print(f"最终测试准确率: {final_acc:.4f} ({final_acc*100:.2f}%)")

    # 保存模型
    os.makedirs(FIGURES_DIR, exist_ok=True)
    model_path = os.path.join(FIGURES_DIR, f"mnist_cnn_{ablate_name}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")

    # Feature Map 可视化
    if args.save_figures:
        visualize_feature_maps(model, test_loader, FIGURES_DIR)

    print(f"\n{'='*50}")
    print(f"配置 [{ablate_name}] 完成 · 准确率 {final_acc*100:.2f}% · 参数 {param_count:,}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()