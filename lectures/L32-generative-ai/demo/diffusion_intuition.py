"""L32 Demo: 扩散模型去噪过程可视化

用纯 NumPy 模拟扩散模型的正向加噪 + 反向去噪过程，
可视化"从噪声中逐步恢复结构"的核心直觉。

用法：
    python diffusion_intuition.py               # 生成去噪过程图
    python diffusion_intuition.py --steps 20    # 20 步去噪

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


def create_target_image(size: int = 32) -> np.ndarray:
    """创建一个简单的目标图像（圆形渐变）。"""
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    r = np.sqrt(x**2 + y**2)
    img = np.clip(1 - r, 0, 1)
    return img


def forward_diffusion(img: np.ndarray, n_steps: int = 10) -> list[np.ndarray]:
    """正向扩散：逐步加噪。"""
    steps = [img.copy()]
    beta = np.linspace(0.02, 0.15, n_steps)
    noisy = img.copy()
    for t in range(n_steps):
        noise = np.random.randn(*img.shape) * np.sqrt(beta[t])
        noisy = np.sqrt(1 - beta[t]) * noisy + noise
        steps.append(noisy.copy())
    return steps


def reverse_diffusion_mock(noisy: np.ndarray, target: np.ndarray,
                           n_steps: int = 10) -> list[np.ndarray]:
    """模拟反向去噪（线性插值模拟，非真实模型推理）。"""
    steps = [noisy.copy()]
    for t in range(1, n_steps + 1):
        alpha = t / n_steps
        denoised = (1 - alpha) * noisy + alpha * target
        # 添加递减噪声模拟去噪不完美性
        residual_noise = np.random.randn(*target.shape) * 0.05 * (1 - alpha)
        denoised += residual_noise
        steps.append(denoised.copy())
    return steps


def plot_process(forward_steps: list, reverse_steps: list,
                 save_path: str = "diffusion_process.png"):
    """可视化正向加噪 + 反向去噪过程。"""
    n_show = min(6, len(forward_steps))
    indices_fwd = np.linspace(0, len(forward_steps) - 1, n_show, dtype=int)
    indices_rev = np.linspace(0, len(reverse_steps) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 2.2, 5))

    for i, idx in enumerate(indices_fwd):
        ax = axes[0, i]
        ax.imshow(forward_steps[idx], cmap="viridis", vmin=-0.5, vmax=1.2)
        ax.set_title(f"t={idx}", fontsize=10)
        ax.axis("off")
    axes[0, 0].set_ylabel("正向加噪 →", fontsize=11, fontweight="bold")

    for i, idx in enumerate(indices_rev):
        ax = axes[1, i]
        ax.imshow(reverse_steps[idx], cmap="viridis", vmin=-0.5, vmax=1.2)
        ax.set_title(f"t={len(reverse_steps)-1-idx}", fontsize=10)
        ax.axis("off")
    axes[1, 0].set_ylabel("← 反向去噪", fontsize=11, fontweight="bold")

    fig.suptitle("扩散模型核心直觉：从结构到噪声，再从噪声中恢复结构",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  图已保存: {save_path}")
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="L32: 扩散模型去噪过程可视化")
    ap.add_argument("--steps", type=int, default=10, help="扩散步数")
    ap.add_argument("--size", type=int, default=32, help="图像尺寸")
    args = ap.parse_args()

    print(f"L32 Demo: 扩散模型直觉可视化 (steps={args.steps})")

    target = create_target_image(args.size)
    forward = forward_diffusion(target, args.steps)
    reverse = reverse_diffusion_mock(forward[-1], target, args.steps)

    print(f"  正向: {len(forward)} 步加噪")
    print(f"  反向: {len(reverse)} 步去噪")

    plot_process(forward, reverse)
