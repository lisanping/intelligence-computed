"""Diffusion denoising step animation — synthetic.

Generates two outputs:
  - diffusion_denoising_animation.gif   20-frame animation of denoising
  - diffusion_denoising_grid.png        9-step strip showing forward + reverse

Pure NumPy + matplotlib (no diffusion model needed). Visualizes the
forward "ink-spreading" process and the reverse "ink-recovering" process
that is the core intuition of DDPM (Ho 2020).

Run:  python diffusion_step_animation.py
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'PingFang SC',
    'Microsoft JhengHei', 'Segoe UI Emoji', 'DejaVu Sans',
]
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)


def synth_clean(size: int = 64) -> np.ndarray:
    """A simple recognizable clean image: stylized 'face'."""
    img = np.full((size, size), 0.85, dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    # head
    head = ((xx - 32) ** 2 + (yy - 32) ** 2) < 22 ** 2
    img[head] = 0.55
    # eyes
    le = ((xx - 25) ** 2 + (yy - 26) ** 2) < 3 ** 2
    re = ((xx - 39) ** 2 + (yy - 26) ** 2) < 3 ** 2
    img[le] = 0.10
    img[re] = 0.10
    # mouth
    mouth = (np.abs(yy - 40) < 1.5) & (np.abs(xx - 32) < 8)
    img[mouth] = 0.10
    return img


def forward_step(x: np.ndarray, t: int, T: int,
                 rng: np.random.Generator) -> np.ndarray:
    """One forward diffusion step. Linear beta schedule."""
    beta = 1e-4 + (0.02 - 1e-4) * t / T
    return np.sqrt(1 - beta) * x + np.sqrt(beta) * rng.standard_normal(x.shape)


def make_forward_chain(clean: np.ndarray, T: int = 200,
                       seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = [clean.copy()]
    x = clean.copy()
    for t in range(1, T + 1):
        x = forward_step(x, t, T, rng)
        out.append(x.copy())
    return out


def make_reverse_chain(forward: list[np.ndarray]) -> list[np.ndarray]:
    """Reverse = play forward backwards (idealized 'oracle' denoiser)."""
    return list(reversed(forward))


def grid_strip() -> None:
    clean = synth_clean()
    fwd = make_forward_chain(clean, T=200, seed=1)
    snaps = [0, 25, 75, 150, 200]
    fig, axes = plt.subplots(2, len(snaps), figsize=(13, 5.5))
    titles_fwd = [f"t={t}" for t in snaps]
    for i, t in enumerate(snaps):
        axes[0, i].imshow(fwd[t], cmap="gray", vmin=-0.5, vmax=1.2)
        axes[0, i].set_title(titles_fwd[i], fontsize=10)
        axes[0, i].axis("off")
    axes[0, 0].text(-0.3, 0.5, "Forward\n(加噪)", transform=axes[0, 0].transAxes,
                    rotation=90, ha="right", va="center", fontsize=12,
                    color="#D62728")

    rev = make_reverse_chain(fwd)
    rsnaps = [0, 50, 125, 175, 200]
    for i, t in enumerate(rsnaps):
        axes[1, i].imshow(rev[t], cmap="gray", vmin=-0.5, vmax=1.2)
        axes[1, i].set_title(f"step {i+1}/5", fontsize=10)
        axes[1, i].axis("off")
    axes[1, 0].text(-0.3, 0.5, "Reverse\n(去噪)", transform=axes[1, 0].transAxes,
                    rotation=90, ha="right", va="center", fontsize=12,
                    color="#2CA02C")

    fig.suptitle(
        "Diffusion 直觉：加噪是物理的，去噪是机器学到的逆过程\n"
        "（DDPM Ho 2020 训练神经网络逐步预测噪声并减去）", fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUT, "diffusion_denoising_grid.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def animation_gif() -> None:
    clean = synth_clean()
    fwd = make_forward_chain(clean, T=200, seed=2)
    rev = make_reverse_chain(fwd)
    frames = (fwd[::10] + rev[::10])

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(frames[0], cmap="gray", vmin=-0.5, vmax=1.2,
                   animated=True)
    title = ax.set_title("Frame 1", fontsize=12)
    ax.axis("off")

    def update(i):
        im.set_array(frames[i])
        if i < len(fwd[::10]):
            title.set_text(f"Forward (加噪)  ·  t={i*10}")
        else:
            j = i - len(fwd[::10])
            title.set_text(f"Reverse (去噪)  ·  step {j+1}")
        return im, title

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=180, blit=False, repeat=True)
    out = os.path.join(OUT, "diffusion_denoising_animation.gif")
    anim.save(out, writer=PillowWriter(fps=6))
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    grid_strip()
    animation_gif()


if __name__ == "__main__":
    main()
