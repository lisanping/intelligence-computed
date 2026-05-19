"""
L35 - Multimodal - CLIP vector space + contrastive matrix visualization
=============================================================
Goal   : Generate CLIP-related visualizations without requiring a GPU
         or actual model weights -- uses synthetic data to illustrate concepts
Figures:
  figures/clip_contrastive_matrix.png  -- CLIP contrastive learning matrix (V1)
  figures/clip_vector_space.png        -- t-SNE projection of image/text embeddings (V2)
  figures/diffusion_steps.png          -- Forward diffusion noise schedule (V3)
  figures/cfg_scale_ablation.png       -- CFG scale effect on generation quality (V8)
Run    : python demo/clip_space_viz.py [--fig matrix|tsne|diffusion|cfg|all]
Deps   : pip install matplotlib numpy
Seed   : 1337
"""
import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(encoding="utf-8")

np.random.seed(1337)
FIG_DIR = Path(__file__).parent / "figures"


# -- V1: CLIP contrastive matrix ---------------------------------
def plot_contrastive_matrix(save: bool = True):
    """NxN image-text similarity matrix -- diagonal = positive pairs."""
    n = 6
    labels_img = [f"img_{i}" for i in range(n)]
    labels_txt = ["a dog", "sunset", "car", "cat", "mountain", "food"]

    # Simulate cosine similarity: high on diagonal, low off-diagonal
    sim = np.random.uniform(0.05, 0.35, (n, n))
    np.fill_diagonal(sim, np.random.uniform(0.75, 0.95, n))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            color = "white" if sim[i, j] > 0.6 else "black"
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                    fontsize=10, color=color, weight="bold" if i == j else "normal")

    ax.set_xticks(range(n)); ax.set_xticklabels(labels_txt, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels_img, fontsize=9)
    ax.set_xlabel("Text Embeddings", fontsize=12)
    ax.set_ylabel("Image Embeddings", fontsize=12)
    ax.set_title("CLIP Contrastive Learning: Image-Text Similarity Matrix",
                 fontsize=13, weight="bold")
    fig.colorbar(im, ax=ax, label="Cosine Similarity", shrink=0.8)
    ax.text(n - 0.5, -0.8, "Diagonal = positive pairs (maximize)\nOff-diagonal = negative pairs (minimize)",
            fontsize=9, ha="right", style="italic")
    plt.tight_layout()
    _save(fig, "clip_contrastive_matrix.png", save)


# -- V2: t-SNE vector space visualization -------------------------
def plot_tsne_space(save: bool = True):
    """Synthetic t-SNE showing image & text embeddings clustering."""
    categories = ["dog", "car", "food", "mountain", "sunset"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
    n_per = 8

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (cat, col) in enumerate(zip(categories, colors)):
        cx, cy = np.random.uniform(-3, 3), np.random.uniform(-3, 3)
        # Image cluster
        ix = cx + np.random.normal(0, 0.5, n_per)
        iy = cy + np.random.normal(0, 0.5, n_per)
        ax.scatter(ix, iy, c=col, marker="o", s=60, alpha=0.7, edgecolors="white", lw=0.5)
        # Text cluster (nearby but slightly offset)
        tx = cx + 0.3 + np.random.normal(0, 0.4, n_per)
        ty = cy + 0.3 + np.random.normal(0, 0.4, n_per)
        ax.scatter(tx, ty, c=col, marker="^", s=60, alpha=0.7, edgecolors="white", lw=0.5)
        ax.annotate(cat, xy=(cx, cy), fontsize=11, weight="bold", color=col,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    # Legend
    ax.scatter([], [], c="gray", marker="o", s=60, label="Image embedding")
    ax.scatter([], [], c="gray", marker="^", s=60, label="Text embedding")
    ax.legend(fontsize=11, loc="upper right")
    ax.set_title("CLIP: Image & Text Embeddings in Shared Vector Space (t-SNE)",
                 fontsize=13, weight="bold")
    ax.set_xlabel("t-SNE dim 1", fontsize=11)
    ax.set_ylabel("t-SNE dim 2", fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    _save(fig, "clip_vector_space.png", save)


# -- V3: Forward diffusion steps ----------------------------------
def plot_diffusion_steps(save: bool = True):
    """Visualize forward diffusion: image -> noise over T steps."""
    T = 6
    fig, axes = plt.subplots(1, T, figsize=(14, 2.5))

    # Create a simple synthetic "image" (gradient pattern)
    x = np.linspace(0, 1, 64)
    y = np.linspace(0, 1, 64)
    xx, yy = np.meshgrid(x, y)
    clean = np.sin(4 * np.pi * xx) * np.cos(4 * np.pi * yy)

    betas = np.linspace(0.0, 1.0, T)
    for t, (ax, beta) in enumerate(zip(axes, betas)):
        noisy = clean * (1 - beta) + np.random.randn(64, 64) * beta
        ax.imshow(noisy, cmap="gray", vmin=-1.5, vmax=1.5)
        ax.set_title(f"t={t}", fontsize=10, weight="bold")
        ax.axis("off")
        if t == 0:
            ax.set_xlabel("Clean", fontsize=9)
        elif t == T - 1:
            ax.set_xlabel("Pure noise", fontsize=9)

    fig.suptitle("Forward Diffusion: Clean Image -> Gaussian Noise",
                 fontsize=13, weight="bold", y=1.05)
    plt.tight_layout()
    _save(fig, "diffusion_steps.png", save)


# -- V8: CFG scale ablation ----------------------------------------
def plot_cfg_ablation(save: bool = True):
    """Simulated CFG scale effect on quality metrics."""
    cfg_scales = [1.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]
    # Simulated: quality peaks around 7.5, drops at extremes
    fidelity = [30, 55, 72, 85, 82, 65, 45]
    diversity = [95, 85, 75, 60, 45, 25, 15]
    clip_score = [40, 60, 78, 88, 86, 70, 50]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(cfg_scales, fidelity, "b-o", lw=2, label="Fidelity (FID-like)")
    ax.plot(cfg_scales, diversity, "r--s", lw=2, label="Diversity")
    ax.plot(cfg_scales, clip_score, "g-^", lw=2, label="CLIP Score")
    ax.axvline(7.5, color="orange", ls=":", lw=2, alpha=0.7)
    ax.annotate("Sweet spot\n(cfg=7.5)", xy=(7.5, 88), xytext=(10, 92),
                fontsize=11, color="orange", weight="bold",
                arrowprops=dict(arrowstyle="->", color="orange", lw=2))
    ax.set_xlabel("CFG Scale", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Classifier-Free Guidance: Scale Ablation",
                 fontsize=13, weight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 22); ax.set_ylim(0, 105)
    plt.tight_layout()
    _save(fig, "cfg_scale_ablation.png", save)


def _save(fig, name, save):
    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
        print(f"  saved -> {FIG_DIR / name}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="L29 Multimodal visualizations")
    p.add_argument("--fig", choices=["matrix", "tsne", "diffusion", "cfg", "all"],
                   default="all")
    args = p.parse_args()
    dispatch = dict(matrix=plot_contrastive_matrix, tsne=plot_tsne_space,
                    diffusion=plot_diffusion_steps, cfg=plot_cfg_ablation)
    if args.fig == "all":
        print("L29 Multimodal -- generating all figures ...")
        for name, fn in dispatch.items():
            print(f"[{name}]")
            fn()
        print("Done.")
    else:
        dispatch[args.fig]()


if __name__ == "__main__":
    main()
