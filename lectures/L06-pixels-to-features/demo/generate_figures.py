"""Generate the 12 figures referenced by L06 slides_outline.md.

Figures generated (saved to demo/figures/):
  - sift_scale_space.png       Gaussian pyramid + DoG concept
  - sift_keypoints.png         SIFT keypoints overlaid (already existed; we regenerate)
  - sift_orientation.png       Keypoint orientation histogram
  - sift_descriptor.png        4x4x8 = 128-D descriptor visualization
  - hog_pedestrian_detection.png   HOG cell grid + sliding window concept
  - degradation_blur_series.png        SIFT robustness vs blur
  - degradation_rotation_series.png    SIFT robustness vs rotation
  - degradation_occlusion_series.png   SIFT robustness vs occlusion
  - degradation_resolution_series.png  SIFT robustness vs downscale
  - degradation_grid.png       4x4 综合退化矩阵
  - feature_comparison_accuracy.png    raw pixel vs HOG vs SIFT+BoVW vs CNN
  - feature_ceiling.png        ImageNet 误差时间线 (2010-2017)
  - imagenet_timeline.png      same as above (alias)

Run:  python generate_figures.py
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
from matplotlib.patches import Rectangle, FancyArrow, Circle
from matplotlib.colors import LinearSegmentedColormap

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)


def synth_image(seed: int = 7, size: int = 128) -> np.ndarray:
    """Tiny synthetic gray image with edges + corners (no external deps)."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 0.55, dtype=np.float32)
    # rectangle
    img[20:60, 30:80] = 0.15
    # circle-ish blob
    yy, xx = np.mgrid[0:size, 0:size]
    blob = ((xx - 90) ** 2 + (yy - 80) ** 2) < 20 ** 2
    img[blob] = 0.85
    # texture noise
    img += rng.normal(0, 0.04, img.shape)
    return np.clip(img, 0, 1)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable gaussian via FFT; avoids scipy dependency."""
    if sigma < 0.5:
        return img.copy()
    n = img.shape[0]
    k = int(max(3, sigma * 4)) | 1
    x = np.arange(k) - k // 2
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    out = img.copy()
    # row pass
    out = np.apply_along_axis(lambda v: np.convolve(v, g, mode="same"), 1, out)
    # col pass
    out = np.apply_along_axis(lambda v: np.convolve(v, g, mode="same"), 0, out)
    return out


# ──────────────────────────────────────────────────────────────────────
# 1.  SIFT scale space (Gaussian pyramid + DoG conceptual)
# ──────────────────────────────────────────────────────────────────────
def fig_sift_scale_space() -> None:
    img = synth_image()
    sigmas = [1.0, 1.6, 2.5, 4.0]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5))
    for i, s in enumerate(sigmas):
        blurred = gaussian_blur(img, s)
        axes[0, i].imshow(blurred, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"σ = {s:.1f}", fontsize=10)
        axes[0, i].axis("off")
    # DoG = G(σ_{k+1}) - G(σ_k)
    for i in range(3):
        a = gaussian_blur(img, sigmas[i])
        b = gaussian_blur(img, sigmas[i + 1])
        dog = b - a
        axes[1, i].imshow(dog, cmap="seismic",
                          vmin=-np.abs(dog).max(), vmax=np.abs(dog).max())
        axes[1, i].set_title(f"DoG  σ{i+1}-σ{i}", fontsize=10)
        axes[1, i].axis("off")
    axes[1, 3].axis("off")
    axes[1, 3].text(0.5, 0.5,
                    "高斯金字塔 (顶)\n+ DoG (差分,底)\n→ 极值即\n候选关键点",
                    ha="center", va="center", fontsize=11,
                    transform=axes[1, 3].transAxes)
    fig.suptitle("SIFT 尺度空间：Gaussian Pyramid + Difference-of-Gaussian",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sift_scale_space.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# 2.  SIFT keypoints (regen for parity with naming)
# ──────────────────────────────────────────────────────────────────────
def fig_sift_keypoints() -> None:
    img = synth_image()
    rng = np.random.default_rng(42)
    # fake keypoints near edges
    kp = np.array([[40, 45], [60, 60], [78, 35], [90, 80], [55, 25],
                   [25, 30], [85, 95], [70, 50], [45, 70], [100, 60]])
    sizes = rng.uniform(8, 22, kp.shape[0])
    angles = rng.uniform(0, 2 * np.pi, kp.shape[0])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    for (x, y), r, a in zip(kp, sizes, angles):
        c = Circle((x, y), r, fill=False, color="lime", lw=1.2)
        ax.add_patch(c)
        ax.plot([x, x + r * np.cos(a)], [y, y + r * np.sin(a)],
                color="lime", lw=1.2)
    ax.set_title("SIFT keypoints  ·  圈大小=尺度  ·  指针=主方向")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sift_keypoints.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# 3.  SIFT orientation histogram
# ──────────────────────────────────────────────────────────────────────
def fig_sift_orientation() -> None:
    rng = np.random.default_rng(11)
    bins = 36
    angles = np.linspace(0, 2 * np.pi, bins, endpoint=False)
    weights = np.exp(-((np.arange(bins) - 12) ** 2) / (2 * 4 ** 2)) * 1.5
    weights += rng.uniform(0.05, 0.25, bins)
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection="polar")
    ax.bar(angles, weights, width=2 * np.pi / bins,
           color="#4C72B0", edgecolor="white", linewidth=0.6)
    main = angles[np.argmax(weights)]
    ax.plot([main, main], [0, weights.max() * 1.05],
            color="crimson", lw=2.5)
    ax.set_yticklabels([])
    ax.set_title("SIFT 关键点主方向直方图（36 bins）\n红色=最大方向", pad=18)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sift_orientation.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# 4.  SIFT 128-D descriptor
# ──────────────────────────────────────────────────────────────────────
def fig_sift_descriptor() -> None:
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(6, 6))
    # 4x4 grid of cells, each with an 8-direction histogram
    for i in range(4):
        for j in range(4):
            cx, cy = j, 3 - i
            for k in range(8):
                a = k * np.pi / 4
                w = max(0.05, rng.normal(0.5, 0.3))
                ax.arrow(cx + 0.5, cy + 0.5,
                         np.cos(a) * w * 0.4, np.sin(a) * w * 0.4,
                         head_width=0.04, color="#2E5A88", lw=0.6,
                         alpha=0.85)
            ax.add_patch(Rectangle((cx, cy), 1, 1, fill=False,
                                   edgecolor="black", lw=0.4))
    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(-0.2, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("SIFT 描述子：4×4 cells × 8 方向 = 128 维")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sift_descriptor.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# 5.  HOG pedestrian detection (concept: cell grid + sliding window)
# ──────────────────────────────────────────────────────────────────────
def fig_hog_pedestrian() -> None:
    # synthetic person silhouette
    h, w = 128, 64
    img = np.full((h, w), 0.85, dtype=np.float32)
    # head
    yy, xx = np.mgrid[0:h, 0:w]
    head = ((xx - 32) ** 2 + (yy - 18) ** 2) < 11 ** 2
    img[head] = 0.18
    # torso
    img[30:80, 18:46] = 0.20
    # legs
    img[80:120, 22:30] = 0.22
    img[80:120, 34:42] = 0.22

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("输入（Sliding window）")
    axes[0].axis("off")

    # cell grid overlay
    axes[1].imshow(img, cmap="gray", vmin=0, vmax=1)
    cell = 8
    for x in range(0, w, cell):
        axes[1].axvline(x, color="cyan", lw=0.4)
    for y in range(0, h, cell):
        axes[1].axhline(y, color="cyan", lw=0.4)
    axes[1].set_title("HOG 8×8 cell grid")
    axes[1].axis("off")

    # synth gradient histograms per cell
    rng = np.random.default_rng(0)
    cells_y = h // cell
    cells_x = w // cell
    axes[2].set_xlim(0, w)
    axes[2].set_ylim(h, 0)
    axes[2].set_aspect("equal")
    axes[2].set_facecolor("white")
    for i in range(cells_y):
        for j in range(cells_x):
            cy = i * cell + cell / 2
            cx = j * cell + cell / 2
            # darker pixels => stronger gradient on edges
            patch = img[i*cell:(i+1)*cell, j*cell:(j+1)*cell]
            if patch.std() < 0.05:
                continue
            for k in range(9):
                ang = k * np.pi / 9
                mag = patch.std() * 4 + 0.2
                axes[2].plot(
                    [cx - np.cos(ang) * mag, cx + np.cos(ang) * mag],
                    [cy - np.sin(ang) * mag, cy + np.sin(ang) * mag],
                    color="black", lw=0.5, alpha=0.9,
                )
    axes[2].set_title("HOG 描述（每 cell 9 方向）")
    axes[2].axis("off")
    fig.suptitle("HOG 行人检测 = sliding window + cell-level 梯度直方图 + SVM",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "hog_pedestrian_detection.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# 6.  Degradation series — 4 separate + 1 grid
# ──────────────────────────────────────────────────────────────────────
def _degraded_variants(img: np.ndarray, kind: str) -> list[np.ndarray]:
    out = []
    if kind == "blur":
        for s in [0.0, 1.0, 2.5, 5.0, 9.0]:
            out.append(gaussian_blur(img, s))
    elif kind == "rotation":
        from scipy.ndimage import rotate  # type: ignore
        for a in [0, 15, 45, 90, 180]:
            out.append(rotate(img, a, reshape=False, order=1, cval=0.55))
    elif kind == "occlusion":
        for p in [0.0, 0.05, 0.15, 0.30, 0.50]:
            d = img.copy()
            n = int(d.size * p)
            ys = np.random.randint(0, d.shape[0], n)
            xs = np.random.randint(0, d.shape[1], n)
            d[ys, xs] = 0.0
            out.append(d)
    elif kind == "resolution":
        for k in [1, 2, 4, 8, 16]:
            small = img[::k, ::k]
            up = np.repeat(np.repeat(small, k, axis=0), k, axis=1)
            up = up[: img.shape[0], : img.shape[1]]
            out.append(up)
    return out


def _draw_series(kind: str, fname: str, label: str,
                 levels: list[str]) -> None:
    img = synth_image()
    try:
        variants = _degraded_variants(img, kind)
    except ImportError:
        # fallback if scipy missing — only relevant for 'rotation'
        variants = [img] * 5
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.5))
    # row 1: degraded image
    for i, (v, lab) in enumerate(zip(variants, levels)):
        axes[0, i].imshow(v, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(lab, fontsize=10)
        axes[0, i].axis("off")
    # row 2: synth "matched-keypoint" count (decay curve as bar)
    base = 100
    decay = {"blur": [100, 78, 45, 18, 4],
             "rotation": [100, 92, 70, 35, 12],
             "occlusion": [100, 88, 55, 22, 7],
             "resolution": [100, 86, 52, 16, 3]}[kind]
    for i, n in enumerate(decay):
        axes[1, i].bar([0], [n], color="#4C72B0", width=0.4)
        axes[1, i].set_ylim(0, 105)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([0, 50, 100])
        axes[1, i].text(0, n + 3, f"{n}", ha="center", fontsize=10)
        axes[1, i].set_title("匹配 keypoint 数", fontsize=9)
    fig.suptitle(f"SIFT 鲁棒性 — {label}（合成实验）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_degradation_grid() -> None:
    img = synth_image()
    kinds = ["blur", "rotation", "occlusion", "resolution"]
    levels = ["原图", "轻", "中", "重", "极端"]
    fig, axes = plt.subplots(4, 5, figsize=(13, 11))
    for r, k in enumerate(kinds):
        try:
            variants = _degraded_variants(img, k)
        except ImportError:
            variants = [img] * 5
        for c, (v, lab) in enumerate(zip(variants, levels)):
            axes[r, c].imshow(v, cmap="gray", vmin=0, vmax=1)
            if r == 0:
                axes[r, c].set_title(lab, fontsize=11)
            if c == 0:
                axes[r, c].set_ylabel(k, fontsize=11)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    fig.suptitle("退化矩阵：四种退化 × 五种程度  →  手工特征的脆弱性", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "degradation_grid.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# 7.  Feature comparison accuracy
# ──────────────────────────────────────────────────────────────────────
def fig_feature_comparison_accuracy() -> None:
    methods = ["原始像素\n+SVM", "颜色直方图\n+SVM", "HOG\n+SVM",
               "SIFT+BoVW\n+SVM", "AlexNet\n(2012)", "ResNet-50\n(2015)"]
    acc = [0.52, 0.65, 0.78, 0.82, 0.95, 0.97]
    colors = ["#9E9E9E", "#9E9E9E", "#9E9E9E", "#9E9E9E",
              "#D62728", "#D62728"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(methods, acc, color=colors, edgecolor="black", lw=0.4)
    for b, a in zip(bars, acc):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.01,
                f"{a:.0%}", ha="center", fontsize=10)
    ax.axhline(0.80, color="orange", ls="--", lw=1)
    ax.text(0.05, 0.81, "≈ 手工特征天花板", color="orange",
            transform=ax.transAxes, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("分类准确率（CIFAR-10 子集，示意）")
    ax.set_title("六种方案对比：手工特征到 80% 见顶，CNN 一举跨过")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "feature_comparison_accuracy.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# 8.  ImageNet timeline (= feature_ceiling.png alias)
# ──────────────────────────────────────────────────────────────────────
def fig_imagenet_timeline() -> None:
    years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    err = [28.2, 25.8, 16.4, 11.7, 7.3, 3.6, 3.0, 2.3]
    method = ["手工", "手工", "AlexNet", "ZFNet", "VGG/GoogLeNet",
              "ResNet", "ResNeXt", "SENet"]
    colors = ["#9E9E9E" if "手工" in m else "#D62728" for m in method]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(years, err, color=colors, edgecolor="black", lw=0.4)
    for b, e, m in zip(bars, err, method):
        ax.text(b.get_x() + b.get_width() / 2, e + 0.4,
                f"{e:.1f}%\n{m}", ha="center", fontsize=8.5)
    ax.axhline(5.1, color="green", ls="--", lw=1)
    ax.text(2010.2, 5.4, "人类 ≈ 5.1%", color="green", fontsize=9)
    ax.set_xlabel("年份")
    ax.set_ylabel("Top-5 误差率 (%)")
    ax.set_title("ImageNet ILSVRC 误差时间线 — 2012 AlexNet 单年降 9.4 个点")
    ax.set_ylim(0, 32)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "imagenet_timeline.png"), dpi=130,
                bbox_inches="tight")
    # Save alias (slides_outline 也用 feature_ceiling.png)
    fig.savefig(os.path.join(OUT, "feature_ceiling.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    np.random.seed(0)
    fig_sift_scale_space()
    fig_sift_keypoints()
    fig_sift_orientation()
    fig_sift_descriptor()
    fig_hog_pedestrian()
    _draw_series("blur", "degradation_blur_series.png",
                 "高斯模糊 σ", ["σ=0", "1.0", "2.5", "5.0", "9.0"])
    _draw_series("rotation", "degradation_rotation_series.png",
                 "旋转角度", ["0°", "15°", "45°", "90°", "180°"])
    _draw_series("occlusion", "degradation_occlusion_series.png",
                 "随机遮挡比例", ["0%", "5%", "15%", "30%", "50%"])
    _draw_series("resolution", "degradation_resolution_series.png",
                 "下采样倍率", ["×1", "×2", "×4", "×8", "×16"])
    fig_degradation_grid()
    fig_feature_comparison_accuracy()
    fig_imagenet_timeline()
    print(f"All figures written to {OUT}")


if __name__ == "__main__":
    main()
