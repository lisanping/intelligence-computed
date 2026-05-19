"""
第 6 讲 · 动手环节 · 手工特征 vs 原始像素分类对比
================================================

目标：用 OpenCV 提取 SIFT / HOG 特征，训练 kNN / SVM 分类器，
     与原始像素输入对比准确率，验证"特征决定上限"。

运行：
    python feature_comparison.py                        # 基线（三路特征全开）
    python feature_comparison.py --ablate no_sift       # 去掉 SIFT 特征
    python feature_comparison.py --ablate no_hog        # 去掉 HOG 特征
    python feature_comparison.py --ablate raw_pixels_only  # 仅用原始像素

所有实验使用 seed=1337。
依赖：opencv-python opencv-contrib-python scikit-learn scikit-image matplotlib numpy
"""
from __future__ import annotations

import argparse
import os
import warnings

import cv2
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[4] / "design" / "meta"))
import cjk_font  # noqa: F401  - applies CJK font config
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skimage.feature import hog

warnings.filterwarnings("ignore")

# ── 固定随机种子 ─────────────────────────────────────────────
SEED = 1337
np.random.seed(SEED)

# ── 命令行参数 ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="L06 手工特征 vs 原始像素分类对比")
parser.add_argument(
    "--ablate", type=str, default=None,
    choices=["no_sift", "no_hog", "raw_pixels_only"],
    help="消融实验开关",
)
parser.add_argument("--n_samples", type=int, default=200, help="每类样本数")
parser.add_argument("--img_size", type=int, default=64, help="统一缩放到的图像尺寸")
parser.add_argument("--n_clusters", type=int, default=50, help="SIFT BoVW 词汇量")
parser.add_argument("--save_figures", action="store_true", help="保存可视化到 figures/")
args = parser.parse_args()

IMG_SIZE = args.img_size
N_CLUSTERS = args.n_clusters
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")


# ═══════════════════════════════════════════════════════════
# 1. 合成数据集（无需外部数据下载）
# ═══════════════════════════════════════════════════════════
def make_synthetic_dataset(n_per_class: int = 200, size: int = 64):
    """生成 3 类合成灰度图像：圆形、矩形、三角形。

    设计目的：
    - 无需下载外部数据集，启动即可跑。
    - 类别之间的区别在于**形状**（边缘/角点/曲率），
      手工特征（SIFT/HOG）应该能捕捉到，原始像素则困难。
    - 添加随机平移、缩放、旋转、噪声，模拟真实场景变化。
    """
    images, labels = [], []
    class_names = ["circle", "rectangle", "triangle"]

    for cls_idx, cls_name in enumerate(class_names):
        for _ in range(n_per_class):
            img = np.zeros((size, size), dtype=np.uint8)
            # 随机中心与大小
            cx = np.random.randint(size // 4, 3 * size // 4)
            cy = np.random.randint(size // 4, 3 * size // 4)
            r = np.random.randint(size // 6, size // 3)
            color = np.random.randint(180, 256)

            if cls_name == "circle":
                cv2.circle(img, (cx, cy), r, int(color), -1)
            elif cls_name == "rectangle":
                half = r
                pt1 = (cx - half, cy - half)
                pt2 = (cx + half, cy + half)
                cv2.rectangle(img, pt1, pt2, int(color), -1)
            elif cls_name == "triangle":
                pts = np.array([
                    [cx, cy - r],
                    [cx - r, cy + r],
                    [cx + r, cy + r],
                ], dtype=np.int32)
                cv2.fillPoly(img, [pts], int(color))

            # 随机旋转
            angle = np.random.uniform(-30, 30)
            M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (size, size))

            # 加高斯噪声
            noise = np.random.normal(0, 15, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            images.append(img)
            labels.append(cls_idx)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels, class_names


# ═══════════════════════════════════════════════════════════
# 2. 特征提取
# ═══════════════════════════════════════════════════════════
def extract_sift_bovw(images: np.ndarray, n_clusters: int = 50, kmeans=None):
    """SIFT 关键点 → Bag-of-Visual-Words 向量。"""
    sift = cv2.SIFT_create()
    all_descriptors = []
    per_image_desc = []

    for img in images:
        kps, desc = sift.detectAndCompute(img, None)
        if desc is not None:
            all_descriptors.append(desc)
            per_image_desc.append(desc)
        else:
            per_image_desc.append(np.zeros((1, 128), dtype=np.float32))

    # 构建视觉词典（仅训练时）
    if kmeans is None:
        all_desc = np.vstack(all_descriptors)
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=SEED, batch_size=1000)
        kmeans.fit(all_desc)

    # 编码为直方图
    features = []
    for desc in per_image_desc:
        words = kmeans.predict(desc)
        hist, _ = np.histogram(words, bins=np.arange(n_clusters + 1))
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-7  # L1 归一化
        features.append(hist)

    return np.array(features), kmeans


def extract_hog_features(images: np.ndarray):
    """HOG 特征向量。"""
    features = []
    hog_images = []
    for img in images:
        feat, hog_img = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=True,
            feature_vector=True,
        )
        features.append(feat)
        hog_images.append(hog_img)
    return np.array(features), hog_images


def extract_raw_pixels(images: np.ndarray):
    """将图像展平为一维向量。"""
    return images.reshape(len(images), -1).astype(np.float32)


# ═══════════════════════════════════════════════════════════
# 3. 分类与评估
# ═══════════════════════════════════════════════════════════
def evaluate(X_train, X_test, y_train, y_test, feature_name: str):
    """训练 kNN 和 SVM，返回准确率。"""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}
    # kNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_s, y_train)
    results["kNN"] = accuracy_score(y_test, knn.predict(X_test_s))

    # SVM
    svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=SEED)
    svm.fit(X_train_s, y_train)
    results["SVM"] = accuracy_score(y_test, svm.predict(X_test_s))

    print(f"  {feature_name:20s} | kNN: {results['kNN']:.1%} | SVM: {results['SVM']:.1%}")
    return results


# ═══════════════════════════════════════════════════════════
# 4. 可视化
# ═══════════════════════════════════════════════════════════
def viz_sift_keypoints(images, class_names, labels):
    """绘制 SIFT 关键点叠加图。"""
    sift = cv2.SIFT_create()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for cls_idx in range(3):
        idx = np.where(labels == cls_idx)[0][0]
        img = images[idx]
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        kps = sift.detect(img, None)
        img_kp = cv2.drawKeypoints(
            img_color, kps, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        axes[cls_idx].imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
        axes[cls_idx].set_title(f"SIFT: {class_names[cls_idx]}")
        axes[cls_idx].axis("off")
    fig.suptitle("V2 · SIFT Keypoints", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if args.save_figures:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "sift_keypoints.png"), dpi=150)
    plt.show()


def viz_hog_visualization(images, hog_images, class_names, labels):
    """绘制 HOG 梯度可视化。"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for cls_idx in range(3):
        idx = np.where(labels == cls_idx)[0][0]
        axes[0, cls_idx].imshow(images[idx], cmap="gray")
        axes[0, cls_idx].set_title(f"Original: {class_names[cls_idx]}")
        axes[0, cls_idx].axis("off")
        axes[1, cls_idx].imshow(hog_images[idx], cmap="gray")
        axes[1, cls_idx].set_title(f"HOG: {class_names[cls_idx]}")
        axes[1, cls_idx].axis("off")
    fig.suptitle("V3 · HOG Gradient Visualization", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if args.save_figures:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "hog_visualization.png"), dpi=150)
    plt.show()


def viz_rgb_decomposition(images, labels):
    """模拟 RGB 通道分解（将灰度图映射到伪彩色通道）。"""
    idx = 0
    img = images[idx]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 模拟彩色版本
    img_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    b, g, r = cv2.split(img_color)

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original (Grayscale)")
    axes[0].axis("off")

    for ch_idx, (ch, cmap, name) in enumerate(
        [(r, "Reds", "R Channel"), (g, "Greens", "G Channel"), (b, "Blues", "B Channel")]
    ):
        axes[ch_idx + 1].imshow(ch, cmap=cmap)
        axes[ch_idx + 1].set_title(name)
        axes[ch_idx + 1].axis("off")

    fig.suptitle("V1 · Channel Decomposition (Simulated RGB)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if args.save_figures:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "rgb_decomposition.png"), dpi=150)
    plt.show()


def viz_degradation_series(images, labels):
    """渐进退化系列：模糊/旋转/遮挡/缩小。"""
    idx = 0
    img = images[idx]
    size = img.shape[0]

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    row_labels = ["Blur", "Rotate", "Occlude", "Resize"]

    # 行 1：模糊
    sigmas = [0, 2, 5, 10, 20]
    for j, sigma in enumerate(sigmas):
        if sigma == 0:
            axes[0, j].imshow(img, cmap="gray")
        else:
            k = int(sigma * 4) | 1  # 确保奇数
            blurred = cv2.GaussianBlur(img, (k, k), sigma)
            axes[0, j].imshow(blurred, cmap="gray")
        axes[0, j].set_title(f"σ={sigma}" if sigma > 0 else "Original")
        axes[0, j].axis("off")

    # 行 2：旋转
    angles = [0, 45, 90, 135, 180]
    for j, angle in enumerate(angles):
        M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (size, size))
        axes[1, j].imshow(rotated, cmap="gray")
        axes[1, j].set_title(f"{angle}°")
        axes[1, j].axis("off")

    # 行 3：遮挡
    occlude_ratios = [0, 0.25, 0.50, 0.75, 0.90]
    for j, ratio in enumerate(occlude_ratios):
        occluded = img.copy()
        h_occ = int(size * ratio)
        if h_occ > 0:
            occluded[:h_occ, :] = 0
        axes[2, j].imshow(occluded, cmap="gray")
        axes[2, j].set_title(f"Occlude {int(ratio * 100)}%")
        axes[2, j].axis("off")

    # 行 4：缩小
    target_sizes = [size, size // 2, size // 4, max(size // 8, 4), max(size // 16, 2)]
    for j, ts in enumerate(target_sizes):
        small = cv2.resize(img, (ts, ts), interpolation=cv2.INTER_AREA)
        big_again = cv2.resize(small, (size, size), interpolation=cv2.INTER_NEAREST)
        axes[3, j].imshow(big_again, cmap="gray")
        axes[3, j].set_title(f"{ts}×{ts}")
        axes[3, j].axis("off")

    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label, fontsize=12, fontweight="bold", rotation=0,
                               labelpad=50, va="center")

    fig.suptitle("V4 · Progressive Degradation Series", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if args.save_figures:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "degradation_series.png"), dpi=150)
    plt.show()


def viz_accuracy_comparison(all_results: dict):
    """绘制准确率对比柱状图。"""
    feature_names = list(all_results.keys())
    knn_accs = [all_results[f]["kNN"] for f in feature_names]
    svm_accs = [all_results[f]["SVM"] for f in feature_names]

    x = np.arange(len(feature_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, knn_accs, width, label="kNN", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, svm_accs, width, label="SVM", color="#DD8452")

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("V6 · Feature Comparison: Handcrafted vs Raw Pixels", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1%}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)

    plt.tight_layout()
    if args.save_figures:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "accuracy_comparison.png"), dpi=150)
    plt.show()


# ═══════════════════════════════════════════════════════════
# 5. 主流程
# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("第 11 讲 · 手工特征 vs 原始像素分类对比")
    print("=" * 60)

    ablate = args.ablate
    if ablate:
        print(f"  消融模式: {ablate}")

    # ── 生成数据 ──
    print("\n[1/4] 生成合成数据集（圆/矩形/三角形）...")
    images, labels, class_names = make_synthetic_dataset(
        n_per_class=args.n_samples, size=IMG_SIZE,
    )
    print(f"  样本总数: {len(images)}  |  类别: {class_names}")

    # ── 划分训练/测试 ──
    idx = np.arange(len(images))
    idx_train, idx_test = train_test_split(idx, test_size=0.3, random_state=SEED, stratify=labels)

    # ── 提取特征 ──
    print("\n[2/4] 提取特征...")
    features = {}

    if ablate != "no_sift" and ablate != "raw_pixels_only":
        sift_feat_train, kmeans = extract_sift_bovw(images[idx_train], n_clusters=N_CLUSTERS)
        sift_feat_test, _ = extract_sift_bovw(images[idx_test], n_clusters=N_CLUSTERS, kmeans=kmeans)
        features["SIFT (BoVW)"] = (sift_feat_train, sift_feat_test)
        print(f"  SIFT BoVW: {sift_feat_train.shape[1]} 维")

    if ablate != "no_hog" and ablate != "raw_pixels_only":
        hog_feat_all, hog_images_all = extract_hog_features(images)
        features["HOG"] = (hog_feat_all[idx_train], hog_feat_all[idx_test])
        print(f"  HOG: {hog_feat_all.shape[1]} 维")

    raw_feat = extract_raw_pixels(images)
    features["Raw Pixels"] = (raw_feat[idx_train], raw_feat[idx_test])
    print(f"  Raw Pixels: {raw_feat.shape[1]} 维")

    # ── 分类 ──
    print("\n[3/4] 训练分类器...")
    print(f"  {'Feature':20s} | {'kNN':>10s} | {'SVM':>10s}")
    print("  " + "-" * 50)

    all_results = {}
    for name, (X_tr, X_te) in features.items():
        all_results[name] = evaluate(X_tr, X_te, labels[idx_train], labels[idx_test], name)

    # ── 可视化 ──
    print("\n[4/4] 生成可视化...")

    viz_rgb_decomposition(images, labels)

    if "SIFT (BoVW)" in features:
        viz_sift_keypoints(images, class_names, labels)

    if "HOG" in features:
        hog_feat_all, hog_images_all = extract_hog_features(images)
        viz_hog_visualization(images, hog_images_all, class_names, labels)

    viz_degradation_series(images, labels)
    viz_accuracy_comparison(all_results)

    # ── 总结 ──
    print("\n" + "=" * 60)
    print("核心洞见：特征决定上限，模型只是逼近上限。")
    if len(all_results) > 1:
        best = max(all_results.items(), key=lambda kv: max(kv[1].values()))
        worst = min(all_results.items(), key=lambda kv: max(kv[1].values()))
        print(f"  最佳特征: {best[0]} (SVM {best[1]['SVM']:.1%})")
        print(f"  最差特征: {worst[0]} (SVM {worst[1]['SVM']:.1%})")
        gap = max(best[1].values()) - max(worst[1].values())
        print(f"  特征差距: {gap:.1%} — 同样的模型，不同的特征，不同的天花板。")
    print("=" * 60)


if __name__ == "__main__":
    main()