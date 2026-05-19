"""SAM 交互式分割 Demo — 第 8 讲：视觉应用全景

使用 SAM (Segment Anything Model) 进行零样本分割。
支持点击提示和自动全图分割。

用法：
    python sam_interactive.py --image sample.jpg             # 自动分割
    python sam_interactive.py --image sample.jpg --mode point # 点提示模式
    python sam_interactive.py --auto                          # 使用内置示例
"""

import argparse
import os

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[4] / "design" / "meta"))
import cjk_font  # noqa: F401  - applies CJK font config
import matplotlib.pyplot as plt
import numpy as np

SEED = 1337


def show_mask(mask, ax, random_color=False):
    """Overlay a segmentation mask on the plot."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    """Show prompt points on the plot."""
    pos = coords[labels == 1]
    neg = coords[labels == 0]
    ax.scatter(pos[:, 0], pos[:, 1], color='green', marker='*',
               s=marker_size, edgecolors='white', linewidths=1.2, zorder=5)
    ax.scatter(neg[:, 0], neg[:, 1], color='red', marker='*',
               s=marker_size, edgecolors='white', linewidths=1.2, zorder=5)


def auto_segment(image_path: str):
    """Run automatic mask generation on an image."""
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        import cv2
    except ImportError:
        print("ERROR: segment_anything not installed.")
        print("Install: pip install segment-anything")
        print("Download model: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        _demo_fallback()
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image: {image_path}")
        _demo_fallback()
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    checkpoint = "sam_vit_h_4b8939.pth"
    if not os.path.exists(checkpoint):
        print(f"SAM checkpoint not found: {checkpoint}")
        print("Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        _demo_fallback()
        return

    sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)

    # Sort by area (largest first)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    for mask in masks:
        show_mask(mask['segmentation'], ax, random_color=True)
    ax.set_title(f"SAM Automatic Segmentation — {len(masks)} masks found",
                 fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/sam_auto.png", dpi=150, bbox_inches='tight')
    print(f"[saved] figures/sam_auto.png ({len(masks)} masks)")
    plt.show()


def _demo_fallback():
    """Fallback demo when SAM is not installed — show concept with synthetic data."""
    print("\n[Fallback] Running SAM concept demo with synthetic data...\n")
    np.random.seed(SEED)

    # Create a synthetic scene
    h, w = 256, 384
    image = np.ones((h, w, 3), dtype=np.uint8) * 200

    # Add "objects"
    objects = [
        {"name": "circle", "center": (100, 80), "radius": 40, "color": (70, 130, 180)},
        {"name": "rect", "tl": (200, 50), "br": (340, 150), "color": (34, 139, 34)},
        {"name": "circle2", "center": (280, 200), "radius": 35, "color": (178, 34, 34)},
    ]

    # Draw objects
    for obj in objects:
        if "radius" in obj:
            yy, xx = np.ogrid[:h, :w]
            cy, cx = obj["center"]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= obj["radius"] ** 2
            image[mask] = obj["color"]
        else:
            y1, x1 = obj["tl"]
            y2, x2 = obj["br"]
            image[x1:x2, y1:y2] = obj["color"]

    # Simulate "SAM masks"
    masks = []
    for obj in objects:
        m = np.zeros((h, w), dtype=bool)
        if "radius" in obj:
            yy, xx = np.ogrid[:h, :w]
            cy, cx = obj["center"]
            m = (xx - cx) ** 2 + (yy - cy) ** 2 <= obj["radius"] ** 2
        else:
            y1, x1 = obj["tl"]
            y2, x2 = obj["br"]
            m[x1:x2, y1:y2] = True
        masks.append(m)

    # Prompt point
    prompt_point = np.array([[100, 80]])
    prompt_label = np.array([1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Original
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontweight='bold')
    axes[0].axis('off')

    # Auto segmentation
    axes[1].imshow(image)
    for m in masks:
        show_mask(m, axes[1], random_color=True)
    axes[1].set_title(f"Auto: {len(masks)} masks", fontweight='bold')
    axes[1].axis('off')

    # Point prompt
    axes[2].imshow(image)
    show_mask(masks[0], axes[2])
    show_points(prompt_point, prompt_label, axes[2])
    axes[2].set_title("Point Prompt → Segment", fontweight='bold')
    axes[2].axis('off')

    fig.suptitle("SAM Concept Demo (synthetic — install segment-anything for real)",
                 fontsize=12)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/sam_concept.png", dpi=150, bbox_inches='tight')
    print("[saved] figures/sam_concept.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="SAM Interactive Segmentation Demo")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--mode", choices=["auto", "point"], default="auto",
                        help="Segmentation mode")
    parser.add_argument("--auto", action="store_true",
                        help="Use built-in synthetic example")
    args = parser.parse_args()

    os.makedirs("figures", exist_ok=True)

    if args.auto or args.image is None:
        _demo_fallback()
    else:
        auto_segment(args.image)


if __name__ == "__main__":
    main()
