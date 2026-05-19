"""
L08 – CV Applications: Canny Edge & Hough Transform Demo
=========================================================
Generates a synthetic image containing lines and circles, then applies
Canny edge detection followed by Hough line / circle detection.
Produces a side-by-side visualisation: Original → Canny → Hough results.

Usage
-----
    python canny_hough_demo.py
    python canny_hough_demo.py --mode lines
    python canny_hough_demo.py --mode circles
    python canny_hough_demo.py --mode both --canny-lo 50 --canny-hi 150
"""

import argparse
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

SEED = 1337
np.random.seed(SEED)

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def make_synthetic(h: int = 400, w: int = 400) -> np.ndarray:
    """White canvas with random lines and circles."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 240
    for _ in range(6):
        pt1 = tuple(np.random.randint(0, w, 2))
        pt2 = tuple(np.random.randint(0, w, 2))
        cv2.line(img, pt1, pt2, (0, 0, 0), 2)
    for _ in range(4):
        centre = tuple(np.random.randint(40, w - 40, 2))
        radius = np.random.randint(20, 80)
        cv2.circle(img, centre, radius, (0, 0, 0), 2)
    return img


def detect_lines(edges, canvas):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=30, maxLineGap=10)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return canvas


def detect_circles(edges, canvas, gray):
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=40, param1=150, param2=30,
                               minRadius=15, maxRadius=100)
    if circles is not None:
        for x, y, r in np.uint16(np.around(circles[0])):
            cv2.circle(canvas, (x, y), r, (0, 0, 255), 2)
            cv2.circle(canvas, (x, y), 2, (0, 0, 255), 3)
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Canny + Hough demo")
    parser.add_argument("--mode", choices=["lines", "circles", "both"],
                        default="both")
    parser.add_argument("--ablate", choices=["no_canny", "no_hough"],
                        default=None,
                        help="Ablation: no_canny=skip edge detection, no_hough=only edges")
    parser.add_argument("--canny-lo", type=int, default=50)
    parser.add_argument("--canny-hi", type=int, default=150)
    parser.add_argument("--image", default=None, help="Path to an input image")
    args = parser.parse_args()

    if args.image and os.path.isfile(args.image):
        img = cv2.imread(args.image)
    else:
        img = make_synthetic()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ablation: skip Canny (feed raw grayscale to Hough)
    if args.ablate == "no_canny":
        edges = gray
        print("[ablate] Skipping Canny — feeding raw grayscale to Hough")
    else:
        edges = cv2.Canny(gray, args.canny_lo, args.canny_hi)

    canvas = img.copy()
    # Ablation: no_hough = show only edges, skip Hough detection
    if args.ablate == "no_hough":
        print("[ablate] Skipping Hough — showing only Canny edges")
    else:
        if args.mode in ("lines", "both"):
            canvas = detect_lines(edges, canvas)
        if args.mode in ("circles", "both"):
            canvas = detect_circles(edges, canvas, gray)

    # --- side-by-side plot ------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title("Canny Edges")
    axes[2].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Hough ({args.mode})")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fname = os.path.join(FIG_DIR, "canny_hough.png")
    fig.savefig(fname, dpi=150)
    print(f"[saved] {fname}")
    plt.show()


if __name__ == "__main__":
    main()
