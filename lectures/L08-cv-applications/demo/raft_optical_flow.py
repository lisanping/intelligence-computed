"""RAFT optical flow — synthetic visualization (no GPU needed).

Generates a side-by-side: two consecutive frames + the colored optical-flow
field (HSV color wheel: hue = direction, value = magnitude).

For the real RAFT model, see the inference template at bottom.

Run:  python raft_optical_flow.py
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
import matplotlib.colors as mcolors

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)


def synth_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frame1, frame2 (with shifted blob), and ground-truth flow (h, w, 2)."""
    h, w = 96, 128
    yy, xx = np.mgrid[0:h, 0:w]
    f1 = np.zeros((h, w), dtype=np.float32)
    f2 = np.zeros((h, w), dtype=np.float32)
    flow = np.zeros((h, w, 2), dtype=np.float32)

    # background gradient
    f1 += yy * 0.005
    f2 += yy * 0.005

    # blob 1: moves right
    blob1 = ((xx - 30) ** 2 + (yy - 40) ** 2) < 12 ** 2
    f1[blob1] = 0.8
    blob1_shift = ((xx - 50) ** 2 + (yy - 40) ** 2) < 12 ** 2
    f2[blob1_shift] = 0.8
    flow[blob1, 0] = 20  # dx
    flow[blob1, 1] = 0   # dy

    # blob 2: moves down-left
    blob2 = ((xx - 90) ** 2 + (yy - 30) ** 2) < 9 ** 2
    f1[blob2] = 0.6
    blob2_shift = ((xx - 80) ** 2 + (yy - 50) ** 2) < 9 ** 2
    f2[blob2_shift] = 0.6
    flow[blob2, 0] = -10
    flow[blob2, 1] = 20

    return f1, f2, flow


def flow_to_color(flow: np.ndarray) -> np.ndarray:
    """Map (h, w, 2) flow to RGB using the standard HSV color wheel."""
    fx, fy = flow[..., 0], flow[..., 1]
    mag = np.sqrt(fx ** 2 + fy ** 2)
    ang = (np.arctan2(fy, fx) + np.pi) / (2 * np.pi)  # 0..1
    mag_norm = mag / (mag.max() + 1e-6)
    hsv = np.stack([ang, np.ones_like(ang), mag_norm], axis=-1)
    return mcolors.hsv_to_rgb(hsv)


def main() -> None:
    f1, f2, flow = synth_pair()
    flow_rgb = flow_to_color(flow)

    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    axes[0].imshow(f1, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Frame t")
    axes[0].axis("off")
    axes[1].imshow(f2, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Frame t+1")
    axes[1].axis("off")
    axes[2].imshow(flow_rgb)
    axes[2].set_title("Optical flow (HSV: hue=方向, val=强度)")
    axes[2].axis("off")

    # color wheel reference
    h, w = 90, 90
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    fy, fx = (yy - cy), (xx - cx)
    wheel = flow_to_color(np.stack([fx, fy], axis=-1).astype(np.float32))
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < (h / 2 - 2) ** 2
    wheel[~mask] = 1.0
    axes[3].imshow(wheel)
    axes[3].set_title("HSV 色轮参考")
    axes[3].axis("off")

    fig.suptitle("RAFT-style optical flow — 合成示例", fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUT, "raft_optical_flow.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ──────────────────────────────────────────────────────────────────────
# Real RAFT inference (uncomment, needs torch + GPU)
# ──────────────────────────────────────────────────────────────────────
"""
# pip install torch torchvision opencv-python
import torch, cv2
from torchvision.models.optical_flow import raft_large
from torchvision.transforms import functional as F
model = raft_large(weights="DEFAULT", progress=False).eval().cuda()
img1 = F.to_tensor(cv2.cvtColor(cv2.imread("frame1.png"), cv2.COLOR_BGR2RGB))
img2 = F.to_tensor(cv2.cvtColor(cv2.imread("frame2.png"), cv2.COLOR_BGR2RGB))
img1 = F.normalize(img1, [0.5]*3, [0.5]*3)[None].cuda()
img2 = F.normalize(img2, [0.5]*3, [0.5]*3)[None].cuda()
with torch.no_grad():
    flow = model(img1, img2)[-1]  # 12 iterative refinements
"""


if __name__ == "__main__":
    main()
