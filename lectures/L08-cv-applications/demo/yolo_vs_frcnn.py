"""YOLO vs Faster-R-CNN: speed/accuracy trade-off visualization.

Generates a comparison chart from published COCO benchmark numbers
(no GPU required). For real-time inference, see the comments at the
bottom for `ultralytics` and `torchvision` install + usage.

Run:  python yolo_vs_frcnn.py
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

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)


# (model, mAP@COCO, FPS on V100 — published numbers)
MODELS = [
    ("Faster R-CNN R50",      37.4,  17, "two-stage"),
    ("Faster R-CNN R101",     39.8,  12, "two-stage"),
    ("Mask R-CNN R101",       40.8,  10, "two-stage"),
    ("Cascade R-CNN R101",    44.3,   8, "two-stage"),
    ("YOLOv5n",               28.0, 159, "one-stage"),
    ("YOLOv5s",               37.4, 156, "one-stage"),
    ("YOLOv5m",               45.4, 121, "one-stage"),
    ("YOLOv5l",               49.0,  99, "one-stage"),
    ("YOLOv5x",               50.7,  68, "one-stage"),
    ("YOLOv8n",               37.3, 280, "one-stage"),
    ("YOLOv8m",               50.2, 145, "one-stage"),
    ("YOLOv8x",               53.9,  80, "one-stage"),
    ("YOLOv11x (2024)",       54.7,  90, "one-stage"),
    ("DETR R50",              42.0,  28, "DETR"),
    ("DINO Swin-L (2023)",    63.3,   4, "DETR"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"two-stage": "#D62728", "one-stage": "#4C72B0",
              "DETR": "#2CA02C"}
    for name, mAP, fps, fam in MODELS:
        ax.scatter(fps, mAP, s=160, color=colors[fam], alpha=0.85,
                   edgecolor="black", lw=0.6)
        ax.annotate(name, (fps, mAP),
                    xytext=(5, -2), textcoords="offset points",
                    fontsize=8, color=colors[fam])
    # legend
    for fam, c in colors.items():
        ax.scatter([], [], color=c, s=120, label=fam,
                   edgecolor="black", lw=0.6)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
    ax.set_xscale("log")
    ax.set_xlabel("FPS (V100 GPU, log scale)")
    ax.set_ylabel("COCO mAP@[.5:.95]")
    ax.set_title("YOLO vs Faster-R-CNN vs DETR — 速度/精度权衡\n"
                 "（数据来源：各模型官方 paper / 仓库 README）")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT, "yolo_vs_frcnn.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ──────────────────────────────────────────────────────────────────────
# Real-time inference templates (uncomment to use; needs GPU)
# ──────────────────────────────────────────────────────────────────────
"""
# --- YOLO ---
# pip install ultralytics
from ultralytics import YOLO
model = YOLO("yolov8m.pt")
results = model("path/to/image.jpg")
results[0].save(filename="out_yolo.jpg")

# --- Faster R-CNN ---
# pip install torchvision pillow
import torch, torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT").eval()
img = torchvision.io.read_image("path/to/image.jpg") / 255.0
with torch.no_grad():
    out = model([img])
print(out[0]["boxes"], out[0]["labels"], out[0]["scores"])
"""


if __name__ == "__main__":
    main()
