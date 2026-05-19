# L08 · CV 应用 — Demo Scripts

> Lecture: [L08 CV 应用](../../README.md) — 边缘检测、目标检测、分割、姿态、3D

## 脚本清单

| 脚本                                           | 类型     | 描述                                                       | 依赖                                   |
| ---------------------------------------------- | -------- | ---------------------------------------------------------- | -------------------------------------- |
| [`canny_hough_demo.py`](canny_hough_demo.py)   | 经典 CV  | Canny 边缘 + Hough 直线/圆检测                             | opencv-python, numpy, matplotlib       |
| [`yolo_realtime.py`](yolo_realtime.py)         | 实时检测 | YOLOv8 摄像头实时检测；需 GPU                              | ultralytics, opencv-python             |
| [`yolo_vs_frcnn.py`](yolo_vs_frcnn.py)         | 性能对比 | YOLO / Faster-R-CNN / DETR 速度-精度散点图（基于公开数据） | matplotlib, numpy                      |
| [`sam_interactive.py`](sam_interactive.py)     | 分割     | SAM (Segment Anything) 交互式点击分割                      | torch, segment_anything, opencv-python |
| [`raft_optical_flow.py`](raft_optical_flow.py) | 光流     | 合成示例 + RAFT 推理代码模板（注释）                       | numpy, matplotlib                      |

## 运行

```bash
# 仅图表（无 GPU 也能跑）
python yolo_vs_frcnn.py
python raft_optical_flow.py
python canny_hough_demo.py

# 需要 GPU + 下载模型
python yolo_realtime.py
python sam_interactive.py
```

## 教学用途

- **canny_hough_demo.py** — 配合 S07 "Hough 变换：投票找几何"
- **yolo_vs_frcnn.py** — 配合 S19 "Faster R-CNN vs YOLO 对比"
- **sam_interactive.py** — 配合 S27 "SAM：分割进入基础模型时代"
- **raft_optical_flow.py** — 配合 S39 "RAFT：迭代光流估计"
