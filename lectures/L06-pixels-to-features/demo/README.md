# L06 · 像素到特征 — Demo Scripts

> Lecture: [L06 像素到特征](../../README.md) — RGB/HOG/SIFT/退化测试/特征天花板

## 脚本清单

| 脚本                                             | 类型     | 描述                                                                                                    | 依赖                                           |
| ------------------------------------------------ | -------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| [`feature_comparison.py`](feature_comparison.py) | 对比实验 | 原始像素 vs HOG vs SIFT 在分类任务上的对比                                                              | numpy, matplotlib, scikit-learn, opencv-python |
| [`generate_figures.py`](generate_figures.py)     | 图像生成 | 生成 12 张教学用合成图：SIFT 尺度空间/方向/描述子、HOG 行人、4 类退化 series、退化矩阵、ImageNet 时间线 | numpy, matplotlib, scipy                       |

## 运行

```bash
pip install numpy matplotlib scipy
python generate_figures.py     # 生成全部 12 张图到 figures/
python feature_comparison.py   # 运行对比实验（较慢）
```

## 教学用途

- **generate_figures.py** — 提供 outline 中显式引用的所有教学图（degradation_grid, sift_scale_space, hog_pedestrian_detection, feature_ceiling/imagenet_timeline 等）
- **feature_comparison.py** — 配合"手工特征 vs CNN"的对比叙事
