# L29 · 多模态 — Demo Scripts

> Lecture: [L29 多模态](../../README.md) — CLIP/Diffusion/VLM/Sora/视频生成

## 脚本清单

| 脚本                                                         | 类型        | 描述                                                                                      | 依赖                      |
| ------------------------------------------------------------ | ----------- | ----------------------------------------------------------------------------------------- | ------------------------- |
| [`clip_space_viz.py`](clip_space_viz.py)                     | CLIP 可视化 | CLIP 向量空间 + 对比矩阵 + CFG 消融 + diffusion 步骤图                                    | numpy, matplotlib         |
| [`diffusion_step_animation.py`](diffusion_step_animation.py) | 动画演示    | Diffusion 加噪/去噪 9 步条 + 20 帧 GIF；生成 figures/diffusion_denoising_grid.png 与 .gif | numpy, matplotlib, Pillow |

## 运行

```bash
python clip_space_viz.py
python diffusion_step_animation.py    # 需要 Pillow 写 GIF
```

## 教学用途

- **clip_space_viz.py** — 配合 S05–S15 CLIP/Diffusion 章节（提供 clip_contrastive_matrix, clip_vector_space, diffusion_steps, cfg_scale_ablation 四张图）
- **diffusion_step_animation.py** — 配合 S13/S14 前向加噪/反向去噪概念卡，动画对建立直觉特别有效

## 完整模型推理

如要跑真实 CLIP / Stable Diffusion 模型：

```bash
pip install torch torchvision open-clip-torch diffusers
# 见各脚本文件底部的注释模板
```
