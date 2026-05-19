# L32 · 生成式 AI — Demo Scripts

> Lecture: [L32 生成式 AI](../../README.md) — Diffusion/Sora/ControlNet/LoRA/商业化

## 脚本清单

| 脚本                                                     | 类型        | 描述                                                                                                        | 依赖              |
| -------------------------------------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------- | ----------------- |
| [`diffusion_intuition.py`](diffusion_intuition.py)       | 直觉演示    | 50 行代码的扩散模型直觉演示；生成 figures/diffusion_process.png                                             | numpy, matplotlib |
| [`controlnet_and_landing.py`](controlnet_and_landing.py) | 架构 + 数据 | ControlNet 架构图 + 2026 商业落地双柱图；生成 figures/controlnet_diagram.png 与 generative_landing_2026.png | matplotlib        |

## 运行

```bash
python diffusion_intuition.py
python controlnet_and_landing.py
```

## 教学用途

- **diffusion_intuition.py** — 配合 S05 "DDPM：Ho 2020 的数学之美"，给学员"50 行代码也能实现扩散"的直觉
- **controlnet_and_landing.py** —
  - 配合 S33 "ControlNet：让生成式 AI 听指挥" 的架构理解
  - 配合 S38 "Canva AI / Figma AI / Notion AI" 的商业落地数据

## 商业数据维护

`controlnet_and_landing.py` 中 `commercial_chart()` 的 ARR / MAU 数字是 **2026-Q1 估计**。建议每季度刷新。
