# L34 · AI 安全 — Demo Scripts

> Lecture: [L34 AI 安全](../../README.md) — Jailbreak/红队/可解释性/RSP

## 脚本清单

| 脚本                                         | 类型     | 描述                                                                                        | 依赖              |
| -------------------------------------------- | -------- | ------------------------------------------------------------------------------------------- | ----------------- |
| [`safety_viz.py`](safety_viz.py)             | 可视化   | 对齐税权衡图；生成 figures/alignment_tradeoff.png                                           | matplotlib        |
| [`generate_figures.py`](generate_figures.py) | 图像生成 | 生成 4 张教学图：red_team_flow / sae_visualization / feature_circuits / golden_gate_example | matplotlib, numpy |
| [`red_team_test.py`](red_team_test.py)       | 红队工具 | 5 类攻击 mock 红队 runner；演示 promptfoo 风格的结构化攻击                                  | 仅标准库          |

## 运行

```bash
python safety_viz.py
python generate_figures.py
python red_team_test.py            # 离线 mock 模式
python red_team_test.py --json     # 输出 JSON 报告
```

## 教学用途

- **generate_figures.py** —
  - red_team_flow.png → S13 "红队测试的五步流程"
  - sae_visualization.png → S24 "Step 2：字典学习 & 稀疏自动编码器"
  - feature_circuits.png → S25 "Step 3：特征回路 & 因果关系"
  - golden_gate_example.png → S26 "Step 4：Golden Gate Claude"
- **red_team_test.py** — 配合 S41 "动手环节：结构化红队测试"
  - 可替换 `MockLLM` 为真实的 OpenAI/Anthropic 客户端进行真实评估

## 真实红队工具

学员作业建议（S67）使用业界工具：

```bash
# Garak (NVIDIA)
pip install garak
garak --model_type openai --model_name gpt-4o --probes dan

# promptfoo (CLI, via npm)
npm install -g promptfoo
promptfoo eval -c redteam.yaml
```
