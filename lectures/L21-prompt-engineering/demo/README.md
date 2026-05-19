# L21 · Prompt Engineering — Demo Scripts

> Lecture: [L21 Prompt Engineering](../../README.md) — Few-shot/CoT/ToT/ReAct

## 脚本清单

| 脚本                                           | 类型     | 描述                                                              | 依赖                        |
| ---------------------------------------------- | -------- | ----------------------------------------------------------------- | --------------------------- |
| [`prompt_techniques.py`](prompt_techniques.py) | 综合     | Zero-shot / Few-shot / CoT / Self-consistency 实测脚本            | openai, anthropic, tiktoken |
| [`cost_calculator.py`](cost_calculator.py)     | 成本分析 | 8 种策略 × 7 种模型的成本对比；生成 figures/cost_calculator.png   | matplotlib (无需 API key)   |
| [`tot_game24.py`](tot_game24.py)               | ToT 演示 | Game-of-24 暴力解 + ToT 搜索树可视化；生成 figures/tot_game24.png | matplotlib                  |

## 运行

```bash
# 不需要 API
python cost_calculator.py
python tot_game24.py

# 需要 OpenAI / Anthropic API key
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
python prompt_techniques.py
```

## 教学用途

- **cost_calculator.py** — 配合 S34 "进阶策略的成本对比"，强调"不同策略的物理账单"
- **tot_game24.py** — 配合 S29 "Tree-of-Thought"，可视化搜索 + 评分 + 剪枝
- **prompt_techniques.py** — 配合 S09–S20 各种 prompt 模式的实测

## 价格数据维护

`cost_calculator.py` 中的 `PRICES` 字典是 **2026-Q2 价格**。建议每季度刷新一次。
