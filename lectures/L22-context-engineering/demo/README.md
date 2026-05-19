# L22 · Context Engineering — Demo Scripts

> Lecture: [L22 Context Engineering](../../README.md) — KV Cache/长上下文/RAG vs LC

## 脚本清单

| 脚本                                               | 类型   | 描述                                                            | 依赖                        |
| -------------------------------------------------- | ------ | --------------------------------------------------------------- | --------------------------- |
| [`context_engineering.py`](context_engineering.py) | 综合   | Token 经济、Prompt Caching、上下文压缩示例                      | openai, anthropic, tiktoken |
| [`context_viz.py`](context_viz.py)                 | 可视化 | 生成 lost_in_middle / needle_in_haystack / token_cost_curves 图 | matplotlib, numpy           |
| [`generate_tradeoff.py`](generate_tradeoff.py)     | 决策图 | RAG vs Long-Context 六维雷达图；生成 figures/tradeoff.png       | matplotlib                  |

## 运行

```bash
python context_viz.py            # 不需要 API
python generate_tradeoff.py      # 不需要 API
python context_engineering.py    # 需要 API key
```

## 教学用途

- **context_viz.py** — 配合 S33 "Lost in the Middle" 与 S32 "Needle in a Haystack"
- **generate_tradeoff.py** — 配合 S34 "长上下文 vs RAG 决策表" 与 S35 "假二选一"
- **context_engineering.py** — 配合整个 KV Cache 与 Prompt Caching 章节
