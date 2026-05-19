# L35 · 预测与未来 — Demo Scripts

> Lecture: [L35 预测与未来](../../README.md) — 反派视角/六大预言/十年规划

## 脚本清单

| 脚本                                                 | 类型     | 描述                                                                                    | 依赖              |
| ---------------------------------------------------- | -------- | --------------------------------------------------------------------------------------- | ----------------- |
| [`prediction_scorecard.py`](prediction_scorecard.py) | 预言卡   | 单张幻灯片用的预言计分卡；生成 figures/prediction_scorecard.png 与 critic_radar.png     | matplotlib, numpy |
| [`generate_finale.py`](generate_finale.py)           | 终章合集 | 生成 5 职业方向表 + 90 天行动清单图 + 6 预言仪表盘 + action_checklist.md + resources.md | matplotlib        |

## 运行

```bash
python prediction_scorecard.py
python generate_finale.py
```

## 输出文件

```
figures/
├── prediction_scorecard.png       (S35 预言验证方法)
├── critic_radar.png               (S20 四位批评者雷达图)
├── 6_predictions_dashboard.png    (S34 六大预言汇总)
├── careers_table.png              (S48 五条职业方向)
└── action_checklist.png           (S49 持续学习清单)

action_checklist.md                90 天可勾选清单 (markdown)
resources.md                       学员长期资源清单 (newsletter/repo/书)
```

## 教学用途

这是课程的**最后一讲**，所有图表都是为"留给学员可执行的下一步"服务：

- **6_predictions_dashboard.png** — 把六大预言压缩到一张可视化
- **careers_table.png** — 让学员明确选哪条路（5 选 1）
- **action_checklist.png** — 立刻可执行的 90 天计划
- **action_checklist.md** — 学员可 fork 到自己仓库勾选

## 维护

预言置信度 (`PREDICTIONS` in `generate_finale.py`) 是**讲师 2026-Q2 个人估计**。建议每年课程开始时刷新。
