# L02 · 经典 AI 时代 — Demo Scripts

> Lecture: [L02 经典 AI](../../README.md) — GOFAI、专家系统、搜索算法

## 脚本清单

| 脚本                                             | 类型       | 描述                                                             | 依赖              |
| ------------------------------------------------ | ---------- | ---------------------------------------------------------------- | ----------------- |
| [`astar_pathfinding.py`](astar_pathfinding.py)   | 算法可视化 | A* 寻路算法演示，输出 figures/astar_result.png                   | numpy, matplotlib |
| [`eliza_replica.py`](eliza_replica.py)           | 交互式     | Weizenbaum 1966 ELIZA 心理咨询机器人复刻；纯模式匹配，无任何学习 | 仅标准库          |
| [`expert_system_mini.py`](expert_system_mini.py) | 规则系统   | 微型 forward-chaining 专家系统，演示 IF-THEN 推理                | 仅标准库          |

## 运行

```bash
# A* 路径规划（生成图像）
python astar_pathfinding.py

# ELIZA 交互（在终端对话；输入 quit 退出）
python eliza_replica.py

# 专家系统
python expert_system_mini.py
```

## 教学用途

- **astar_pathfinding.py** — 配合 slides_outline S?? 关于 A* 算法的卡片
- **eliza_replica.py** — 配合 GOFAI 时代叙事；让学员体会"无学习的对话"
- **expert_system_mini.py** — 配合专家系统幻灯片，理解符号 AI 的天花板
