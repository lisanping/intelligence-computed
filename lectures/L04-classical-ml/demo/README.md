# L04 · 经典 ML — Demo Scripts

> Lecture: [L04 经典机器学习](../../README.md) — KNN/决策树/SVM/朴素贝叶斯/集成方法

## 脚本清单

| 脚本                                       | 类型     | 描述                                                                                  | 依赖                                    |
| ------------------------------------------ | -------- | ------------------------------------------------------------------------------------- | --------------------------------------- |
| [`algorithm_arena.py`](algorithm_arena.py) | 综合对比 | 8 种经典算法在多个合成数据集上的决策边界与精度对比；生成 figures/algorithm_arena.png  | numpy, matplotlib, scikit-learn         |
| [`titanic_xgboost.py`](titanic_xgboost.py) | 实战     | Titanic 数据集上 LR/RF/XGBoost/LightGBM 横向对比；生成 figures/titanic_comparison.png | pandas, scikit-learn, xgboost, lightgbm |

## 运行

```bash
pip install -r ../../../../tools/demo-requirements.txt   # 包含所有依赖
python algorithm_arena.py
python titanic_xgboost.py     # 需要联网下载 Titanic 数据
```

## 教学用途

- **algorithm_arena.py** — 配合 S05 "8 种算法 = L03 公式的 8 种填法" 的统一框架表
- **titanic_xgboost.py** — 配合 S32 "XGBoost：表格数据的'作弊代码'" 故事卡
