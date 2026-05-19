# L23 · Embedding & Retrieval — Demo Scripts

> Lecture: [L23 Embedding 与检索](../../README.md) — BM25/Dense/HNSW/IVF/Hybrid

## 脚本清单

| 脚本                                             | 类型     | 描述                                                                                                 | 依赖                             |
| ------------------------------------------------ | -------- | ---------------------------------------------------------------------------------------------------- | -------------------------------- |
| [`vector_search_demo.py`](vector_search_demo.py) | 综合     | BM25 vs Dense vs Hybrid 检索对比                                                                     | sentence-transformers, rank_bm25 |
| [`ann_benchmark.py`](ann_benchmark.py)           | ANN 实测 | Flat / IVF / HNSW 召回率-延迟曲线（toy implementations，无需 faiss）；生成 figures/ann_benchmark.png | numpy, matplotlib                |

## 运行

```bash
python ann_benchmark.py            # 仅 numpy/matplotlib
python vector_search_demo.py       # 需下载 sentence-transformers 模型
```

## 教学用途

- **ann_benchmark.py** — 配合 S48 "动手环节 ②：HNSW 召回率 vs 延迟曲线"
- **vector_search_demo.py** — 配合 S47 "动手环节 ①：语义检索 vs BM25 对比"

## 注意

`ann_benchmark.py` 中的 HNSW 是**教学用 toy 实现**（O(n²) 建索引）。生产环境请用 faiss、qdrant 或 pgvector。
