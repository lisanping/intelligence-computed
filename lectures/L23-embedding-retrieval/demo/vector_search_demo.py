"""
第 23 讲 · 向量知识库构建与语义搜索 Demo
==========================================

演示：
  1. 语义搜索 vs 关键词搜索（同义词/多义词/跨语言）
  2. ANN 索引对比（暴力搜索 vs HNSW，速度与精度消融）
  3. 贯穿项目第三步：为 AI 助手构建向量知识库

依赖：pip install sentence-transformers chromadb numpy scikit-learn matplotlib
无需 API key，纯本地运行。
种子：所有实验使用 seed=1337
"""

import argparse
import time
import numpy as np

# ──────────────────────────────────────────────
# 0. 参数解析
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vector search demo for L29")
    p.add_argument("--mode", type=str, default="all",
                   choices=["all", "semantic-vs-keyword", "ann-comparison",
                            "build-knowledge-base"],
                   help="Which demo to run (default: all)")
    p.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                   help="Sentence Transformer model (default: all-MiniLM-L6-v2)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--top-k", type=int, default=3,
                   help="Number of results to return (default: 3)")
    return p.parse_args()


# ──────────────────────────────────────────────
# 1. 模拟知识库文档（中文客服场景）
# ──────────────────────────────────────────────
KNOWLEDGE_BASE = [
    # 退货与售后
    "我们的退货政策允许购买后 30 天内退货。请确保商品未拆封并保留购物凭证。",
    "如需申请退款，请联系客服并提供订单号。退款将在 5-7 个工作日内原路返回。",
    "商品质量问题可享受免费换货服务，无需承担运费。",
    # 产品信息
    "我们的旗舰产品 Pro Max 配备 12GB 内存和 256GB 存储空间，售价 4999 元。",
    "标准版产品配备 8GB 内存和 128GB 存储空间，售价 2999 元，性价比极高。",
    "所有产品均提供一年质保，可额外购买延保服务延长至三年。",
    # 折扣与会员
    "新用户首单可享受 9 折优惠，使用优惠码 WELCOME10。",
    "年度会员可享受全场 8.5 折优惠，并获得专属客服通道和优先发货权益。",
    "每年双十一和 618 大促期间，部分商品低至 5 折。",
    # 技术支持
    "如遇到网络连接超时问题，请先检查 Wi-Fi 连接，然后重启路由器。如仍无法解决，请拨打技术支持热线。",
    "API 接口超时的默认阈值为 30 秒，可通过配置文件 config.yaml 中的 timeout 字段调整。",
    "设备固件可通过设置 > 系统更新 > 检查更新进行升级。建议连接稳定的 Wi-Fi 网络后操作。",
    # 配送与物流
    "标准配送 3-5 个工作日送达，加急配送 1-2 个工作日送达（需额外支付 20 元运费）。",
    "支持全国配送，港澳台及海外地区暂不支持。偏远地区可能需要额外 2-3 天。",
    # 账户与隐私
    "您可以在个人中心 > 账户设置中修改密码和绑定手机号。",
    "我们严格遵守个人信息保护法，您的数据仅用于提供服务，不会与第三方共享。",
    # 企业服务
    "企业客户可申请批量采购折扣，10 台以上享 7 折优惠，请联系企业销售团队。",
    "我们提供企业定制化解决方案，包括专属 API 接口、私有化部署和 7×24 技术支持。",
    # 杂项
    "客服工作时间：周一至周五 9:00-18:00，周末及法定节假日 10:00-16:00。",
    "欢迎关注我们的官方公众号获取最新产品资讯和优惠活动信息。",
]


# ──────────────────────────────────────────────
# 2. 辅助函数
# ──────────────────────────────────────────────
def load_model(model_name: str):
    """加载 Sentence Transformer 模型。"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not found.")
        print("Run: pip install sentence-transformers")
        raise SystemExit(1)
    print(f"加载模型: {model_name} ...")
    model = SentenceTransformer(model_name)
    print(f"模型加载完成（维度: {model.get_sentence_embedding_dimension()}）")
    return model


def keyword_search(query: str, documents: list[str], top_k: int = 3) -> list[tuple[int, str, float]]:
    """简易关键词搜索：统计查询分词在文档中出现的次数作为分数。"""
    # 简单按字符级 n-gram 分词（教学用途，非生产级）
    query_terms = set(query)
    results = []
    for i, doc in enumerate(documents):
        score = sum(1 for term in query_terms if term in doc)
        results.append((i, doc, score))
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_k]


def print_results(title: str, results: list, show_score: bool = True):
    """格式化打印搜索结果。"""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    for rank, item in enumerate(results, 1):
        if show_score and len(item) >= 3:
            idx, doc, score = item[0], item[1], item[2]
            score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
            print(f"  #{rank} [score={score_str}] {doc[:80]}...")
        else:
            print(f"  #{rank} {item[1][:80]}...")
    print()


# ──────────────────────────────────────────────
# 3. Demo 1: 语义搜索 vs 关键词搜索
# ──────────────────────────────────────────────
def demo_semantic_vs_keyword(model, args):
    """对比语义搜索和关键词搜索在同义词、多义词、跨语言场景的差异。"""
    print("\n" + "=" * 60)
    print("  Demo 1: 语义搜索 vs 关键词搜索")
    print("=" * 60)

    # 编码知识库
    print("\n编码知识库文档...")
    t0 = time.perf_counter()
    doc_embeddings = model.encode(KNOWLEDGE_BASE, show_progress_bar=False)
    t1 = time.perf_counter()
    print(f"编码 {len(KNOWLEDGE_BASE)} 条文档耗时: {t1 - t0:.3f}s")

    test_queries = [
        ("如何解决连接超时", "同义词 + 意图理解"),
        ("价格太贵了", "意图理解（抱怨 → 折扣方案）"),
        ("how to return a product", "跨语言（英文查中文知识库）"),
        ("退货需要几天", "直接语义匹配"),
    ]

    for query, scenario in test_queries:
        print(f"\n{'━' * 60}")
        print(f"  查询: \"{query}\"")
        print(f"  场景: {scenario}")
        print(f"{'━' * 60}")

        # 关键词搜索
        kw_results = keyword_search(query, KNOWLEDGE_BASE, args.top_k)
        print_results("关键词搜索结果", kw_results)

        # 语义搜索
        query_embedding = model.encode([query], show_progress_bar=False)
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:args.top_k]
        sem_results = [(int(i), KNOWLEDGE_BASE[i], float(similarities[i]))
                       for i in top_indices]
        print_results("语义搜索结果", sem_results)


# ──────────────────────────────────────────────
# 4. Demo 2: ANN 索引对比（速度 vs 精度消融）
# ──────────────────────────────────────────────
def demo_ann_comparison(model, args):
    """对比暴力搜索 vs ANN 索引的速度和精度。"""
    print("\n" + "=" * 60)
    print("  Demo 2: ANN 索引对比（暴力搜索 vs HNSW）")
    print("=" * 60)

    np.random.seed(args.seed)
    dim = model.get_sentence_embedding_dimension()

    # 生成模拟数据（教学用，用随机向量 + 知识库真实向量混合）
    n_vectors = 50_000
    print(f"\n生成 {n_vectors:,} 个 {dim} 维模拟向量...")

    # 用知识库真实 embedding 作为种子，加噪声生成更多向量
    real_embeddings = model.encode(KNOWLEDGE_BASE, show_progress_bar=False)
    synthetic = []
    for _ in range(n_vectors // len(KNOWLEDGE_BASE)):
        noise = np.random.randn(len(KNOWLEDGE_BASE), dim).astype(np.float32) * 0.3
        synthetic.append(real_embeddings + noise)
    all_vectors = np.vstack(synthetic)[:n_vectors].astype(np.float32)

    # 归一化（余弦相似度等价于归一化后的内积）
    norms = np.linalg.norm(all_vectors, axis=1, keepdims=True)
    all_vectors = all_vectors / norms

    query_text = "退货需要几天"
    query_vec = model.encode([query_text], show_progress_bar=False).astype(np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)

    # --- 暴力搜索 ---
    print(f"\n暴力搜索 ({n_vectors:,} 向量)...")
    t0 = time.perf_counter()
    n_trials = 100
    for _ in range(n_trials):
        scores = all_vectors @ query_vec.T
        brute_top = np.argsort(scores.ravel())[::-1][:args.top_k]
    t1 = time.perf_counter()
    brute_time_ms = (t1 - t0) / n_trials * 1000
    brute_set = set(brute_top.tolist())
    print(f"  平均耗时: {brute_time_ms:.2f} ms/query")
    print(f"  Top-{args.top_k} indices: {brute_top.tolist()}")

    # --- HNSW (via sklearn NearestNeighbors with ball_tree 作为近似) ---
    # 教学用途：用 sklearn 的 BallTree 模拟 ANN 的"近似但更快"效果
    # 生产环境应使用 FAISS 或向量数据库的 HNSW 实现
    try:
        from sklearn.neighbors import NearestNeighbors

        print(f"\n构建 Ball Tree 索引 ({n_vectors:,} 向量)...")
        t0 = time.perf_counter()
        nn = NearestNeighbors(n_neighbors=args.top_k, metric='cosine',
                              algorithm='ball_tree')
        nn.fit(all_vectors)
        t1 = time.perf_counter()
        build_time = t1 - t0
        print(f"  索引构建耗时: {build_time:.3f}s")

        print(f"\n索引搜索 ({n_vectors:,} 向量)...")
        t0 = time.perf_counter()
        for _ in range(n_trials):
            distances, indices = nn.kneighbors(query_vec)
        t1 = time.perf_counter()
        ann_time_ms = (t1 - t0) / n_trials * 1000
        ann_set = set(indices[0].tolist())

        recall = len(brute_set & ann_set) / len(brute_set)
        print(f"  平均耗时: {ann_time_ms:.2f} ms/query")
        print(f"  Top-{args.top_k} indices: {indices[0].tolist()}")
        print(f"  Recall@{args.top_k}: {recall:.2%}")

        speedup = brute_time_ms / ann_time_ms if ann_time_ms > 0 else float('inf')
        print(f"\n{'━' * 60}")
        print(f"  速度对比：暴力 {brute_time_ms:.2f}ms vs 索引 {ann_time_ms:.2f}ms")
        print(f"  加速比: {speedup:.1f}x")
        print(f"  Recall: {recall:.2%}")
        print(f"  结论: 索引快 {speedup:.1f} 倍，Recall {recall:.0%}")
        print(f"{'━' * 60}")

    except ImportError:
        print("sklearn not available, skipping ANN comparison")

    # --- 可视化 ---
    try:
        import sys as _sys, pathlib as _pathlib
        _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[4] / "design" / "meta"))
        import cjk_font  # noqa: F401  - applies CJK font config
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        print("\n生成 t-SNE/PCA 可视化...")
        pca = PCA(n_components=2, random_state=args.seed)
        reduced = pca.fit_transform(all_vectors[:2000])
        query_reduced = pca.transform(query_vec)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.scatter(reduced[:, 0], reduced[:, 1], s=3, alpha=0.3,
                   c='steelblue', label='文档向量')
        ax.scatter(query_reduced[:, 0], query_reduced[:, 1], s=200,
                   c='red', marker='*', zorder=5, label=f'查询: "{query_text}"')

        # 标注 top-k 结果
        for idx in brute_top:
            if idx < 2000:
                ax.scatter(reduced[idx, 0], reduced[idx, 1], s=80,
                           c='orange', edgecolors='red', linewidths=2,
                           zorder=4, label='Top-K 结果' if idx == brute_top[0] else '')

        ax.set_title(f'向量空间可视化（PCA 降维，{n_vectors:,} 向量中取前 2000）',
                     fontsize=13)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.tight_layout()
        plt.savefig('demo/figures/vector_space_pca.png', dpi=150)
        print("可视化已保存: demo/figures/vector_space_pca.png")
        plt.close()

    except ImportError:
        print("matplotlib not available, skipping visualization")


# ──────────────────────────────────────────────
# 5. Demo 3: 贯穿项目第三步 — 构建向量知识库
# ──────────────────────────────────────────────
def demo_build_knowledge_base(model, args):
    """用 Chroma 构建向量知识库，模拟贯穿项目第三步。"""
    print("\n" + "=" * 60)
    print("  Demo 3: 贯穿项目第三步 — 为 Sage 构建向量知识库")
    print("=" * 60)

    try:
        import chromadb
    except ImportError:
        print("ERROR: chromadb not found. Run: pip install chromadb")
        return

    # Step 1: 创建 Chroma 客户端（内存模式，不写磁盘）
    print("\n[Step 1] 创建 Chroma 客户端（内存模式）...")
    client = chromadb.Client()

    # Step 2: 创建 Collection
    print("[Step 2] 创建 Collection 'sage_knowledge_base'...")
    collection = client.create_collection(
        name="sage_knowledge_base",
        metadata={"hnsw:space": "cosine"},  # 使用余弦距离
    )

    # Step 3: 编码并存入文档
    print(f"[Step 3] 编码 {len(KNOWLEDGE_BASE)} 条文档并存入 Chroma...")
    t0 = time.perf_counter()
    embeddings = model.encode(KNOWLEDGE_BASE, show_progress_bar=False).tolist()
    t1 = time.perf_counter()

    collection.add(
        ids=[f"doc_{i}" for i in range(len(KNOWLEDGE_BASE))],
        embeddings=embeddings,
        documents=KNOWLEDGE_BASE,
        metadatas=[{"category": _categorize(doc)} for doc in KNOWLEDGE_BASE],
    )
    t2 = time.perf_counter()
    print(f"  Embedding 耗时: {t1 - t0:.3f}s")
    print(f"  存入 Chroma 耗时: {t2 - t1:.3f}s")
    print(f"  知识库大小: {collection.count()} 条文档")

    # Step 4: 语义搜索
    print("\n[Step 4] 语义搜索测试")

    queries = [
        "退货需要几天？",
        "有没有折扣？",
        "如何解决网络超时问题？",
        "你们的产品配置是什么？",
        "个人数据安全吗？",
    ]

    for query in queries:
        query_embedding = model.encode([query], show_progress_bar=False).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=args.top_k,
            include=["documents", "distances", "metadatas"],
        )

        print(f"\n{'━' * 60}")
        print(f"  查询: \"{query}\"")
        print(f"{'━' * 60}")
        for rank in range(len(results["documents"][0])):
            doc = results["documents"][0][rank]
            dist = results["distances"][0][rank]
            cat = results["metadatas"][0][rank]["category"]
            # Chroma 返回的是距离（越小越好），转成相似度
            similarity = 1 - dist
            print(f"  #{rank + 1} [sim={similarity:.4f}] [{cat}] {doc[:70]}...")

    # Step 5: 带元数据过滤的搜索
    print(f"\n{'━' * 60}")
    print("  [Step 5] 带元数据过滤的搜索")
    print(f"{'━' * 60}")
    filter_query = "有什么优惠？"
    query_embedding = model.encode([filter_query], show_progress_bar=False).tolist()

    print(f'\n  查询: "{filter_query}" (过滤: category="折扣与会员")')
    results_filtered = collection.query(
        query_embeddings=query_embedding,
        n_results=args.top_k,
        where={"category": "折扣与会员"},
        include=["documents", "distances", "metadatas"],
    )
    for rank in range(len(results_filtered["documents"][0])):
        doc = results_filtered["documents"][0][rank]
        dist = results_filtered["distances"][0][rank]
        similarity = 1 - dist
        print(f"  #{rank + 1} [sim={similarity:.4f}] {doc[:70]}...")

    print(f"\n✅ 向量知识库构建完成！共 {collection.count()} 条文档。")
    print("  下一步（L30 RAG）：把检索结果注入 LLM 上下文，生成最终回答。")


def _categorize(doc: str) -> str:
    """简单的文档分类（教学用途）。"""
    if any(w in doc for w in ["退货", "退款", "换货"]):
        return "退货与售后"
    if any(w in doc for w in ["内存", "存储", "质保", "配备"]):
        return "产品信息"
    if any(w in doc for w in ["折", "优惠", "会员", "促"]):
        return "折扣与会员"
    if any(w in doc for w in ["超时", "固件", "API", "连接", "更新"]):
        return "技术支持"
    if any(w in doc for w in ["配送", "物流", "送达", "运费"]):
        return "配送与物流"
    if any(w in doc for w in ["密码", "隐私", "数据", "账户"]):
        return "账户与隐私"
    if any(w in doc for w in ["企业", "批量", "定制"]):
        return "企业服务"
    return "其他"


# ──────────────────────────────────────────────
# 6. 主函数
# ──────────────────────────────────────────────
def main():
    args = parse_args()
    np.random.seed(args.seed)
    model = load_model(args.model)

    if args.mode in ("all", "semantic-vs-keyword"):
        demo_semantic_vs_keyword(model, args)

    if args.mode in ("all", "ann-comparison"):
        demo_ann_comparison(model, args)

    if args.mode in ("all", "build-knowledge-base"):
        demo_build_knowledge_base(model, args)

    print("\n" + "=" * 60)
    print("  所有 Demo 运行完成 ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()