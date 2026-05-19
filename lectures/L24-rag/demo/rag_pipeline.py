"""
第 24 讲 · 完整 RAG Pipeline Demo
==================================

演示：
  1. 分块策略对比（固定 vs 递归）
  2. Hybrid Search（BM25 + 向量检索 + RRF 融合）
  3. Cross-Encoder Reranking（精排 vs 无精排）
  4. 完整 RAG Pipeline（检索 + Reranking + LLM 生成）
  5. 消融实验（逐个开关模块观察效果变化）

依赖：pip install sentence-transformers chromadb numpy scikit-learn rank-bm25 openai matplotlib
无需 API key 即可运行检索部分；LLM 生成需要 OPENAI_API_KEY（可选）。
种子：所有实验使用 seed=1337
"""

import argparse
import os
import re
import time
import numpy as np
from typing import Optional

# ──────────────────────────────────────────────
# 0. 参数解析
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG Pipeline demo for L30")
    p.add_argument("--mode", type=str, default="all",
                   choices=["all", "chunking-comparison", "hybrid-search",
                            "reranking-comparison", "full-pipeline", "ablation"],
                   help="Which demo to run (default: all)")
    p.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                   help="Sentence Transformer model (default: all-MiniLM-L6-v2)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--top-k", type=int, default=5,
                   help="Number of retrieval results (default: 5)")
    p.add_argument("--top-n", type=int, default=3,
                   help="Number of results after reranking (default: 3)")
    return p.parse_args()


# ──────────────────────────────────────────────
# 1. 模拟知识库文档（中文客服场景 — 与 L29 共享）
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
# 2. 分块策略
# ──────────────────────────────────────────────
def fixed_size_chunk(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """固定大小分块。"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def recursive_character_chunk(text: str, chunk_size: int = 200,
                              overlap: int = 50) -> list[str]:
    """递归字符分块：按段落→句子→空格→字符优先级切割。"""
    separators = ["\n\n", "\n", "。", ".", "！", "!", "？", "?", "；", ";", " "]

    def _split(text: str, seps: list[str]) -> list[str]:
        if not seps or len(text) <= chunk_size:
            return [text] if text.strip() else []
        sep = seps[0]
        parts = text.split(sep)
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > chunk_size:
                    chunks.extend(_split(part, seps[1:]))
                else:
                    current = part
        if current:
            chunks.append(current)
        return [c.strip() for c in chunks if c.strip()]

    return _split(text, separators)


# ──────────────────────────────────────────────
# 3. 检索组件
# ──────────────────────────────────────────────
def bm25_search(query: str, documents: list[str], top_k: int = 5) -> list[tuple[int, str, float]]:
    """BM25 关键词检索。"""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("WARNING: rank-bm25 not found. Run: pip install rank-bm25")
        return []
    tokenized_docs = [list(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = list(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(i), documents[i], float(scores[i])) for i in top_indices]


def vector_search(query: str, documents: list[str], model,
                  top_k: int = 5) -> list[tuple[int, str, float]]:
    """向量语义检索。"""
    from sklearn.metrics.pairwise import cosine_similarity
    doc_embeddings = model.encode(documents, show_progress_bar=False)
    query_embedding = model.encode([query], show_progress_bar=False)
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(int(i), documents[i], float(similarities[i])) for i in top_indices]


def reciprocal_rank_fusion(
    result_lists: list[list[tuple[int, str, float]]],
    k: int = 60,
    top_k: int = 5,
) -> list[tuple[int, str, float]]:
    """RRF 融合多个检索结果列表。"""
    rrf_scores: dict[int, float] = {}
    doc_map: dict[int, str] = {}
    for results in result_lists:
        for rank, (idx, doc, _score) in enumerate(results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
            doc_map[idx] = doc
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(idx, doc_map[idx], score) for idx, score in sorted_items[:top_k]]


def hybrid_search(query: str, documents: list[str], model,
                  top_k: int = 5) -> list[tuple[int, str, float]]:
    """混合检索：BM25 + 向量 → RRF 融合。"""
    bm25_results = bm25_search(query, documents, top_k=top_k * 2)
    vec_results = vector_search(query, documents, model, top_k=top_k * 2)
    return reciprocal_rank_fusion([bm25_results, vec_results], top_k=top_k)


# ──────────────────────────────────────────────
# 4. Cross-Encoder Reranking
# ──────────────────────────────────────────────
def rerank_cross_encoder(
    query: str,
    candidates: list[tuple[int, str, float]],
    top_n: int = 3,
) -> list[tuple[int, str, float]]:
    """用 Cross-Encoder 对候选结果重排序。"""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("WARNING: sentence-transformers CrossEncoder not available.")
        return candidates[:top_n]
    ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    pairs = [(query, doc) for _, doc, _ in candidates]
    scores = ce_model.predict(pairs)
    scored = [(candidates[i][0], candidates[i][1], float(scores[i]))
              for i in range(len(candidates))]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_n]


# ──────────────────────────────────────────────
# 5. LLM 生成（可选 — 需要 OPENAI_API_KEY）
# ──────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """你是一个专业的客服助手。请根据以下参考资料回答用户的问题。

## 规则
1. 只使用参考资料中的信息来回答，不要使用你自己的知识。
2. 如果参考资料中没有相关信息，请明确说"根据现有资料，我无法回答这个问题"。
3. 回答时引用信息来源，格式为 [来源: 文档编号]。
4. 如果多个来源信息矛盾，请指出矛盾并给出各方说法。

## 参考资料
{context}

## 用户问题
{question}

## 回答"""


def generate_answer(question: str, documents: list[tuple[int, str, float]]) -> str:
    """用 LLM 生成回答（需要 OPENAI_API_KEY）。"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        context_str = "\n".join(
            f"[文档{i+1}] {doc}" for i, (_, doc, _) in enumerate(documents)
        )
        return (
            f"[LLM 生成跳过 — 未设置 OPENAI_API_KEY]\n\n"
            f"以下是将发送给 LLM 的 Prompt 中的参考资料部分：\n{context_str}"
        )
    try:
        from openai import OpenAI
    except ImportError:
        return "[ERROR: openai package not installed. Run: pip install openai]"

    context_str = "\n".join(
        f"[文档{i+1}] {doc}" for i, (_, doc, _) in enumerate(documents)
    )
    prompt = RAG_PROMPT_TEMPLATE.format(context=context_str, question=question)

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500,
    )
    return response.choices[0].message.content or "[Empty response]"


# ──────────────────────────────────────────────
# 6. 辅助打印
# ──────────────────────────────────────────────
def print_results(title: str, results: list[tuple[int, str, float]]):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    for rank, (idx, doc, score) in enumerate(results, 1):
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        doc_preview = doc[:70].replace("\n", " ")
        print(f"  #{rank} [score={score_str}] {doc_preview}...")
    print()


def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ──────────────────────────────────────────────
# 7. Demo 模块
# ──────────────────────────────────────────────
def demo_chunking_comparison(args):
    """对比固定分块 vs 递归分块。"""
    print_header("Demo: 分块策略对比（固定 vs 递归）")

    # 模拟一段跨段落的长文档
    long_doc = (
        "退货政策说明\n\n"
        "一、退货条件\n"
        "购买后 30 天内可申请退货。商品必须未拆封，且需保留购物凭证。"
        "如果商品已拆封但存在质量问题，仍可申请退货。\n\n"
        "二、退款流程\n"
        "请联系客服提供订单号。审核通过后，退款将在 5-7 个工作日内原路返回。"
        "如使用信用卡支付，退款可能需要额外 1-2 个工作日到账。\n\n"
        "三、换货服务\n"
        "质量问题可享受免费换货，无需承担运费。"
        "换货商品将在 3 个工作日内寄出。"
    )

    print(f"\n原始文档 ({len(long_doc)} 字符):")
    print(f"  {long_doc[:100]}...")

    fixed_chunks = fixed_size_chunk(long_doc, chunk_size=100, overlap=20)
    recursive_chunks = recursive_character_chunk(long_doc, chunk_size=100, overlap=20)

    print(f"\n固定分块 ({len(fixed_chunks)} 块, chunk_size=100):")
    for i, chunk in enumerate(fixed_chunks):
        print(f"  块 {i+1}: [{len(chunk):3d} 字符] {chunk[:60]}...")

    print(f"\n递归分块 ({len(recursive_chunks)} 块, chunk_size=100):")
    for i, chunk in enumerate(recursive_chunks):
        print(f"  块 {i+1}: [{len(chunk):3d} 字符] {chunk[:60]}...")

    print("\n对比：")
    print("  固定分块 → 可能在句子中间切断（检查块边界）")
    print("  递归分块 → 沿段落/句子边界切割，语义更完整")


def demo_hybrid_search(model, args):
    """演示 Hybrid Search（BM25 + 向量 + RRF）。"""
    print_header("Demo: Hybrid Search（BM25 + 向量 + RRF 融合）")

    test_queries = [
        ("getCustomerById 接口超时", "精确 API 名称 + 语义意图"),
        ("退货需要几天", "语义匹配为主"),
        ("价格太贵了怎么办", "意图理解"),
    ]

    for query, scenario in test_queries:
        print(f"\n{'━' * 60}")
        print(f"  查询: \"{query}\"  |  场景: {scenario}")
        print(f"{'━' * 60}")

        bm25_results = bm25_search(query, KNOWLEDGE_BASE, args.top_k)
        vec_results = vector_search(query, KNOWLEDGE_BASE, model, args.top_k)
        hybrid_results = reciprocal_rank_fusion(
            [bm25_results, vec_results], top_k=args.top_k
        )

        print_results("BM25 关键词检索", bm25_results[:3])
        print_results("向量语义检索", vec_results[:3])
        print_results("Hybrid Search (RRF 融合)", hybrid_results[:3])


def demo_reranking_comparison(model, args):
    """对比有/无 Reranking 的检索质量。"""
    print_header("Demo: Reranking 对比（有 vs 无 Cross-Encoder）")

    test_queries = [
        "退货需要几天",
        "如何解决连接超时",
        "对比标准版和 Pro Max 的配置",
    ]

    for query in test_queries:
        print(f"\n{'━' * 60}")
        print(f"  查询: \"{query}\"")
        print(f"{'━' * 60}")

        vec_results = vector_search(query, KNOWLEDGE_BASE, model, top_k=10)
        print_results(f"向量检索 Top-{args.top_n}（无 Reranking）", vec_results[:args.top_n])

        reranked = rerank_cross_encoder(query, vec_results, top_n=args.top_n)
        print_results(f"Cross-Encoder Reranking Top-{args.top_n}", reranked)


def demo_full_pipeline(model, args):
    """完整 RAG Pipeline：检索 + Reranking + 生成。"""
    print_header("Demo: 完整 RAG Pipeline")

    test_queries = [
        "退货需要几天？",
        "对比标准版和 Pro Max 的配置和价格",
        "明天北京的天气怎么样？",
    ]

    for query in test_queries:
        print(f"\n{'━' * 60}")
        print(f"  用户问题: \"{query}\"")
        print(f"{'━' * 60}")

        # Step 1: Hybrid Search
        results = hybrid_search(query, KNOWLEDGE_BASE, model, top_k=10)
        print_results("Step 1 - Hybrid Search Top-10", results[:5])

        # Step 2: Reranking
        reranked = rerank_cross_encoder(query, results, top_n=args.top_n)
        print_results(f"Step 2 - Reranking Top-{args.top_n}", reranked)

        # Step 3: Generate
        print(f"  Step 3 - LLM 生成回答:")
        answer = generate_answer(query, reranked)
        print(f"  {answer}\n")


def demo_ablation(model, args):
    """消融实验：逐个开关 RAG 模块。"""
    print_header("Demo: 消融实验（逐个加模块观察效果变化）")

    test_queries = [
        "退货需要几天",
        "如何解决连接超时问题",
        "对比标准版和 Pro Max",
    ]

    configs = [
        ("Naive: 纯向量检索",            {"hybrid": False, "rerank": False}),
        ("+ Hybrid Search",              {"hybrid": True,  "rerank": False}),
        ("+ Hybrid + Reranking",         {"hybrid": True,  "rerank": True}),
    ]

    for query in test_queries:
        print(f"\n{'━' * 60}")
        print(f"  查询: \"{query}\"")
        print(f"{'━' * 60}")

        for config_name, flags in configs:
            if flags["hybrid"]:
                results = hybrid_search(query, KNOWLEDGE_BASE, model, top_k=10)
            else:
                results = vector_search(query, KNOWLEDGE_BASE, model, top_k=10)

            if flags["rerank"]:
                results = rerank_cross_encoder(query, results, top_n=args.top_n)
            else:
                results = results[:args.top_n]

            docs_preview = " | ".join(doc[:30] for _, doc, _ in results)
            print(f"  [{config_name}]")
            print(f"    Top-{args.top_n}: {docs_preview}")
        print()


# ──────────────────────────────────────────────
# 8. 主函数
# ──────────────────────────────────────────────
def load_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not found.")
        print("Run: pip install sentence-transformers")
        raise SystemExit(1)
    print(f"加载模型: {model_name} ...")
    m = SentenceTransformer(model_name)
    print(f"模型加载完成（维度: {m.get_sentence_embedding_dimension()}）")
    return m


def main():
    args = parse_args()
    np.random.seed(args.seed)

    mode = args.mode
    need_model = mode != "chunking-comparison"

    model = load_model(args.model) if need_model else None

    if mode in ("all", "chunking-comparison"):
        demo_chunking_comparison(args)

    if mode in ("all", "hybrid-search"):
        demo_hybrid_search(model, args)

    if mode in ("all", "reranking-comparison"):
        demo_reranking_comparison(model, args)

    if mode in ("all", "full-pipeline"):
        demo_full_pipeline(model, args)

    if mode in ("all", "ablation"):
        demo_ablation(model, args)

    print("\n" + "=" * 60)
    print("  所有 demo 运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()