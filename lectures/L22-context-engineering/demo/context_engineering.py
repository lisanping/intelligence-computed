"""
L28 · Context Engineering — 四种上下文管理策略对比
=============================================================
Goal   : 模拟 50 轮对话，对比截断/摘要/关键信息提取三种策略
         在"回忆早期信息"任务上的表现差异
Run    : python demo/context_engineering.py                  # 全部策略
         python demo/context_engineering.py --strategy trunc # 仅截断
Deps   : pip install tiktoken  (可选，用于精确 token 计数)
Seed   : 1337
"""
import argparse
import random
import textwrap

random.seed(1337)

# ── Simulated conversation data ──────────────────────────────
# Early turns contain key personal info; later turns are generic filler
EARLY_TOPICS = [
    ("我对猫过敏，所以我养了两只柯基", "preference"),
    ("我的生日是 3 月 14 日，圆周率日", "fact"),
    ("下周三要交项目报告，题目是量子计算综述", "todo"),
    ("我最喜欢的编程语言是 Rust", "preference"),
    ("今天天气真好，出去跑了 5 公里", "chitchat"),
    ("帮我翻译一段英文论文摘要", "task"),
    ("周末打算去西湖骑行", "chitchat"),
    ("我正在学习 Transformer 架构", "fact"),
    ("能推荐一本深度学习入门书吗", "task"),
    ("我的预算上限是 500 元", "preference"),
]

FILLER_TOPICS = [
    "帮我写一段 Python 排序代码",
    "解释一下什么是梯度下降",
    "今天中午吃什么好",
    "帮我检查这段代码的 bug",
    "推荐一部科幻电影",
    "这个公式怎么推导",
    "总结一下这篇文章的要点",
    "帮我写一封邮件",
]

RECALL_QUESTIONS = [
    ("我对什么动物过敏？", "猫", 0),
    ("我的生日是哪天？", "3 月 14 日", 1),
    ("我最喜欢的编程语言是？", "Rust", 3),
    ("我的预算上限是多少？", "500 元", 9),
]


def generate_conversation(n_turns: int = 50) -> list[dict]:
    """Generate a simulated multi-turn conversation.
    Early turns contain key info; later turns are generic filler."""
    messages = [{"role": "system", "content": "你是一个有记忆的 AI 助手。"}]
    for i in range(n_turns):
        if i < len(EARLY_TOPICS):
            topic = EARLY_TOPICS[i][0]
        else:
            topic = FILLER_TOPICS[(i - len(EARLY_TOPICS)) % len(FILLER_TOPICS)]
        user_msg = f"[Turn {i+1}] {topic}"
        assistant_msg = f"好的，收到。这是第 {i+1} 轮对话的回复。"
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    return messages


# ── Strategy 1: Truncation ───────────────────────────────────
def strategy_truncate(messages: list[dict], max_turns: int = 10) -> list[dict]:
    """Keep system prompt + last N turns (2N messages)."""
    system = [m for m in messages if m["role"] == "system"]
    history = [m for m in messages if m["role"] != "system"]
    return system + history[-(max_turns * 2):]


# ── Strategy 2: Summary Compression ──────────────────────────
def strategy_summarize(messages: list[dict], threshold: int = 20) -> list[dict]:
    """Compress old messages into a summary when exceeding threshold turns."""
    system = [m for m in messages if m["role"] == "system"]
    history = [m for m in messages if m["role"] != "system"]
    n_turns = len(history) // 2
    if n_turns <= threshold:
        return messages

    old = history[:-(threshold * 2)]
    recent = history[-(threshold * 2):]
    # Simulated summary (in production: LLM call)
    summary = _mock_summary(old)
    return system + [{"role": "system", "content": f"[对话摘要]\n{summary}"}] + recent


def _mock_summary(old_messages: list[dict]) -> str:
    """Mock LLM summary — extracts key info from old messages."""
    facts = []
    for m in old_messages:
        if m["role"] == "user":
            for topic, ttype in EARLY_TOPICS:
                if topic[:6] in m["content"] and ttype != "chitchat":
                    facts.append(topic)
    return "用户历史信息：" + "；".join(set(facts)) if facts else "（无重要信息）"


# ── Strategy 3: Key Info Extraction ──────────────────────────
def strategy_key_info(messages: list[dict], max_recent: int = 10) -> list[dict]:
    """Extract preferences/facts/TODOs, discard chitchat."""
    system = [m for m in messages if m["role"] == "system"]
    history = [m for m in messages if m["role"] != "system"]

    # Simulated extraction (in production: LLM call with extraction prompt)
    extracted = {"preferences": [], "facts": [], "todos": []}
    for m in history:
        if m["role"] != "user":
            continue
        for topic, ttype in EARLY_TOPICS:
            if topic[:6] in m["content"]:
                if ttype == "preference":
                    extracted["preferences"].append(topic)
                elif ttype == "fact":
                    extracted["facts"].append(topic)
                elif ttype == "todo":
                    extracted["todos"].append(topic)

    profile = "【用户画像】\n"
    for k, vs in extracted.items():
        if vs:
            profile += f"  {k}: {'; '.join(set(vs))}\n"

    recent = history[-(max_recent * 2):]
    return system + [{"role": "system", "content": profile}] + recent


# ── Evaluation ───────────────────────────────────────────────
def evaluate_recall(context: list[dict], questions: list) -> dict:
    """Check if key info from early turns is still in context."""
    context_text = " ".join(m["content"] for m in context)
    results = {}
    for q, answer, _turn_idx in questions:
        found = answer in context_text
        results[q] = {"answer": answer, "found": found}
    return results


def count_tokens(messages: list[dict]) -> int:
    """Approximate token count (4 chars ≈ 1 token for CJK)."""
    return sum(len(m["content"]) for m in messages) // 2


# ── Main ─────────────────────────────────────────────────────
STRATEGIES = {
    "trunc": ("Truncation (last 10 turns)", strategy_truncate),
    "summary": ("Summary Compression", strategy_summarize),
    "keyinfo": ("Key Info Extraction", strategy_key_info),
}


def run(strategy_name: str):
    print("=" * 62)
    print("L22 Demo: 上下文管理策略对比 (50 轮对话)")
    print("=" * 62)

    conversation = generate_conversation(50)
    full_tokens = count_tokens(conversation)
    print(f"\n原始对话: {len(conversation)} 条消息, ~{full_tokens:,} tokens\n")

    targets = STRATEGIES if strategy_name == "all" else {strategy_name: STRATEGIES[strategy_name]}

    for key, (label, fn) in targets.items():
        ctx = fn(conversation)
        tokens = count_tokens(ctx)
        recall = evaluate_recall(ctx, RECALL_QUESTIONS)
        hits = sum(1 for r in recall.values() if r["found"])

        print(f"{'─' * 62}")
        print(f"策略: {label}")
        print(f"  消息数: {len(ctx):>4}  |  ~Tokens: {tokens:>6,}"
              f"  |  压缩比: {full_tokens/max(tokens,1):.1f}×")
        print(f"  早期信息召回: {hits}/{len(RECALL_QUESTIONS)}")
        for q, r in recall.items():
            status = "✅" if r["found"] else "❌"
            print(f"    {status} {q} → {r['answer']}")

    print(f"\n{'=' * 62}")
    print("观察要点:")
    print("  1. 截断: 零成本但完全丢失早期信息 (amnesia)")
    print("  2. 摘要: 保留关键事实，但需要额外 LLM 调用")
    print("  3. 关键信息提取: 最佳召回，适合生产环境")
    print("  4. Token 数差异 → 直接影响 API 成本")
    print("=" * 62)


def main():
    ap = argparse.ArgumentParser(description="L22: Context management strategies")
    ap.add_argument("--strategy", choices=["trunc", "summary", "keyinfo", "all"],
                    default="all", help="which strategy to demo")
    run(ap.parse_args().strategy)


if __name__ == "__main__":
    main()
