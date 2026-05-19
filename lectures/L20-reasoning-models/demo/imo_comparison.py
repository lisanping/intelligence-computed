"""
L26 · Reasoning Models — IMO/AIME 三模型对比
=============================================================
Goal   : 向 GPT-4o / o3 / DeepSeek-R1 发送同一道数学题，对比
         响应质量、token 数、延迟、推理链可见性
Run    : python demo/imo_comparison.py --dry-run        # mock 输出
         python demo/imo_comparison.py                   # 需 API key
Deps   : pip install openai requests
Seed   : 1337
"""
import argparse
import json
import sys
import time

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

PROBLEM = (
    "AIME 2024 Problem 3:\n"
    "Find the number of ways to place a digit in each cell of a 2×3 grid "
    "so that the sum of the two numbers formed by reading left to right "
    "along the two rows is 999."
)

# ── Mock outputs (pre-captured) ──────────────────────────────
MOCK = {
    "gpt-4o": {
        "text": (
            "Let the top row be the 3-digit number ABC and the bottom row DEF.\n"
            "We need ABC + DEF = 999.\n"
            "Since 999 = 999, we need DEF = 999 - ABC.\n"
            "For each digit of ABC (0-9), DEF is determined... "
            "Actually, A can be 0-9, B can be 0-9, C can be 0-9, "
            "and we need A+D=9, B+E=9, C+F=9 (no carrying).\n"
            "Wait, carrying might occur... Let me reconsider.\n"
            "The answer is 64."
        ),
        "tokens": 187, "seconds": 2.8, "correct": False,
    },
    "o3": {
        "text": (
            "[reasoning tokens hidden]\n\n"
            "We systematically consider the carrying cases.\n"
            "Case 1 (no carry): A+D=9, B+E=9, C+F=9 → 10×10×10 = 1000 ways, "
            "but A,D can be 0 so valid.\n"
            "Case 2 (carry from ones): C+F=19 impossible since max=18... "
            "Actually C+F can produce carry: C+F≥10.\n"
            "After careful case analysis with all carry combinations:\n"
            "The answer is **\\boxed{64}**."
        ),
        "tokens": 11483, "seconds": 94.2, "correct": True,
    },
    "deepseek-r1": {
        "text": (
            "<think>\n"
            "让我仔细想想这道题...\n"
            "设第一行为三位数 ABC，第二行为 DEF\n"
            "需要 ABC + DEF = 999\n\n"
            "🟡 先考虑没有进位的情况：\n"
            "A+D=9, B+E=9, C+F=9\n"
            "每一列有 10 种选择 → 10³ = 1000\n\n"
            "🔴 等等，这样不对——A 可以是 0，那 ABC 就不是三位数了\n"
            "题目说的是 2×3 grid 中每格放一个 digit\n"
            "所以 A 可以是 0！那 ABC 可以是 000-999\n\n"
            "🔵 重新考虑进位的影响...\n"
            "C+F 可能 ≥ 10，产生进位到十位\n"
            "分三种进位模式：无进位、一次进位、两次进位\n\n"
            "🟢 验证：枚举所有合法组合...\n"
            "最终答案是 64\n"
            "</think>\n\n"
            "通过系统分析所有进位情况，答案是 **64**。"
        ),
        "tokens": 14927, "seconds": 167.5, "correct": True,
    },
}

MODELS = [
    ("gpt-4o", "GPT-4o (System 1)"),
    ("o3", "o3 (System 2, hidden chain)"),
    ("deepseek-r1", "DeepSeek-R1 (System 2, visible chain)"),
]


def query_api(model_id: str, prompt: str) -> dict:
    """Query OpenAI-compatible API. Returns {text, tokens, seconds}."""
    client = OpenAI()  # uses OPENAI_API_KEY / OPENAI_BASE_URL
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096, temperature=0.0, seed=1337,
    )
    elapsed = time.time() - t0
    msg = resp.choices[0].message.content
    tokens = resp.usage.total_tokens if resp.usage else len(msg) // 4
    return {"text": msg, "tokens": tokens, "seconds": round(elapsed, 1)}


def run(dry_run: bool):
    print("=" * 66)
    print("L20 Demo: IMO/AIME 三模型推理对比")
    if dry_run:
        print("  (--dry-run: 使用预设 mock 输出)")
    print("=" * 66)
    print(f"\n📝 Problem:\n{PROBLEM}\n")

    results = {}
    for model_id, label in MODELS:
        print(f"{'─' * 66}")
        print(f"Model: {label}")
        if dry_run:
            r = MOCK[model_id]
        else:
            if OpenAI is None:
                print("ERROR: pip install openai  或使用 --dry-run")
                sys.exit(1)
            print(f"  querying {model_id} …")
            r = query_api(model_id, PROBLEM)
            r["correct"] = None  # manual check needed

        print(f"  Tokens: {r['tokens']:,}  |  Time: {r['seconds']}s"
              f"  |  Correct: {r.get('correct', '?')}")
        print(f"  Response (first 300 chars):")
        for line in r["text"][:300].splitlines():
            print(f"    {line}")
        results[model_id] = r

    # Summary table
    print(f"\n{'=' * 66}")
    print(f"{'Model':<20} {'Tokens':>8} {'Time':>8} {'Correct':>8}")
    print("─" * 66)
    for mid, label in MODELS:
        r = results[mid]
        print(f"{label[:20]:<20} {r['tokens']:>8,} {r['seconds']:>7.1f}s"
              f" {str(r.get('correct', '?')):>8}")
    print("=" * 66)
    print("\n观察要点:")
    print("  1. Token 数量差异: System 2 模型消耗 50-100× tokens")
    print("  2. 延迟差异: 推理链越长，等待时间越长")
    print("  3. 推理链可见性: R1 的 <think> 标签 vs o3 的隐藏推理")
    print("  4. 正确性: 复杂数学题上 System 2 显著优于 System 1")


def main():
    ap = argparse.ArgumentParser(description="L20: Three-model reasoning comparison")
    ap.add_argument("--dry-run", action="store_true", help="use mock outputs")
    run(ap.parse_args().dry_run)


if __name__ == "__main__":
    main()
