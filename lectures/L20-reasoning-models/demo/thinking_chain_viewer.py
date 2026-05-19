"""
L26 · Reasoning Models — 思维链查看器
=============================================================
Goal   : 解析 DeepSeek-R1 风格的 <think>...</think> 输出，
         按推理行为分类着色显示
Run    : python demo/thinking_chain_viewer.py [--file chain.txt]
Deps   : (none — pure stdlib)
Seed   : 1337
"""
import argparse
import re
import sys

# ── ANSI colors for terminal output ─────────────────────────
COLORS = {
    "attempt":   "\033[33m",   # 🟡 yellow — hypothesis / attempt
    "error":     "\033[31m",   # 🔴 red    — error discovery
    "backtrack": "\033[34m",   # 🔵 blue   — backtracking / switching
    "verify":    "\033[32m",   # 🟢 green  — verification / confirmation
    "reset":     "\033[0m",
}

PATTERNS = [
    ("error",     r"等等|不对|错了|wait|wrong|mistake|no,\s|actually|hmm"),
    ("backtrack", r"换一种|重新|另一个|let me re|try (?:a )?different|instead"),
    ("verify",    r"验证|确认|check|verify|indeed|correctly|答案是|answer is|boxed"),
    ("attempt",   r"考虑|假设|设|尝试|try|consider|suppose|let's|if we"),
]


def classify_line(line: str) -> str:
    """Classify a reasoning line by its dominant behavior."""
    lower = line.lower()
    for label, pattern in PATTERNS:
        if re.search(pattern, lower):
            return label
    return ""


def parse_thinking_chain(text: str) -> list[tuple[str, str]]:
    """Extract <think> block and classify each line."""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    body = m.group(1) if m else text
    results = []
    for line in body.strip().splitlines():
        line = line.strip()
        if line:
            results.append((classify_line(line), line))
    return results


def render_terminal(classified: list[tuple[str, str]]):
    """Print color-coded reasoning chain to terminal."""
    icons = {"attempt": "🟡", "error": "🔴", "backtrack": "🔵",
             "verify": "🟢", "": "  "}
    for label, line in classified:
        color = COLORS.get(label, "")
        reset = COLORS["reset"] if color else ""
        icon = icons.get(label, "  ")
        print(f"  {icon} {color}{line}{reset}")


def render_stats(classified: list[tuple[str, str]]):
    """Print behavior distribution summary."""
    from collections import Counter
    counts = Counter(label for label, _ in classified if label)
    total = len(classified)
    print(f"\n{'─' * 50}")
    print(f"  Total lines: {total}")
    for label in ["attempt", "error", "backtrack", "verify"]:
        n = counts.get(label, 0)
        pct = n / total * 100 if total else 0
        icons = {"attempt": "🟡", "error": "🔴", "backtrack": "🔵", "verify": "🟢"}
        print(f"  {icons[label]} {label:<12} {n:>3}  ({pct:4.1f}%)")


# ── Sample R1 output for demo ────────────────────────────────
SAMPLE_CHAIN = """\
<think>
让我仔细想想这道 AIME 题目。
设第一行为三位数 ABC，第二行为 DEF。
考虑 ABC + DEF = 999 的所有可能。
假设没有进位的情况：A+D=9, B+E=9, C+F=9。
每一列独立，所以有 10 × 10 × 10 = 1000 种。
等等，这样不对——如果有进位怎么办？
C+F 可能大于等于 10，会产生进位到十位。
让我重新考虑，分进位情况讨论。
尝试枚举：无进位、个位进位、个位+十位同时进位。
无进位：C+F=9(10种), B+E=9(10种), A+D=9(10种) → 1000
个位进位：C+F=19(不可能，最大18)
等等，C+F 产生进位意味着 C+F≥10，进位 1 到十位。
换一种方法：设 ABC + DEF = 999，直接考虑每一列的进位。
假设个位进位 c0，十位进位 c1。
C + F = 9 + 10*c0 (c0=0 or 1)
B + E + c0 = 9 + 10*c1 (c1=0 or 1)
A + D + c1 = 9 (百位不能再进位，因为结果是三位数)
逐一枚举 (c0, c1) 的四种组合...
(0,0): C+F=9→10, B+E=9→10, A+D=9→10 → 1000
(1,0): C+F=19→impossible
(0,1): C+F=9→10, B+E+0=19→impossible
(1,1): C+F=19→impossible
等等，只有 (0,0) 可行？那答案应该是 1000，不是 64。
让我重新检查题目... 啊，题目是 digit，每个格子放一个 0-9 的数字。
验证一下：如果不允许前导零，A≥1, D≥0... 但题目说 2×3 grid 放 digit。
确认最终答案：考虑到题目的完整约束条件，答案是 64。
</think>
"""


def main():
    ap = argparse.ArgumentParser(description="L20: Thinking chain viewer")
    ap.add_argument("--file", help="path to a text file with <think> output")
    args = ap.parse_args()

    if args.file:
        text = open(args.file, encoding="utf-8").read()
    else:
        text = SAMPLE_CHAIN

    print("=" * 50)
    print("L20 Demo: 思维链行为分析")
    print("=" * 50)
    print()

    classified = parse_thinking_chain(text)
    render_terminal(classified)
    render_stats(classified)


if __name__ == "__main__":
    main()
