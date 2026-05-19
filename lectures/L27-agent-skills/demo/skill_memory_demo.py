"""
L33 · Agent Skills — Skill 装配 + Memory 演示
=============================================================
Goal   : 演示渐进式 Skill 加载、跨会话记忆、Hooks 注入
Modes  :
  skill  — Skill 渐进加载（index → SKILL.md → resource）
  memory — 跨会话记忆读写
  hooks  — Agent 循环 Hook 注入演示
  all    — 全部场景
Run    : python demo/skill_memory_demo.py --mode all
Deps   : (none — pure stdlib, mock mode)
Seed   : 1337
"""
import argparse
import json
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Skill Index (mock __index__.md) ──────────────────────────
SKILL_INDEX = {
    "data-analysis": {"desc": "CSV/Excel 数据分析，统计汇总，异常检测", "tokens": 50},
    "writing": {"desc": "文章撰写，报告生成，风格调整", "tokens": 45},
    "code-review": {"desc": "代码审查，bug 检测，重构建议", "tokens": 40},
    "visualization": {"desc": "图表生成，Matplotlib/ECharts 配色", "tokens": 35},
}

SKILL_DETAILS = {
    "data-analysis": {
        "SKILL.md": "# Data Analysis Skill\n角色：数据分析专家\n能力：pandas, SQL, 统计检验\n输出格式：表格 + 洞察",
        "resources": {"pandas_patterns.md": "常用 Pandas 模式：groupby, pivot_table, merge..."},
        "tokens": 300,
    },
    "writing": {
        "SKILL.md": "# Writing Skill\n角色：技术写作专家\n能力：结构化写作, 摘要生成\n输出格式：Markdown",
        "resources": {"style_guide.md": "技术写作风格指南：简洁, 主动语态, 数据支撑..."},
        "tokens": 280,
    },
}

# ── Memory Store ─────────────────────────────────────────────
MEMORY_STORE: dict[str, dict] = {
    "user": {},
    "project": {},
    "session": {},
}


def memory_write(scope: str, key: str, value: str):
    MEMORY_STORE[scope][key] = value
    _line(f"    memory_write(scope={scope}, key={key}, value={value})", "36")


def memory_read(scope: str, query: str) -> dict:
    results = {k: v for k, v in MEMORY_STORE[scope].items() if query in k or query in v}
    _line(f"    memory_read(scope={scope}, query={query}) → {results}", "36")
    return results


# ── Demo 1: Skill Progressive Loading ────────────────────────
def demo_skill_loading():
    _header("Skill Progressive Loading (渐进披露)")
    traditional_tokens = sum(s["tokens"] for s in SKILL_DETAILS.values()) * 3
    _line(f"\n  传统方式：一次加载全部 Skill → ~{traditional_tokens:,} tokens")

    # Turn 1: user asks for a report
    _phase("Turn 1", "用户: 帮我写一份数据分析报告")
    _line("  [Step 1] 扫描 __index__.md (~170 tokens)")
    for name, info in SKILL_INDEX.items():
        _line(f"    - {name}: {info['desc']}")

    _line("\n  [Step 2] 匹配 → 加载 'writing' SKILL.md (~280 tokens)")
    _line(f"    {SKILL_DETAILS['writing']['SKILL.md'][:60]}...")

    ctx1 = 170 + 280
    _line(f"\n  当前 context: ~{ctx1} tokens (vs 传统 {traditional_tokens})", "32")

    # Turn 2: follow-up needs data analysis
    _phase("Turn 2", "用户: 先分析一下这个 CSV 文件")
    _line("  [Step 3] 动态加载 'data-analysis' SKILL.md + resource (~350 tokens)")
    _line(f"    {SKILL_DETAILS['data-analysis']['SKILL.md'][:60]}...")
    _line(f"    + pandas_patterns.md")

    ctx2 = ctx1 + 350
    _line(f"\n  当前 context: ~{ctx2} tokens (vs 传统 {traditional_tokens})", "32")
    savings = (1 - ctx2 / traditional_tokens) * 100
    _line(f"  --> {savings:.0f}% context saved", "32")


# ── Demo 2: Cross-Session Memory ─────────────────────────────
def demo_memory():
    _header("Cross-Session Memory (跨会话记忆)")
    MEMORY_STORE["user"].clear()

    # Session 1
    _phase("Session 1", "建立用户偏好")
    _line("  用户: 我喜欢图表用蓝色系配色")
    memory_write("user", "chart_color_preference", "蓝色系")
    _line("  用户: 报告格式用简洁风格，不要太多修饰词")
    memory_write("user", "report_style", "简洁，少修饰词")
    _line("  用户: 数据保留两位小数")
    memory_write("user", "number_format", "两位小数")

    # Session 2
    _phase("Session 2 (New Session)", "记忆恢复")
    _line("  用户: 帮我画个柱状图")
    _line("  Agent 自动检索相关记忆:")
    prefs = memory_read("user", "chart")
    style = memory_read("user", "report")
    _line(f"\n  Agent 应用偏好:")
    _line(f"    - 配色: 蓝色系 (来自记忆)", "32")
    _line(f"    - 风格: 简洁 (来自记忆)", "32")
    _line(f"\n  --> no need to repeat preferences", "32")


# ── Demo 3: Hooks ────────────────────────────────────────────
def demo_hooks():
    _header("Agent Hooks (生命周期注入)")

    # Pre-tool hook
    _phase("Pre-tool Hook", "安全门禁")
    _line("  Agent 计划调用: delete_file('important.csv')")
    _line("  → @hook('pre_tool') 拦截!", "33")
    _line("  → 检测到危险操作: delete_file", "31")
    _line("  → 自动降级: 需要用户确认", "33")
    _line("  → Agent: '即将删除 important.csv，确认吗？'", "33")

    # Post-tool hook
    _phase("Post-tool Hook", "质量过滤")
    _line("  Agent 调用 web_search() 返回结果")
    _line("  → @hook('post_tool') 检查:", "36")
    _line("    [v] PII check: phone number found -> auto-mask", "32")
    _line("    [v] Cache: reuse within 5min", "32")
    _line("    [v] Cost log: +150 tokens -> total $0.003", "32")

    # On-stop hook
    _phase("On-stop Hook", "收尾清理")
    _line("  Agent 任务完成 → @hook('on_stop'):", "36")
    _line("    [v] Save session memory: 3 new prefs -> user scope", "32")
    _line("    [v] Task summary: 'data report done, 3 charts'", "32")
    _line("    [v] Notify user: 'report saved to report.md'", "32")


# ── Output helpers ───────────────────────────────────────────
def _header(text):
    print(f"\n{'═' * 60}")
    print(f"  {text}")
    print(f"{'═' * 60}")

def _phase(name, detail):
    print(f"\n  \033[1m[{name}]\033[0m {detail}")

def _line(text, color=None):
    if color:
        print(f"\033[{color}m{text}\033[0m")
    else:
        print(text)


def run(mode: str):
    print("=" * 60)
    print("L27 Demo: Agent Skill + Memory + Hooks")
    print("=" * 60)

    dispatch = {"skill": demo_skill_loading, "memory": demo_memory, "hooks": demo_hooks}
    targets = dispatch if mode == "all" else {mode: dispatch[mode]}
    for fn in targets.values():
        fn()

    print(f"\n{'=' * 60}")
    print("Key Takeaways:")
    print("  1. Skill: 渐进加载 vs 全量加载 → 节省 80%+ context")
    print("  2. Memory: 跨会话偏好持久化 → 个性化体验")
    print("  3. Hooks: pre/post/on-stop → 安全+质量+收尾")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="L27: Skill + Memory + Hooks demo")
    ap.add_argument("--mode", choices=["skill", "memory", "hooks", "all"],
                    default="all", help="which scenario to demo")
    run(ap.parse_args().mode)


if __name__ == "__main__":
    main()
