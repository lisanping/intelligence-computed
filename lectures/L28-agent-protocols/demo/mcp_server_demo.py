"""
L34 · Agent Protocols — MCP Server 开发 + Client 集成演示
=============================================================
Goal   : 演示 MCP 三原语（Resources/Tools/Prompts）、Server 生命周期、
         Client 动态工具发现、description 消融实验
Modes  :
  primitives — 三原语控制权对比
  server     — MCP Server 生命周期（注册→发现→调用→响应）
  client     — MCP Client 动态工具集成（多 Server 聚合）
  ablation   — bad-description 消融（description 质量 → 工具可用性）
  all        — 全部场景
Run    : python demo/mcp_server_demo.py --mode all
         python demo/mcp_server_demo.py --ablate bad-description
Deps   : (none — pure stdlib, mock mode)
Seed   : 1337
"""
import argparse
import json
import random
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(encoding="utf-8")

random.seed(1337)

# ═══════════════════════════════════════════════════════════════
# Mock MCP primitives
# ═══════════════════════════════════════════════════════════════

RESOURCES = [
    {"uri": "notes://linear-algebra.md", "name": "linear-algebra.md",
     "desc": "线性代数笔记", "mime": "text/markdown"},
    {"uri": "notes://transformer.md", "name": "transformer.md",
     "desc": "Transformer 架构笔记", "mime": "text/markdown"},
    {"uri": "notes://rlhf.md", "name": "rlhf.md",
     "desc": "RLHF 对齐笔记", "mime": "text/markdown"},
]

TOOLS_GOOD = [
    {"name": "create_note", "desc": "Create a new course note with title and markdown content",
     "params": {"title": "str", "content": "str"}},
    {"name": "search_notes", "desc": "Search notes by keyword, returns list of matching filenames",
     "params": {"query": "str"}},
    {"name": "delete_note", "desc": "Delete a course note by filename (requires confirmation)",
     "params": {"filename": "str"}},
]

TOOLS_BAD = [
    {"name": "create_note", "desc": "create", "params": {"title": "str", "content": "str"}},
    {"name": "search_notes", "desc": "search", "params": {"query": "str"}},
    {"name": "delete_note", "desc": "delete", "params": {"filename": "str"}},
]

PROMPTS = [
    {"name": "analyze_sprint", "desc": "Analyze current sprint progress and blockers",
     "args": [{"name": "sprint_id", "required": True}]},
    {"name": "weekly_summary", "desc": "Generate weekly learning summary from notes",
     "args": [{"name": "week_number", "required": True}]},
]

# User queries to match against tools (English — simulating LLM keyword extraction)
TOOL_MATCH_QUERIES = [
    ("Create a note about CNN architectures", "create_note"),
    ("Search my notes for attention mechanism", "search_notes"),
    ("Delete the file draft.md from my notes", "delete_note"),
    ("I want to create a new note on RAG", "create_note"),
    ("Find notes that mention embedding", "search_notes"),
    ("Which notes mention loss function?", "search_notes"),
    ("Write a new Transformer study note", "create_note"),
    ("Remove that temporary note file", "delete_note"),
    ("Search for positional encoding", "search_notes"),
    ("Create a summary note about fine-tuning", "create_note"),
]

# ═══════════════════════════════════════════════════════════════
# Mock MCP Server / Client
# ═══════════════════════════════════════════════════════════════

MCP_SERVERS = {
    "notes": {
        "command": "python notes_server.py",
        "tools": [
            {"name": "create_note", "desc": "Create a new course note", "schema": {"title": "str", "content": "str"}},
            {"name": "search_notes", "desc": "Search notes by keyword", "schema": {"query": "str"}},
        ],
        "resources": 3,
    },
    "github": {
        "command": "npx -y @modelcontextprotocol/server-github",
        "tools": [
            {"name": "create_issue", "desc": "Create a GitHub issue", "schema": {"repo": "str", "title": "str"}},
            {"name": "list_prs", "desc": "List open pull requests", "schema": {"repo": "str"}},
            {"name": "get_file", "desc": "Get file content from repo", "schema": {"repo": "str", "path": "str"}},
        ],
        "resources": 5,
    },
    "postgres": {
        "command": "npx -y @modelcontextprotocol/server-postgres",
        "tools": [
            {"name": "query", "desc": "Run SQL query", "schema": {"sql": "str"}},
        ],
        "resources": 10,
    },
}


# ═══════════════════════════════════════════════════════════════
# Demo 1: Three Primitives
# ═══════════════════════════════════════════════════════════════

def demo_primitives():
    _header("MCP 三原语 — Resources / Tools / Prompts")

    _phase("Resources (资源)", "控制权: 应用")
    _line("  应用代码决定何时暴露给模型 — 类比 RAG 检索结果")
    _line("  Server 暴露的 Resources:")
    for r in RESOURCES:
        _line(f"    [R] {r['uri']}  ->  {r['desc']}", "36")
    _line("\n  模型不能自主读取 Resource — 由 Host 应用代码控制", "33")

    _phase("Tools (工具)", "控制权: 模型")
    _line("  模型看到工具描述后自主决定是否调用 — 类比 Function Calling")
    _line("  Server 暴露的 Tools:")
    for t in TOOLS_GOOD:
        _line(f"    [T] {t['name']}({', '.join(t['params'])})  ->  {t['desc']}", "34")
    _line("\n  模型自主选择调用哪个 Tool — 由 LLM 决策", "33")

    _phase("Prompts (提示模板)", "控制权: 用户")
    _line("  用户在 Host UI 中手动选择触发 — 类比 Prompt Template 库")
    _line("  Server 暴露的 Prompts:")
    for p in PROMPTS:
        args = ", ".join(a["name"] for a in p["args"])
        _line(f"    [P] {p['name']}({args})  ->  {p['desc']}", "35")
    _line("\n  用户手动选择模板并填参数 — 由用户控制", "33")

    _line("\n  ┌─────────────┬──────────┬─────────────────────┐")
    _line("  │ 原语        │ 控制权   │ 类比                │")
    _line("  ├─────────────┼──────────┼─────────────────────┤")
    _line("  │ Resources   │ 应用     │ RAG 的检索结果      │")
    _line("  │ Tools       │ 模型     │ Function Calling    │")
    _line("  │ Prompts     │ 用户     │ Prompt Template 库  │")
    _line("  └─────────────┴──────────┴─────────────────────┘")
    _line("\n  Three control scopes = balance between safety and autonomy", "32")


# ═══════════════════════════════════════════════════════════════
# Demo 2: MCP Server Lifecycle
# ═══════════════════════════════════════════════════════════════

def demo_server():
    _header("MCP Server 生命周期 — 注册 → 发现 → 调用 → 响应")

    _phase("Step 1", "Host 启动 Server 进程")
    _line('  config: {"mcpServers": {"course-notes": {"command": "python", "args": ["notes_server.py"]}}}')
    _line("  → Host spawn 子进程，建立 stdio 通道", "36")

    _phase("Step 2", "Client 初始化握手 (JSON-RPC)")
    _line("  → initialize({protocolVersion: '2024-11-05', capabilities: {...}})")
    _line("  ← {protocolVersion: '2024-11-05', serverInfo: {name: 'course-notes'}}", "32")

    _phase("Step 3", "Client 发现工具 (list_tools)")
    _line("  → tools/list")
    _line("  ← 返回 2 个工具:", "32")
    for t in TOOLS_GOOD[:2]:
        _line(f"      [T] {t['name']}: {t['desc']}", "32")

    _phase("Step 4", "模型决定调用 (call_tool)")
    _line('  用户: "帮我搜索关于 attention 的笔记"')
    _line('  模型: → 调用 search_notes(query="attention")')
    _line("  → tools/call {name: 'search_notes', arguments: {query: 'attention'}}")
    results = ["transformer.md", "rlhf.md"]
    _line(f"  ← {json.dumps(results)}", "32")

    _phase("Step 5", "模型处理结果并回复用户")
    _line(f'  模型: "找到 {len(results)} 个相关笔记: {", ".join(results)}"', "32")

    _line(f"\n  Full server = 70 lines Python -- one afternoon to build", "32")


# ═══════════════════════════════════════════════════════════════
# Demo 3: MCP Client — Dynamic Tool Discovery
# ═══════════════════════════════════════════════════════════════

def demo_client():
    _header("MCP Client 集成 — 动态工具发现 + 多 Server 聚合")

    all_tools = []

    for name, server in MCP_SERVERS.items():
        _phase(f"Connect: {name}", server["command"])
        _line(f"  → stdio_client({name}) 连接成功")
        _line(f"  → list_tools() 发现 {len(server['tools'])} 个工具:")
        for t in server["tools"]:
            full_name = f"{name}__{t['name']}"
            all_tools.append(full_name)
            _line(f"      [T] {full_name}: {t['desc']}", "36")
        _line(f"  → list_resources() 发现 {server['resources']} 个资源", "36")

    _phase("聚合结果", f"共 {len(all_tools)} 个工具可用")
    for t in all_tools:
        _line(f"    [v] {t}", "32")

    _phase("命名空间隔离", "server__tool 避免冲突")
    _line("  notes__create_note  ≠  github__create_issue")
    _line("  Same-name tools separated by server prefix -> safe aggregation", "32")

    _phase("格式转换", "MCP → OpenAI Function Calling 格式")
    sample = {
        "type": "function",
        "function": {
            "name": "notes__search_notes",
            "description": "Search notes by keyword",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }
    _line(f"  {json.dumps(sample, indent=4, ensure_ascii=False)}", "36")
    _line("\n  Dynamic discovery -> plug & play, no hardcoding", "32")


# ═══════════════════════════════════════════════════════════════
# Demo 4: Ablation — Bad Description
# ═══════════════════════════════════════════════════════════════

def _simulate_tool_match(query: str, tools: list[dict]) -> str | None:
    """Simulate LLM tool selection via keyword overlap (proxy for real model)."""
    q_lower = query.lower()
    best, best_score = None, 0
    for t in tools:
        # Score = keyword overlap between query and description
        desc_words = set(t["desc"].lower().split())
        q_words = set(q_lower.replace("，", " ").replace("。", " ").split())
        score = len(desc_words & q_words)
        # Boost from tool name substring match
        if t["name"].replace("_", " ") in q_lower or any(w in t["name"] for w in q_words):
            score += 2
        if score > best_score:
            best, best_score = t["name"], score
    return best if best_score > 0 else None


def demo_ablation():
    _header("消融实验 — Tool Description 质量 → 工具可用性")

    _phase("Good Description", "详细、语义丰富")
    for t in TOOLS_GOOD:
        _line(f"    {t['name']}: \"{t['desc']}\"", "32")

    _phase("Bad Description", "极简、信息不足")
    for t in TOOLS_BAD:
        _line(f"    {t['name']}: \"{t['desc']}\"", "31")

    _phase("测试 10 个用户查询", "模拟模型工具选择")
    good_correct, bad_correct = 0, 0

    for query, expected in TOOL_MATCH_QUERIES:
        good_pick = _simulate_tool_match(query, TOOLS_GOOD)
        bad_pick = _simulate_tool_match(query, TOOLS_BAD)
        g_ok = good_pick == expected
        b_ok = bad_pick == expected
        if g_ok:
            good_correct += 1
        if b_ok:
            bad_correct += 1
        g_sym = "[v]" if g_ok else "[x]"
        b_sym = "[v]" if b_ok else "[x]"
        g_col = "32" if g_ok else "31"
        b_col = "32" if b_ok else "31"
        _line(f"    Q: {query}")
        _line(f"      Good -> {good_pick or 'None'} {g_sym}  "
              f"Bad -> {bad_pick or 'None'} {b_sym}",
              g_col if g_ok and b_ok else b_col)

    g_pct = good_correct / len(TOOL_MATCH_QUERIES) * 100
    b_pct = bad_correct / len(TOOL_MATCH_QUERIES) * 100
    _line(f"\n  ┌─────────────────────┬──────────────┐")
    _line(f"  │ Description 质量    │ 匹配准确率   │")
    _line(f"  ├─────────────────────┼──────────────┤")
    _line(f"  │ Good (详细语义)     │ {g_pct:5.0f}%       │", "32")
    _line(f"  │ Bad  (极简一词)     │ {b_pct:5.0f}%       │", "31")
    _line(f"  └─────────────────────┴──────────────┘")
    _line(f"\n  Description = 模型理解工具的唯一入口", "33")
    _line(f"  Description quality = tool usability", "32")


# ═══════════════════════════════════════════════════════════════
# Output helpers
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def run(mode: str, ablate: str | None):
    print("=" * 60)
    print("L28 Demo: MCP Server 开发 + Client 集成")
    print("=" * 60)

    if ablate == "bad-description":
        demo_ablation()
        return

    dispatch = {
        "primitives": demo_primitives,
        "server": demo_server,
        "client": demo_client,
        "ablation": demo_ablation,
    }
    targets = dispatch if mode == "all" else {mode: dispatch[mode]}
    for fn in targets.values():
        fn()

    print(f"\n{'=' * 60}")
    print("Key Takeaways:")
    print("  1. 三原语: Resources(应用) / Tools(模型) / Prompts(用户)")
    print("  2. Server: 70 行 Python → 一个下午写一个 MCP Server")
    print("  3. Client: 动态发现 → 连上就能用，namespace 避免冲突")
    print("  4. Description 质量 = 工具可用性（消融: 详细 vs 极简）")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="L28: MCP Server + Client demo")
    ap.add_argument("--mode", choices=["primitives", "server", "client", "ablation", "all"],
                    default="all", help="which scenario to demo")
    ap.add_argument("--ablate", choices=["bad-description"], default=None,
                    help="run ablation experiment")
    args = ap.parse_args()
    run(args.mode, args.ablate)


if __name__ == "__main__":
    main()
