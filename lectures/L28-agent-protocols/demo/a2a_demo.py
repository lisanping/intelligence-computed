"""
L34 · Agent Protocols — A2A 协议协作演示
=============================================================
Goal   : 演示 Agent Card 发现、Task 生命周期（状态机）、
         多轮协商（input-required）、流式响应、MCP+A2A 协同
Modes  :
  card      — Agent Card 结构与发现
  task      — Task 生命周期状态机（submitted→working→completed/failed）
  negotiate — input-required 多轮协商
  streaming — 长任务 + 流式响应
  collab    — MCP + A2A 协同选型
  all       — 全部场景
Run    : python demo/a2a_demo.py --mode all
Deps   : (none — pure stdlib, mock mode)
Seed   : 1337
"""
import argparse
import json
import sys
import time
import random

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(encoding="utf-8")

random.seed(1337)

# ═══════════════════════════════════════════════════════════════
# Mock Agent Cards
# ═══════════════════════════════════════════════════════════════

AGENT_CARDS = {
    "DataAnalysisAgent": {
        "name": "DataAnalysisAgent",
        "description": "Analyzes structured data, produces statistical summaries and insights",
        "url": "https://data-agent.example.com",
        "version": "1.0.0",
        "capabilities": {"streaming": True, "pushNotifications": True, "stateTransitionHistory": True},
        "authentication": {"schemes": ["OAuth2"]},
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["text/plain", "application/json"],
        "skills": [
            {"id": "statistical-analysis", "name": "Statistical Analysis",
             "description": "Performs descriptive and inferential statistics on tabular data",
             "tags": ["statistics", "data", "analysis"],
             "examples": ["Analyze this CSV and find outliers", "Run regression on columns A and B"]},
            {"id": "data-cleaning", "name": "Data Cleaning",
             "description": "Handles missing values, deduplication, and type conversion",
             "tags": ["cleaning", "preprocessing"],
             "examples": ["Clean this dataset", "Handle missing values in column X"]},
        ],
    },
    "ReportAgent": {
        "name": "ReportAgent",
        "description": "Generates professional reports from data and analysis results",
        "url": "https://report-agent.example.com",
        "version": "1.2.0",
        "capabilities": {"streaming": True, "pushNotifications": False, "stateTransitionHistory": False},
        "authentication": {"schemes": ["OAuth2"]},
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["text/plain", "application/pdf"],
        "skills": [
            {"id": "report-generation", "name": "Report Generation",
             "description": "Creates structured reports with charts and executive summaries",
             "tags": ["report", "writing", "summary"],
             "examples": ["Generate Q3 analysis report", "Create executive summary from data"]},
        ],
    },
    "VizAgent": {
        "name": "VizAgent",
        "description": "Creates data visualizations, charts, and interactive dashboards",
        "url": "https://viz-agent.example.com",
        "version": "0.9.0",
        "capabilities": {"streaming": False, "pushNotifications": True, "stateTransitionHistory": True},
        "authentication": {"schemes": ["API-Key"]},
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["image/png", "text/html"],
        "skills": [
            {"id": "chart-creation", "name": "Chart Creation",
             "description": "Generates bar, line, scatter, and pie charts from data",
             "tags": ["chart", "visualization", "plot"],
             "examples": ["Create a bar chart of sales by region", "Plot trends over time"]},
        ],
    },
}


# ═══════════════════════════════════════════════════════════════
# Demo 1: Agent Card Discovery
# ═══════════════════════════════════════════════════════════════

def demo_card():
    _header("Agent Card — Agent 的简历 / 名片")

    _phase("发现 Agent", "GET /.well-known/agent.json")
    for name, card in AGENT_CARDS.items():
        _line(f"\n  🤖 {card['name']} ({card['url']})")
        _line(f"     描述: {card['description']}", "36")
        _line(f"     Skills:")
        for s in card["skills"]:
            _line(f"       • {s['name']}: {s['description']}", "36")
            _line(f"         tags={s['tags']}, examples={s['examples'][:1]}")
        caps = card["capabilities"]
        _line(f"     Capabilities: streaming={caps['streaming']}, "
              f"push={caps['pushNotifications']}")
        _line(f"     Auth: {card['authentication']['schemes']}")

    _phase("Agent Card vs OpenAPI", "类比对照")
    _line("  ┌──────────────┬───────────────────────┬───────────────────────┐")
    _line("  │              │ OpenAPI               │ Agent Card            │")
    _line("  ├──────────────┼───────────────────────┼───────────────────────┤")
    _line("  │ 消费者       │ 人类开发者            │ 其他 Agent            │")
    _line("  │ 发现方式     │ API 文档网站          │ /.well-known/agent.json│")
    _line("  │ 描述内容     │ 端点、参数、返回值    │ 技能、能力、示例      │")
    _line("  │ 交互模式     │ 请求→响应             │ 多轮协商 + 流式       │")
    _line("  └──────────────┴───────────────────────┴───────────────────────┘")

    _phase("语义匹配", "任务 → 找合适的 Agent (呼应 L29 Embedding)")
    task = "generate a report with charts"
    _line(f'  任务: "{task}"')
    match = _find_agent(task)
    _line(f"  匹配结果: {match} ✓", "32")


# ═══════════════════════════════════════════════════════════════
# Demo 2: Task Lifecycle State Machine
# ═══════════════════════════════════════════════════════════════

TASK_STATES = ["submitted", "working", "input-required", "working", "completed"]
TASK_STATES_FAIL = ["submitted", "working", "failed"]

def demo_task():
    _header("Task 生命周期 — 状态机")

    _phase("成功路径", "submitted → working → completed")
    task_id = "task-001"
    _line(f'  Client Agent: "分析 Q3 销售数据"')
    _line(f"  → POST /tasks/send {{id: '{task_id}', message: ...}}")

    for i, state in enumerate(["submitted", "working", "completed"]):
        sym = {"submitted": "⏳", "working": "🔄", "completed": "✅"}[state]
        col = {"submitted": "37", "working": "34", "completed": "32"}[state]
        _line(f"    [{i+1}] {sym} {state}", col)

    artifact = {"name": "Q3 Analysis Report",
                "parts": [{"type": "text", "text": "## Q3 Sales Analysis\nTop: Northeast (+23%), Southwest (+18%)..."}]}
    _line(f"\n  Artifact (最终产出):")
    _line(f"    {json.dumps(artifact, indent=4, ensure_ascii=False)}", "32")

    _phase("失败路径", "submitted → working → failed")
    for i, state in enumerate(TASK_STATES_FAIL):
        sym = {"submitted": "⏳", "working": "🔄", "failed": "❌"}[state]
        col = {"submitted": "37", "working": "34", "failed": "31"}[state]
        _line(f"    [{i+1}] {sym} {state}", col)
    _line('    error: "Data source unavailable — connection timeout"', "31")

    _phase("取消", "任意状态 → canceled")
    _line("    ⏳ submitted → 🚫 canceled", "33")
    _line('    reason: "User canceled the task"', "33")

    _phase("Message vs Artifact", "过程沟通 vs 最终产出")
    _line("  ┌──────────────┬─────────────────────┬─────────────────────┐")
    _line("  │              │ Message             │ Artifact            │")
    _line("  ├──────────────┼─────────────────────┼─────────────────────┤")
    _line("  │ 用途         │ 对话、协商、追问    │ 最终结果、报告      │")
    _line("  │ 时机         │ 任务进行中          │ 任务完成时          │")
    _line("  │ 类比         │ 微信聊天记录        │ 最终交付的 PPT      │")
    _line("  └──────────────┴─────────────────────┴─────────────────────┘")


# ═══════════════════════════════════════════════════════════════
# Demo 3: Multi-turn Negotiation (input-required)
# ═══════════════════════════════════════════════════════════════

def demo_negotiate():
    _header("多轮协商 — input-required 状态")

    _phase("场景", "Client Agent 请求 DataAnalysisAgent 分析数据")

    steps = [
        ("Client", "user", "分析这份 Q3 销售数据", "submitted"),
        ("Server", None, None, "working"),
        ("Server", "agent", "你想分析哪些维度？可选：销售额、利润率、客户数、退货率", "input-required"),
        ("Client", "user", "销售额和利润率", "working"),
        ("Server", "agent", "需要按什么维度分组？可选：地区、产品线、月份", "input-required"),
        ("Client", "user", "按地区分组", "working"),
        ("Server", None, None, "completed"),
    ]

    state_colors = {
        "submitted": "37", "working": "34",
        "input-required": "33", "completed": "32",
    }

    for who, role, msg, state in steps:
        col = state_colors[state]
        state_sym = {"submitted": "⏳", "working": "🔄", "input-required": "❓", "completed": "✅"}[state]
        if msg:
            _line(f"    {who:>6}: \"{msg}\"")
        _line(f"           → state: {state_sym} {state}", col)

    _line("\n  关键洞察:", "33")
    _line("    简单 HTTP: 发一次请求 → 等一次响应 ❌", "31")
    _line("    A2A Task:  多轮协商 → Agent 可主动追问 ✅", "32")
    _line("    input-required = 协作的核心设计 ✓", "32")


# ═══════════════════════════════════════════════════════════════
# Demo 4: Long Task + Streaming
# ═══════════════════════════════════════════════════════════════

def demo_streaming():
    _header("长任务 + 流式响应 — SSE Streaming")

    _phase("长任务", "30 分钟数据分析 — 不能傻等")
    _line("  方案 A: 轮询 → GET /tasks/{id} 每 30 秒检查状态")
    _line("  方案 B: 推送 → Server 完成后主动通知 Client (pushNotifications: true)")

    _phase("流式响应", "POST /tasks/sendSubscribe → SSE 事件流")
    _line("  报告生成 Agent 边做边输出：")

    sse_events = [
        ("status", {"state": "working"}, 0.0),
        ("artifact", {"name": "Executive Summary", "text": "## Executive Summary\nQ3 revenue grew 18%..."}, 0.3),
        ("artifact", {"name": "Regional Breakdown", "text": "## Regional Analysis\nNortheast: +23%..."}, 0.3),
        ("artifact", {"name": "Recommendations", "text": "## Recommendations\n1. Expand Southwest..."}, 0.3),
        ("status", {"state": "completed"}, 0.1),
    ]

    for event_type, data, delay in sse_events:
        time.sleep(delay)
        if event_type == "status":
            sym = "🔄" if data["state"] == "working" else "✅"
            col = "34" if data["state"] == "working" else "32"
            _line(f"    event: {event_type}", col)
            _line(f"    data:  {sym} state={data['state']}", col)
        else:
            _line(f"    event: artifact", "36")
            _line(f"    data:  📄 {data['name']}: {data['text'][:50]}...", "36")

    _line("\n  用户不用等全部完成就能看到中间结果 ✓", "32")
    _line("  同 LLM streaming response 思路 — 渐进披露", "32")


# ═══════════════════════════════════════════════════════════════
# Demo 5: MCP + A2A Collaboration Pattern
# ═══════════════════════════════════════════════════════════════

def demo_collab():
    _header("MCP + A2A 协同 — 纵向工具 + 横向协作")

    _phase("架构图", "纵向 MCP + 横向 A2A")
    _line("  ┌────────────────────────────────────────────────────┐")
    _line("  │              [用户的 Agent (Host)]                 │")
    _line("  │                  ↕ A2A        ↕ A2A               │")
    _line("  │     [DataAnalysisAgent]    [ReportAgent]          │")
    _line("  │          ↕ MCP                 ↕ MCP              │")
    _line("  │   [PostgreSQL Server]     [Google Docs Server]    │")
    _line("  │   [Jira Server]           [Slack Server]          │")
    _line("  └────────────────────────────────────────────────────┘")
    _line("     纵向: MCP 连工具 (Agent→Tool)", "34")
    _line("     横向: A2A 连 Agent (Agent↔Agent)", "32")

    _phase("协议选型决策树", "该用哪个？")
    _line("  你要连接的对象是什么？")
    _line("  ├── 🔧 工具/API/数据源")
    _line("  │   ├── 只在你的项目里用 → Function Calling", "36")
    _line("  │   ├── 要跨项目共享     → MCP Server (stdio)", "34")
    _line("  │   └── 要跨组织公开     → MCP Server (Streamable HTTP)", "34")
    _line("  ├── 🤖 另一个 Agent")
    _line("  │   ├── 同一代码库/框架  → Sub-agent (L33 Handoff)", "32")
    _line("  │   └── 跨框架/跨组织    → A2A", "32")
    _line("  └── 🖥️  没有 API 的系统")
    _line("      ├── Web 界面         → Browser Use", "33")
    _line("      └── 桌面 GUI         → Computer Use", "33")

    _phase("选型矩阵", "四维对比")
    _line("  ┌──────────────┬──────────┬──────────┬──────────┬──────────────┐")
    _line("  │ 维度         │ FC       │ MCP      │ A2A      │ Computer Use │")
    _line("  ├──────────────┼──────────┼──────────┼──────────┼──────────────┤")
    _line("  │ 对象         │ 单个工具 │ 工具生态 │ Agent 态 │ 任何 GUI     │")
    _line("  │ 标准化       │ 厂商     │ 开放     │ 开放     │ 无           │")
    _line("  │ 复用性       │ 单项目   │ 跨团队   │ 跨组织   │ 不可复用     │")
    _line("  │ 多轮协商     │ ✗        │ ✗        │ ✓        │ N/A          │")
    _line("  │ 成本/调用    │ ~$0.001  │ ~$0.001  │ ~$0.01   │ ~$0.05-0.10  │")
    _line("  │ 成熟度       │ 成熟     │ 成长中   │ 早期     │ 实验性       │")
    _line("  └──────────────┴──────────┴──────────┴──────────┴──────────────┘")

    _phase("经验法则", "一句话选型")
    _line("    有 API        → 用 API (FC)", "32")
    _line("    有 MCP Server → 用 MCP", "32")
    _line("    有 Agent      → 用 A2A", "32")
    _line("    什么都没有    → Computer Use (兜底)", "33")
    _line("\n  不是互斥 — 成熟系统同时用所有四种", "32")
    _line("  就像电脑同时有 USB / WiFi / 蓝牙 / HDMI ✓", "32")


# ═══════════════════════════════════════════════════════════════
# Agent matching helper
# ═══════════════════════════════════════════════════════════════

def _find_agent(task_desc: str) -> str | None:
    """Simple keyword matching (production: use Embedding — L29)."""
    desc_lower = task_desc.lower()
    for name, card in AGENT_CARDS.items():
        for skill in card.get("skills", []):
            for tag in skill.get("tags", []):
                if tag in desc_lower:
                    return name
    return None


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

def run(mode: str):
    print("=" * 60)
    print("L28 Demo: A2A 协议 — Agent 间发现、协商、协作")
    print("=" * 60)

    dispatch = {
        "card": demo_card,
        "task": demo_task,
        "negotiate": demo_negotiate,
        "streaming": demo_streaming,
        "collab": demo_collab,
    }
    targets = dispatch if mode == "all" else {mode: dispatch[mode]}
    for fn in targets.values():
        fn()

    print(f"\n{'=' * 60}")
    print("Key Takeaways:")
    print("  1. Agent Card: Agent 的简历 → /.well-known/agent.json")
    print("  2. Task 生命周期: submitted→working→[input-required]→completed")
    print("  3. input-required: 多轮协商的关键设计 (区别于简单 HTTP)")
    print("  4. 流式: SSE 渐进输出 → 不用等全部完成")
    print("  5. MCP(纵向工具) + A2A(横向协作) = 完整 Agent 生态")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="L28: A2A protocol collaboration demo")
    ap.add_argument("--mode", choices=["card", "task", "negotiate", "streaming", "collab", "all"],
                    default="all", help="which scenario to demo")
    run(ap.parse_args().mode)


if __name__ == "__main__":
    main()
