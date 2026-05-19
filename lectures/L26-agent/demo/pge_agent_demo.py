"""
L32 · Agent — PGE 闭环 Agent 演示
=============================================================
Goal   : 演示 Plan → Generate → Evaluate 三角闭环，三种场景：
         simple  — Evaluator 直接放行（单轮）
         reflect — Evaluator 驳回 → 反馈 → 修正通过（多轮）
         fail    — 达到最大循环次数 → 优雅退出
Run    : python demo/pge_agent_demo.py --mode all       # 全部场景
         python demo/pge_agent_demo.py --mode reflect   # 仅反思场景
Deps   : (none — pure stdlib, mock mode)
Seed   : 1337
"""
import argparse
import json
import textwrap

# ── Mock Tools ───────────────────────────────────────────────
TOOLS = {
    "get_weather": {
        "desc": "查询城市天气",
        "mock": lambda city: f"{city}：晴，28°C，湿度 45%，微风",
    },
    "web_search": {
        "desc": "网络搜索",
        "mock": lambda q: f"搜索结果 [{q}]: 量子计算利用量子比特的叠加和纠缠态进行并行计算...",
    },
    "write_text": {
        "desc": "生成文本段落",
        "mock": lambda topic: f"[生成文本] {topic}: 量子计算是利用量子力学原理进行信息处理的新型计算范式。"
                              f"与经典计算不同，它使用量子比特，可同时处于 0 和 1 的叠加态。",
    },
    "fact_check": {
        "desc": "事实核查",
        "mock": lambda text: "核查结果：文本基本准确，建议补充量子纠错相关内容。",
    },
}

# ── PGE Agent Core ───────────────────────────────────────────
SCENARIOS = {
    "simple": {
        "task": "查今天北京的天气",
        "plans": [
            [{"tool": "get_weather", "args": "北京", "reason": "直接查询天气"}],
        ],
        "eval_pass": [True],  # round 0: pass
    },
    "reflect": {
        "task": "写一段关于量子计算的科普文章，确保事实准确",
        "plans": [
            [{"tool": "web_search", "args": "量子计算基础概念", "reason": "收集背景知识"},
             {"tool": "write_text", "args": "量子计算科普", "reason": "生成初稿"}],
            [{"tool": "fact_check", "args": "初稿内容", "reason": "根据反馈核查事实"},
             {"tool": "write_text", "args": "修订版量子计算科普", "reason": "补充量子纠错内容"}],
        ],
        "eval_pass": [False, True],  # round 0: reject, round 1: pass
        "feedback": ["初稿缺少量子纠错的内容，事实准确性 70%。请补充纠错码相关知识并重新生成。"],
    },
    "fail": {
        "task": "预测明天特斯拉股票的收盘价",
        "plans": [
            [{"tool": "web_search", "args": "特斯拉股票预测", "reason": "搜索市场分析"}],
            [{"tool": "web_search", "args": "TSLA 技术分析", "reason": "尝试不同搜索角度"}],
            [{"tool": "web_search", "args": "特斯拉财报分析", "reason": "尝试基本面分析"}],
        ],
        "eval_pass": [False, False, False],
        "feedback": [
            "搜索结果无法提供可靠的股票价格预测。准确性评分：20%。",
            "技术分析数据不足以做出精确预测。准确性评分：25%。",
            "基本面分析也无法预测明天的具体价格。建议放弃此任务。",
        ],
    },
}


def pge_agent(scenario: dict, max_rounds: int = 3):
    """Plan → Generate → Evaluate loop with mock responses."""
    task = scenario["task"]
    plans = scenario["plans"]
    eval_pass = scenario["eval_pass"]
    feedbacks = scenario.get("feedback", [])

    _header(f"Task: {task}")

    for rnd in range(max_rounds):
        plan_idx = min(rnd, len(plans) - 1)
        plan = plans[plan_idx]

        # ── Plan ──
        _phase("Planner", f"Round {rnd + 1}/{max_rounds}")
        if rnd > 0 and rnd - 1 < len(feedbacks):
            _line(f"  收到反馈: {feedbacks[rnd - 1]}", "31")  # red
        for i, step in enumerate(plan):
            _line(f"  Step {i+1}: {step['tool']}({step['args']}) — {step['reason']}")

        # ── Generate ──
        _phase("Generator", "执行计划")
        results = []
        for step in plan:
            tool_fn = TOOLS[step["tool"]]["mock"]
            result = tool_fn(step["args"])
            results.append(result)
            _line(f"  [{step['tool']}] → {result[:80]}")

        # ── Evaluate ──
        eval_idx = min(rnd, len(eval_pass) - 1)
        passed = eval_pass[eval_idx]
        _phase("Evaluator", "质量评审")

        if passed:
            _line("  ✅ 评审通过 — 任务完成!", "32")  # green
            return True

        if rnd < max_rounds - 1:
            fb = feedbacks[rnd] if rnd < len(feedbacks) else "结果质量不足，请改进。"
            _line(f"  ❌ 评审未通过 — 进入下一轮", "33")  # yellow
            _line(f"  反馈: {fb}", "33")
        else:
            _line("  ❌ 达到最大循环次数 — 优雅退出", "31")
            _line("  Agent 判断: 当前工具和能力无法完成此任务。", "31")
            return False

    return False


# ── Output helpers ───────────────────────────────────────────
def _header(text):
    print(f"\n{'═' * 62}")
    print(f"  {text}")
    print(f"{'═' * 62}")

def _phase(name, detail):
    colors = {"Planner": "34", "Generator": "36", "Evaluator": "35"}
    c = colors.get(name, "0")
    print(f"\n  \033[{c};1m[{name}]\033[0m {detail}")

def _line(text, color=None):
    if color:
        print(f"\033[{color}m{text}\033[0m")
    else:
        print(text)


# ── CLI ──────────────────────────────────────────────────────
def run(mode: str):
    print("=" * 62)
    print("L26 Demo: PGE 闭环 Agent (Plan → Generate → Evaluate)")
    print("=" * 62)

    targets = SCENARIOS if mode == "all" else {mode: SCENARIOS[mode]}
    for key, scenario in targets.items():
        pge_agent(scenario, max_rounds=3)

    print(f"\n{'=' * 62}")
    print("观察要点:")
    print("  1. simple  — Evaluator 一次放行，无需循环")
    print("  2. reflect — Evaluator 驳回 → 具体反馈 → Planner 调整 → 通过")
    print("  3. fail    — max_rounds 后优雅退出，不死循环")
    print("  4. 核心: 一个能自我纠错的弱模型 > 不会反思的强模型")
    print("=" * 62)


def main():
    ap = argparse.ArgumentParser(description="L26: PGE Agent demo")
    ap.add_argument("--mode", choices=["simple", "reflect", "fail", "all"],
                    default="all", help="which scenario to demo")
    run(ap.parse_args().mode)


if __name__ == "__main__":
    main()
