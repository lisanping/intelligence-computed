"""L26 · AI Agent — 30 行最小 Agent 实现 + ReAct 完整版

两个脚本合并：
  - minimal_agent.py 部分：30 行 Python 实现一个会调用工具的 Agent (mock LLM)
  - react_agent.py 部分：完整的 Reason-Act-Observe 循环
  - 同时生成 swe_bench_progress.png 图（2024-2026 SWE-Bench Verified 解题率进展）

本文件可直接运行；无需 API key（使用 mock LLM）。
"""
from __future__ import annotations
import sys, pathlib, re
ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "design" / "meta"))
import cjk_font  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# 第一部分：30 行最小 Agent
# ─────────────────────────────────────────────────────────────────────
def minimal_agent_demo():
    """30 行实现一个能调用工具的 Agent（用 mock LLM 演示）。"""
    print("\n--- 30 行最小 Agent (mock LLM) ---")

    # 工具定义
    def calculator(expr: str) -> str:
        try:
            return str(eval(expr, {"__builtins__": {}}, {}))
        except Exception as e:
            return f"ERROR: {e}"

    def get_weather(city: str) -> str:
        # mock 数据
        return f"{city} 今日 22°C 多云"

    tools = {"calculator": calculator, "weather": get_weather}

    # Mock LLM：解析 user 问题并选择工具
    def mock_llm(prompt: str) -> str:
        if "天气" in prompt:
            city = prompt.split("天气")[0].strip()
            return f"调用 weather('{city}')"
        if any(c in prompt for c in "+-*/"):
            expr = re.sub(r"[^\d\+\-\*\/\(\)\.\s]", "", prompt).strip()
            return f"调用 calculator('{expr}')"
        return "我不知道"

    # Agent 主循环（最小化）
    def agent(question: str, max_steps: int = 3):
        for step in range(max_steps):
            decision = mock_llm(question)
            print(f"  [step {step}] LLM 决策: {decision}")
            m = re.match(r"调用 (\w+)\('(.*)'\)", decision)
            if m:
                tool, arg = m.group(1), m.group(2)
                result = tools[tool](arg)
                print(f"  [step {step}] 工具返回: {result}")
                return result
            else:
                return decision
        return "达到最大步数"

    print("\nUser: 计算 (2+3)*4")
    print(f"Final: {agent('计算 (2+3)*4')}")
    print("\nUser: 北京天气")
    print(f"Final: {agent('北京天气')}")


# ─────────────────────────────────────────────────────────────────────
# 第二部分：ReAct (Reason-Act-Observe) 完整循环
# ─────────────────────────────────────────────────────────────────────
def react_agent_demo():
    """完整 ReAct 循环 — 多步推理 + 工具使用。"""
    print("\n--- ReAct Agent 完整循环 ---")

    # 知识库（mock RAG）
    knowledge = {
        "Python": "Python 由 Guido van Rossum 于 1989 年设计。",
        "Transformer": "Transformer 由 Google 团队 2017 年发表。",
        "GPT-4": "GPT-4 是 OpenAI 2023 年发布的多模态模型。",
    }

    def search(query: str) -> str:
        for k, v in knowledge.items():
            if k.lower() in query.lower():
                return v
        return f"未找到与 '{query}' 相关的内容"

    def calc(expr: str) -> str:
        try:
            return str(eval(expr, {"__builtins__": {}}, {}))
        except Exception as e:
            return f"ERROR: {e}"

    tools = {"search": search, "calc": calc}

    def react_step(state: dict) -> dict:
        """单步：根据 state 产生下一步 action。"""
        thought = state.get("last_thought", "")
        history = state.get("history", [])

        # Mock 思考逻辑：根据原问题与 history 做下一步
        question = state["question"]
        if not history:
            if "年" in question or "几岁" in question or "多少年" in question:
                state["last_thought"] = "需要查找时间信息"
                state["next_action"] = ("search", question.split("？")[0].strip())
            else:
                state["last_thought"] = "试试搜索"
                state["next_action"] = ("search", question)
        elif len(history) == 1:
            # 第二步：根据 search 结果决定计算
            obs = history[-1]["observation"]
            if "1989" in obs or "2017" in obs or "2023" in obs:
                year = int(re.search(r"(19\d{2}|20\d{2})", obs).group())
                state["last_thought"] = f"找到了 {year}，现在算到 2026 的差"
                state["next_action"] = ("calc", f"2026-{year}")
            else:
                state["last_thought"] = "无法继续"
                state["next_action"] = ("done", obs)
        else:
            state["last_thought"] = "已经有结果"
            state["next_action"] = ("done", history[-1]["observation"])
        return state

    def react_loop(question: str, max_steps: int = 5):
        state = {"question": question, "history": []}
        for step in range(max_steps):
            state = react_step(state)
            tool, arg = state["next_action"]
            print(f"\n  [step {step}]")
            print(f"  Thought: {state['last_thought']}")
            print(f"  Action: {tool}({arg!r})")
            if tool == "done":
                print(f"  Final answer: {arg}")
                return arg
            obs = tools[tool](arg)
            print(f"  Observation: {obs}")
            state["history"].append({"action": tool, "arg": arg,
                                      "observation": obs})
        return "达到最大步数"

    print("\nQuestion: Python 出生到现在多少年了？")
    react_loop("Python 出生到现在多少年了？")
    print("\nQuestion: Transformer 出生到现在多少年了？")
    react_loop("Transformer 出生到现在多少年了？")


# ─────────────────────────────────────────────────────────────────────
# 第三部分：SWE-Bench Verified 解题率进展图
# ─────────────────────────────────────────────────────────────────────
def swe_bench_progress():
    months = ["2023-09\nGPT-4 baseline",
              "2024-01\nDevin alpha",
              "2024-06\nClaude 3 Opus",
              "2024-10\nClaude 3.5 + AGENTLESS",
              "2025-02\nClaude 3.7 Sonnet",
              "2025-05\nGPT-4.5 Agent",
              "2025-09\nClaude 4 Sonnet",
              "2026-01\nClaude 4.5 + Computer",
              "2026-04\nGPT-5 + Aider"]
    score = [1.96, 13.86, 38.0, 49.0, 62.3, 64.6, 70.5, 75.2, 78.0]
    cols = ["#bbb"] * 3 + ["#27AE60"] * 6

    fig, ax = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    bars = ax.bar(range(len(months)), score, color=cols, edgecolor="black")
    for i, v in enumerate(score):
        ax.text(i, v + 1.5, f"{v}%", ha="center", fontsize=10,
                fontweight="bold")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_ylabel("SWE-Bench Verified 解题率 (%)", fontsize=11)
    ax.set_title("SWE-Bench Verified 进展 (2023-09 → 2026-04)\n"
                 "30 个月内从 ~2% 攀升至 ~78%（人类基线 ≈ 90%）",
                 fontsize=13, fontweight="bold")
    ax.axhline(90, ls="--", color="#C0392B", lw=1.2)
    ax.text(8, 92, "人类基线 ~90%", color="#C0392B", fontsize=9, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    out = OUT / "swe_bench_progress.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────
# 第四部分：Reflexion 消融图（已存在 reflexion_ablation.png 占位）
# ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("=== L26 minimal agent + ReAct + SWE-Bench progress ===")
    minimal_agent_demo()
    react_agent_demo()
    swe_bench_progress()
    print("\nDone.")
