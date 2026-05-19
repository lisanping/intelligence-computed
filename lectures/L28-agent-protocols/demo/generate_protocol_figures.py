"""L28 · Agent 协议层 — 5 张协议图

生成：
  - mcp_lifecycle.png              (MCP server-client 生命周期序列图)
  - protocol_stack_comparison.png  (MCP/A2A/ACP/AGNTCY 4 协议栈对比表)
  - security_defense_layers.png    (5 层防御深度示意)
  - protocol_adoption_matrix.png   (12 公司 × 4 协议采用矩阵)
  - computer_use_workflow.png      (Computer Use 工作流：截图 → LLM → 点击/键入)

依赖：numpy, matplotlib
"""
from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "design" / "meta"))
import cjk_font  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# 1. mcp_lifecycle.png — MCP 生命周期序列
# ─────────────────────────────────────────────────────────────────────
def mcp_lifecycle():
    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")

    # 三个垂直 lifeline
    actors = ["AI Client\n(Claude / Cursor)", "MCP Protocol\n(JSON-RPC 2.0)",
              "MCP Server\n(本地或远程)"]
    x_pos = [2, 6, 10]
    for x, name in zip(x_pos, actors):
        # box at top
        ax.add_patch(Rectangle((x - 1.2, 9), 2.4, 0.8,
                                facecolor="#1A1F2B", edgecolor="black"))
        ax.text(x, 9.4, name, ha="center", va="center", color="white",
                fontsize=10, fontweight="bold")
        # vertical line
        ax.plot([x, x], [0.5, 9], color="#444", lw=1.2, ls=":")

    # 消息序列 (y, from_idx, to_idx, label, color)
    msgs = [
        (8.4, 0, 1, "1. initialize", "#27AE60"),
        (8.0, 1, 2, "→ initialize", "#27AE60"),
        (7.6, 2, 1, "← capabilities", "#27AE60"),
        (7.2, 1, 0, "← capabilities", "#27AE60"),
        (6.6, 0, 1, "2. tools/list", "#2980B9"),
        (6.2, 1, 2, "→ tools/list", "#2980B9"),
        (5.8, 2, 1, "← [tool_def]", "#2980B9"),
        (5.4, 1, 0, "← [tool_def]", "#2980B9"),
        (4.8, 0, 1, "3. tools/call (read_file)", "#E67E22"),
        (4.4, 1, 2, "→ tools/call", "#E67E22"),
        (4.0, 2, 1, "← {result}", "#E67E22"),
        (3.6, 1, 0, "← {result}", "#E67E22"),
        (3.0, 0, 1, "4. resources/read", "#9B59B6"),
        (2.6, 1, 2, "→ resources/read", "#9B59B6"),
        (2.2, 2, 1, "← content", "#9B59B6"),
        (1.8, 1, 0, "← content", "#9B59B6"),
        (1.2, 0, 1, "5. shutdown", "#C0392B"),
        (0.8, 1, 2, "→ shutdown", "#C0392B"),
    ]
    for y, fi, ti, lbl, col in msgs:
        x1, x2 = x_pos[fi], x_pos[ti]
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
        ax.text((x1 + x2) / 2, y + 0.07, lbl, ha="center", fontsize=8.5,
                color=col)

    fig.suptitle("MCP 生命周期序列图  ·  Initialize → Discover → Invoke → Read → Shutdown",
                 fontsize=13, fontweight="bold")
    out = OUT / "mcp_lifecycle.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────
# 2. protocol_stack_comparison.png — 4 协议对比表
# ─────────────────────────────────────────────────────────────────────
def protocol_stack_comparison():
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    ax.axis("off")

    headers = ["协议", "提出方", "时间", "通信主体", "传输", "用途", "成熟度"]
    rows = [
        ["MCP", "Anthropic", "2024-11", "AI <-> Tool/Data", "stdio + SSE", "工具/资源调用", "★★★★★"],
        ["A2A", "Google", "2025-04", "Agent <-> Agent", "HTTP/gRPC", "跨 Agent 协作", "★★★☆☆"],
        ["ACP", "IBM Research", "2025-03", "Agent <-> Agent", "HTTP", "跨厂商互操作", "★★☆☆☆"],
        ["AGNTCY", "OASIS / 联盟", "2025-09", "Agent 平台", "多种", "标准化框架", "★★☆☆☆"],
    ]

    table = ax.table(
        cellText=rows, colLabels=headers,
        cellLoc="center", loc="center",
        colWidths=[0.10, 0.13, 0.10, 0.16, 0.13, 0.20, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.4)

    # 表头着色
    for j, _ in enumerate(headers):
        cell = table[(0, j)]
        cell.set_facecolor("#1A1F2B")
        cell.set_text_props(color="white", fontweight="bold")
    # 协议名着色
    proto_colors = ["#27AE60", "#2980B9", "#E67E22", "#9B59B6"]
    for i, col in enumerate(proto_colors, start=1):
        cell = table[(i, 0)]
        cell.set_facecolor(col)
        cell.set_text_props(color="white", fontweight="bold")

    fig.suptitle("Agent 协议栈对比 (2024-2026)",
                 fontsize=14, fontweight="bold")
    out = OUT / "protocol_stack_comparison.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────
# 3. security_defense_layers.png — 5 层防御深度
# ─────────────────────────────────────────────────────────────────────
def security_layers():
    layers = [
        ("L5 · 审计与回滚",  "操作日志 + 回滚机制 + 责任追溯",        "#1A1F2B"),
        ("L4 · 行为限制",    "rate limit + 危险操作白名单 + 二次确认", "#2C3E50"),
        ("L3 · 工具沙箱",    "权限最小化 + 文件系统隔离 + 网络限制",   "#34495E"),
        ("L2 · 输入消毒",    "Prompt Injection 检测 + 内容过滤",       "#5D6D7E"),
        ("L1 · 协议层签名",  "Server 证书 + 工具描述完整性校验",       "#7F8C8D"),
    ]

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")

    for i, (title, content, col) in enumerate(layers):
        y = 8.5 - i * 1.5
        ax.add_patch(Rectangle((1, y), 10, 1.2, facecolor=col,
                                edgecolor="white", lw=2))
        ax.text(2, y + 0.6, title, fontsize=13, fontweight="bold",
                color="white", va="center")
        ax.text(7, y + 0.6, content, fontsize=11, color="white",
                va="center")

    # 攻击者
    ax.text(6, 0.5, "[攻击] 恶意 MCP Server / Prompt Injection / Tool Poisoning",
            ha="center", fontsize=12, color="#C0392B", fontweight="bold")
    ax.annotate("", xy=(6, 1.4), xytext=(6, 0.9),
                arrowprops=dict(arrowstyle="->", color="#C0392B", lw=2))
    # 用户
    ax.text(6, 9.7, "[保护] 用户敏感数据 / 系统资源",
            ha="center", fontsize=12, color="#27AE60", fontweight="bold")

    fig.suptitle("Agent 协议安全 · 5 层纵深防御 (Defense in Depth)",
                 fontsize=14, fontweight="bold")
    out = OUT / "security_defense_layers.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────
# 4. protocol_adoption_matrix.png — 12 公司 × 4 协议
# ─────────────────────────────────────────────────────────────────────
def adoption_matrix():
    companies = ["Anthropic", "OpenAI", "Google", "Microsoft", "Meta",
                 "AWS", "GitHub", "Cursor", "Continue", "Zed",
                 "JetBrains", "DeepSeek"]
    protocols = ["MCP", "A2A", "ACP", "AGNTCY"]

    # 采用度：3=完全支持, 2=部分, 1=声明意向, 0=无
    matrix = np.array([
        [3, 1, 0, 0],   # Anthropic — MCP 主推
        [2, 2, 0, 1],   # OpenAI
        [2, 3, 0, 0],   # Google — A2A 主推
        [3, 1, 0, 0],   # Microsoft
        [2, 0, 0, 1],   # Meta
        [2, 1, 0, 0],   # AWS
        [3, 0, 0, 0],   # GitHub
        [3, 0, 0, 0],   # Cursor
        [3, 0, 0, 0],   # Continue
        [3, 0, 0, 0],   # Zed
        [3, 0, 0, 0],   # JetBrains
        [2, 0, 0, 0],   # DeepSeek
    ])

    fig, ax = plt.subplots(figsize=(8, 9), constrained_layout=True)
    im = ax.imshow(matrix, cmap="Greens", vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(len(protocols)))
    ax.set_xticklabels(protocols, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(companies)))
    ax.set_yticklabels(companies, fontsize=11)
    ax.set_xlabel("协议", fontsize=12)

    # 注释每格
    labels = {0: "—", 1: "意向", 2: "部分", 3: "完全"}
    for i in range(len(companies)):
        for j in range(len(protocols)):
            v = matrix[i, j]
            ax.text(j, i, labels[v], ha="center", va="center",
                    color="black" if v < 2 else "white",
                    fontsize=10, fontweight="bold")

    ax.set_title("协议采用矩阵 (2026 Q1)\nMCP 已成事实标准；A2A 在 Google 系；ACP/AGNTCY 仍早期",
                 fontsize=12, fontweight="bold")

    out = OUT / "protocol_adoption_matrix.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ─────────────────────────────────────────────────────────────────────
# 5. computer_use_workflow.png — Computer Use 工作流
# ─────────────────────────────────────────────────────────────────────
def computer_use_workflow():
    fig, ax = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")

    # 5 步流程
    steps = [
        (1, "1. 截屏\n(screenshot)", "#3498DB",
         "OS API 抓当前屏幕"),
        (4, "2. 视觉解析\n(VLM 分析)", "#9B59B6",
         "GPT-4V / Claude\n识别 UI 元素 + 文字"),
        (7, "3. 决策\n(LLM)", "#E67E22",
         "决定下一步操作\n(点击 / 键入 / 滚动)"),
        (10, "4. 动作生成\n(Action)", "#27AE60",
         "click(x,y) / type('text')\nscroll(direction)"),
        (13, "5. 执行 + 验证\n(回到 step 1)", "#C0392B",
         "OS API 发出鼠标键盘事件\n截图验证"),
    ]
    for x, label, col, content in steps:
        ax.add_patch(Rectangle((x - 0.9, 4), 1.8, 2.2,
                                facecolor=col, edgecolor="black", lw=1.5))
        ax.text(x, 5.6, label, ha="center", va="center",
                color="white", fontsize=10, fontweight="bold")
        ax.text(x, 4.5, content, ha="center", va="center",
                color="white", fontsize=8)

    # 横向箭头
    for x_from, x_to in zip([1, 4, 7, 10], [4, 7, 10, 13]):
        ax.annotate("", xy=(x_to - 1.0, 5.1), xytext=(x_from + 1.0, 5.1),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#444"))

    # 循环箭头
    ax.annotate("", xy=(1, 6.5), xytext=(13, 6.5),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#888",
                               connectionstyle="arc3,rad=-0.3"))
    ax.text(7, 7.5, "循环直到任务完成（典型任务 5-50 轮）",
            ha="center", fontsize=11, color="#666", style="italic")

    # 底部说明
    ax.text(7, 2.5,
            "代表实现：Anthropic Claude Computer Use (2024-10)、"
            "OpenAI Operator (2025-01)、Google Project Mariner (2024-12)",
            ha="center", fontsize=10, color="#444")
    ax.text(7, 1.8,
            "用途：在没有 API 的应用上自动化（PDF 阅读器、桌面办公、老旧 ERP 等）",
            ha="center", fontsize=10, color="#444")
    ax.text(7, 1.0,
            "[警告] 安全风险：屏幕截图可能含密码 / 个人数据；点击/键入是不可撤销的物理操作",
            ha="center", fontsize=10, color="#C0392B", fontweight="bold")

    fig.suptitle("Computer Use  ·  把屏幕给 AI 看 + 让 AI 像人一样操作",
                 fontsize=14, fontweight="bold")
    out = OUT / "computer_use_workflow.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


if __name__ == "__main__":
    print("=== L28 protocol figures ===")
    mcp_lifecycle()
    protocol_stack_comparison()
    security_layers()
    adoption_matrix()
    computer_use_workflow()
    print("Done.")
