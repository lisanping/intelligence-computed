"""L35 finale visuals + actionable career-path artefacts.

Outputs:
  - careers_table.png            5 career paths × 7 column matrix
  - action_checklist.png         Student "next-90-days" checklist visual
  - 6_predictions_dashboard.png  Six predictions × confidence dashboard

Also writes:
  - action_checklist.md          machine-readable next-step plan
  - resources.md                 curated subscriptions & repos to follow

Run:  python generate_finale.py
"""
from __future__ import annotations
import os
import textwrap
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'PingFang SC',
    'Microsoft JhengHei', 'Segoe UI Emoji', 'DejaVu Sans',
]
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = os.path.dirname(__file__)
FIG = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# 1.  Careers table
# ──────────────────────────────────────────────────────────────────────
CAREERS = [
    {
        "name": "AI 应用工程师",
        "level": "广 + 浅",
        "primary_lectures": "L21–L31",
        "core_skills": "Prompt / RAG / Agent / LLMOps",
        "demand_2026": "★★★★★",
        "salary_band": "$120K–$220K (US) ／ ¥40W–¥80W",
        "first_step": "做一个真实业务场景的 RAG → Agent 流水线",
    },
    {
        "name": "ML / DL 研究员",
        "level": "深 + 窄",
        "primary_lectures": "L13–L20, L29",
        "core_skills": "Pre-training / RL / 评估 / 论文",
        "demand_2026": "★★★★",
        "salary_band": "$200K–$1M+ (顶尖实验室)",
        "first_step": "复现一篇近 6 个月的 NeurIPS / ICLR 论文",
    },
    {
        "name": "AI 安全 / 对齐工程师",
        "level": "深 + 横",
        "primary_lectures": "L17, L18, L34",
        "core_skills": "红队 / 对齐 / 可解释性 / 治理",
        "demand_2026": "★★★★ (爆发中)",
        "salary_band": "$150K–$400K (Anthropic / OpenAI)",
        "first_step": "贡献一个 jailbreak 到 promptfoo 或 garak",
    },
    {
        "name": "AI 产品经理",
        "level": "横 + 商业",
        "primary_lectures": "L00, L21–L30",
        "core_skills": "需求 / UX / cost / 用户教育",
        "demand_2026": "★★★★",
        "salary_band": "$140K–$280K",
        "first_step": "做 5 次产品 hackathon；写 10 篇 PRD",
    },
    {
        "name": "AI 教育 / 内容创作",
        "level": "广 + 沟通",
        "primary_lectures": "L00, L01, L35",
        "core_skills": "讲解 / 写作 / 视频 / 社群",
        "demand_2026": "★★★",
        "salary_band": "$50K–∞ (头部大 V)",
        "first_step": "每周写 1 篇笔记；3 个月攒 100 篇",
    },
]


def careers_table() -> None:
    rows = ["职业方向", "性格匹配", "对应讲次", "核心技能",
            "2026 需求", "薪酬区间", "第一步"]
    data = [[c["name"], c["level"], c["primary_lectures"],
             c["core_skills"], c["demand_2026"], c["salary_band"],
             c["first_step"]] for c in CAREERS]

    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.axis("off")
    table = ax.table(cellText=data, colLabels=rows,
                     cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    # Style: header row
    palette = ["#1565C0", "#2E7D32", "#C62828", "#6A1B9A", "#EF6C00"]
    for j, _ in enumerate(rows):
        cell = table[(0, j)]
        cell.set_facecolor("#263238")
        cell.set_text_props(color="white", fontweight="bold")
    for i in range(1, len(data) + 1):
        for j in range(len(rows)):
            cell = table[(i, j)]
            if j == 0:
                cell.set_facecolor(palette[i - 1])
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#FAFAFA" if i % 2 else "#FFFFFF")

    # Auto-fit column widths
    for j in range(len(rows)):
        table.auto_set_column_width(j)

    fig.suptitle("五条职业方向  ·  与本课程讲次的对应",
                 fontsize=14, fontweight="bold", y=0.95)
    fig.tight_layout()
    out = os.path.join(FIG, "careers_table.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ──────────────────────────────────────────────────────────────────────
# 2.  Action checklist visual
# ──────────────────────────────────────────────────────────────────────
ACTION_PHASES = [
    ("第 1-7 天", "立刻开始 — 工具到位",
     ["选定 1 条职业方向（5 选 1）",
      "注册 Claude / GPT / Gemini 三家 API 账号",
      "fork 一个本课程仓库；跑通至少 3 个 demo",
      "订阅 5 个 newsletter（见 resources.md）"]),
    ("第 8-30 天", "首个项目 — 端到端跑通",
     ["从 L24 RAG 出发：用一个真实文档库做 chatbot",
      "评估：用 promptfoo 写 20 个 golden cases",
      "成本：用 cost_calculator.py 算月费",
      "把项目写成博客，贴 GitHub"]),
    ("第 31-60 天", "深入 — 跟一个真前沿方向",
     ["每周读 2 篇 arxiv / 1 篇 Anthropic / OpenAI blog",
      "复现 1 篇近 6 个月的论文（不需完全复现，关键实验即可）",
      "向 1 个开源项目提 PR（哪怕是文档）",
      "找到 1 个学习社群（Discord / Slack / 微信群）"]),
    ("第 61-90 天", "立旗 — 公开你存在",
     ["开始公开输出：Twitter / 知乎 / 公众号 / B站 选 1",
      "主动联系 3 个本领域工程师做 1 对 1 chat",
      "申请 1 个相关岗位（哪怕实习 / 兼职）",
      "回答这门课的最后一个问题：你接下来要做什么？"]),
]


def action_checklist() -> None:
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.set_axis_off()

    palette = ["#1976D2", "#43A047", "#F57C00", "#D32F2F"]
    x = 0.5
    for i, (phase, title, items) in enumerate(ACTION_PHASES):
        color = palette[i]
        # phase header bar
        ax.add_patch(FancyBboxPatch((x, 8.0), 12.0, 0.7,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor="black",
                                    lw=0.6))
        ax.text(x + 0.3, 8.35, phase, fontsize=11, fontweight="bold",
                color="white", va="center")
        ax.text(x + 2.0, 8.35, "·  " + title, fontsize=12,
                color="white", va="center")
        # items
        for j, it in enumerate(items):
            y = 7.4 - j * 0.45
            ax.text(x + 0.6, y, "[ ]", fontsize=12, color=color, va="center",
                    family="monospace")
            ax.text(x + 1.2, y, it, fontsize=10.5, va="center")
        # advance to next phase row
        ax.text(x, 5.4 - 0.0, "", fontsize=8)
        # vertical offset: re-anchor by drawing in stacked figure tier
        # (we simulate by saving and re-issuing — use simpler y math)

    # The above stacks all phases at top; redo with proper y-step
    ax.cla()
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.set_axis_off()
    y = 8.4
    for i, (phase, title, items) in enumerate(ACTION_PHASES):
        color = palette[i]
        ax.add_patch(FancyBboxPatch((0.5, y - 0.5), 12, 0.6,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor="black",
                                    lw=0.6))
        ax.text(0.8, y - 0.2, phase, fontsize=11, fontweight="bold",
                color="white", va="center")
        ax.text(2.4, y - 0.2, "·  " + title, fontsize=12,
                color="white", va="center")
        y -= 0.85
        for it in items:
            ax.text(0.9, y, "[ ]", fontsize=12, color=color, va="center",
                    family="monospace")
            ax.text(1.55, y, it, fontsize=10.5, va="center")
            y -= 0.42
        y -= 0.2

    ax.text(6.5, 0.3,
            "[CORE] 核心规则：90 天内必须有 1 个公开可证明的产出（项目/博客/PR/视频）。",
            ha="center", fontsize=11, color="#444444", style="italic")
    fig.suptitle("90 天行动清单  ·  把课程变成你的下一份工作",
                 fontsize=14, fontweight="bold", y=0.96)
    out = os.path.join(FIG, "action_checklist.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ──────────────────────────────────────────────────────────────────────
# 3.  Six predictions dashboard
# ──────────────────────────────────────────────────────────────────────
PREDICTIONS = [
    ("① 推理模型成为标配 + 廉价",          0.85, "已发生 (GPT-5/Claude 4.5)"),
    ("② Agent 成为新工业自动化层",        0.70, "进行中 (Devin/Cursor 量产)"),
    ("③ 多模态成为默认输入",              0.85, "已发生 (omni 实时对话)"),
    ("④ 具身智能商业化加速",              0.55, "早期 (Figure/宇树出货)"),
    ("⑤ 教育/创作/编程被重构",            0.75, "进行中 (v0/Khanmigo/Cursor)"),
    ("⑥ 监管成熟 + 安全成为产品差异",     0.65, "进行中 (EU AI Act 已生效)"),
]


def predictions_dashboard() -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    names = [p[0] for p in PREDICTIONS]
    confs = [p[1] for p in PREDICTIONS]
    notes = [p[2] for p in PREDICTIONS]
    colors = ["#2CA02C" if c >= 0.75 else
              "#F9A825" if c >= 0.6 else "#D62728" for c in confs]
    bars = ax.barh(names, confs, color=colors, edgecolor="black", lw=0.4)
    for b, c, n in zip(bars, confs, notes):
        ax.text(c + 0.01, b.get_y() + b.get_height() / 2,
                f"{int(c*100)}%   ·  {n}", va="center", fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("置信度（讲师 2026-Q2 个人估计）")
    ax.set_title("六个预言  ·  到 2030 年为止的当前置信度\n"
                 "（绿=高置信  黄=中  红=低；置信度本身也是预言）",
                 fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    out = os.path.join(FIG, "6_predictions_dashboard.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ──────────────────────────────────────────────────────────────────────
# 4.  Markdown artefacts
# ──────────────────────────────────────────────────────────────────────
def write_action_checklist_md() -> None:
    lines = ["# 90 天行动清单", "",
             "> 把这门课变成你的下一份工作。"]
    for phase, title, items in ACTION_PHASES:
        lines.append(f"\n## {phase} — {title}\n")
        for it in items:
            lines.append(f"- [ ] {it}")
    lines.append("\n---\n\n**核心规则**：90 天内必须有 1 个公开可证明的产出"
                 "（项目/博客/PR/视频）。")
    out = os.path.join(OUT_DIR, "action_checklist.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out}")


def write_resources_md() -> None:
    text = textwrap.dedent("""\
        # 学员长期资源清单（2026-Q2）

        > 以下都是讲师亲测、目前仍活跃维护的来源。
        > 优先级：先订满第 1 类（4-5 个），再扩展。

        ## 1️⃣ Newsletter（每周必读）

        - **Import AI** (Jack Clark, Anthropic 联合创始人) — 每周一篇政策 + 技术
        - **The Batch** (Andrew Ng) — DeepLearning.AI 出品，技术 + 教育视角
        - **Stratechery** (Ben Thompson) — AI 商业战略最深刻
        - **Last Week in AI** — 周更，覆盖工业落地
        - **Latent Space** — Practitioner 视角的 podcast + newsletter

        ## 2️⃣ 实验室博客（精读，不要订全文 RSS）

        - Anthropic Research / OpenAI Research / DeepMind Blog
        - METR / UK AISI / NIST AISI（安全方向必读）
        - Hugging Face Blog（开源 + 教程）

        ## 3️⃣ Twitter / X 关注列表（10-20 人足够）

        - 从你已知的最佳工程师/研究员开始；用 list 隔离信息流
        - 不要追求"全网最新"——选 10-20 人深读

        ## 4️⃣ 社群

        - Hugging Face Discord（开源最大）
        - LocalLLaMA Reddit（端侧 + 量化）
        - r/MachineLearning（学术为主）
        - 中文：知乎机器学习专栏 + AI 公众号 5-10 个（避免娱乐化的）

        ## 5️⃣ 必看代码仓库（star + 偶尔 git pull）

        - vllm-project/vllm（推理引擎）
        - langchain-ai/langgraph 或 vercel/ai 或 mastra-ai/mastra（agent）
        - microsoft/markitdown（文档处理）
        - anthropic-ai/courses & quickstarts
        - openai/openai-cookbook
        - Significant-Gravitas/AutoGPT（保留教学价值）

        ## 6️⃣ 课程外的进阶阅读（按时序）

        1. *Bishop*, Pattern Recognition and ML（数学基础）
        2. *Goodfellow et al*, Deep Learning（系统化深度学习）
        3. *Sutton & Barto*, RL: An Introduction（RL 圣经）
        4. *Stuart Russell*, Human Compatible（对齐哲学）
        5. *Bostrom*, Superintelligence（远期 x-risk 视角）

        ## ⚠️ 反建议

        - 不要订阅"AI 日报"类聚合 — 会让你每天读 100 条噪音
        - 不要追每一个新模型 — 等社区评估 1 周再看
        - 不要在 Twitter 上吵架 — 写代码时间 > 辩论时间
        - 不要相信任何"AGI 倒计时"具体年份 — 包括我说的

        ---

        **最后**：信息消费 ≤ 1 小时/天。其余时间写代码、做项目、找人聊。
    """)
    out = os.path.join(OUT_DIR, "resources.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {out}")


def main() -> None:
    careers_table()
    action_checklist()
    predictions_dashboard()
    write_action_checklist_md()
    write_resources_md()


if __name__ == "__main__":
    main()
