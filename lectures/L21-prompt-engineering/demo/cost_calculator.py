"""Token cost calculator for prompt engineering decisions.

Estimates cost / latency of various prompting strategies (zero-shot,
few-shot, CoT, self-consistency, ToT) using 2026-Q2 published prices.
Generates a comparison bar chart.

Run:  python cost_calculator.py
"""
from __future__ import annotations
import os
from dataclasses import dataclass
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'PingFang SC',
    'Microsoft JhengHei', 'Segoe UI Emoji', 'DejaVu Sans',
]
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)


# Prices in USD per 1M tokens (as of 2026-Q2 — refresh quarterly)
PRICES = {
    "GPT-5":           {"in":  3.00, "out":  12.00},
    "GPT-5-mini":      {"in":  0.30, "out":   1.20},
    "Claude 4.5 Sonnet": {"in": 3.00, "out": 15.00},
    "Claude 4.5 Haiku": {"in":  0.80, "out":   4.00},
    "Gemini 3 Pro":    {"in":  1.25, "out":   5.00},
    "Gemini 3 Flash":  {"in":  0.10, "out":   0.40},
    "DeepSeek V3.5":   {"in":  0.27, "out":   1.10},
}


@dataclass
class Strategy:
    name: str
    in_tokens_factor: float   # multiplier on prompt tokens
    out_tokens_factor: float  # multiplier on output tokens
    samples: int = 1          # for self-consistency / best-of-N

    def cost(self, in_t: int, out_t: int, prices: dict) -> float:
        ti = in_t * self.in_tokens_factor * self.samples
        to = out_t * self.out_tokens_factor * self.samples
        return (ti / 1e6 * prices["in"]) + (to / 1e6 * prices["out"])


STRATEGIES = [
    Strategy("Zero-shot",       1.0, 1.0, 1),
    Strategy("Few-shot (3)",    2.5, 1.0, 1),
    Strategy("Few-shot (10)",   6.0, 1.0, 1),
    Strategy("CoT (let's think)",  1.05, 3.0, 1),
    Strategy("Few-shot CoT (3)",   3.5, 3.0, 1),
    Strategy("Self-Consistency-5", 1.05, 3.0, 5),
    Strategy("Tree-of-Thought-3x3", 2.0, 6.0, 9),
    Strategy("Reflexion (3 iters)", 2.0, 4.0, 3),
]


def main() -> None:
    # baseline: 500 input tokens, 200 output tokens per query
    in_t, out_t = 500, 200
    queries_per_day = 10_000
    days = 30

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Chart 1: per-query cost across strategies (using GPT-5)
    prices = PRICES["GPT-5"]
    costs = [s.cost(in_t, out_t, prices) * 1000 for s in STRATEGIES]
    names = [s.name for s in STRATEGIES]
    bars = axes[0].barh(names, costs, color="#4C72B0",
                        edgecolor="black", lw=0.4)
    for b, c in zip(bars, costs):
        axes[0].text(c + max(costs) * 0.01,
                     b.get_y() + b.get_height() / 2,
                     f"${c:.3f}", va="center", fontsize=9)
    axes[0].set_xlabel("成本 (mUSD per 1000 queries)  ·  GPT-5 prices")
    axes[0].set_title(f"Per-strategy cost  ({in_t}-in, {out_t}-out tokens)")
    axes[0].grid(axis="x", alpha=0.3)
    axes[0].invert_yaxis()

    # Chart 2: monthly cost across providers for "CoT" strategy
    cot = STRATEGIES[3]
    provider_costs = []
    for prov, p in PRICES.items():
        c = cot.cost(in_t, out_t, p) * queries_per_day * days
        provider_costs.append((prov, c))
    provider_costs.sort(key=lambda x: x[1])
    pnames, pvals = zip(*provider_costs)
    bars2 = axes[1].barh(pnames, pvals, color="#D62728",
                         edgecolor="black", lw=0.4)
    for b, c in zip(bars2, pvals):
        axes[1].text(c + max(pvals) * 0.01,
                     b.get_y() + b.get_height() / 2,
                     f"${c:,.0f}", va="center", fontsize=9)
    axes[1].set_xlabel("月成本 (USD)")
    axes[1].set_title(f"CoT 策略月费  ·  {queries_per_day:,} queries/day × {days} d")
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Prompt 工程的物理账单 — 策略 × 模型选择都是成本决策\n"
        "（价格 2026-Q2，需季度刷新）", fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUT, "cost_calculator.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

    # Print breakdown
    print("\nPer-strategy cost breakdown (GPT-5, USD per 1000 queries):")
    for s, c in zip(STRATEGIES, costs):
        print(f"  {s.name:30s} ${c:.4f}  (×{s.samples} samples)")


if __name__ == "__main__":
    main()
