"""
L31 - LLMOps - Dashboard mock: latency, cost, monitoring
=============================================================
Goal   : Generate production monitoring visualizations
Figures:
  figures/latency_distribution.png  -- TTFT/TPS P50/P95/P99 (V4)
  figures/cost_waterfall.png        -- Inference cost breakdown (V6)
  figures/monitoring_panel.png      -- 4-panel monitoring dashboard (V10)
Run    : python demo/llmops_dashboard.py [--fig latency|cost|monitor|all]
Deps   : pip install matplotlib numpy
Seed   : 1337
"""
import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(encoding="utf-8")

np.random.seed(1337)
FIG_DIR = Path(__file__).parent / "figures"


# -- V4: Latency distribution ------------------------------------
def plot_latency(save: bool = True):
    """TTFT and TPS distributions with percentile markers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # TTFT (Time to First Token)
    ttft = np.random.lognormal(mean=5.5, sigma=0.6, size=2000)
    ttft = ttft[ttft < 1500]  # clip outliers
    ax = axes[0]
    ax.hist(ttft, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    for pct, col in [(50, "green"), (95, "orange"), (99, "red")]:
        val = np.percentile(ttft, pct)
        ax.axvline(val, color=col, ls="--", lw=2, label=f"P{pct}: {val:.0f}ms")
    ax.set_xlabel("TTFT (ms)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Time to First Token Distribution", fontsize=12, weight="bold")
    ax.legend(fontsize=10)

    # TPS (Tokens per Second)
    tps = np.random.normal(loc=45, scale=12, size=2000)
    tps = tps[tps > 0]
    ax = axes[1]
    ax.hist(tps, bins=50, color="seagreen", alpha=0.7, edgecolor="white")
    for pct, col in [(50, "green"), (5, "orange"), (1, "red")]:
        val = np.percentile(tps, pct) if pct > 10 else np.percentile(tps, pct)
        label = f"P{100-pct}" if pct < 50 else f"P{pct}"
        ax.axvline(val, color=col, ls="--", lw=2, label=f"{label}: {val:.0f} tok/s")
    ax.set_xlabel("Throughput (tokens/s)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Throughput Distribution", fontsize=12, weight="bold")
    ax.legend(fontsize=10)

    fig.suptitle("Production Latency Monitoring", fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "latency_distribution.png", save)


# -- V6: Cost waterfall ------------------------------------------
def plot_cost_waterfall(save: bool = True):
    """Inference cost breakdown waterfall chart."""
    items = ["GPU Compute", "KV Cache\nMemory", "Network\nI/O",
             "Embedding\nLookup", "Sampling\n+ Detokenize", "Total"]
    costs = [4.2, 1.8, 0.6, 0.3, 0.1, 7.0]
    cumulative = [0]
    for c in costs[:-1]:
        cumulative.append(cumulative[-1] + c)
    cumulative.append(0)  # total bar starts from 0

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#3498db", "#3498db", "#3498db", "#3498db", "#3498db", "#2ecc71"]
    for i, (item, cost, base) in enumerate(zip(items, costs, cumulative)):
        bar = ax.bar(i, cost, bottom=base, color=colors[i], alpha=0.85,
                      edgecolor="white", width=0.6)
        ax.text(i, base + cost / 2, f"${cost:.1f}", ha="center", va="center",
                fontsize=11, weight="bold", color="white" if cost > 0.5 else "black")

    # Connect lines between waterfall segments
    for i in range(len(items) - 2):
        ax.plot([i + 0.3, i + 0.7], [cumulative[i] + costs[i]] * 2,
                "k-", lw=0.8, alpha=0.4)

    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(items, fontsize=10)
    ax.set_ylabel("Cost per 1M tokens ($)", fontsize=12)
    ax.set_title("Inference Cost Breakdown (per 1M output tokens)",
                 fontsize=13, weight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 8.5)
    plt.tight_layout()
    _save(fig, "cost_waterfall.png", save)


# -- V10: 4-panel monitoring dashboard ----------------------------
def plot_monitoring_panel(save: bool = True):
    """Mock 4-panel LLMOps monitoring dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    hours = np.arange(0, 24, 0.25)

    # Panel 1: Request rate
    ax = axes[0, 0]
    base = 100 + 80 * np.sin(np.pi * hours / 12 - np.pi / 3)
    rate = base + np.random.normal(0, 10, len(hours))
    ax.plot(hours, rate, "steelblue", lw=1.5)
    ax.fill_between(hours, rate, alpha=0.2, color="steelblue")
    ax.set_title("Request Rate (req/min)", fontsize=11, weight="bold")
    ax.set_ylabel("Requests/min")
    ax.grid(True, alpha=0.3)

    # Panel 2: Latency percentiles
    ax = axes[0, 1]
    p50 = 200 + 50 * np.sin(np.pi * hours / 12) + np.random.normal(0, 15, len(hours))
    p95 = p50 * 1.8 + np.random.normal(0, 20, len(hours))
    p99 = p50 * 3.0 + np.random.normal(0, 30, len(hours))
    ax.plot(hours, p50, "g-", lw=1.5, label="P50")
    ax.plot(hours, p95, color="orange", lw=1.5, label="P95")
    ax.plot(hours, p99, "r-", lw=1.5, label="P99")
    ax.axhline(1000, color="red", ls="--", alpha=0.5, label="SLA (1s)")
    ax.set_title("Latency Percentiles (ms)", fontsize=11, weight="bold")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)

    # Panel 3: Token consumption
    ax = axes[1, 0]
    tokens_in = 50000 + 30000 * np.sin(np.pi * hours / 12 - np.pi / 3) + np.random.normal(0, 3000, len(hours))
    tokens_out = tokens_in * 0.4 + np.random.normal(0, 2000, len(hours))
    ax.stackplot(hours, [tokens_in, tokens_out], labels=["Input tokens", "Output tokens"],
                 colors=["#3498db", "#e74c3c"], alpha=0.7)
    ax.set_title("Token Consumption (per 15min)", fontsize=11, weight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel 4: Error rate + cost
    ax = axes[1, 1]
    error_rate = np.random.exponential(0.5, len(hours))
    error_rate[40:48] = np.random.uniform(3, 8, 8)  # spike at hour 10-12
    ax.bar(hours, error_rate, width=0.2, color="red", alpha=0.6, label="Error %")
    ax2 = ax.twinx()
    cost = np.cumsum(rate * 0.002 * 0.25)
    ax2.plot(hours, cost, "g-", lw=2, label="Cumulative cost ($)")
    ax.set_title("Error Rate & Cost", fontsize=11, weight="bold")
    ax.set_ylabel("Error %", color="red")
    ax2.set_ylabel("Cost ($)", color="green")
    ax.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Hour of Day")
        ax.set_xlim(0, 24)

    fig.suptitle("LLMOps Monitoring Dashboard (24h)", fontsize=14, weight="bold")
    plt.tight_layout()
    _save(fig, "monitoring_panel.png", save)


def _save(fig, name, save):
    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
        print(f"  saved -> {FIG_DIR / name}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="L31 LLMOps dashboard visualizations")
    p.add_argument("--fig", choices=["latency", "cost", "monitor", "all"],
                   default="all")
    args = p.parse_args()
    dispatch = dict(latency=plot_latency, cost=plot_cost_waterfall,
                    monitor=plot_monitoring_panel)
    if args.fig == "all":
        print("L31 LLMOps -- generating all figures ...")
        for name, fn in dispatch.items():
            print(f"[{name}]")
            fn()
        print("Done.")
    else:
        dispatch[args.fig]()


if __name__ == "__main__":
    main()
