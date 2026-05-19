"""
L22 - Open Source & Quantization - Visualizations
=============================================================
Goal   : Generate 6 matplotlib visualizations for lecture slides
Figures:
  figures/open_source_timeline.png      -- V1 open vs closed gap trend
  figures/family_radar.png              -- V2 four-family benchmark radar
  figures/memory_bar.png                -- V3 70B model memory by precision
  figures/quantization_intuition.png    -- V4 quantization process visual
  figures/quant_tradeoff.png            -- V5 precision-quality-memory curve
  figures/llamacpp_stars.png            -- V6 llama.cpp GitHub star growth
Run    : python demo/quantization_viz.py [--fig timeline|radar|memory|quant|tradeoff|stars|all]
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
from matplotlib.patches import FancyBboxPatch

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(encoding="utf-8")

np.random.seed(1337)
FIG_DIR = Path(__file__).parent / "figures"


# -- V1: Open-source vs closed-source gap timeline ---------------
def plot_timeline(save: bool = True):
    dates = ["2023.02", "2023.07", "2023.09", "2023.12",
             "2024.04", "2024.07", "2024.10", "2025.01"]
    gap_months = [18, 12, 9, 6, 4, 3, 2, 1]  # months behind closed-source SOTA
    events = {0: "LLaMA 1", 1: "LLaMA 2", 3: "Mixtral 8x7B",
              5: "LLaMA 3.1", 7: "DeepSeek-R1"}

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(dates)), gap_months, "o-", color="steelblue", lw=2.5, ms=8)
    ax.fill_between(range(len(dates)), gap_months, alpha=0.15, color="steelblue")

    for idx, label in events.items():
        ax.annotate(label, xy=(idx, gap_months[idx]),
                    xytext=(idx, gap_months[idx] + 2),
                    fontsize=9, weight="bold", ha="center", color="darkblue",
                    arrowprops=dict(arrowstyle="->", color="darkblue", lw=1.2))

    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Gap vs Closed-Source SOTA (months)", fontsize=11)
    ax.set_title("Open-Source is Catching Up: Performance Gap Over Time",
                 fontsize=13, weight="bold")
    ax.set_ylim(0, 22)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="green", ls="--", alpha=0.5, lw=1.5)
    ax.text(6.5, 0.8, "Parity", color="green", fontsize=10, weight="bold")
    plt.tight_layout()
    _save(fig, "open_source_timeline.png", save)


# -- V2: Four-family radar chart ---------------------------------
def plot_radar(save: bool = True):
    categories = ["Coding", "Math", "Reasoning", "Multilingual", "Knowledge"]
    families = {
        "LLaMA 3.1 70B":    [78, 72, 80, 65, 82],
        "Mistral Large":     [75, 68, 76, 72, 78],
        "Qwen 2.5 72B":     [80, 82, 78, 85, 76],
        "DeepSeek-V2 67B":  [82, 85, 82, 60, 74],
    }
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))
    for (name, vals), col in zip(families.items(), colors):
        vals_closed = vals + vals[:1]
        ax.plot(angles, vals_closed, "o-", lw=2, label=name, color=col)
        ax.fill(angles, vals_closed, alpha=0.08, color=col)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(50, 95)
    ax.set_title("Open-Source LLM Families: Benchmark Comparison",
                 fontsize=13, weight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    _save(fig, "family_radar.png", save)


# -- V3: 70B model memory by precision ---------------------------
def plot_memory_bar(save: bool = True):
    precisions = ["FP32", "FP16/BF16", "INT8", "Q4_K_M", "Q2_K"]
    memory_gb = [280, 140, 70, 40, 25]
    colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(precisions, memory_gb, color=colors, alpha=0.85,
                   edgecolor="white", width=0.55)
    for bar, mem in zip(bars, memory_gb):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f"{mem} GB", ha="center", fontsize=11, weight="bold")

    # Reference lines for common hardware
    hw_lines = [(96, "MacBook Pro M3 Max (96GB)", "orange"),
                (24, "RTX 4090 (24GB)", "red"),
                (192, "2x A100 (192GB)", "gray")]
    for mem, label, col in hw_lines:
        ax.axhline(mem, color=col, ls="--", alpha=0.6, lw=1.5)
        ax.text(len(precisions) - 0.5, mem + 2, label, fontsize=8,
                ha="right", color=col, style="italic")

    ax.set_ylabel("Memory Required (GB)", fontsize=12)
    ax.set_title("LLaMA 70B: Memory Footprint by Precision",
                 fontsize=13, weight="bold")
    ax.set_ylim(0, 310)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "memory_bar.png", save)


# -- V4: Quantization intuition (3-panel) -------------------------
def plot_quantization_intuition(save: bool = True):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: Weight distribution → bucketing
    ax = axes[0]
    weights = np.concatenate([np.random.normal(0, 0.5, 900),
                               np.random.normal(0, 0.5, 90),
                               np.array([3.2, -2.8, 4.1, -3.5, 2.9,
                                         -4.0, 3.7, -3.1, 3.3, -2.9])])
    ax.hist(weights, bins=60, color="steelblue", alpha=0.7, edgecolor="white", density=True)
    # Mark outliers
    outlier_mask = np.abs(weights) > 2.5
    ax.scatter(weights[outlier_mask], np.zeros(outlier_mask.sum()) + 0.02,
               color="red", s=30, zorder=3, label=f"Outliers ({outlier_mask.sum()})")
    ax.set_title("FP16 Weight Distribution", fontsize=11, weight="bold")
    ax.set_xlabel("Weight value")
    ax.legend(fontsize=8)

    # Panel 2: INT8 uniform quantization (staircase)
    ax = axes[1]
    x = np.linspace(-4, 4, 500)
    levels = 256
    scale = 8.0 / levels
    quantized = np.round(x / scale) * scale
    ax.plot(x, x, "b--", alpha=0.3, label="FP16 (original)")
    ax.plot(x, quantized, "r-", lw=1.5, label="INT8 (quantized)")
    ax.fill_between(x, x, quantized, alpha=0.1, color="red")
    ax.set_title("Linear Quantization: Staircase", fontsize=11, weight="bold")
    ax.set_xlabel("Original value")
    ax.set_ylabel("Quantized value")
    ax.legend(fontsize=8)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # Panel 3: Group quantization (per-group scale)
    ax = axes[2]
    group_size = 32
    n_groups = 4
    for g in range(n_groups):
        gw = np.random.normal(g * 0.3 - 0.5, 0.3 + g * 0.1, group_size)
        x_pos = np.arange(g * group_size, (g + 1) * group_size)
        ax.bar(x_pos, gw, width=0.8, alpha=0.6,
               color=plt.cm.Set2(g / n_groups), label=f"Group {g+1}")
        # Group scale/zero markers
        ax.axhline(gw.max(), xmin=g / n_groups, xmax=(g + 1) / n_groups,
                    color="red", ls=":", alpha=0.5)
        ax.axhline(gw.min(), xmin=g / n_groups, xmax=(g + 1) / n_groups,
                    color="blue", ls=":", alpha=0.5)
    ax.set_title("Group Quantization (per-32 weights)", fontsize=11, weight="bold")
    ax.set_xlabel("Weight index")
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Quantization Intuition: From FP16 to INT4",
                 fontsize=13, weight="bold", y=1.03)
    plt.tight_layout()
    _save(fig, "quantization_intuition.png", save)


# -- V5: Precision-quality-memory trade-off -----------------------
def plot_tradeoff(save: bool = True):
    bits = [2, 3, 4, 5, 8, 16]
    quality_pct = [72, 88, 96, 98, 99.5, 100]  # % of FP16 quality retained
    memory_gb = [18, 26, 40, 50, 70, 140]       # for 70B model

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    color1, color2 = "steelblue", "seagreen"

    ax1.plot(bits, quality_pct, "o-", color=color1, lw=2.5, ms=10, label="Quality retained (%)")
    ax1.set_xlabel("Quantization Bits", fontsize=12)
    ax1.set_ylabel("Quality Retained (%)", fontsize=12, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(60, 105)

    ax2 = ax1.twinx()
    ax2.plot(bits, memory_gb, "s--", color=color2, lw=2.5, ms=10, label="Memory (GB)")
    ax2.set_ylabel("Memory (GB) for 70B model", fontsize=12, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Annotate sweet spot
    ax1.annotate("Sweet spot\n(Q4_K_M)", xy=(4, 96), xytext=(5.5, 80),
                 fontsize=11, color="orange", weight="bold",
                 arrowprops=dict(arrowstyle="->", color="orange", lw=2))
    # Precision cliff
    ax1.axvspan(1.5, 3.5, alpha=0.08, color="red")
    ax1.text(2.5, 65, "Precision\ncliff", fontsize=10, ha="center",
             color="red", weight="bold", style="italic")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="center left")

    ax1.set_title("Quantization Trade-off: Quality vs Memory (70B Model)",
                  fontsize=13, weight="bold")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "quant_tradeoff.png", save)


# -- V6: llama.cpp GitHub star growth ----------------------------
def plot_stars(save: bool = True):
    months = ["2023.03", "2023.06", "2023.09", "2023.12",
              "2024.03", "2024.06", "2024.09", "2024.12"]
    stars_k = [2, 12, 28, 42, 52, 58, 63, 71]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(months)), stars_k, "o-", color="darkorange", lw=2.5, ms=8)
    ax.fill_between(range(len(months)), stars_k, alpha=0.15, color="orange")

    events = {0: "Initial release\n(ggml)", 1: "GGUF format",
              3: "Q4_K_M\nintroduced", 6: "Vulkan\nbackend"}
    for idx, label in events.items():
        ax.annotate(label, xy=(idx, stars_k[idx]),
                    xytext=(idx + 0.3, stars_k[idx] + 6),
                    fontsize=8, ha="center",
                    arrowprops=dict(arrowstyle="->", lw=1))

    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("GitHub Stars (K)", fontsize=11)
    ax.set_title("llama.cpp: The Engine of Local AI",
                 fontsize=13, weight="bold")
    ax.set_ylim(0, 85)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "llamacpp_stars.png", save)


def _save(fig, name, save):
    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
        print(f"  saved -> {FIG_DIR / name}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="L16 Open Source & Quantization visualizations")
    p.add_argument("--fig", choices=["timeline", "radar", "memory", "quant",
                                      "tradeoff", "stars", "all"],
                   default="all")
    args = p.parse_args()
    dispatch = dict(timeline=plot_timeline, radar=plot_radar, memory=plot_memory_bar,
                    quant=plot_quantization_intuition, tradeoff=plot_tradeoff,
                    stars=plot_stars)
    if args.fig == "all":
        print("L16 Open Source & Quantization -- generating all figures ...")
        for name, fn in dispatch.items():
            print(f"[{name}]")
            fn()
        print("Done.")
    else:
        dispatch[args.fig]()


if __name__ == "__main__":
    main()
