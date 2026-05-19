"""
第 14 讲 Demo · Scaling Law 可视化

七组图：
  1. Kaplan 幂律曲线（Loss vs Compute / Params / Data）
  2. GPT 系列参数量时间线
  3. Chinchilla 最优前沿（IsoFLOP curves）
  4. 涌现能力阶跃图
  5. 涌现 vs 度量幻觉（Schaeffer 对比）
  6. 三面墙趋势图（数据墙 / 能源墙 / 算力墙）
  7. 训练成本阶梯图

依赖: numpy matplotlib
无需 GPU，无需下载模型，纯可视化脚本。
"""

import argparse
import os
import numpy as np
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[4] / "design" / "meta"))
import cjk_font  # noqa: F401  - applies CJK font config
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── style ────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#0d1117",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "font.size": 11,
})

COLORS = {
    "blue": "#58a6ff",
    "green": "#3fb950",
    "orange": "#d29922",
    "red": "#f85149",
    "purple": "#bc8cff",
    "cyan": "#39d2c0",
    "pink": "#f778ba",
}

OUT_DIR = "figures"


def ensure_output_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


# ── Plot 1: Kaplan Power Laws ───────────────────────────────────────

def plot_kaplan_power_laws():
    """Three log-log plots: Loss vs Compute, Params, Data."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Loss vs Compute
    compute = np.logspace(17, 24, 200)
    alpha_c = 0.050
    L_c = 8.0 * (compute / 1e17) ** (-alpha_c)
    axes[0].loglog(compute, L_c, color=COLORS["blue"], linewidth=2.5)
    axes[0].set_xlabel("Compute (FLOPs)")
    axes[0].set_ylabel("Test Loss")
    axes[0].set_title(r"$L(C) \propto C^{-0.050}$", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    # annotate GPT-3 approximate position
    axes[0].scatter([3.64e23], [8.0 * (3.64e23 / 1e17) ** (-alpha_c)],
                    color=COLORS["orange"], s=80, zorder=5)
    axes[0].annotate("GPT-3", xy=(3.64e23, 8.0 * (3.64e23 / 1e17) ** (-alpha_c)),
                     xytext=(5e22, 3.2), fontsize=9, color=COLORS["orange"],
                     arrowprops=dict(arrowstyle="->", color=COLORS["orange"]))

    # Loss vs Parameters
    params = np.logspace(6, 12, 200)
    alpha_n = 0.076
    L_n = 6.0 * (params / 1e6) ** (-alpha_n)
    axes[1].loglog(params, L_n, color=COLORS["green"], linewidth=2.5)
    axes[1].set_xlabel("Parameters (N)")
    axes[1].set_title(r"$L(N) \propto N^{-0.076}$", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    # annotate model sizes
    for name, n in [("GPT-1", 1.17e8), ("GPT-2", 1.5e9), ("GPT-3", 1.75e11)]:
        loss_val = 6.0 * (n / 1e6) ** (-alpha_n)
        axes[1].scatter([n], [loss_val], color=COLORS["orange"], s=60, zorder=5)
        axes[1].annotate(name, xy=(n, loss_val), fontsize=8, color=COLORS["orange"],
                         xytext=(n * 0.15, loss_val * 1.15))

    # Loss vs Data
    data = np.logspace(8, 13, 200)
    alpha_d = 0.095
    L_d = 7.0 * (data / 1e8) ** (-alpha_d)
    axes[2].loglog(data, L_d, color=COLORS["purple"], linewidth=2.5)
    axes[2].set_xlabel("Dataset Size (tokens)")
    axes[2].set_title(r"$L(D) \propto D^{-0.095}$", fontsize=13, fontweight="bold")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Kaplan et al. (2020): Scaling Laws for Neural Language Models",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "kaplan_power_laws.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[Plot 1] Kaplan power laws → {out}")
    plt.close()


# ── Plot 2: GPT Series Timeline ─────────────────────────────────────

def plot_gpt_timeline():
    """GPT-1 → GPT-4 parameter count staircase."""
    fig, ax = plt.subplots(figsize=(12, 6))

    models = ["GPT-1\n(2018.06)", "GPT-2\n(2019.02)", "GPT-3\n(2020.05)", "GPT-4\n(2023.03)"]
    params = [1.17e8, 1.5e9, 1.75e11, 1.8e12]  # GPT-4 MoE estimate
    colors = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["red"]]
    capabilities = [
        "Fine-tune to be useful",
        "Zero-shot emerges",
        "Few-shot learning\n(emergent!)",
        "Expert-level reasoning\n(bar exam, SAT, USMLE)",
    ]

    bars = ax.bar(models, params, color=colors, width=0.6, edgecolor="#30363d", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("Parameters (log scale)")
    ax.set_title("GPT Series: Parameter Count & Emergent Capabilities",
                 fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, cap, p in zip(bars, capabilities, params):
        ax.text(bar.get_x() + bar.get_width() / 2, p * 2.5, cap,
                ha="center", va="bottom", fontsize=9, color="#c9d1d9",
                style="italic")
        ax.text(bar.get_x() + bar.get_width() / 2, p * 0.4,
                f"{p:.0e}".replace("+0", "").replace("+", ""),
                ha="center", va="top", fontsize=10, fontweight="bold",
                color="white")

    ax.set_ylim(1e7, 1e14)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "gpt_timeline.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[Plot 2] GPT timeline → {out}")
    plt.close()


# ── Plot 3: Chinchilla Optimal Frontier ──────────────────────────────

def plot_chinchilla_frontier():
    """IsoFLOP curves with optimal frontier line."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # IsoFLOP curves: C ≈ 6 * N * D  →  D = C / (6N)
    param_range = np.logspace(9, 12, 300)

    flop_budgets = [1e21, 1e22, 1e23, 3.64e23, 1e24]
    flop_labels = ["$10^{21}$", "$10^{22}$", "$10^{23}$",
                   "$3.64×10^{23}$\n(GPT-3)", "$10^{24}$"]
    iso_colors = ["#30363d", "#484f58", "#8b949e", COLORS["orange"], COLORS["blue"]]

    for C, label, color in zip(flop_budgets, flop_labels, iso_colors):
        D = C / (6 * param_range)
        valid = D > 1e9  # filter unreasonable
        ax.loglog(param_range[valid], D[valid], color=color, linewidth=1.5,
                  alpha=0.7, linestyle="--", label=f"C = {label}")

    # Chinchilla optimal frontier: D_opt ≈ 20 * N
    N_opt = np.logspace(9, 12, 100)
    D_opt = 20 * N_opt
    ax.loglog(N_opt, D_opt, color=COLORS["green"], linewidth=3,
              label="Chinchilla Optimal (D ≈ 20N)", zorder=5)

    # Mark specific models
    models = {
        "GPT-3\n(over-param.)": (1.75e11, 3e11, COLORS["red"]),
        "Chinchilla": (7e10, 1.4e12, COLORS["green"]),
        "LLaMA-1 65B": (6.5e10, 1.4e12, COLORS["cyan"]),
    }
    for name, (n, d, c) in models.items():
        ax.scatter([n], [d], color=c, s=120, zorder=6, edgecolors="white", linewidth=1.5)
        offset = (1.3, 1.3) if "GPT" in name else (0.3, 1.5)
        ax.annotate(name, xy=(n, d),
                    xytext=(n * offset[0], d * offset[1]),
                    fontsize=10, color=c, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=c))

    ax.set_xlabel("Parameters (N)", fontsize=12)
    ax.set_ylabel("Training Tokens (D)", fontsize=12)
    ax.set_title("Chinchilla (Hoffmann 2022): Optimal Parameter–Data Frontier",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1e9, 2e12)
    ax.set_ylim(1e10, 1e14)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "chinchilla_frontier.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[Plot 3] Chinchilla frontier → {out}")
    plt.close()


# ── Plot 4: Emergent Abilities ───────────────────────────────────────

def _sigmoid(x, x0, k):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def plot_emergence():
    """Emergent abilities: accuracy vs model scale for multiple tasks."""
    fig, ax = plt.subplots(figsize=(11, 6))

    log_params = np.linspace(7, 12, 300)  # log10(params)

    tasks = [
        ("3-digit addition",        10.5, 3.0, COLORS["blue"]),
        ("IPA transliteration",     10.0, 2.5, COLORS["green"]),
        ("Multi-step reasoning",    11.0, 2.0, COLORS["orange"]),
        ("Code generation",         10.8, 2.2, COLORS["purple"]),
        ("Word unscrambling",        9.5, 4.0, COLORS["cyan"]),
    ]

    for name, x0, k, color in tasks:
        acc = _sigmoid(log_params, x0, k) * 100
        # add small noise for realism
        rng = np.random.RandomState(hash(name) % 2**31)
        noise = rng.normal(0, 1.5, len(acc))
        acc_noisy = np.clip(acc + noise, 0, 100)
        ax.plot(log_params, acc_noisy, color=color, linewidth=2, label=name)

    # threshold line
    ax.axhline(y=25, color=COLORS["red"], linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(7.2, 27, "random baseline", fontsize=9, color=COLORS["red"], alpha=0.7)

    # emergence zone
    ax.axvspan(9.8, 11.2, alpha=0.08, color=COLORS["orange"])
    ax.text(10.1, 90, "Emergence\nZone", fontsize=11, fontweight="bold",
            color=COLORS["orange"], alpha=0.6)

    xtick_vals = [7, 8, 9, 10, 11, 12]
    xtick_labels = ["$10^7$", "$10^8$", "$10^9$", "$10^{10}$", "$10^{11}$", "$10^{12}$"]
    ax.set_xticks(xtick_vals)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel("Model Parameters (log scale)", fontsize=12)
    ax.set_ylabel("Task Accuracy (%)", fontsize=12)
    ax.set_title("Emergent Abilities: Sudden Capability Jumps at Scale",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "emergence.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[Plot 4] Emergence → {out}")
    plt.close()


# ── Plot 5: Emergence vs Metric Mirage (Schaeffer) ──────────────────

def plot_emergence_mirage():
    """Same data, two metrics: exact-match (step) vs log-likelihood (smooth)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    log_params = np.linspace(8, 12, 200)

    # Underlying smooth capability
    smooth_capability = (log_params - 8) / (12 - 8)  # linear 0→1

    # Left: exact match accuracy (nonlinear threshold)
    threshold = 0.6
    exact_match = np.where(smooth_capability > threshold,
                           (smooth_capability - threshold) / (1 - threshold),
                           0.0)
    exact_match = np.clip(exact_match ** 2 * 100, 0, 100)
    rng = np.random.RandomState(42)
    axes[0].plot(log_params, exact_match + rng.normal(0, 1.5, len(exact_match)),
                 color=COLORS["red"], linewidth=2.5)
    axes[0].set_title("Exact-Match Accuracy\n→ Looks like emergence!",
                      fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].axhline(y=0, color="#30363d", linewidth=0.8)

    # Right: token-level log-likelihood (smooth)
    log_likelihood = smooth_capability * 4 - 3  # linear improvement
    axes[1].plot(log_params, log_likelihood + rng.normal(0, 0.05, len(log_likelihood)),
                 color=COLORS["green"], linewidth=2.5)
    axes[1].set_title("Token Log-Likelihood\n→ Smooth improvement!",
                      fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Log-Likelihood")

    for ax in axes:
        xtick_vals = [8, 9, 10, 11, 12]
        xtick_labels = ["$10^8$", "$10^9$", "$10^{10}$", "$10^{11}$", "$10^{12}$"]
        ax.set_xticks(xtick_vals)
        ax.set_xticklabels(xtick_labels)
        ax.set_xlabel("Model Parameters (log scale)")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Schaeffer et al. (2023): Same Data, Different Metric → Different Story",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "emergence_mirage.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[Plot 5] Emergence mirage → {out}")
    plt.close()


# ── Plot 6: Three Walls ─────────────────────────────────────────────

def plot_three_walls():
    """Data wall, energy wall, compute wall trend charts."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    years = np.arange(2018, 2031)

    # Data wall
    # Training data demand (tokens, log scale) vs available high-quality text
    demand = 3e11 * (4.0 ** (years - 2020))  # ~4x per year
    supply_ceiling = np.full_like(years, 3e13, dtype=float)  # ~30T tokens
    axes[0].semilogy(years, demand, color=COLORS["blue"], linewidth=2.5,
                     marker="o", markersize=5, label="Training data demand")
    axes[0].semilogy(years, supply_ceiling, color=COLORS["red"], linewidth=2,
                     linestyle="--", label="Available high-quality text")
    axes[0].fill_between(years, supply_ceiling, demand,
                         where=(demand > supply_ceiling),
                         alpha=0.15, color=COLORS["red"])
    axes[0].set_title("Data Wall", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Tokens")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(1e10, 1e16)

    # Energy wall
    models_years = [2019, 2020, 2022, 2023, 2025]
    energy_mwh = [0.3, 1300, 3000, 50000, 150000]  # rough estimates
    model_names_e = ["GPT-2", "GPT-3", "PaLM", "GPT-4", "Next-gen?"]
    axes[1].semilogy(models_years, energy_mwh, color=COLORS["orange"], linewidth=2.5,
                     marker="s", markersize=8)
    for yr, e, name in zip(models_years, energy_mwh, model_names_e):
        axes[1].annotate(name, xy=(yr, e), xytext=(yr + 0.2, e * 1.8),
                         fontsize=8, color=COLORS["orange"])
    axes[1].set_title("Energy Wall", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Training Energy (MWh)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(2018, 2027)

    # Compute wall
    gpu_perf_years = np.arange(2018, 2028)
    # GPU peak TFLOPS (FP16): V100=125, A100=312, H100=990, B100~1800 ...
    gpu_perf = 125 * (1.35 ** (gpu_perf_years - 2018))  # ~35% annual increase
    demand_tflops = 125 * (3.5 ** (gpu_perf_years - 2018))  # ~3.5x annual demand growth
    axes[2].semilogy(gpu_perf_years, gpu_perf, color=COLORS["green"], linewidth=2.5,
                     marker="o", markersize=5, label="GPU peak TFLOPS (FP16)")
    axes[2].semilogy(gpu_perf_years, demand_tflops, color=COLORS["red"], linewidth=2.5,
                     marker="^", markersize=5, label="Training compute demand")
    axes[2].fill_between(gpu_perf_years, gpu_perf, demand_tflops,
                         where=(demand_tflops > gpu_perf),
                         alpha=0.15, color=COLORS["red"])
    axes[2].set_title("Compute Wall", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("TFLOPS / Demand index")
    axes[2].legend(fontsize=8, loc="upper left")
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel("Year")

    fig.suptitle("Three Walls: Physical Limits of Scaling",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "three_walls.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[Plot 6] Three walls → {out}")
    plt.close()


# ── Plot 7: Training Cost Staircase ─────────────────────────────────

def plot_training_cost():
    """Training cost per model (estimated)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ["GPT-2\n(2019)", "GPT-3\n(2020)", "Chinchilla\n(2022)",
              "GPT-4\n(2023)", "Gemini Ultra\n(2024)"]
    costs = [5e4, 4.6e6, 1e7, 1e8, 2e8]  # USD
    colors = [COLORS["blue"], COLORS["green"], COLORS["cyan"],
              COLORS["orange"], COLORS["red"]]

    bars = ax.bar(models, costs, color=colors, width=0.6,
                  edgecolor="#30363d", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("Estimated Training Cost (USD, log scale)", fontsize=11)
    ax.set_title("Training Cost Per Model: ~10× Every 2 Years",
                 fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, cost in zip(bars, costs):
        if cost >= 1e6:
            label = f"${cost / 1e6:.1f}M"
        else:
            label = f"${cost / 1e3:.0f}K"
        ax.text(bar.get_x() + bar.get_width() / 2, cost * 1.8, label,
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color="white")

    ax.set_ylim(1e3, 1e10)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "training_cost.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[Plot 7] Training cost → {out}")
    plt.close()


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="L14 Demo: Scaling Law Visualizations")
    parser.add_argument("--only", type=int, default=0,
                        help="Only generate plot N (1-7). 0 = all.")
    args = parser.parse_args()

    ensure_output_dir()

    plots = [
        (1, "Kaplan Power Laws", plot_kaplan_power_laws),
        (2, "GPT Timeline", plot_gpt_timeline),
        (3, "Chinchilla Frontier", plot_chinchilla_frontier),
        (4, "Emergence", plot_emergence),
        (5, "Emergence Mirage", plot_emergence_mirage),
        (6, "Three Walls", plot_three_walls),
        (7, "Training Cost", plot_training_cost),
    ]

    for idx, name, fn in plots:
        if args.only == 0 or args.only == idx:
            print(f"\n{'=' * 50}")
            print(f"Generating Plot {idx}: {name}")
            print(f"{'=' * 50}")
            fn()

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()