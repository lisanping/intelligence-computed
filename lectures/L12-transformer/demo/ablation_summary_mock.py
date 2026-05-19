"""
第 12 讲 · V10 消融汇总图（模拟版）
===============================

在真正跑完五组训练之前，先用手工拟合的典型曲线占位。
真实数据来了以后，读 `log_*.txt` 覆盖即可（脚本见 `ablations.md`）。

曲线形状依据：
    - baseline    : 训练稳定下降到 ~1.5
    - 1head       : 略差，停在 ~1.65
    - no_scale    : 早期震荡明显，后期勉强追上 ~1.8
    - no_pe       : 几乎卡住 ~2.4
    - no_resid    : 4 层深度梯度难传，卡在 ~2.7
"""
from __future__ import annotations

from pathlib import Path

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[4] / "design" / "meta"))
import cjk_font  # noqa: F401  - applies CJK font config
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

steps = np.arange(0, 3001, 100)


def curve(final: float, noise: float, plateau_at: int | None = None, shock: bool = False):
    start = 4.2
    base = start - (start - final) * (1 - np.exp(-steps / 900))
    if plateau_at is not None:
        base = np.minimum.accumulate(base * 0 + start)  # reset
        base = start - (start - final) * (1 - np.exp(-steps / plateau_at))
    if shock:
        rng = np.random.default_rng(0)
        base = base + rng.normal(0, 0.08, base.shape)
        spike_mask = (steps > 300) & (steps < 900)
        base[spike_mask] += rng.normal(0, 0.25, spike_mask.sum())
    return base


runs = {
    "baseline":   curve(final=1.50, noise=0.02),
    "1 head":     curve(final=1.65, noise=0.02),
    "no sqrt(d)": curve(final=1.80, noise=0.05, shock=True),
    "no PE":      curve(final=2.40, noise=0.02, plateau_at=1500),
    "no residual": curve(final=2.75, noise=0.02, plateau_at=2000),
}

fig, ax = plt.subplots(figsize=(8, 5))
for name, vals in runs.items():
    ax.plot(steps, vals, label=name, linewidth=2)
ax.set_xlabel("training step")
ax.set_ylabel("validation loss")
ax.set_title("Ablation: what each Transformer component actually does\n(mocked curves — replace with real logs via ablations.md)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
out_path = OUT / "ablation_summary_mock.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"saved {out_path}")