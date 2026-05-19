# 第 12 讲 · 消融实验脚本

> 讲稿 [26:00] 对应的四个 "如果去掉它会怎样" 实验。
> **建议课前全部跑完并保存 loss 曲线图**，课上直接播图；现场跑风险太高（时间不可控）。

## 实验矩阵

| 实验 | 命令 | 预期 val loss（3000 步） | 现象 |
|---|---|---|---|
| 基线 | `python nanogpt_from_scratch.py` | ≈ 1.5 | 生成像样的莎士比亚 |
| 去掉位置编码 | `python nanogpt_from_scratch.py --ablate no_pe` | ≈ 2.5（几乎卡住） | 生成几乎全是高频字符堆 |
| 去掉 √d_k 缩放 | `python nanogpt_from_scratch.py --ablate no_scale` | 早期不稳，后期勉强追上 | loss 曲线有明显震荡 |
| 去掉残差连接 | `python nanogpt_from_scratch.py --ablate no_resid` | ≈ 2.8+ | 深层梯度消失，完全学不动 |
| 单头 vs 多头 | `python nanogpt_from_scratch.py --n_head 1` | ≈ 1.6 | 略差于 6 头，差距随深度放大 |

## 一键全跑（Linux / macOS）

```bash
cd lectures/L12-transformer/demo
for ab in baseline no_pe no_scale no_resid; do
    if [ "$ab" = "baseline" ]; then
        python nanogpt_from_scratch.py | tee "log_${ab}.txt"
    else
        python nanogpt_from_scratch.py --ablate "$ab" | tee "log_${ab}.txt"
    fi
done
python nanogpt_from_scratch.py --n_head 1 | tee log_1head.txt
```

## Windows PowerShell

```powershell
cd lectures\L12-transformer\demo
python nanogpt_from_scratch.py                  | Tee-Object log_baseline.txt
python nanogpt_from_scratch.py --ablate no_pe   | Tee-Object log_no_pe.txt
python nanogpt_from_scratch.py --ablate no_scale| Tee-Object log_no_scale.txt
python nanogpt_from_scratch.py --ablate no_resid| Tee-Object log_no_resid.txt
python nanogpt_from_scratch.py --n_head 1       | Tee-Object log_1head.txt
```

## 汇总脚本（产出 V10 柱状图 / 折线图）

跑完后，用下面这段把 `log_*.txt` 中的 `step ... | val ...` 行解析出来，统一画到一张图：

```python
import re
from pathlib import Path
import matplotlib.pyplot as plt

pat = re.compile(r"step\s+(\d+)\s+\|\s+train\s+([\d.]+)\s+\|\s+val\s+([\d.]+)")
runs = ["baseline", "no_pe", "no_scale", "no_resid", "1head"]

plt.figure(figsize=(8, 5))
for name in runs:
    log = Path(f"log_{name}.txt").read_text()
    steps, vals = [], []
    for m in pat.finditer(log):
        steps.append(int(m.group(1)))
        vals.append(float(m.group(3)))
    plt.plot(steps, vals, label=name)
plt.xlabel("step"); plt.ylabel("val loss"); plt.legend(); plt.grid(alpha=0.3)
plt.title("Ablation: Transformer 组件的作用")
plt.tight_layout(); plt.savefig("ablation_summary.png", dpi=150)
```