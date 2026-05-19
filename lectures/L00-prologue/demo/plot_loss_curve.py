"""Plot the training loss curve referenced in L00 lecture notes.

Mirrors shakespeare_50lines_torch.py exactly (same seed, hyperparams,
data loader) but logs every step's loss and writes a PNG annotated with
the random-baseline (-ln(1/V)) and the "starts learning structure" floor.

Run:  python plot_loss_curve.py
Out:  demo/figures/loss_curve_prologue.png
"""
import math
from pathlib import Path
from urllib.request import urlopen

import torch  # must precede matplotlib on Windows: matplotlib loads OpenMP DLLs that break torch's shm.dll loader if imported first
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(1337)

text = urlopen(
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
).read().decode()
chars = sorted(set(text))
V = len(chars)
s2i = {c: i for i, c in enumerate(chars)}
data = torch.tensor([s2i[c] for c in text], dtype=torch.long)
device = "cuda" if torch.cuda.is_available() else "cpu"

B, T, D, H = 64, 64, 128, 4


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(V, D)
        self.pos_emb = nn.Embedding(T, D)
        self.attn = nn.MultiheadAttention(D, H, dropout=0.1, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        self.register_buffer(
            "mask", torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        )

    def forward(self, idx):
        t = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(t, device=idx.device))
        x = x + self.attn(
            self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=self.mask[:t, :t]
        )[0]
        x = x + self.ff(self.ln2(x))
        return self.head(x)


model = TinyGPT().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

losses: list[float] = []
for step in range(3000):
    ix = torch.randint(len(data) - T - 1, (B,))
    xb = torch.stack([data[i : i + T] for i in ix]).to(device)
    yb = torch.stack([data[i + 1 : i + T + 1] for i in ix]).to(device)
    loss = F.cross_entropy(model(xb).view(-1, V), yb.view(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if step % 200 == 0:
        print(f"step {step:4d} | loss {loss.item():.3f}")

baseline = math.log(V)  # uniform-guess cross-entropy = -ln(1/V) = ln(V)

out_path = Path(__file__).parent / "figures" / "loss_curve_prologue.png"
out_path.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(losses, color="#2563eb", linewidth=1.0, alpha=0.45, label="per-step loss")
window = 50
if len(losses) >= window:
    smooth = torch.tensor(losses).unfold(0, window, 1).mean(dim=1).tolist()
    ax.plot(range(window - 1, len(losses)), smooth, color="#1e3a8a", linewidth=2.0, label=f"moving avg ({window})")
ax.axhline(baseline, color="#9ca3af", linestyle="--", linewidth=1.0)
ax.text(len(losses) * 0.98, baseline + 0.05, f"random baseline = ln({V}) ≈ {baseline:.2f}",
        ha="right", va="bottom", color="#6b7280", fontsize=9)
ax.set_xlabel("training step")
ax.set_ylabel("cross-entropy loss")
ax.set_title("TinyGPT on Tiny-Shakespeare — 50-line PyTorch demo")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(out_path, dpi=150)
print(f"\nsaved → {out_path}")
