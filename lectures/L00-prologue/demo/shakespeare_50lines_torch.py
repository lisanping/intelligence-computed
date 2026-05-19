"""50行PyTorch训练一个最小语言模型 — 课程序章开场实验"""
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(1337)  # 固定随机种子，保证可复现
# --- 数据准备 ---
from urllib.request import urlopen
text = urlopen("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").read().decode()
chars = sorted(set(text)); V = len(chars)
s2i = {c:i for i,c in enumerate(chars)}; i2s = {i:c for c,i in s2i.items()}
data = torch.tensor([s2i[c] for c in text], dtype=torch.long)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# --- 模型：单层 Transformer Decoder ---
B, T, D, H = 64, 64, 128, 4  # batch, context_len, embed_dim, heads
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(V, D)           # 词嵌入："给每个字符一个身份证"
        self.pos_emb = nn.Embedding(T, D)            # 位置编码："给每个位置一个座位号"
        self.attn = nn.MultiheadAttention(D, H, dropout=0.1, batch_first=True)  # 注意力
        self.ff = nn.Sequential(nn.Linear(D, 4*D), nn.GELU(), nn.Linear(4*D, D))  # FFN
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=False)      # 输出头：预测下一个字符
        self.register_buffer('mask', torch.triu(torch.ones(T,T,dtype=torch.bool), 1))  # 因果掩码
    def forward(self, idx):
        t = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(t, device=idx.device))
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=self.mask[:t,:t])[0]
        x = x + self.ff(self.ln2(x))
        return self.head(x)
model = TinyGPT().to(device); opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
# --- 训练 ---
for step in range(3000):
    ix = torch.randint(len(data)-T-1, (B,))
    xb = torch.stack([data[i:i+T]   for i in ix]).to(device)
    yb = torch.stack([data[i+1:i+T+1] for i in ix]).to(device)
    loss = F.cross_entropy(model(xb).view(-1, V), yb.view(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 200 == 0: print(f"step {step:4d} | loss {loss.item():.3f}")
# --- 生成 ---
@torch.no_grad()
def generate(start="ROMEO:", length=500):
    idx = torch.tensor([[s2i[c] for c in start]], device=device)
    for _ in range(length):
        logits = model(idx[:, -T:])[:, -1]
        idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
    return ''.join(i2s[i] for i in idx[0].tolist())
print("\n--- 生成结果 ---\n"); print(generate())
