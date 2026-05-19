"""50行代码训练一个最小语言模型 — 序章开场实验用"""
import numpy as np
np.random.seed(1337)  # 固定随机种子，保证可复现
# --- 数据准备 ---
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
try:
    from urllib.request import urlopen; text = urlopen(url).read().decode()
except Exception:
    text = open("input.txt").read()
chars = sorted(set(text)); V = len(chars)
s2i = {c:i for i,c in enumerate(chars)}; i2s = {i:c for c,i in s2i.items()}
data = np.array([s2i[c] for c in text])
# --- 超参数 ---
B, T, D, H = 64, 32, 64, 4          # batch, context, embed_dim, heads
lr, steps = 1e-3, 3000
# --- 参数初始化 ---
def randn(*s): return np.random.randn(*s).astype(np.float32) * 0.02
emb = randn(V, D); pos = randn(T, D)                        # 词嵌入 + 位置编码
Wq, Wk, Wv, Wo = [randn(D, D) for _ in range(4)]            # 注意力
W1, b1, W2, b2 = randn(D, 4*D), np.zeros(4*D, dtype=np.float32), randn(4*D, D), np.zeros(D, dtype=np.float32)  # FFN
Wout = randn(D, V)                                           # 输出头
params = [emb, pos, Wq, Wk, Wv, Wo, W1, b1, W2, b2, Wout]
mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), 1)   # 因果掩码
# --- 工具函数 ---
def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True)); return e / e.sum(axis=axis, keepdims=True)
def gelu(x): return 0.5 * x * (1 + np.tanh(0.7978846 * (x + 0.044715 * x**3)))
def layer_norm(x, eps=1e-5):
    mu = x.mean(-1, keepdims=True); sigma = x.std(-1, keepdims=True); return (x - mu) / (sigma + eps)
# --- 前向传播 ---
def forward(idx):                                             # idx: (B, T)
    x = emb[idx] + pos[:idx.shape[1]]                        # (B, T, D)
    # Self-Attention
    q, k, v = [np.einsum('btd,dh->bth', layer_norm(x), W).reshape(x.shape[0], -1, H, D//H).transpose(0,2,1,3)
                for W in (Wq, Wk, Wv)]                       # (B, H, T, D/H)
    att = softmax(q @ k.transpose(0,1,3,2) / (D//H)**0.5 + mask[:idx.shape[1],:idx.shape[1]])
    x = x + (att @ v).transpose(0,2,1,3).reshape(x.shape) @ Wo
    # FFN
    x = x + gelu(layer_norm(x) @ W1 + b1) @ W2 + b2
    logits = layer_norm(x) @ Wout                             # (B, T, V)
    return logits
# --- 训练循环 ---
for step in range(steps):
    ix = np.random.randint(0, len(data)-T-1, B)
    xb = np.stack([data[i:i+T] for i in ix])
    yb = np.stack([data[i+1:i+T+1] for i in ix])
    logits = forward(xb)
    # 交叉熵损失
    probs = softmax(logits.reshape(-1, V))
    loss = -np.log(probs[np.arange(len(yb.ravel())), yb.ravel()] + 1e-9).mean()
    if step % 200 == 0: print(f"step {step:4d} | loss {loss:.3f}")
    # 数值梯度（为简洁用有限差分，实际课程可换成 PyTorch autograd）
    for p in params:
        grad = np.zeros_like(p); flat = p.ravel(); g = grad.ravel()
        for j in np.random.choice(len(flat), min(50, len(flat)), replace=False):
            old = flat[j]; flat[j] = old + 1e-4
            l2 = -np.log(softmax(forward(xb).reshape(-1,V))[np.arange(len(yb.ravel())),yb.ravel()]+1e-9).mean()
            g[j] = (l2 - loss) / 1e-4; flat[j] = old
        p -= lr * grad
# --- 生成 ---
def generate(start="ROMEO:", length=300):
    idx = np.array([[s2i[c] for c in start]])
    for _ in range(length):
        logits = forward(idx[:, -T:])
        p = softmax(logits[0, -1]); nxt = np.random.choice(V, p=p)
        idx = np.concatenate([idx, [[nxt]]], axis=1)
    return ''.join(i2s[i] for i in idx[0])
print("\n--- 生成结果 ---\n"); print(generate())
