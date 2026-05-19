"""ANN benchmark — Flat / IVF / HNSW recall vs latency.

Generates a small synthetic embedding dataset (no real model needed),
indexes it with three approaches and produces:
  - ann_benchmark.png   recall@10 vs latency curve (per index)

Pure-Python implementations, no faiss dependency. The HNSW here is a
toy multi-layer graph for didactic purposes — for production use faiss/qdrant.

Run:  python ann_benchmark.py
"""
from __future__ import annotations
import os
import time
import heapq
import math
import random
import numpy as np
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


# ──────────────────────────────────────────────────────────────────────
# Tiny synthetic embedding dataset
# ──────────────────────────────────────────────────────────────────────
def make_dataset(n: int = 5_000, dim: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # 8 cluster centers
    centers = rng.normal(0, 1, (8, dim))
    labels = rng.integers(0, 8, n)
    X = centers[labels] + rng.normal(0, 0.3, (n, dim))
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X.astype(np.float32)


def make_queries(X: np.ndarray, n_queries: int = 100,
                 seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), n_queries, replace=False)
    Q = X[idx] + rng.normal(0, 0.05, (n_queries, X.shape[1]))
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
    return Q.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# Index implementations
# ──────────────────────────────────────────────────────────────────────
class FlatIndex:
    name = "Flat (brute force)"

    def __init__(self, X: np.ndarray):
        self.X = X

    def search(self, q: np.ndarray, k: int):
        sims = self.X @ q
        top = np.argpartition(-sims, k)[:k]
        return top[np.argsort(-sims[top])]


class IVFIndex:
    """Cluster into nlist groups; search nprobe nearest clusters."""
    name = "IVF"

    def __init__(self, X: np.ndarray, nlist: int = 64):
        self.X = X
        self.nlist = nlist
        # k-means via random init + 2 iterations (toy)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), nlist, replace=False)
        centers = X[idx].copy()
        for _ in range(3):
            sims = X @ centers.T
            assign = sims.argmax(axis=1)
            for c in range(nlist):
                mask = assign == c
                if mask.any():
                    centers[c] = X[mask].mean(axis=0)
                    centers[c] /= np.linalg.norm(centers[c]) + 1e-9
        self.centers = centers
        self.lists = [np.where(assign == c)[0] for c in range(nlist)]

    def search(self, q: np.ndarray, k: int, nprobe: int = 4):
        sims = self.centers @ q
        probe_clusters = np.argpartition(-sims, nprobe)[:nprobe]
        cands = np.concatenate([self.lists[c] for c in probe_clusters])
        if len(cands) == 0:
            return np.array([], dtype=int)
        sims_c = self.X[cands] @ q
        order = np.argsort(-sims_c)[:k]
        return cands[order]


class ToyHNSW:
    """Pedagogical HNSW: 2 layers, M neighbors, ef_search."""
    name = "HNSW"

    def __init__(self, X: np.ndarray, M: int = 16, seed: int = 0):
        self.X = X
        self.M = M
        rng = random.Random(seed)
        n = len(X)
        # layer 1 = sqrt(n) random subset
        self.upper = rng.sample(range(n), max(1, int(math.sqrt(n))))
        self.entry = self.upper[0]
        # build neighbors: each upper-layer node connects to M nearest in upper
        self.upper_nb = {}
        for u in self.upper:
            sims = X[self.upper] @ X[u]
            self.upper_nb[u] = [self.upper[i]
                                for i in np.argsort(-sims)[1:M + 1]]
        # base layer: each node connects to M nearest globally (toy)
        self.base_nb = {}
        for i in range(n):
            sims = X @ X[i]
            self.base_nb[i] = list(np.argsort(-sims)[1:M + 1])

    def _greedy(self, q: np.ndarray, start: int, layer_nb: dict) -> int:
        cur = start
        cur_sim = float(self.X[cur] @ q)
        improved = True
        while improved:
            improved = False
            for nb in layer_nb[cur]:
                s = float(self.X[nb] @ q)
                if s > cur_sim:
                    cur = nb
                    cur_sim = s
                    improved = True
        return cur

    def search(self, q: np.ndarray, k: int, ef: int = 32):
        # Step 1: greedy in upper layer
        entry = self._greedy(q, self.entry, self.upper_nb)
        # Step 2: best-first in base layer
        visited = {entry}
        cand = [(-(self.X[entry] @ q), entry)]
        result = [(self.X[entry] @ q, entry)]
        while cand:
            neg_s, node = heapq.heappop(cand)
            if -neg_s < min(s for s, _ in result) and len(result) >= ef:
                break
            for nb in self.base_nb[node]:
                if nb in visited:
                    continue
                visited.add(nb)
                s = float(self.X[nb] @ q)
                heapq.heappush(cand, (-s, nb))
                result.append((s, nb))
                if len(result) > ef:
                    result.sort(reverse=True)
                    result = result[:ef]
        result.sort(reverse=True)
        return np.array([n for _, n in result[:k]])


# ──────────────────────────────────────────────────────────────────────
# Benchmark
# ──────────────────────────────────────────────────────────────────────
def benchmark(index, queries: np.ndarray, gt: np.ndarray, k: int,
              **kwargs) -> tuple[float, float]:
    """Returns (mean_latency_ms, mean_recall@k)."""
    recalls = []
    t0 = time.perf_counter()
    for q, g in zip(queries, gt):
        pred = set(int(x) for x in index.search(q, k, **kwargs))
        recalls.append(len(pred & set(int(x) for x in g)) / k)
    t = (time.perf_counter() - t0) / len(queries) * 1000
    return t, float(np.mean(recalls))


def main() -> None:
    print("Building dataset (5000 × 64-D)…")
    X = make_dataset(5000, 64)
    Q = make_queries(X, 200)

    flat = FlatIndex(X)
    print("  computing ground truth via Flat…")
    gt = np.array([flat.search(q, 10) for q in Q])

    points = []  # (name, latency_ms, recall, marker_label)

    # Flat baseline
    t, r = benchmark(flat, Q, gt, 10)
    points.append(("Flat", t, r, "Flat"))
    print(f"Flat:  {t:.2f} ms, recall@10 = {r:.3f}")

    # IVF: vary nprobe
    ivf = IVFIndex(X, nlist=64)
    for nprobe in [1, 2, 4, 8, 16, 32]:
        t, r = benchmark(ivf, Q, gt, 10, nprobe=nprobe)
        points.append((f"IVF nprobe={nprobe}", t, r, "IVF"))
        print(f"IVF nprobe={nprobe:2d}: {t:.2f} ms, recall@10 = {r:.3f}")

    # HNSW: vary ef
    print("Building HNSW (toy implementation, slow build)…")
    hnsw = ToyHNSW(X, M=16)
    for ef in [16, 32, 64, 128, 256]:
        t, r = benchmark(hnsw, Q, gt, 10, ef=ef)
        points.append((f"HNSW ef={ef}", t, r, "HNSW"))
        print(f"HNSW ef={ef:3d}: {t:.2f} ms, recall@10 = {r:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    color_of = {"Flat": "#9E9E9E", "IVF": "#4C72B0", "HNSW": "#D62728"}
    family_pts = {}
    for name, t, r, fam in points:
        family_pts.setdefault(fam, []).append((t, r, name))
    for fam, pts in family_pts.items():
        pts.sort()
        ax.plot([t for t, _, _ in pts], [r for _, r, _ in pts],
                "o-", color=color_of[fam], lw=2, ms=8, label=fam)
        for t, r, name in pts:
            ax.annotate(name, (t, r), xytext=(5, 4),
                        textcoords="offset points", fontsize=7.5,
                        color=color_of[fam])
    ax.set_xscale("log")
    ax.set_xlabel("查询延迟 (ms, log scale)")
    ax.set_ylabel("Recall@10")
    ax.set_title("ANN benchmark: Flat / IVF / HNSW recall vs latency\n"
                 "（5000 vectors × 64-D, 200 queries; toy in-process impl）")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=11)
    fig.tight_layout()
    out = os.path.join(OUT, "ann_benchmark.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
