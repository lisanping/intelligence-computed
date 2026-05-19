"""
L05 – Structured Data: Mini Recommender System (User-CF & SVD)
==============================================================
Demonstrates collaborative filtering on a tiny built-in rating matrix.
User-CF predicts via k-nearest neighbours; SVD via truncated matrix
factorisation.  Compares RMSE and visualises the matrix before/after.

Usage
-----
    python recommender_mini.py
    python recommender_mini.py --method svd
    python recommender_mini.py --method both --k 2
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

SEED = 1337
np.random.seed(SEED)

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# 6 users × 5 items  (0 = missing)
RATINGS = np.array([
    [5, 3, 0, 1, 0],
    [4, 0, 0, 1, 0],
    [1, 1, 0, 5, 0],
    [0, 0, 5, 4, 0],
    [0, 1, 4, 0, 5],
    [1, 0, 3, 0, 4],
], dtype=float)

USERS = [f"U{i}" for i in range(RATINGS.shape[0])]
ITEMS = [f"I{j}" for j in range(RATINGS.shape[1])]


# ---------- helpers -------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else 0.0


def rmse(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    diff = (pred - true)[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


# ---------- User-CF -------------------------------------------------------

def user_cf(R: np.ndarray, k: int = 3) -> np.ndarray:
    n_users, n_items = R.shape
    pred = np.copy(R)
    mean = np.where(R > 0, R, np.nan)
    user_mean = np.nanmean(mean, axis=1)

    for u in range(n_users):
        sims = np.array([cosine_sim(R[u], R[v]) for v in range(n_users)])
        neighbours = np.argsort(sims)[::-1][1:k + 1]
        for i in range(n_items):
            if R[u, i] == 0:
                num = sum(sims[v] * (R[v, i] - user_mean[v])
                          for v in neighbours if R[v, i] > 0)
                den = sum(abs(sims[v]) for v in neighbours if R[v, i] > 0)
                pred[u, i] = user_mean[u] + (num / den if den else 0)
    return np.clip(pred, 1, 5)


# ---------- SVD -----------------------------------------------------------

def svd_complete(R: np.ndarray, rank: int = 2) -> np.ndarray:
    filled = np.where(R > 0, R, np.nanmean(R[R > 0]))
    U, s, Vt = np.linalg.svd(filled, full_matrices=False)
    S = np.diag(s[:rank])
    pred = U[:, :rank] @ S @ Vt[:rank, :]
    return np.clip(pred, 1, 5)


# ---------- visualisation -------------------------------------------------

def plot_matrices(original, completed, title, fname):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, data, lab in zip(axes, [original, completed],
                             ["Original (0 = missing)", f"Completed ({title})"]):
        im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=5, aspect="auto")
        ax.set_xticks(range(len(ITEMS)), ITEMS)
        ax.set_yticks(range(len(USERS)), USERS)
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                ax.text(c, r, f"{data[r, c]:.1f}", ha="center", va="center", fontsize=8)
        ax.set_title(lab)
    fig.colorbar(im, ax=axes, shrink=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    print(f"[saved] {FIG_DIR}/{fname}")


# ---------- main ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mini recommender demo")
    parser.add_argument("--method", choices=["user_cf", "svd", "both"],
                        default="both")
    parser.add_argument("--ablate", choices=["no_cf", "no_svd"],
                        default=None, help="Ablation: disable one method")
    parser.add_argument("--k", type=int, default=3, help="Neighbours for CF")
    args = parser.parse_args()

    # --ablate overrides --method
    if args.ablate == "no_cf":
        args.method = "svd"
    elif args.ablate == "no_svd":
        args.method = "user_cf"

    known = RATINGS > 0

    if args.method in ("user_cf", "both"):
        pred_cf = user_cf(RATINGS, k=args.k)
        err = rmse(pred_cf, RATINGS, known)
        print(f"User-CF  RMSE (on known): {err:.4f}")
        plot_matrices(RATINGS, pred_cf, "User-CF", "rec_user_cf.png")

    if args.method in ("svd", "both"):
        pred_svd = svd_complete(RATINGS)
        err = rmse(pred_svd, RATINGS, known)
        print(f"SVD      RMSE (on known): {err:.4f}")
        plot_matrices(RATINGS, pred_svd, "SVD", "rec_svd.png")

    plt.show()


if __name__ == "__main__":
    main()
