"""L11 Demo: Word Vector Visualization with t-SNE
用 GloVe 预训练词向量做语义类比 + t-SNE 可视化
所有实验使用 seed=1337
"""
import argparse
import os
import zipfile
import numpy as np
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[4] / "design" / "meta"))
import cjk_font  # noqa: F401  - applies CJK font config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SEED = 1337
np.random.seed(SEED)
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_FILE = "glove.6B.50d.txt"


def load_glove(path: str = GLOVE_FILE, dim: int = 50, max_words: int = 50000):
    """Load GloVe vectors. Auto-download if not present."""
    if not os.path.exists(path):
        print(f"[INFO] {path} not found. Generating random mock vectors for demo.")
        print(f"       For real results, download from {GLOVE_URL}")
        # Generate mock vectors with semantic structure for demo purposes
        words = _get_demo_words()
        vectors = {}
        for i, w in enumerate(words):
            vectors[w] = np.random.randn(dim).astype(np.float32) * 0.1
        # Inject structure: make related words closer
        _inject_structure(vectors, dim)
        return vectors

    print(f"[INFO] Loading GloVe from {path}...")
    vectors = {}
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_words:
                break
            parts = line.strip().split()
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            vectors[word] = vec
    print(f"[INFO] Loaded {len(vectors)} word vectors (dim={dim})")
    return vectors


def _get_demo_words():
    """Words for mock demo when GloVe is not available."""
    return [
        # Royalty
        "king", "queen", "prince", "princess", "royal", "throne", "crown",
        # Gender
        "man", "woman", "boy", "girl", "he", "she", "father", "mother",
        # Countries
        "china", "japan", "france", "germany", "italy", "spain", "russia",
        "beijing", "tokyo", "paris", "berlin", "rome", "madrid", "moscow",
        # Animals
        "cat", "dog", "bird", "fish", "horse", "lion", "tiger", "elephant",
        # Numbers
        "one", "two", "three", "four", "five", "six", "seven", "eight",
        # Verbs
        "run", "walk", "jump", "swim", "fly", "eat", "drink", "sleep",
        # Tech
        "computer", "software", "algorithm", "data", "network", "model",
        # Colors
        "red", "blue", "green", "yellow", "black", "white",
    ]


def _inject_structure(vectors, dim):
    """Inject semantic structure into mock vectors for demonstration."""
    # Create semantic directions
    gender_dir = np.random.randn(dim).astype(np.float32) * 0.5
    royalty_dir = np.random.randn(dim).astype(np.float32) * 0.5
    capital_dir = np.random.randn(dim).astype(np.float32) * 0.5

    # Royalty + gender structure
    base = np.random.randn(dim).astype(np.float32) * 0.1
    if "king" in vectors:
        vectors["king"] = base + royalty_dir + gender_dir
    if "queen" in vectors:
        vectors["queen"] = base + royalty_dir - gender_dir
    if "man" in vectors:
        vectors["man"] = base + gender_dir
    if "woman" in vectors:
        vectors["woman"] = base - gender_dir

    # Country-capital structure
    for country, capital in [("china", "beijing"), ("japan", "tokyo"),
                              ("france", "paris"), ("germany", "berlin")]:
        if country in vectors and capital in vectors:
            c_base = np.random.randn(dim).astype(np.float32) * 0.1
            vectors[country] = c_base
            vectors[capital] = c_base + capital_dir


def analogy(vectors, a, b, c, topn=5):
    """Solve: a - b + c = ?  (e.g., king - man + woman = ?)"""
    if a not in vectors or b not in vectors or c not in vectors:
        print(f"[WARN] Missing word(s): {[w for w in [a,b,c] if w not in vectors]}")
        return []
    target = vectors[a] - vectors[b] + vectors[c]
    exclude = {a, b, c}
    scores = []
    for word, vec in vectors.items():
        if word in exclude:
            continue
        cos_sim = np.dot(target, vec) / (np.linalg.norm(target) * np.linalg.norm(vec) + 1e-9)
        scores.append((word, cos_sim))
    scores.sort(key=lambda x: -x[1])
    return scores[:topn]


def plot_tsne(vectors, word_groups: dict, filename="tsne_clusters.png"):
    """t-SNE visualization of word groups with color coding."""
    from sklearn.manifold import TSNE

    all_words = []
    all_vecs = []
    group_labels = []

    for group_name, words in word_groups.items():
        for w in words:
            if w in vectors:
                all_words.append(w)
                all_vecs.append(vectors[w])
                group_labels.append(group_name)

    if len(all_vecs) < 5:
        print("[WARN] Too few words for t-SNE")
        return

    X = np.array(all_vecs)
    perplexity = min(30, len(X) - 1)
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=perplexity, max_iter=1000)
    X_2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(14, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(word_groups)))

    for i, (group_name, _) in enumerate(word_groups.items()):
        mask = [g == group_name for g in group_labels]
        idx = np.where(mask)[0]
        ax.scatter(X_2d[idx, 0], X_2d[idx, 1], c=[colors[i]], label=group_name,
                   s=100, alpha=0.7, edgecolors='white', linewidth=0.5)
        for j in idx:
            ax.annotate(all_words[j], (X_2d[j, 0] + 0.5, X_2d[j, 1] + 0.5),
                       fontsize=9, alpha=0.85)

    ax.legend(fontsize=12, loc='upper right')
    ax.set_title("t-SNE Visualization of Word Vectors", fontsize=16)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved {FIGURES_DIR / filename}")
    plt.close()


def plot_analogy(vectors, a, b, c, d, filename="analogy_vectors.png"):
    """Plot the parallelogram formed by an analogy."""
    from sklearn.manifold import TSNE

    words = [a, b, c, d]
    vecs = [vectors[w] for w in words if w in vectors]
    if len(vecs) < 4:
        print(f"[WARN] Missing words for analogy plot")
        return

    X = np.array(vecs)
    if X.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=2, max_iter=1000)
        X_2d = tsne.fit_transform(X)
    else:
        X_2d = X

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_2d[:, 0], X_2d[:, 1], s=200, c=['tab:blue', 'tab:orange', 'tab:blue', 'tab:orange'],
               zorder=5, edgecolors='black', linewidth=1)

    for i, w in enumerate(words):
        ax.annotate(w.upper(), (X_2d[i, 0], X_2d[i, 1]),
                   fontsize=14, fontweight='bold',
                   textcoords="offset points", xytext=(10, 10))

    # Draw parallelogram
    ax.annotate("", xy=X_2d[1], xytext=X_2d[0],
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=2))
    ax.annotate("", xy=X_2d[3], xytext=X_2d[2],
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=2))
    ax.annotate("", xy=X_2d[2], xytext=X_2d[0],
                arrowprops=dict(arrowstyle="->", color="tab:green", lw=2, ls='--'))
    ax.annotate("", xy=X_2d[3], xytext=X_2d[1],
                arrowprops=dict(arrowstyle="->", color="tab:green", lw=2, ls='--'))

    ax.set_title(f"Word Analogy: {a} − {b} + {c} ≈ {d}", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved {FIGURES_DIR / filename}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="L11: Word Vector Visualization")
    parser.add_argument("--glove-path", default=GLOVE_FILE, help="Path to GloVe file")
    parser.add_argument("--ablate", choices=["random_vectors"],
                        help="random_vectors: use random vectors to show no structure")
    args = parser.parse_args()

    vectors = load_glove(args.glove_path)

    if args.ablate == "random_vectors":
        print("[ABLATION] Replacing all vectors with random noise...")
        for word in vectors:
            vectors[word] = np.random.randn(len(list(vectors.values())[0])).astype(np.float32)

    # --- Analogy demo ---
    print("\n=== Word Analogies ===")
    analogies = [
        ("king", "man", "woman"),     # → queen
        ("paris", "france", "japan"), # → tokyo
    ]
    for a, b, c in analogies:
        results = analogy(vectors, a, b, c)
        print(f"  {a} - {b} + {c} = {[f'{w}({s:.3f})' for w, s in results[:3]]}")

    # --- t-SNE visualization ---
    print("\n=== Generating t-SNE visualization ===")
    word_groups = {
        "Royalty": ["king", "queen", "prince", "princess", "crown", "throne"],
        "Gender": ["man", "woman", "boy", "girl", "father", "mother"],
        "Countries": ["china", "japan", "france", "germany", "italy", "spain"],
        "Capitals": ["beijing", "tokyo", "paris", "berlin", "rome", "madrid"],
        "Animals": ["cat", "dog", "bird", "fish", "horse", "lion", "tiger"],
    }
    suffix = "_random" if args.ablate == "random_vectors" else ""
    plot_tsne(vectors, word_groups, f"tsne_clusters{suffix}.png")

    # --- Analogy parallelogram ---
    if "king" in vectors and "queen" in vectors:
        print("\n=== Generating analogy parallelogram ===")
        plot_analogy(vectors, "king", "man", "woman", "queen", f"analogy_king_queen{suffix}.png")

    print("\n[DONE] All figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
