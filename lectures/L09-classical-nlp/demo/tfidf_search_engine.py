"""
L09 – Classical NLP: Mini TF-IDF Search Engine
===============================================
Builds a tiny search engine over 10 built-in AI/ML documents using
sklearn's TfidfVectorizer.  Given a query, ranks documents by cosine
similarity and displays a TF-IDF heatmap of the corpus.

Usage
-----
    python tfidf_search_engine.py --query "neural network training"
    python tfidf_search_engine.py --query "reinforcement learning" --top 5
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SEED = 1337
np.random.seed(SEED)

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DOCS = [
    "Deep learning uses neural networks with many layers to learn features.",
    "Reinforcement learning trains agents via rewards in an environment.",
    "Decision trees split data using feature thresholds for classification.",
    "Convolutional neural networks excel at image recognition tasks.",
    "Recurrent neural networks model sequential data like text and speech.",
    "Support vector machines find optimal hyperplanes for classification.",
    "Gradient descent optimises model parameters by following the loss gradient.",
    "Transfer learning reuses pretrained models on new downstream tasks.",
    "Natural language processing enables machines to understand human language.",
    "Generative adversarial networks produce realistic synthetic images.",
]


def main():
    parser = argparse.ArgumentParser(description="TF-IDF search engine demo")
    parser.add_argument("--query", type=str, default="neural network training")
    parser.add_argument("--top", type=int, default=5, help="Top-K results")
    args = parser.parse_args()

    vec = TfidfVectorizer(stop_words="english")
    tfidf = vec.fit_transform(DOCS)

    q_vec = vec.transform([args.query])
    scores = cosine_similarity(q_vec, tfidf).flatten()
    ranked = np.argsort(scores)[::-1]

    print(f"\nQuery: \"{args.query}\"\n{'─' * 50}")
    for rank, idx in enumerate(ranked[: args.top], 1):
        print(f"  {rank}. (score {scores[idx]:.4f})  {DOCS[idx]}")

    # --- heatmap ----------------------------------------------------------
    terms = vec.get_feature_names_out()
    mat = tfidf.toarray()
    fig, ax = plt.subplots(figsize=(max(8, len(terms) * 0.45), 4))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(terms)), terms, rotation=90, fontsize=6)
    ax.set_yticks(range(len(DOCS)), [f"Doc{i}" for i in range(len(DOCS))], fontsize=7)
    ax.set_title("TF-IDF Matrix")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fname = os.path.join(FIG_DIR, "tfidf_heatmap.png")
    fig.savefig(fname, dpi=150)
    print(f"\n[saved] {fname}")
    plt.show()


if __name__ == "__main__":
    main()
