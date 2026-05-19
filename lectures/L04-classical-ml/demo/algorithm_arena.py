"""多算法擂台赛 Demo — 第 4 讲：经典机器学习

Scikit-learn 6 种算法在 Iris 数据集上的决策边界可视化 + 准确率对比。
支持消融实验：K 值 / 树深度 / SVM 核函数。

用法：
    python algorithm_arena.py                   # 6 种算法并排
    python algorithm_arena.py --ablate knn_k     # K 值影响
    python algorithm_arena.py --ablate tree_depth # 树深度影响
    python algorithm_arena.py --ablate svm_kernel # 核函数影响
"""

import argparse
import os

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[4] / "design" / "meta"))
import cjk_font  # noqa: F401  - applies CJK font config
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

SEED = 1337
np.random.seed(SEED)


def load_data():
    """Load Iris with 2 features for 2D visualization."""
    iris = load_iris()
    X = iris.data[:, [0, 2]]  # sepal length, petal length
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler, iris.feature_names


def plot_decision_boundary(ax, clf, X, y, title, h=0.02):
    """Plot decision boundary for a 2D classifier."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu,
               edgecolors='k', s=20, linewidths=0.5)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])


def main_arena(X_train, X_test, y_train, y_test):
    """6 algorithms side by side."""
    classifiers = [
        ("KNN (K=5)", KNeighborsClassifier(n_neighbors=5)),
        ("Decision Tree", DecisionTreeClassifier(max_depth=5, random_state=SEED)),
        ("Logistic Reg.", LogisticRegression(max_iter=200, random_state=SEED)),
        ("SVM (RBF)", SVC(kernel='rbf', random_state=SEED)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=SEED)),
        ("GBDT (sklearn)", GradientBoostingClassifier(n_estimators=100, random_state=SEED)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()

    print(f"\n{'Algorithm':<20} {'Train Acc':>10} {'Test Acc':>10}")
    print("=" * 42)

    for ax, (name, clf) in zip(axes, classifiers):
        clf.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"{name:<20} {train_acc:>10.3f} {test_acc:>10.3f}")
        plot_decision_boundary(ax, clf, X_train, y_train,
                               f"{name}\nTest Acc: {test_acc:.3f}")

    fig.suptitle("Algorithm Arena — Decision Boundaries on Iris (2 features)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("figures/algorithm_arena.png", dpi=150, bbox_inches='tight')
    print("\n[saved] figures/algorithm_arena.png")
    plt.show()


def ablate_knn_k(X_train, X_test, y_train, y_test):
    """Show KNN decision boundary for different K values."""
    ks = [1, 5, 15, 50]
    fig, axes = plt.subplots(1, len(ks), figsize=(16, 4))
    for ax, k in zip(axes, ks):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        plot_decision_boundary(ax, clf, X_train, y_train,
                               f"KNN K={k}\nAcc: {acc:.3f}")
    fig.suptitle("Ablation: KNN — Effect of K", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("figures/ablate_knn_k.png", dpi=150, bbox_inches='tight')
    print("[saved] figures/ablate_knn_k.png")
    plt.show()


def ablate_tree_depth(X_train, X_test, y_train, y_test):
    """Show decision tree for different max depths."""
    depths = [1, 3, 5, 20]
    fig, axes = plt.subplots(1, len(depths), figsize=(16, 4))
    for ax, d in zip(axes, depths):
        clf = DecisionTreeClassifier(max_depth=d, random_state=SEED)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        plot_decision_boundary(ax, clf, X_train, y_train,
                               f"Tree depth={d}\nAcc: {acc:.3f}")
    fig.suptitle("Ablation: Decision Tree — Effect of Max Depth",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("figures/ablate_tree_depth.png", dpi=150, bbox_inches='tight')
    print("[saved] figures/ablate_tree_depth.png")
    plt.show()


def ablate_svm_kernel(X_train, X_test, y_train, y_test):
    """Show SVM with different kernels."""
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    fig, axes = plt.subplots(1, len(kernels), figsize=(16, 4))
    for ax, k in zip(axes, kernels):
        clf = SVC(kernel=k, random_state=SEED)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        plot_decision_boundary(ax, clf, X_train, y_train,
                               f"SVM kernel={k}\nAcc: {acc:.3f}")
    fig.suptitle("Ablation: SVM — Effect of Kernel Function",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("figures/ablate_svm_kernel.png", dpi=150, bbox_inches='tight')
    print("[saved] figures/ablate_svm_kernel.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="ML Algorithm Arena")
    parser.add_argument("--ablate", choices=["knn_k", "tree_depth", "svm_kernel"],
                        default=None, help="Run ablation study")
    args = parser.parse_args()

    os.makedirs("figures", exist_ok=True)
    X_train, X_test, y_train, y_test, _, _ = load_data()

    if args.ablate == "knn_k":
        ablate_knn_k(X_train, X_test, y_train, y_test)
    elif args.ablate == "tree_depth":
        ablate_tree_depth(X_train, X_test, y_train, y_test)
    elif args.ablate == "svm_kernel":
        ablate_svm_kernel(X_train, X_test, y_train, y_test)
    else:
        main_arena(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
