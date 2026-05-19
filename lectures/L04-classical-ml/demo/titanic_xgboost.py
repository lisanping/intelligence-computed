"""Titanic 生存预测 Demo — 第 4 讲：经典机器学习

完整的表格数据 ML 管线：数据清洗 → 特征工程 → 多算法对比。
展示 Gradient Boosting 在表格数据上的统治地位。

用法：
    python titanic_xgboost.py                # 完整管线 + 多算法对比
    python titanic_xgboost.py --no-feature-eng  # 不做特征工程（对比）
"""

import argparse
import os
import warnings

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[4] / "design" / "meta"))
import cjk_font  # noqa: F401  - applies CJK font config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import fetch_openml

warnings.filterwarnings('ignore')
SEED = 1337


def load_titanic():
    """Load Titanic dataset from sklearn/openml."""
    data = fetch_openml('titanic', version=1, as_frame=True, parser='auto')
    df = data.frame
    return df


def preprocess(df: pd.DataFrame, feature_eng: bool = True) -> tuple:
    """Clean and prepare features."""
    # Select useful columns
    cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'survived']
    df = df[cols].copy()

    # Target
    y = df['survived'].astype(int).values
    df = df.drop('survived', axis=1)

    # Fill missing values
    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].fillna(df['fare'].median())
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

    # Encode categoricals
    df['sex'] = LabelEncoder().fit_transform(df['sex'])
    df['embarked'] = LabelEncoder().fit_transform(df['embarked'])

    # Feature engineering
    if feature_eng:
        df['family_size'] = df['sibsp'] + df['parch'] + 1
        df['is_alone'] = (df['family_size'] == 1).astype(int)
        df['fare_per_person'] = df['fare'] / df['family_size']
        df['age_bin'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 100],
                                labels=[0, 1, 2, 3, 4]).astype(int)

    X = df.values.astype(float)
    feature_names = list(df.columns)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, feature_names


def run_comparison(X, y, feature_eng_label: str):
    """Run 6 algorithms with cross-validation."""
    classifiers = [
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
        ("Decision Tree", DecisionTreeClassifier(max_depth=5, random_state=SEED)),
        ("Logistic Reg.", LogisticRegression(max_iter=500, random_state=SEED)),
        ("SVM (RBF)", SVC(kernel='rbf', random_state=SEED)),
        ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=SEED)),
        ("GBDT (sklearn)", GradientBoostingClassifier(n_estimators=200, random_state=SEED)),
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}

    print(f"\n{'='*50}")
    print(f"  Titanic Survival Prediction — {feature_eng_label}")
    print(f"{'='*50}")
    print(f"  Features: {X.shape[1]} | Samples: {X.shape[0]}")
    print(f"\n{'Algorithm':<20} {'Mean CV Acc':>12} {'Std':>8}")
    print("-" * 42)

    for name, clf in classifiers:
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        results[name] = scores.mean()
        print(f"  {name:<18} {scores.mean():>10.4f}   ±{scores.std():.4f}")

    # Winner
    winner = max(results, key=results.get)
    print(f"\n  [Winner] {winner} ({results[winner]:.4f})")

    return results


def plot_comparison(results_with_fe, results_without_fe):
    """Bar chart comparing algorithms with and without feature engineering."""
    names = list(results_with_fe.keys())
    accs_with = [results_with_fe[n] for n in names]
    accs_without = [results_without_fe[n] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, accs_without, width, label='Without Feature Eng.',
                    color='#95a5a6', edgecolor='white')
    bars2 = ax.bar(x + width/2, accs_with, width, label='With Feature Eng.',
                    color='#2ecc71', edgecolor='white')

    ax.set_ylabel('5-Fold CV Accuracy')
    ax.set_title('Titanic: Algorithm Comparison\n(Feature Engineering > Algorithm Choice)',
                  fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0.7, 0.85)
    ax.axhline(y=max(accs_with), color='green', linestyle='--', alpha=0.3)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig("figures/titanic_comparison.png", dpi=150, bbox_inches='tight')
    print("\n[saved] figures/titanic_comparison.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Titanic ML Pipeline")
    parser.add_argument("--no-feature-eng", action="store_true",
                        help="Skip feature engineering (for comparison)")
    args = parser.parse_args()

    os.makedirs("figures", exist_ok=True)

    print("Loading Titanic dataset...")
    df = load_titanic()

    if args.no_feature_eng:
        X, y, features = preprocess(df, feature_eng=False)
        run_comparison(X, y, "Without Feature Engineering")
    else:
        # Run both for comparison
        X_no, y, _ = preprocess(df, feature_eng=False)
        X_yes, y, features = preprocess(df, feature_eng=True)
        print(f"\nFeatures (with eng): {features}")
        results_no = run_comparison(X_no, y, "Without Feature Engineering")
        results_yes = run_comparison(X_yes, y, "With Feature Engineering")
        plot_comparison(results_yes, results_no)


if __name__ == "__main__":
    main()
