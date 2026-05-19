"""
L09 – Classical NLP: CRF Named-Entity Recognition Demo
=======================================================
Trains a tiny CRF sequence labeller (BIO scheme) on built-in Chinese
example sentences.  Uses sklearn_crfsuite when available; falls back to
a simple rule-based tagger otherwise.

Usage
-----
    python crf_ner_demo.py
    python crf_ner_demo.py --ablate   # skip CRF, use rule-based only
"""

import argparse
import os
import warnings

SEED = 1337
warnings.filterwarnings("ignore")

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Built-in corpus: list of (tokens, BIO-tags)
CORPUS = [
    (list("张三去了北京大学"),    ["B-PER", "I-PER", "O", "O", "B-ORG", "I-ORG", "I-ORG", "I-ORG"]),
    (list("李四在上海工作"),      ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "O", "O"]),
    (list("王五就读于清华大学"),  ["B-PER", "I-PER", "O", "O", "O", "B-ORG", "I-ORG", "I-ORG", "I-ORG"]),
    (list("赵六住在深圳南山"),    ["B-PER", "I-PER", "O", "O", "B-LOC", "I-LOC", "B-LOC", "I-LOC"]),
    (list("孙七加入了百度公司"),  ["B-PER", "I-PER", "O", "O", "O", "B-ORG", "I-ORG", "I-ORG", "I-ORG"]),
    (list("周八来自杭州西湖"),    ["B-PER", "I-PER", "O", "O", "B-LOC", "I-LOC", "B-LOC", "I-LOC"]),
]


def word_features(sent, i):
    """Extract per-character features for CRF."""
    c = sent[i]
    feats = {"char": c, "bias": 1.0, "is_digit": c.isdigit()}
    if i > 0:
        feats["prev_char"] = sent[i - 1]
    else:
        feats["BOS"] = True
    if i < len(sent) - 1:
        feats["next_char"] = sent[i + 1]
    else:
        feats["EOS"] = True
    return feats


def sent_to_features(sent):
    return [word_features(sent, i) for i in range(len(sent))]


def rule_based_tag(sent):
    """Naive fallback: label every character as O."""
    return ["O"] * len(sent)


def evaluate(y_true, y_pred):
    total = correct = 0
    for yt, yp in zip(y_true, y_pred):
        for a, b in zip(yt, yp):
            total += 1
            correct += a == b
    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser(description="CRF NER demo")
    parser.add_argument("--ablate", action="store_true",
                        help="Use rule-based fallback instead of CRF")
    args = parser.parse_args()

    sents = [s for s, _ in CORPUS]
    labels = [t for _, t in CORPUS]

    X = [sent_to_features(s) for s in sents]
    y = labels

    if args.ablate:
        print("[ablate] Using rule-based tagger (all O)")
        preds = [rule_based_tag(s) for s in sents]
    else:
        try:
            import sklearn_crfsuite
            crf = sklearn_crfsuite.CRF(algorithm="lbfgs", max_iterations=200,
                                       all_possible_transitions=True)
            crf.fit(X, y)
            preds = crf.predict(X)
            print("[CRF] Model trained successfully")
        except ImportError:
            print("[warn] sklearn_crfsuite not installed – falling back to rules")
            preds = [rule_based_tag(s) for s in sents]

    acc = evaluate(y, preds)
    print(f"Token accuracy: {acc:.2%}\n")

    # Show predictions for each sentence
    for sent, gold, pred in zip(sents, labels, preds):
        print("".join(sent))
        print("  Gold:", " ".join(gold))
        print("  Pred:", " ".join(pred))
        print()


if __name__ == "__main__":
    main()
