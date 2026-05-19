"""
第 13 讲 Demo · BERT vs GPT-2 — 同一段文本，两种处理方式

三组实验：
  1. 填空对决：BERT MLM  vs  GPT-2 next-token prediction
  2. 文本生成：GPT-2 自回归生成（BERT 做不到）
  3. 注意力模式：双向满矩阵 vs 因果下三角

依赖: transformers torch matplotlib numpy
首次运行会下载 bert-base-uncased (~440 MB) 和 gpt2 (~500 MB)

所有实验使用 seed=1337
"""

import argparse
import torch
import numpy as np
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[4] / "design" / "meta"))
import cjk_font  # noqa: F401  - applies CJK font config
import matplotlib.pyplot as plt
from transformers import (
    BertTokenizer, BertForMaskedLM,
    GPT2Tokenizer, GPT2LMHeadModel,
)

torch.manual_seed(1337)

# ── helpers ──────────────────────────────────────────────────────────

def top_k_predictions(logits: torch.Tensor, tokenizer, k: int = 5):
    """Return top-k (token, probability) pairs from a logit vector."""
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k)
    return [(tokenizer.decode([tid]).strip(), p.item()) for tid, p in zip(top_ids, top_probs)]


# ── Experiment 1: Fill-in-the-blank ──────────────────────────────────

def experiment_fill_blank(bert_model, bert_tok, gpt_model, gpt_tok):
    print("\n" + "=" * 60)
    print("实验 1 · 填空对决：BERT MLM  vs  GPT-2 next-token")
    print("=" * 60)

    sentence = "The capital of France is [MASK]."
    print(f"\n输入: \"{sentence}\"\n")

    # ── BERT: predict [MASK] ──
    inputs = bert_tok(sentence, return_tensors="pt")
    mask_idx = (inputs["input_ids"] == bert_tok.mask_token_id).nonzero(as_tuple=True)[1].item()
    with torch.no_grad():
        logits = bert_model(**inputs).logits[0, mask_idx]
    bert_preds = top_k_predictions(logits, bert_tok)
    print("BERT top-5:")
    for token, prob in bert_preds:
        print(f"  {token:>12s}  {prob:.4f}")

    # ── GPT-2: predict next token after "The capital of France is" ──
    prompt = "The capital of France is"
    inputs = gpt_tok(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = gpt_model(**inputs).logits[0, -1]
    gpt_preds = top_k_predictions(logits, gpt_tok)
    print(f"\nGPT-2 top-5 (prompt: \"{prompt}\"):")
    for token, prob in gpt_preds:
        print(f"  {token:>12s}  {prob:.4f}")


# ── Experiment 2: Text generation (GPT-2 only) ──────────────────────

def experiment_generation(gpt_model, gpt_tok, temperature: float = 0.8):
    print("\n" + "=" * 60)
    print("实验 2 · GPT-2 文本生成（BERT 做不到）")
    print("=" * 60)

    prompt = "In a shocking finding, scientists discovered a new species of dinosaur"
    print(f"\nPrompt: \"{prompt}\"\n")

    input_ids = gpt_tok.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = gpt_model.generate(
            input_ids,
            max_new_tokens=80,
            temperature=temperature,
            top_k=40,
            do_sample=True,
            pad_token_id=gpt_tok.eos_token_id,
        )
    generated = gpt_tok.decode(output[0], skip_special_tokens=True)
    print(f"[temperature={temperature}]\n{generated}\n")


# ── Experiment 3: Attention patterns ─────────────────────────────────

def experiment_attention(bert_model, bert_tok, gpt_model, gpt_tok):
    print("\n" + "=" * 60)
    print("实验 3 · 注意力模式：双向 vs 因果")
    print("=" * 60)

    sentence = "The cat sat on the mat because it was tired"
    print(f"\n输入: \"{sentence}\"")

    # ── BERT attention ──
    inputs_b = bert_tok(sentence, return_tensors="pt")
    with torch.no_grad():
        out_b = bert_model(**inputs_b, output_attentions=True)
    # average over heads in last layer
    attn_bert = out_b.attentions[-1][0].mean(dim=0).numpy()
    tokens_bert = bert_tok.convert_ids_to_tokens(inputs_b["input_ids"][0])

    # ── GPT-2 attention ──
    inputs_g = gpt_tok(sentence, return_tensors="pt")
    with torch.no_grad():
        out_g = gpt_model(**inputs_g, output_attentions=True)
    attn_gpt = out_g.attentions[-1][0].mean(dim=0).numpy()
    tokens_gpt = gpt_tok.convert_ids_to_tokens(inputs_g["input_ids"][0])

    # ── plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(attn_bert, cmap="Blues")
    axes[0].set_xticks(range(len(tokens_bert)))
    axes[0].set_yticks(range(len(tokens_bert)))
    axes[0].set_xticklabels(tokens_bert, rotation=45, ha="right", fontsize=7)
    axes[0].set_yticklabels(tokens_bert, fontsize=7)
    axes[0].set_title("BERT (bidirectional) — last layer avg")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(attn_gpt, cmap="Oranges")
    axes[1].set_xticks(range(len(tokens_gpt)))
    axes[1].set_yticks(range(len(tokens_gpt)))
    axes[1].set_xticklabels(tokens_gpt, rotation=45, ha="right", fontsize=7)
    axes[1].set_yticklabels(tokens_gpt, fontsize=7)
    axes[1].set_title("GPT-2 (causal) — last layer avg")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.suptitle("Attention Patterns: BERT vs GPT-2", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = "figures/attention_comparison.png"
    import os
    os.makedirs("figures", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\n注意力对比图已保存 → {out_path}")
    plt.show()


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="L13 Demo: BERT vs GPT-2")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="GPT-2 generation temperature (default 0.8)")
    parser.add_argument("--skip-plot", action="store_true",
                        help="Skip attention visualization (useful on headless servers)")
    args = parser.parse_args()

    print("加载模型 …")
    bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased", attn_implementation="eager")
    bert_model.eval()

    gpt_tok = GPT2Tokenizer.from_pretrained("gpt2")
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    gpt_model.eval()
    print("模型加载完成\n")

    experiment_fill_blank(bert_model, bert_tok, gpt_model, gpt_tok)
    experiment_generation(gpt_model, gpt_tok, temperature=args.temperature)

    if not args.skip_plot:
        experiment_attention(bert_model, bert_tok, gpt_model, gpt_tok)


if __name__ == "__main__":
    main()