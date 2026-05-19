"""
第 19 讲 · QLoRA 微调 7B 模型 — 完整训练骨架
==============================================

用 QLoRA (4-bit NF4 + LoRA) 微调一个 7B Instruct 模型，
使其成为领域专家（示例：中医诊疗助手）。

依赖：pip install torch transformers peft bitsandbytes datasets trl accelerate
硬件：1× GPU ≥ 12GB VRAM（RTX 3060+）; --dry-run 可在 CPU 上验证流程
种子：所有实验使用 seed=1337
"""

import argparse
import json
import torch
from pathlib import Path


# ──────────────────────────────────────────────
# 0. 参数解析
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tuning skeleton")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                   help="HuggingFace model id")
    p.add_argument("--dataset_path", type=str, default=None,
                   help="Path to JSONL training data (messages format). "
                        "If None, uses a built-in toy dataset for demo.")
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dry-run", action="store_true",
                   help="Run on CPU with tiny data to verify the pipeline")
    p.add_argument("--ablate", type=str, default=None,
                   choices=["rank", "modules"],
                   help="Run ablation: 'rank' sweeps r in {4,8,16,32,64}; "
                        "'modules' compares qv-only vs all-linear")
    return p.parse_args()


# ──────────────────────────────────────────────
# 1. 内置示例数据（仅供 demo / dry-run）
# ──────────────────────────────────────────────
DEMO_DATA = [
    {
        "messages": [
            {"role": "system", "content": "你是一位经验丰富的中医师，根据患者描述给出诊疗建议。"},
            {"role": "user", "content": "我最近总是失眠多梦，口干舌燥，手脚心发热。"},
            {"role": "assistant", "content": (
                "根据您描述的症状——失眠多梦、口干舌燥、手脚心热——"
                "这是典型的阴虚内热证候，属于心肾不交型失眠。\n"
                "辨证：肾阴虚，虚火上扰心神。\n"
                "建议处方：天王补心丹加减——"
                "生地15g、当归10g、五味子6g、麦冬12g、酸枣仁15g、柏子仁10g。\n"
                "生活建议：睡前避免剧烈运动，忌辛辣燥热食物，可适量饮用酸枣仁汤。"
            )},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "你是一位经验丰富的中医师，根据患者描述给出诊疗建议。"},
            {"role": "user", "content": "我经常感到疲倦乏力，食欲不振，大便溏稀。"},
            {"role": "assistant", "content": (
                "您的症状——疲倦乏力、食欲不振、大便溏稀——指向脾气虚证。\n"
                "辨证：脾虚运化失常，中气不足。\n"
                "建议处方：四君子汤加减——"
                "党参15g、白术10g、茯苓15g、甘草6g、山药15g、薏苡仁20g。\n"
                "饮食建议：忌生冷油腻，可食山药粥、红枣桂圆汤等健脾食物。"
            )},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "你是一位经验丰富的中医师，根据患者描述给出诊疗建议。"},
            {"role": "user", "content": "能帮我开一副减肥药吗？要那种吃了马上瘦的。"},
            {"role": "assistant", "content": (
                "抱歉，中医不提倡使用'速效减肥药'，这类药物可能损害肝肾功能。\n"
                "如果您有减重需求，建议到正规医疗机构做体质辨识，"
                "根据体质制定调理方案（如痰湿体质可用二陈汤化裁）。\n"
                "健康减重需要饮食、运动、作息的综合调整，请勿自行用药。"
            )},
        ]
    },
]


# ──────────────────────────────────────────────
# 2. 加载数据
# ──────────────────────────────────────────────
def load_dataset(dataset_path: str | None, dry_run: bool):
    from datasets import Dataset

    if dataset_path is None or dry_run:
        print("[INFO] Using built-in demo data "
              f"({len(DEMO_DATA)} samples, {'dry-run' if dry_run else 'no dataset_path'})")
        return Dataset.from_list(DEMO_DATA)

    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    print(f"[INFO] Loaded {len(records)} samples from {dataset_path}")
    return Dataset.from_list(records)


# ──────────────────────────────────────────────
# 3. 加载模型（4-bit 量化）
# ──────────────────────────────────────────────
def load_model(model_name: str, dry_run: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dry_run:
        print("[DRY-RUN] Loading model in FP32 on CPU (no quantization)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    return model, tokenizer


# ──────────────────────────────────────────────
# 4. 注入 LoRA
# ──────────────────────────────────────────────
def inject_lora(model, r: int = 16, alpha: int = 32,
                target_modules: list[str] | None = None):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)

    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config


# ──────────────────────────────────────────────
# 5. 训练
# ──────────────────────────────────────────────
def train(model, tokenizer, dataset, peft_config, args: argparse.Namespace):
    from trl import SFTConfig, SFTTrainer

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1 if getattr(args, "dry_run", False) else 10,
        save_strategy="epoch",
        bf16=not getattr(args, "dry_run", False),
        fp16=False,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        max_steps=3 if getattr(args, "dry_run", False) else -1,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50 + "\n")

    trainer.train()

    print("\n[INFO] Training complete. Saving adapter...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return trainer


# ──────────────────────────────────────────────
# 6. 推理测试
# ──────────────────────────────────────────────
def test_inference(model, tokenizer, dry_run: bool):
    test_prompt = "我最近总是头晕目眩，耳鸣，腰膝酸软。请问是什么问题？"
    messages = [
        {"role": "system", "content": "你是一位经验丰富的中医师，根据患者描述给出诊疗建议。"},
        {"role": "user", "content": test_prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"<|system|>{messages[0]['content']}<|user|>{messages[1]['content']}<|assistant|>"

    device = "cpu" if dry_run else "cuda"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:],
                                skip_special_tokens=True)
    print("\n" + "=" * 50)
    print(f"[TEST] Prompt: {test_prompt}")
    print(f"[TEST] Response:\n{response}")
    print("=" * 50 + "\n")


# ──────────────────────────────────────────────
# 7. 合并 LoRA 并导出
# ──────────────────────────────────────────────
def merge_and_export(model, tokenizer, output_dir: str):
    merged_dir = str(Path(output_dir) / "merged")
    print(f"[INFO] Merging LoRA weights and saving to {merged_dir}")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"[INFO] Merged model saved. Can be loaded directly or converted for Ollama/vLLM.")


# ──────────────────────────────────────────────
# 8. 消融实验
# ──────────────────────────────────────────────
def run_ablation(args: argparse.Namespace):
    if args.ablate == "rank":
        print("\n[ABLATION] Sweeping LoRA rank: r ∈ {4, 8, 16, 32, 64}")
        for r in [4, 8, 16, 32, 64]:
            print(f"\n--- r = {r} ---")
            args.lora_r = r
            args.lora_alpha = 2 * r
            main_pipeline(args, skip_merge=True)

    elif args.ablate == "modules":
        print("\n[ABLATION] Comparing target modules:")
        configs = {
            "qv_only": ["q_proj", "v_proj"],
            "all_linear": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        }
        for name, modules in configs.items():
            print(f"\n--- {name}: {modules} ---")
            args._target_modules = modules
            main_pipeline(args, skip_merge=True)


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────
def main_pipeline(args: argparse.Namespace, skip_merge: bool = False):
    dry_run = getattr(args, "dry_run", False)

    # Step 1-2: Load data
    dataset = load_dataset(args.dataset_path, dry_run)

    # Step 3: Load model
    model, tokenizer = load_model(args.model_name, dry_run)

    # Step 4: Inject LoRA
    target_modules = getattr(args, "_target_modules", None)
    model, peft_config = inject_lora(
        model, r=args.lora_r, alpha=args.lora_alpha,
        target_modules=target_modules,
    )

    # Step 5: Train
    train(model, tokenizer, dataset, peft_config, args)

    # Step 6: Test inference
    test_inference(model, tokenizer, dry_run)

    # Step 7: Merge & export
    if not skip_merge and not dry_run:
        merge_and_export(model, tokenizer, args.output_dir)


# ──────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    # Normalize dry-run flag
    args.dry_run = getattr(args, "dry_run", False)

    if args.ablate:
        run_ablation(args)
    else:
        main_pipeline(args)