"""QLoRA fine-tuning on full v2 corpus (20K examples).

Uses proven BitsAndBytes 4-bit + SDPA pipeline from fast run.
Optimized for 4GB VRAM RTX A2000.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class V2FullConfig:
    model_path: str = "D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B"
    output_dir: str = "D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B-AVA-v2"
    corpus_path: str = "D:/AVA/experiments/exp4_finetune/corpora/ava_exp4_finetune_v2_augmented.jsonl"

    # LoRA config (same as v1 fast — proven stable)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0

    # Training — 1 epoch on 20K examples ≈ 2586 steps at batch=8
    num_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1.5e-4  # Slightly lower for larger corpus
    warmup_steps: int = 80  # ~3% of total steps
    max_seq_length: int = 384
    logging_steps: int = 20
    save_steps: int = 300  # Save every ~1 hour
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    seed: int = 42


def load_corpus(path: str) -> list[dict]:
    """Load JSONL corpus into list of dicts."""
    examples = []
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def format_for_training(examples: list[dict], tokenizer) -> list[str]:
    """Format examples as chat-templated strings."""
    formatted = []
    for ex in examples:
        prompt = ex.get("prompt", ex.get("input", ""))
        response = ex.get("response", ex.get("output", ex.get("completion", "")))
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        formatted.append(text)
    return formatted


def run_v2_finetune(config: V2FullConfig | None = None) -> dict:
    config = config or V2FullConfig()

    print("=" * 60)
    print("AVA Experiment 4: V2 Full Corpus QLoRA Fine-tuning")
    print("=" * 60)
    print(f"Model: {config.model_path}")
    print(f"Corpus: {config.corpus_path}")
    print(f"Output: {config.output_dir}")

    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Set CC for Triton kernel compilation (FLA fast path)
    # Also set to bundled TinyCC as fallback
    if not os.environ.get("CC"):
        candidates = [
            r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36014\bin\Hostx64\x64\cl.exe",
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe",
            os.path.join(os.path.dirname(__import__('triton').__file__), "runtime", "tcc", "tcc.exe"),
        ]
        for cc in candidates:
            if os.path.exists(cc):
                os.environ["CC"] = cc
                print(f"  CC set to: {cc}")
                break

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model with BitsAndBytes 4-bit + SDPA attention...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    print(f"GPU memory after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Manual freeze (avoids OOM from prepare_model_for_kbit_training)
    for param in model.parameters():
        param.requires_grad = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Apply LoRA
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    trainable_params, total_params = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"GPU memory after LoRA: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Load and format corpus
    print("\nLoading corpus...")
    examples = load_corpus(config.corpus_path)
    texts = format_for_training(examples, tokenizer)
    print(f"Formatted {len(texts)} examples")

    # Create dataset
    from datasets import Dataset
    dataset = Dataset.from_dict({"text": texts})

    # Split train/eval
    n_eval = min(200, len(dataset) // 10)
    ds = dataset.train_test_split(test_size=n_eval, seed=config.seed)
    print(f"Train: {len(ds['train'])}, Eval: {len(ds['test'])}")

    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

    print("Tokenizing dataset...")

    def tokenize_fn(example):
        encoded = tokenizer(
            example["text"],
            truncation=True,
            max_length=config.max_seq_length,
            padding=False,
        )
        return encoded

    ds = ds.map(tokenize_fn, remove_columns=["text"], batched=False)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        fp16=False,
        bf16=True,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=4,
        max_grad_norm=1.0,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        eval_strategy="no",  # Eval OOMs on 4GB VRAM (248K vocab logits.float())
        seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=data_collator,
    )

    n_steps = len(ds["train"]) // (config.per_device_train_batch_size * config.gradient_accumulation_steps)
    print(f"\nStarting training...")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Effective batch: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"  LR: {config.learning_rate}")
    print(f"  LoRA rank: {config.lora_r}")
    print(f"  Total optimizer steps: ~{n_steps}")

    # Resume from latest checkpoint if one exists (e.g. after crash/restart)
    last_ckpt = None
    ckpt_dir = Path(config.output_dir)
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if ckpts:
            last_ckpt = str(ckpts[-1])
            print(f"  Resuming from checkpoint: {last_ckpt}")

    start = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=last_ckpt)
    elapsed = time.perf_counter() - start

    # Save
    print(f"\nSaving model to {config.output_dir}...")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    result = {
        "engine": "sdpa_qlora",
        "version": "v2_full",
        "model": config.model_path,
        "corpus": config.corpus_path,
        "output": config.output_dir,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "train_loss": train_result.training_loss,
        "train_examples": len(ds["train"]),
        "epochs": config.num_epochs,
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_minutes": round(elapsed / 60, 1),
        "steps_per_second": round(train_result.metrics.get("train_steps_per_second", 0), 2),
        "lora_r": config.lora_r,
        "learning_rate": config.learning_rate,
        "max_seq_length": config.max_seq_length,
        "gpu_memory_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
    }

    report_path = Path(config.output_dir) / "training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Train loss: {train_result.training_loss:.4f}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Steps/sec: {train_result.metrics.get('train_steps_per_second', 0):.2f}")
    print(f"  Saved to: {config.output_dir}")

    return result


if __name__ == "__main__":
    run_v2_finetune()
