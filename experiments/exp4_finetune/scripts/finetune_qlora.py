"""QLoRA fine-tuning for Qwen3.5-2B on AVA corpora.

Designed for 4GB VRAM (RTX A2000 Laptop GPU).
Uses 4-bit quantization + LoRA adapters for memory-efficient training.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from torch.utils.data import Dataset


@dataclass
class FinetuneConfig:
    model_path: str = "D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B"
    output_dir: str = "D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B-AVA-v1"
    corpus_path: str = "D:/AVA/corpora/ava_v3_rich_posttrain_v1/examples.jsonl"

    # QLoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None  # auto-detect

    # Training config
    num_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    max_seq_length: int = 1024
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 200
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning from JSONL with prompt/response fields."""

    def __init__(
        self,
        corpus_path: str,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 1024,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples: list[dict] = []

        path = Path(corpus_path)
        if path.suffix == ".jsonl":
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.examples.append(json.loads(line))
        elif path.suffix == ".json":
            self.examples = json.loads(path.read_text(encoding="utf-8"))

        print(f"Loaded {len(self.examples)} examples from {corpus_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        prompt = ex.get("prompt", ex.get("input", ""))
        response = ex.get("response", ex.get("output", ex.get("completion", "")))

        # Build chat messages
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )

        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None,
        )

        # For causal LM, labels = input_ids (shifted internally)
        encoded["labels"] = encoded["input_ids"].copy()

        return encoded


def run_finetune(config: FinetuneConfig | None = None) -> dict:
    config = config or FinetuneConfig()

    print(f"=" * 60)
    print(f"AVA Experiment 4: QLoRA Fine-tuning")
    print(f"=" * 60)
    print(f"Model: {config.model_path}")
    print(f"Corpus: {config.corpus_path}")
    print(f"Output: {config.output_dir}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in 4-bit
    print("Loading model in 4-bit quantization...")
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
    )

    # Skip prepare_model_for_kbit_training to avoid OOM from float32 upcast
    # Instead, manually freeze and enable gradient checkpointing
    for param in model.parameters():
        param.requires_grad = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Configure LoRA — target only text model layers
    target_modules = config.target_modules
    if target_modules is None:
        # Target attention + MLP projections in text model only
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=None,
    )

    model = get_peft_model(model, lora_config)
    trainable_params, total_params = model.get_nb_trainable_parameters()
    print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"GPU memory after LoRA: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Load dataset
    print("\nLoading dataset...")
    dataset = SupervisedDataset(config.corpus_path, tokenizer, config.max_seq_length)

    # Split into train/eval
    n_eval = min(100, len(dataset) // 10)
    n_train = len(dataset) - n_eval
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_eval], generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {n_train} examples, Eval: {n_eval} examples")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        max_grad_norm=config.max_grad_norm,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        gradient_checkpointing=config.gradient_checkpointing,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=config.save_steps,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    print(f"\nStarting training...")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.per_device_train_batch_size} x {config.gradient_accumulation_steps} = {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  LoRA rank: {config.lora_r}")

    start = time.perf_counter()
    train_result = trainer.train()
    elapsed = time.perf_counter() - start

    # Save
    print(f"\nSaving model to {config.output_dir}...")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Eval
    eval_result = trainer.evaluate()

    result = {
        "model": config.model_path,
        "corpus": config.corpus_path,
        "output": config.output_dir,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "train_loss": train_result.training_loss,
        "eval_loss": eval_result.get("eval_loss"),
        "train_examples": n_train,
        "eval_examples": n_eval,
        "epochs": config.num_epochs,
        "elapsed_seconds": round(elapsed, 1),
        "lora_r": config.lora_r,
        "learning_rate": config.learning_rate,
        "gpu_memory_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
    }

    # Save training report
    report_path = Path(config.output_dir) / "training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Train loss: {train_result.training_loss:.4f}")
    print(f"  Eval loss: {eval_result.get('eval_loss', 'N/A')}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Saved to: {config.output_dir}")
    print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    return result


if __name__ == "__main__":
    run_finetune()
