"""DPO (Direct Preference Optimization) fine-tuning.

Trains the model to prefer correct reasoning over incorrect reasoning,
using (chosen, rejected) pairs from build_dpo_corpus.py.

This is Phase 3 of the training pipeline:
  Phase 1: SFT (supervised fine-tuning) → finetune_unsloth.py / finetune_v2_full.py
  Phase 2: Generate DPO pairs → build_dpo_corpus.py
  Phase 3: DPO training → this script

Usage:
    python -u scripts/finetune_dpo.py > training_dpo.log 2>&1 &
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class DPOTrainingConfig:
    # Start from the SFT model (not raw Qwen)
    model_path: str = "D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B"
    sft_adapter_path: str = "D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B-AVA-v2"
    output_dir: str = "D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B-AVA-v2-dpo"
    dpo_corpus_path: str = "D:/AVA/experiments/exp4_finetune/corpora/ava_exp4_dpo_gsm8k.jsonl"

    # DPO hyperparameters
    beta: float = 0.1  # KL divergence weight (lower = more preference learning)
    lora_r: int = 8  # Smaller rank for DPO (less aggressive changes)
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Training
    num_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5  # Lower LR for DPO
    warmup_steps: int = 20
    max_seq_length: int = 512  # DPO needs longer sequences (prompt + chosen + rejected)
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    seed: int = 42


def load_dpo_corpus(path: str) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def format_dpo_dataset(examples: list[dict], tokenizer) -> dict:
    """Format DPO pairs for TRL's DPOTrainer."""
    prompts = []
    chosen_responses = []
    rejected_responses = []

    for ex in examples:
        prompt = ex["prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        # Format as chat messages
        prompt_messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

        prompts.append(prompt_text)
        chosen_responses.append(chosen)
        rejected_responses.append(rejected)

    return {
        "prompt": prompts,
        "chosen": chosen_responses,
        "rejected": rejected_responses,
    }


def run_dpo_training(config: DPOTrainingConfig | None = None):
    config = config or DPOTrainingConfig()

    print("=" * 60)
    print("AVA DPO Training")
    print("=" * 60)
    print(f"Base model: {config.model_path}")
    print(f"SFT adapter: {config.sft_adapter_path}")
    print(f"DPO corpus: {config.dpo_corpus_path}")
    print(f"Output: {config.output_dir}")

    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType

    # Check if TRL is available
    try:
        from trl import DPOTrainer, DPOConfig as TRLDPOConfig
    except ImportError:
        print("ERROR: TRL library required for DPO training.")
        print("Install with: pip install trl")
        sys.exit(1)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with SFT adapter merged
    print("\nLoading model...")
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

    # Load and merge SFT adapter if it exists
    sft_adapter_config = Path(config.sft_adapter_path) / "adapter_config.json"
    if sft_adapter_config.exists():
        from peft import PeftModel
        print(f"Loading SFT adapter from {config.sft_adapter_path}...")
        model = PeftModel.from_pretrained(model, config.sft_adapter_path)
        model = model.merge_and_unload()
        print("SFT adapter merged!")

    print(f"GPU: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Manual freeze
    for param in model.parameters():
        param.requires_grad = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Apply new LoRA for DPO (smaller rank than SFT)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # Load DPO corpus
    print(f"\nLoading DPO corpus...")
    corpus = load_dpo_corpus(config.dpo_corpus_path)
    print(f"  {len(corpus)} preference pairs")

    # Format dataset
    from datasets import Dataset
    dpo_data = format_dpo_dataset(corpus, tokenizer)
    dataset = Dataset.from_dict(dpo_data)

    # DPO training config
    training_args = TRLDPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        bf16=True,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        beta=config.beta,
        max_length=config.max_seq_length,
        max_prompt_length=256,
        eval_strategy="no",
        seed=config.seed,
    )

    # Create DPO Trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"\nStarting DPO training...")
    print(f"  Beta: {config.beta}")
    print(f"  LR: {config.learning_rate}")
    print(f"  LoRA rank: {config.lora_r}")

    start = time.perf_counter()
    train_result = trainer.train()
    elapsed = time.perf_counter() - start

    # Save
    print(f"\nSaving to {config.output_dir}...")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    print(f"\n{'='*60}")
    print(f"DPO Training Complete!")
    print(f"  Loss: {train_result.training_loss:.4f}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Saved to: {config.output_dir}")


if __name__ == "__main__":
    run_dpo_training()
