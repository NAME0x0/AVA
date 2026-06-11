"""QAT recovery pass for AVA v2 — the Gemma-4-QAT analog on a 4 GB GPU.

Goal: make the 4-bit GGUF (Q4_0 / Q4_K_M) of AVA v2 match Q8_0 quality.
Google's Gemma QAT releases do this by fine-tuning with quantization noise
while distilling from the bf16 checkpoint. We reproduce the recipe within a
4 GB VRAM budget in three stages:

  1. cache  — teacher pass. The merged bf16 model is loaded in 8-bit
              (bitsandbytes int8, fits the A2000 and matches the shipped
              Q8_0 reference) and run over the training corpus once,
              caching top-k logits to disk.
  2. train  — student pass. The same base is loaded in 4-bit (NF4) with a
              LoRA on every linear projection. The LoRA learns to cancel
              4-bit quantization noise by minimising KL against the cached
              teacher logits (plus a small CE term on the data).
  3. merge  — the recovery LoRA is merged into the bf16 weights on CPU and
              saved; convert_hf_to_gguf.py + llama-quantize (with imatrix)
              turn it into Q4_0/Q4_K_M GGUFs.

Caveat (documented honestly): bitsandbytes NF4 is not the exact Q4_0 grid,
so noise cancellation transfers approximately, not exactly. The final word
is the perplexity/benchmark comparison of the resulting GGUF against the
plain imatrix quants — ship whichever wins.

Usage:
  python qat_recover.py cache   [--max-examples 20000]
  python qat_recover.py train   [--epochs 2]
  python qat_recover.py merge
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

MERGED_DIR = Path(r"D:\AVA\gguf_build\merged")
CORPORA = Path(r"D:\AVA\corpora")
WORK_DIR = Path(r"D:\AVA\gguf_build\qat")
CACHE_FILE = WORK_DIR / "teacher_topk.pt"
LORA_DIR = WORK_DIR / "recovery_lora"
QAT_MERGED_DIR = Path(r"D:\AVA\gguf_build\merged_qat")

SOURCES = [
    CORPORA / "ava_v2_posttrain_mix_v1" / "examples.jsonl",
    CORPORA / "ava_v2_posttrain_assistant_v1" / "examples.jsonl",
    CORPORA / "ava_v2_posttrain_generalist_v1" / "examples.jsonl",
    CORPORA / "gsm8k_train_reasoning_support_v1" / "examples.jsonl",
    CORPORA / "arc_train_support_v1" / "examples.jsonl",
]

CHAT = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
SEQ_LEN = 384  # matches the original v2 fine-tune
TOP_K = 64
KL_TEMPERATURE = 1.0
KL_WEIGHT = 0.75  # rest is CE on the corpus text


def load_examples(max_examples: int, seed: int = 20260611) -> list[str]:
    rng = random.Random(seed)
    texts: list[str] = []
    for path in SOURCES:
        if not path.exists():
            print(f"skip missing {path}")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("prompt") and d.get("response"):
                texts.append(CHAT.format(prompt=d["prompt"], response=d["response"]))
    rng.shuffle(texts)
    return texts[:max_examples]


def batch_tokens(tokenizer, texts: list[str], batch_size: int):
    """Yield (input_ids, attention_mask) batches of SEQ_LEN."""
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i : i + batch_size],
            max_length=SEQ_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        yield enc["input_ids"], enc["attention_mask"]


def stage_cache(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_DIR,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval()

    texts = load_examples(args.max_examples)
    print(f"caching teacher top-{TOP_K} logits for {len(texts)} examples")

    all_ids, all_mask, all_topk_ids, all_topk_logits = [], [], [], []
    done = 0
    with torch.no_grad():
        for input_ids, attention_mask in batch_tokens(tokenizer, texts, args.batch_size):
            out = model(
                input_ids=input_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),
            )
            topk = out.logits.float().topk(TOP_K, dim=-1)
            all_ids.append(input_ids.to(torch.int32))
            all_mask.append(attention_mask.to(torch.bool))
            all_topk_ids.append(topk.indices.to(torch.int32).cpu())
            all_topk_logits.append(topk.values.to(torch.float16).cpu())
            done += input_ids.shape[0]
            if done % (args.batch_size * 50) == 0:
                print(f"  {done}/{len(texts)}")

    torch.save(
        {
            "input_ids": torch.cat(all_ids),
            "attention_mask": torch.cat(all_mask),
            "topk_ids": torch.cat(all_topk_ids),
            "topk_logits": torch.cat(all_topk_logits),
        },
        CACHE_FILE,
    )
    print(f"wrote {CACHE_FILE} ({CACHE_FILE.stat().st_size / 1e6:.0f} MB)")


def kl_to_cached_teacher(
    student_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """KL(teacher ‖ student) over the teacher's cached top-k support."""
    t = KL_TEMPERATURE
    teacher_p = F.softmax(topk_logits.float() / t, dim=-1)
    student_on_topk = student_logits.gather(-1, topk_ids.long())
    # log-softmax over the full vocab, evaluated at the top-k ids
    student_logz = torch.logsumexp(student_logits.float() / t, dim=-1, keepdim=True)
    student_logp = student_on_topk.float() / t - student_logz
    kl = (teacher_p * (torch.log(teacher_p + 1e-9) - student_logp)).sum(-1)
    return (kl * mask).sum() / mask.sum().clamp(min=1)


def stage_train(args: argparse.Namespace) -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cache = torch.load(CACHE_FILE, map_location="cpu")
    n = cache["input_ids"].shape[0]
    print(f"loaded teacher cache: {n} sequences")

    tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_DIR,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * (n // args.batch_size)
    )

    step = 0
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        for i in range(0, n - args.batch_size + 1, args.batch_size):
            idx = perm[i : i + args.batch_size]
            input_ids = cache["input_ids"][idx].long().to(model.device)
            mask = cache["attention_mask"][idx].to(model.device)
            topk_ids = cache["topk_ids"][idx].to(model.device)
            topk_logits = cache["topk_logits"][idx].to(model.device)

            out = model(input_ids=input_ids, attention_mask=mask)
            kl = kl_to_cached_teacher(out.logits, topk_ids, topk_logits, mask)

            labels = input_ids.masked_fill(~mask.bool(), -100)
            ce = F.cross_entropy(
                out.logits[:, :-1].flatten(0, 1).float(),
                labels[:, 1:].flatten(),
                ignore_index=-100,
            )
            loss = KL_WEIGHT * kl + (1 - KL_WEIGHT) * ce
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % 50 == 0:
                print(
                    f"epoch {epoch} step {step} "
                    f"loss {loss.item():.4f} kl {kl.item():.4f} ce {ce.item():.4f}"
                )

    LORA_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(LORA_DIR)
    tokenizer.save_pretrained(LORA_DIR)
    print(f"saved recovery LoRA to {LORA_DIR}")


def stage_merge(args: argparse.Namespace) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("merging recovery LoRA into bf16 weights on CPU")
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_DIR, dtype=torch.bfloat16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(model, LORA_DIR)
    model = model.merge_and_unload()
    QAT_MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(QAT_MERGED_DIR)
    AutoTokenizer.from_pretrained(MERGED_DIR).save_pretrained(QAT_MERGED_DIR)
    print(f"saved QAT-recovered model to {QAT_MERGED_DIR}")
    print("next: convert_hf_to_gguf.py + llama-quantize --imatrix → Q4_0/Q4_K_M")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    p_cache = sub.add_parser("cache", help="cache teacher top-k logits (8-bit pass)")
    p_cache.add_argument("--max-examples", type=int, default=20000)
    p_cache.add_argument("--batch-size", type=int, default=8)
    p_cache.set_defaults(fn=stage_cache)

    p_train = sub.add_parser("train", help="train 4-bit recovery LoRA against cache")
    p_train.add_argument("--epochs", type=int, default=2)
    p_train.add_argument("--batch-size", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=5e-5)
    p_train.set_defaults(fn=stage_train)

    p_merge = sub.add_parser("merge", help="merge recovery LoRA into bf16")
    p_merge.set_defaults(fn=stage_merge)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
