"""Full benchmark suite for fine-tuned AVA models.

Tests ARC-Challenge and GSM8K with proper evaluation methods.
Supports base model, LoRA adapter, and merged model paths.
"""
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from ava.external_benchmarks import load_external_benchmark_tasks


@dataclass
class BenchmarkConfig:
    model_path: str = "D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B"
    adapter_path: str | None = None
    arc_limit: int = 100
    gsm8k_limit: int = 50
    output_dir: str = "D:/AVA/experiments/exp4_finetune/results"
    run_arc: bool = True
    run_gsm8k: bool = True
    enable_thinking: bool = False


def load_model(config: BenchmarkConfig):
    """Load model with optional LoRA adapter."""
    print(f"Loading tokenizer from {config.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)

    print("Loading model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if not os.environ.get("CC"):
        candidates = [
            r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36014\bin\Hostx64\x64\cl.exe",
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe",
        ]
        for cc in candidates:
            if os.path.exists(cc):
                os.environ["CC"] = cc
                break

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    model_name = Path(config.model_path).name

    if config.adapter_path:
        adapter_config_file = Path(config.adapter_path) / "adapter_config.json"
        if adapter_config_file.exists():
            from peft import PeftModel
            print(f"Loading LoRA adapter from {config.adapter_path}...")
            model = PeftModel.from_pretrained(model, config.adapter_path)
            model = model.merge_and_unload()
            model_name = Path(config.adapter_path).name
            print("LoRA adapter loaded and merged!")
        else:
            print(f"WARNING: No adapter found at {config.adapter_path}")

    model.eval()
    print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer, model_name


@torch.no_grad()
def eval_arc_logprobs(model, tokenizer, tasks, enable_thinking: bool = False):
    """Evaluate ARC using logprob scoring (most reliable for MCQ)."""
    results = []
    correct = 0

    for i, task in enumerate(tasks):
        messages = [{"role": "user", "content": task.prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)

        scores = {}
        for label, _choice_text in task.choices:
            token_ids = tokenizer.encode(label, add_special_tokens=False)
            if token_ids:
                scores[label] = float(log_probs[token_ids[0]].item())
            else:
                scores[label] = float("-inf")

        pred = max(scores, key=scores.get)
        matched = pred.strip().lower() == task.expected.strip().lower()
        if matched:
            correct += 1

        results.append({
            "task_id": task.task_id,
            "category": task.category,
            "expected": task.expected,
            "predicted": pred,
            "matched": matched,
        })

        if (i + 1) % 20 == 0:
            print(f"  ARC [{i+1}/{len(tasks)}] {correct}/{i+1} = {correct/(i+1)*100:.1f}%")

    return correct, results


@torch.no_grad()
def eval_gsm8k_generation(model, tokenizer, tasks, enable_thinking: bool = False):
    """Evaluate GSM8K using generation + answer extraction."""
    results = []
    correct = 0

    for i, task in enumerate(tasks):
        # Build prompt encouraging step-by-step reasoning
        prompt = task.prompt.strip()
        if "step by step" not in prompt.lower():
            prompt += "\n\nSolve this step by step and give the final numeric answer."

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=768,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Extract numeric answer
        predicted = extract_numeric_answer(response)
        expected = task.expected.strip()

        # Try to match
        try:
            matched = abs(float(predicted) - float(expected)) < 0.01 if predicted else False
        except (ValueError, TypeError):
            matched = str(predicted) == str(expected)

        if matched:
            correct += 1

        results.append({
            "task_id": task.task_id,
            "category": task.category,
            "expected": expected,
            "predicted": predicted,
            "response_snippet": response[:200],
            "matched": matched,
        })

        if (i + 1) % 10 == 0:
            print(f"  GSM8K [{i+1}/{len(tasks)}] {correct}/{i+1} = {correct/(i+1)*100:.1f}%")

    return correct, results


def extract_numeric_answer(text: str) -> str | None:
    """Extract the final numeric answer from generated text.

    Uses a priority-ordered list of patterns, preferring explicit answer markers
    over positional heuristics. Strips trailing periods/whitespace.
    """
    # Priority-ordered patterns (most explicit first)
    patterns = [
        r"(?:the answer is|the final answer is)\s*\$?([-\d,]+\.?\d*)",
        r"(?:####)\s*\$?([-\d,]+\.?\d*)",
        r"(?:final answer:?)\s*\$?([-\d,]+\.?\d*)",
        r"(?:answer:)\s*\$?([-\d,]+\.?\d*)",
        r"(?:therefore|thus|so|hence)[,:]?\s+\$?([-\d,]+\.?\d*)",
        r"\\boxed\{([-\d,]+\.?\d*)\}",
        r"=\s*\$?([-\d,]+\.?\d*)\s*$",
        r"\$?([-\d,]+\.?\d*)\s*(?:dollars|miles|hours|people|items|pieces|years)?\s*\.?\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            val = match.group(1).replace(",", "").rstrip(".")
            if val and val != "-":
                return val

    # Last resort: find the last number in the text (skip very small step numbers)
    numbers = re.findall(r"[-\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "").rstrip(".")

    return None


def run_benchmarks(config: BenchmarkConfig | None = None) -> dict:
    config = config or BenchmarkConfig()

    print("=" * 60)
    print("AVA Experiment 4: Full Benchmark Suite")
    print("=" * 60)

    model, tokenizer, model_name = load_model(config)
    all_results = {"model": model_name, "benchmarks": {}}

    if config.run_arc:
        print(f"\n--- ARC-Challenge ({config.arc_limit} items) ---")
        tasks = load_external_benchmark_tasks("arc-challenge", split=None, limit=config.arc_limit, offset=0)
        start = time.perf_counter()
        arc_correct, arc_results = eval_arc_logprobs(model, tokenizer, tasks, config.enable_thinking)
        arc_time = time.perf_counter() - start

        arc_acc = arc_correct / len(tasks)
        print(f"\n  ARC-Challenge: {arc_correct}/{len(tasks)} = {arc_acc*100:.1f}% ({arc_time:.0f}s)")

        all_results["benchmarks"]["arc_challenge"] = {
            "accuracy": round(arc_acc, 3),
            "correct": arc_correct,
            "total": len(tasks),
            "elapsed_seconds": round(arc_time, 1),
            "results": arc_results,
        }

    if config.run_gsm8k:
        print(f"\n--- GSM8K ({config.gsm8k_limit} items) ---")
        tasks = load_external_benchmark_tasks("gsm8k", split=None, limit=config.gsm8k_limit, offset=0)
        start = time.perf_counter()
        gsm_correct, gsm_results = eval_gsm8k_generation(model, tokenizer, tasks, config.enable_thinking)
        gsm_time = time.perf_counter() - start

        gsm_acc = gsm_correct / len(tasks)
        print(f"\n  GSM8K: {gsm_correct}/{len(tasks)} = {gsm_acc*100:.1f}% ({gsm_time:.0f}s)")

        all_results["benchmarks"]["gsm8k"] = {
            "accuracy": round(gsm_acc, 3),
            "correct": gsm_correct,
            "total": len(tasks),
            "elapsed_seconds": round(gsm_time, 1),
            "results": gsm_results,
        }

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"benchmark_{model_name}.json"
    output_file.write_text(json.dumps(all_results, indent=2) + "\n", encoding="utf-8")

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY: {model_name}")
    print(f"{'='*60}")
    for bench_name, bench_data in all_results["benchmarks"].items():
        print(f"  {bench_name}: {bench_data['accuracy']*100:.1f}%")
    print(f"\nSaved to: {output_file}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--arc-limit", type=int, default=100)
    parser.add_argument("--gsm8k-limit", type=int, default=50)
    parser.add_argument("--output-dir", default="D:/AVA/experiments/exp4_finetune/results")
    parser.add_argument("--no-arc", action="store_true")
    parser.add_argument("--no-gsm8k", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()

    run_benchmarks(BenchmarkConfig(
        model_path=args.model,
        adapter_path=args.adapter,
        arc_limit=args.arc_limit,
        gsm8k_limit=args.gsm8k_limit,
        output_dir=args.output_dir,
        run_arc=not args.no_arc,
        run_gsm8k=not args.no_gsm8k,
        enable_thinking=args.thinking,
    ))
