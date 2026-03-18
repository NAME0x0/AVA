"""Quick quality evaluation of a fine-tuned model.

Tests a few questions from ARC and GSM8K to quickly assess
model quality without running the full benchmark suite.
"""
import sys
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B"
ADAPTER_PATH = "D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B-AVA-v1"

# Quick test questions
ARC_QUESTIONS = [
    {
        "prompt": "Which of the following is the best conductor of electricity?\n\nOptions:\nA. wood\nB. copper\nC. rubber\nD. glass",
        "expected": "B",
    },
    {
        "prompt": "What is the main function of the roots of a plant?\n\nOptions:\nA. to make food\nB. to absorb water and nutrients\nC. to produce flowers\nD. to release oxygen",
        "expected": "B",
    },
    {
        "prompt": "Which planet in our solar system has the most moons?\n\nOptions:\nA. Earth\nB. Mars\nC. Jupiter\nD. Saturn",
        "expected": "D",
    },
    {
        "prompt": "What happens to water when it freezes?\n\nOptions:\nA. It contracts\nB. It expands\nC. It evaporates\nD. It stays the same size",
        "expected": "B",
    },
    {
        "prompt": "Sound travels fastest through which medium?\n\nOptions:\nA. air\nB. water\nC. steel\nD. vacuum",
        "expected": "C",
    },
]

GSM8K_QUESTIONS = [
    {
        "prompt": "If a book costs $12 and you buy 5 books, how much do you spend in total?",
        "expected": "60",
    },
    {
        "prompt": "Sarah has 24 cookies. She gives 1/3 of them to her friend. How many cookies does Sarah have left?",
        "expected": "16",
    },
    {
        "prompt": "A train travels at 60 mph. How far does it travel in 2.5 hours?",
        "expected": "150",
    },
]

TOOL_QUESTIONS = [
    {
        "prompt": "What is 847 * 23?",
        "check": "tool_call",
    },
    {
        "prompt": "Who are you?",
        "check": "AVA",
    },
]


def load_model(adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, dtype=torch.bfloat16,
    )
    model_name = "Qwen3.5-2B (base)"

    if adapter_path and (Path(adapter_path) / "adapter_config.json").exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        model_name = f"AVA-v1 (LoRA from {Path(adapter_path).name})"

    model.eval()
    return model, tokenizer, model_name


@torch.no_grad()
def score_mcq(model, tokenizer, prompt, choices="ABCD"):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    logits = model(**inputs).logits[0, -1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    scores = {}
    for c in choices:
        ids = tokenizer.encode(c, add_special_tokens=False)
        scores[c] = float(log_probs[ids[0]].item()) if ids else float("-inf")
    return max(scores, key=scores.get), scores


@torch.no_grad()
def generate_answer(model, tokenizer, prompt, max_tokens=256):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.1, do_sample=True,
                          pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run_quick_eval(adapter_path=None):
    print("Loading model...")
    model, tokenizer, model_name = load_model(adapter_path)
    print(f"Model: {model_name}")
    print(f"GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB\n")

    # ARC MCQ
    print("=== ARC-Challenge Quick Test ===")
    arc_correct = 0
    for q in ARC_QUESTIONS:
        pred, scores = score_mcq(model, tokenizer, q["prompt"])
        ok = pred == q["expected"]
        arc_correct += ok
        status = "OK" if ok else "WRONG"
        print(f"  [{status}] Expected: {q['expected']}, Got: {pred}")
    print(f"  ARC: {arc_correct}/{len(ARC_QUESTIONS)} = {arc_correct/len(ARC_QUESTIONS)*100:.0f}%\n")

    # GSM8K Generation
    print("=== GSM8K Quick Test ===")
    gsm_correct = 0
    for q in GSM8K_QUESTIONS:
        response = generate_answer(model, tokenizer, q["prompt"])
        # Check if expected answer appears in response
        ok = q["expected"] in response
        gsm_correct += ok
        status = "OK" if ok else "WRONG"
        print(f"  [{status}] Expected: {q['expected']}")
        print(f"    Response: {response[:150]}")
    print(f"  GSM8K: {gsm_correct}/{len(GSM8K_QUESTIONS)} = {gsm_correct/len(GSM8K_QUESTIONS)*100:.0f}%\n")

    # Tool use and identity
    print("=== Tool Use & Identity Test ===")
    for q in TOOL_QUESTIONS:
        response = generate_answer(model, tokenizer, q["prompt"])
        has_marker = q["check"].lower() in response.lower()
        status = "OK" if has_marker else "MISS"
        print(f"  [{status}] Looking for: '{q['check']}'")
        print(f"    Response: {response[:150]}")

    print(f"\n{'='*40}")
    print(f"Quick eval complete for: {model_name}")


if __name__ == "__main__":
    adapter = ADAPTER_PATH if "--adapter" in sys.argv else None
    if "--checkpoint" in sys.argv:
        # Use a specific checkpoint
        idx = sys.argv.index("--checkpoint")
        adapter = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
    run_quick_eval(adapter)
