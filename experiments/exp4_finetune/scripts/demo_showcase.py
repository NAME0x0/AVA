"""Showcase demo of fine-tuned AVA model capabilities.

Runs a series of tests comparing base model vs fine-tuned model
across different capability categories.
"""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B"
ADAPTER_PATH = "D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B-AVA-v1"

DEMO_PROMPTS = {
    "Math (CoT)": [
        "If a train travels at 60 mph for 2.5 hours and then at 80 mph for 1.5 hours, what is the total distance traveled?",
        "A store has a 25% off sale. If a jacket originally costs $120, and you also have a $10 coupon, how much do you pay?",
    ],
    "Science (MCQ)": [
        "Which of the following is NOT a renewable energy source?\n\nOptions:\nA. Solar power\nB. Natural gas\nC. Wind power\nD. Hydroelectric power",
        "What is the primary function of mitochondria in a cell?\n\nOptions:\nA. Protein synthesis\nB. Energy production\nC. Cell division\nD. DNA replication",
    ],
    "Tool Use": [
        "What is 1847 * 293?",
        "Calculate the factorial of 12.",
    ],
    "Identity": [
        "Who are you?",
        "What makes you different from other AI assistants?",
    ],
    "Reasoning": [
        "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?",
        "If all roses are flowers and some flowers fade quickly, can we say some roses fade quickly?",
    ],
}


def load_model(adapter_path=None):
    """Load model with optional LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, dtype=torch.bfloat16,
    )
    name = "Qwen3.5-2B (base)"

    if adapter_path and (Path(adapter_path) / "adapter_config.json").exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        name = "AVA-v1 (fine-tuned)"

    model.eval()
    return model, tokenizer, name


@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=300):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_tokens,
        temperature=0.3, do_sample=True, top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run_demo(adapter_path=None):
    print("Loading model...")
    model, tokenizer, name = load_model(adapter_path)
    print(f"Model: {name}")
    print(f"GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB\n")

    results = {}
    for category, prompts in DEMO_PROMPTS.items():
        print(f"\n{'='*60}")
        print(f"  {category}")
        print(f"{'='*60}")
        results[category] = []

        for prompt in prompts:
            print(f"\n  Q: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
            start = time.perf_counter()
            response = generate(model, tokenizer, prompt)
            elapsed = time.perf_counter() - start

            # Truncate for display
            display = response[:300]
            if len(response) > 300:
                display += "..."

            print(f"  A: {display}")
            print(f"  ({elapsed:.1f}s)")

            results[category].append({
                "prompt": prompt,
                "response": response,
                "time": round(elapsed, 1),
            })

    # Save results
    output_dir = Path("D:/AVA/experiments/exp4_finetune/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "finetuned" if adapter_path else "base"
    output_file = output_dir / f"demo_{tag}.json"
    output_file.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    adapter = ADAPTER_PATH if "--adapter" in sys.argv else None
    if "--checkpoint" in sys.argv:
        idx = sys.argv.index("--checkpoint")
        adapter = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
    run_demo(adapter)
