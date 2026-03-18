"""Quick inference test for Qwen3.5-2B on 4GB VRAM."""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Vocab size: {tokenizer.vocab_size}")

print("Loading model in 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# Print memory usage
mem_alloc = torch.cuda.memory_allocated() / 1e9
mem_reserved = torch.cuda.memory_reserved() / 1e9
print(f"GPU memory allocated: {mem_alloc:.2f} GB")
print(f"GPU memory reserved: {mem_reserved:.2f} GB")
print(f"Model loaded successfully!")

# Quick generation test
prompt = "What is the capital of France? Answer in one word:"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

print(f"\nPrompt: {prompt}")
print("Generating...")

start = time.perf_counter()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
        temperature=1.0,
    )
elapsed = time.perf_counter() - start
generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
n_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
tok_per_sec = n_tokens / elapsed if elapsed > 0 else 0

print(f"Response: {generated}")
print(f"Generated {n_tokens} tokens in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)")

# ARC-style question test
arc_prompt = """Which of the following statements best explains why magnets usually stick to a refrigerator door?

Options:
A. The refrigerator door is smooth.
B. The refrigerator door contains iron.
C. The refrigerator door is a good conductor.
D. The refrigerator door has electric wires in it.

Reply with only the correct option label."""

messages = [{"role": "user", "content": arc_prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

start = time.perf_counter()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
    )
elapsed = time.perf_counter() - start
generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
n_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
tok_per_sec = n_tokens / elapsed if elapsed > 0 else 0

print(f"\nARC Question:")
print(f"Response: {generated}")
print(f"Generated {n_tokens} tokens in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)")

# GSM8K-style question test
gsm_prompt = """Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

Reply with only the final numerical answer."""

messages = [{"role": "user", "content": gsm_prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

start = time.perf_counter()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
    )
elapsed = time.perf_counter() - start
generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
n_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
tok_per_sec = n_tokens / elapsed if elapsed > 0 else 0

print(f"\nGSM8K Question:")
print(f"Response: {generated}")
print(f"Generated {n_tokens} tokens in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)")

# Final memory report
print(f"\nFinal GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
