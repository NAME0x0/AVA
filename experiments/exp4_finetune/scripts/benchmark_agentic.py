"""Agentic benchmark: GSM8K with tool use.

Tests whether the fine-tuned model can use calculator/python tools
to solve math problems it might otherwise get wrong. This is the
real power of the agentic harness — 2B models shouldn't do mental
arithmetic when they have tools.

Compares: raw generation vs agentic (with tool execution).
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ava.external_benchmarks import load_external_benchmark_tasks
from harness.engine import CalculatorTool, PythonTool

BASE_MODEL = "D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B"
RESULTS_DIR = Path("D:/AVA/experiments/exp4_finetune/results")

TOOL_SYSTEM_PROMPT = """You are AVA, an intelligent AI assistant with tool-use capabilities.
When solving math problems, use the calculator tool for precise computation.

You have access to the following tools:
- calculator: Evaluate a mathematical expression. Supports +, -, *, /, **, sqrt, abs, round.
- python: Execute Python code and return the output. Use print() to show results.

To use a tool, write: <tool_call>{"name": "tool_name", "arguments": {"arg": "value"}}</tool_call>
Wait for the tool result before giving your final answer.

Always end with: The answer is [number]."""


@dataclass
class AgenticConfig:
    model_path: str = BASE_MODEL
    adapter_path: str | None = None
    gsm8k_limit: int = 50
    max_new_tokens: int = 512
    max_tool_rounds: int = 3


def load_model(config: AgenticConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, dtype=torch.bfloat16,
    )
    name = Path(config.model_path).name
    if config.adapter_path:
        ac = Path(config.adapter_path) / "adapter_config.json"
        if ac.exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, config.adapter_path)
            model = model.merge_and_unload()
            name = Path(config.adapter_path).name + " (LoRA)"
    model.eval()
    return model, tokenizer, name


def parse_tool_calls(text):
    """Extract tool calls from generated text."""
    calls = []
    for match in re.finditer(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        inner = match.group(1).strip()
        try:
            calls.append(json.loads(inner))
            continue
        except (json.JSONDecodeError, ValueError):
            pass
        # Qwen native XML format
        func_match = re.search(r'<function=(\w+)>(.*?)</function>', inner, re.DOTALL)
        if func_match:
            func_name = func_match.group(1)
            params_text = func_match.group(2)
            args = {}
            for param in re.finditer(r'<parameter=(\w+)>\n?(.*?)\n?</parameter>', params_text, re.DOTALL):
                args[param.group(1)] = param.group(2).strip()
            calls.append({"name": func_name, "arguments": args})
    return calls


def execute_tool(call):
    """Execute a tool call and return the result string."""
    tools = {
        "calculator": CalculatorTool(),
        "python": PythonTool(),
    }
    name = call.get("name", "")
    args = call.get("arguments", {})
    if name not in tools:
        return f"Unknown tool: {name}"
    try:
        return tools[name].execute(**args)
    except Exception as e:
        return f"Error: {e}"


def extract_numeric(text):
    """Extract final numeric answer."""
    patterns = [
        r"(?:the answer is|the final answer is)\s*\$?([-\d,]+\.?\d*)",
        r"(?:####)\s*\$?([-\d,]+\.?\d*)",
        r"(?:final answer:?)\s*\$?([-\d,]+\.?\d*)",
        r"(?:answer:)\s*\$?([-\d,]+\.?\d*)",
        r"(?:therefore|thus|so|hence)[,:]?\s+\$?([-\d,]+\.?\d*)",
        r"\\boxed\{([-\d,]+\.?\d*)\}",
        r"=\s*\$?([-\d,]+\.?\d*)\s*$",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
        if m:
            val = m.group(1).replace(",", "").rstrip(".")
            if val and val != "-":
                return val
    nums = re.findall(r"[-\d,]+\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


@torch.no_grad()
def generate_raw(model, tokenizer, prompt, max_tokens=768):
    """Plain generation without tool execution."""
    messages = [
        {"role": "user", "content": prompt + "\n\nSolve this step by step and give the final numeric answer."},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_tokens, temperature=0.1, do_sample=True,
        top_p=0.95, pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def generate_agentic(model, tokenizer, prompt, config: AgenticConfig):
    """Agentic generation with tool execution loop."""
    messages = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": prompt + "\n\nSolve this step by step using tools when helpful."},
    ]

    full_response = ""
    for round_idx in range(config.max_tool_rounds):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=config.max_new_tokens, temperature=0.1,
            do_sample=True, top_p=0.95, pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        full_response += response + "\n"

        # Check for tool calls
        tool_calls = parse_tool_calls(response)
        if not tool_calls:
            break  # No tool calls, this is the final answer

        # Add assistant response and tool results to messages
        messages.append({"role": "assistant", "content": response})
        for call in tool_calls:
            result = execute_tool(call)
            messages.append({"role": "user", "content": f"Tool result: {result}"})

    return full_response.strip()


def check_match(predicted, expected):
    """Check if numeric answer matches."""
    if not predicted:
        return False
    try:
        return abs(float(predicted) - float(expected)) < 0.01
    except (ValueError, TypeError):
        return str(predicted) == str(expected)


def run_agentic_benchmark(config: AgenticConfig | None = None):
    config = config or AgenticConfig()

    print("=" * 60)
    print("AVA Agentic Benchmark: GSM8K with Tool Use")
    print("=" * 60)

    model, tokenizer, model_name = load_model(config)
    print(f"Model: {model_name}")
    print(f"GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"GSM8K limit: {config.gsm8k_limit}\n")

    tasks = load_external_benchmark_tasks("gsm8k", split=None, limit=config.gsm8k_limit, offset=0)

    raw_correct = 0
    agentic_correct = 0
    results = []

    for i, task in enumerate(tasks):
        expected = task.expected.strip()

        # Raw generation
        raw_resp = generate_raw(model, tokenizer, task.prompt, max_tokens=768)
        raw_pred = extract_numeric(raw_resp)
        raw_ok = check_match(raw_pred, expected)
        if raw_ok:
            raw_correct += 1

        # Agentic generation (with tools)
        agent_resp = generate_agentic(model, tokenizer, task.prompt, config)
        agent_pred = extract_numeric(agent_resp)
        agent_ok = check_match(agent_pred, expected)
        if agent_ok:
            agentic_correct += 1

        used_tools = "<tool_call>" in agent_resp

        results.append({
            "task_id": task.task_id,
            "expected": expected,
            "raw_pred": raw_pred,
            "raw_ok": raw_ok,
            "agent_pred": agent_pred,
            "agent_ok": agent_ok,
            "used_tools": used_tools,
            "raw_snippet": raw_resp[:200],
            "agent_snippet": agent_resp[:300],
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(tasks)}] Raw: {raw_correct}/{i+1} ({raw_correct/(i+1)*100:.0f}%) "
                  f"| Agentic: {agentic_correct}/{i+1} ({agentic_correct/(i+1)*100:.0f}%)")

    # Summary
    n = len(tasks)
    raw_acc = raw_correct / n
    agent_acc = agentic_correct / n
    tools_used = sum(1 for r in results if r["used_tools"])
    tool_helped = sum(1 for r in results if r["agent_ok"] and not r["raw_ok"])
    tool_hurt = sum(1 for r in results if not r["agent_ok"] and r["raw_ok"])

    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"  Raw GSM8K:     {raw_correct}/{n} = {raw_acc*100:.1f}%")
    print(f"  Agentic GSM8K: {agentic_correct}/{n} = {agent_acc*100:.1f}%")
    print(f"  Delta:         {(agent_acc-raw_acc)*100:+.1f}pp")
    print(f"  Tool calls:    {tools_used}/{n} problems used tools")
    print(f"  Tool helped:   {tool_helped} (wrong→right)")
    print(f"  Tool hurt:     {tool_hurt} (right→wrong)")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = Path(config.adapter_path).name if config.adapter_path else "base"
    out = RESULTS_DIR / f"agentic_gsm8k_{tag}.json"
    summary = {
        "model": model_name,
        "raw_accuracy": round(raw_acc, 3),
        "agentic_accuracy": round(agent_acc, 3),
        "delta_pp": round((agent_acc - raw_acc) * 100, 1),
        "tools_used": tools_used,
        "tool_helped": tool_helped,
        "tool_hurt": tool_hurt,
        "results": results,
    }
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved to: {out}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    run_agentic_benchmark(AgenticConfig(
        model_path=args.model,
        adapter_path=args.adapter,
        gsm8k_limit=args.limit,
    ))
