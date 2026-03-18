"""Interactive CLI for the AVA agentic harness."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import AVAEngine, CalculatorTool, PythonTool, GenerationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ava-chat", description="AVA Experiment 4 - Agentic Chat")
    parser.add_argument("--model", default="D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B",
                        help="Path to model or LoRA adapter")
    parser.add_argument("--base-model", default=None,
                        help="Base model path when using LoRA adapter")
    parser.add_argument("--memory", default="D:/AVA/experiments/exp4_finetune/memory",
                        help="Path to memory store")
    parser.add_argument("--no-tools", action="store_true", help="Disable tool use")
    parser.add_argument("--no-think", action="store_true", help="Disable thinking mode")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    tools = []
    if not args.no_tools:
        tools = [CalculatorTool(), PythonTool()]

    # Determine model vs adapter paths
    model_path = args.model
    adapter_path = None
    if args.base_model:
        model_path = args.base_model
        adapter_path = args.model

    print("Loading AVA Engine...")
    engine = AVAEngine(
        model_path=model_path,
        adapter_path=adapter_path,
        memory_path=args.memory,
        tools=tools,
        quantize_4bit=True,
    )

    vram = engine.vram_report()
    print(f"AVA ready. VRAM: {vram.get('allocated_gb', '?')} GB / {vram.get('total_gb', '?')} GB")
    print(f"Tools: {', '.join(engine.tools.keys()) if engine.tools else 'none'}")
    print(f"Type 'quit' to exit, 'reset' to clear conversation, 'memory' to view memory.\n")

    config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=args.temperature > 0,
        enable_thinking=not args.no_think,
    )

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            engine.reset()
            print("[Conversation reset]")
            continue
        if user_input.lower() == "memory":
            if engine.memory:
                recent = engine.memory.get_recent(5)
                for m in recent:
                    print(f"  [{m['timestamp']}] {m['content'][:100]}")
            else:
                print("  No memory store.")
            continue

        response = engine.chat(user_input, config)
        print(f"\nAVA: {response}\n")


if __name__ == "__main__":
    main()
