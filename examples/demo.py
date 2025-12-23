#!/usr/bin/env python3
"""
AVA Demo Script
===============

This script demonstrates how to use AVA's Python API for:
- Basic chat interactions
- Deep thinking mode (Cortex)
- Tool usage
- Response inspection

Prerequisites:
    1. Ollama must be running: ollama serve
    2. Required model: ollama pull gemma3:4b
    3. Install AVA: pip install -e .

Usage:
    python examples/demo.py
    python examples/demo.py --deep  # Force deep thinking for all queries
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ava import AVA


async def basic_demo():
    """Demonstrate basic chat functionality."""
    print("\n" + "=" * 60)
    print("AVA Basic Demo")
    print("=" * 60)

    ava = AVA()

    try:
        print("\n[*] Starting AVA...")
        success = await ava.start()

        if not success:
            print("[!] Failed to start AVA. Is Ollama running?")
            print("    Try: ollama serve")
            return False

        print("[+] AVA started successfully!\n")

        # Simple greeting
        print("-" * 40)
        print("Example 1: Simple Chat")
        print("-" * 40)
        response = await ava.chat("Hello! What can you help me with?")
        print(f"AVA: {response.text}")
        print(f"  [Cortex used: {response.used_cortex}]")
        print(f"  [Confidence: {response.confidence:.2f}]")
        print(f"  [Cognitive State: {response.cognitive_state}]")

        # Math question
        print("\n" + "-" * 40)
        print("Example 2: Quick Calculation")
        print("-" * 40)
        response = await ava.chat("What is 15 * 7 + 23?")
        print(f"AVA: {response.text}")
        print(f"  [Cortex used: {response.used_cortex}]")
        if response.tools_used:
            print(f"  [Tools: {', '.join(response.tools_used)}]")

        # Deep thinking
        print("\n" + "-" * 40)
        print("Example 3: Deep Thinking Mode")
        print("-" * 40)
        print("[*] This uses the Cortex for complex reasoning...")
        response = await ava.think("Why is the sky blue? Explain briefly.")
        print(f"AVA: {response.text}")
        print(f"  [Cortex used: {response.used_cortex}]")
        print(f"  [Thinking time: {response.thinking_time_ms:.0f}ms]")

        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n[!] Error: {e}")
        return False

    finally:
        await ava.stop()
        print("\n[*] AVA stopped.")


async def interactive_demo():
    """Interactive chat session."""
    print("\n" + "=" * 60)
    print("AVA Interactive Mode")
    print("=" * 60)
    print("Commands:")
    print("  /think <msg> - Force deep thinking")
    print("  /status      - Show AVA status")
    print("  /quit        - Exit")
    print("=" * 60)

    ava = AVA()

    try:
        print("\n[*] Starting AVA...")
        success = await ava.start()

        if not success:
            print("[!] Failed to start AVA. Is Ollama running?")
            return

        print("[+] Ready! Type your message:\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() == "/quit":
                print("Goodbye!")
                break

            if user_input.lower() == "/status":
                print(f"  Running: {ava.is_running}")
                continue

            # Check for /think command
            if user_input.lower().startswith("/think "):
                message = user_input[7:]
                response = await ava.think(message)
            else:
                response = await ava.chat(user_input)

            print(f"AVA: {response.text}")
            print(f"  [{response.cognitive_state} | Cortex: {response.used_cortex}]\n")

    finally:
        await ava.stop()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AVA Demo Script")
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--deep", "-d",
        action="store_true",
        help="Force deep thinking for all queries"
    )
    args = parser.parse_args()

    if args.interactive:
        await interactive_demo()
    else:
        success = await basic_demo()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Windows event loop policy
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
