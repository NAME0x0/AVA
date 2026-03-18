"""Test the AVA agentic harness with Qwen3.5-2B."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.engine import AVAEngine, CalculatorTool, PythonTool, GenerationConfig

MODEL_PATH = "D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B"
MEMORY_PATH = "D:/AVA/experiments/exp4_finetune/memory"

print("Initializing AVA Engine...")
engine = AVAEngine(
    model_path=MODEL_PATH,
    memory_path=MEMORY_PATH,
    tools=[CalculatorTool(), PythonTool()],
    quantize_4bit=True,
)

print(f"VRAM: {engine.vram_report()}")
print("\n" + "=" * 60)
print("AVA Agentic Harness Test")
print("=" * 60)

config = GenerationConfig(
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
    enable_thinking=True,
)

# Test 1: Simple question
print("\n[Test 1] Simple question")
response = engine.chat("What is the square root of 144?", config)
print(f"Response: {response}")
engine.reset()

# Test 2: Tool use (calculator)
print("\n[Test 2] Calculator tool use")
response = engine.chat("Can you calculate 17 * 23 + 456 / 12 using the calculator tool?", config)
print(f"Response: {response}")
engine.reset()

# Test 3: Multi-turn conversation
print("\n[Test 3] Multi-turn conversation")
r1 = engine.chat("My name is Afsah and I'm building an AI system called AVA.", config)
print(f"Turn 1: {r1[:200]}")
r2 = engine.chat("What was my name again?", config)
print(f"Turn 2: {r2[:200]}")
engine.reset()

# Test 4: ARC-style reasoning
print("\n[Test 4] ARC-style reasoning")
response = engine.chat(
    "Which of the following is most likely to cause a decrease in the number of birds in a forest?\n"
    "A. An increase in sunlight\n"
    "B. A decrease in the number of predators\n"
    "C. An increase in the number of trees\n"
    "D. A decrease in the amount of available water\n\n"
    "Think step by step and give the answer.",
    config,
)
print(f"Response: {response[:300]}")
engine.reset()

# Test 5: Code execution
print("\n[Test 5] Python tool use")
response = engine.chat("Use the python tool to compute the first 10 Fibonacci numbers.", config)
print(f"Response: {response[:300]}")
engine.reset()

print("\n" + "=" * 60)
print("All tests complete!")
print(f"Memory entries: {len(engine.memory.memories) if engine.memory else 0}")
print(f"Final VRAM: {engine.vram_report()}")
