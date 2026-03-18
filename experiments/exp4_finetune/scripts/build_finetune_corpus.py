"""Build a rich fine-tuning corpus for AVA Experiment 4.

Combines existing AVA corpora with new agentic/tool-use examples.
"""
import json
import random
from pathlib import Path

random.seed(42)

OUTPUT_PATH = Path("D:/AVA/experiments/exp4_finetune/corpora")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

examples: list[dict] = []


# === 1. Load existing rich posttrain corpus ===
rich_path = Path("D:/AVA/corpora/ava_v3_rich_posttrain_v1/examples.jsonl")
if rich_path.exists():
    with open(rich_path, encoding="utf-8") as f:
        rich_examples = [json.loads(line) for line in f if line.strip()]
    # Filter to reasonable length for Qwen tokenizer (longer context available)
    kept = [ex for ex in rich_examples if len(ex.get("prompt", "") + ex.get("response", "")) < 3000]
    examples.extend(kept)
    print(f"Loaded {len(kept)} / {len(rich_examples)} from rich posttrain corpus")


# === 2. Generate tool-use training examples ===
tool_use_examples = [
    # Calculator tool
    {
        "prompt": "What is 847 * 23 + 156?",
        "response": 'Let me calculate that.\n\n<tool_call>{"name": "calculator", "arguments": {"expression": "847 * 23 + 156"}}</tool_call>\n\nThe result is 19,637.',
    },
    {
        "prompt": "Calculate the compound interest on $5000 at 8% annual rate for 3 years.",
        "response": 'I\'ll use the compound interest formula: A = P(1 + r)^n\n\n<tool_call>{"name": "calculator", "arguments": {"expression": "5000 * (1 + 0.08) ** 3"}}</tool_call>\n\nThe compound amount is $6,298.56, so the interest earned is $1,298.56.',
    },
    {
        "prompt": "What's the square root of 2048?",
        "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "sqrt(2048)"}}</tool_call>\n\nThe square root of 2048 is approximately 45.25.',
    },
    {
        "prompt": "If I drive 340 miles at 65 mph, how many hours will it take?",
        "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "340 / 65"}}</tool_call>\n\nIt will take approximately 5.23 hours, or about 5 hours and 14 minutes.',
    },
    {
        "prompt": "Convert 98.6 Fahrenheit to Celsius.",
        "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "(98.6 - 32) * 5 / 9"}}</tool_call>\n\n98.6°F is 37.0°C — normal human body temperature.',
    },
    # Python tool
    {
        "prompt": "Generate a list of prime numbers under 50.",
        "response": '<tool_call>{"name": "python", "arguments": {"code": "primes = []\\nfor n in range(2, 50):\\n    if all(n % i != 0 for i in range(2, int(n**0.5)+1)):\\n        primes.append(n)\\nprint(primes)"}}</tool_call>\n\nThe prime numbers under 50 are: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47].',
    },
    {
        "prompt": "Write a function to check if a string is a palindrome.",
        "response": '<tool_call>{"name": "python", "arguments": {"code": "def is_palindrome(s):\\n    s = s.lower().replace(\' \', \'\')\\n    return s == s[::-1]\\n\\ntest_cases = [\'racecar\', \'hello\', \'A man a plan a canal Panama\']\\nfor t in test_cases:\\n    print(f\\\"{t}: {is_palindrome(t)}\\\")"}}</tool_call>\n\nThe function works correctly: "racecar" and "A man a plan a canal Panama" are palindromes, while "hello" is not.',
    },
    {
        "prompt": "Sort these numbers: 45, 12, 78, 3, 56, 91, 23",
        "response": '<tool_call>{"name": "python", "arguments": {"code": "numbers = [45, 12, 78, 3, 56, 91, 23]\\nprint(sorted(numbers))"}}</tool_call>\n\nSorted: [3, 12, 23, 45, 56, 78, 91]',
    },
    # Memory search
    {
        "prompt": "What did we discuss about machine learning yesterday?",
        "response": 'Let me search my memory for our previous conversation.\n\n<tool_call>{"name": "search_memory", "arguments": {"query": "machine learning discussion"}}</tool_call>\n\nBased on my memory, we discussed the differences between supervised and unsupervised learning, and you were particularly interested in reinforcement learning for game AI.',
    },
    # Multi-tool use
    {
        "prompt": "I need to calculate the monthly payment for a $250,000 mortgage at 6.5% over 30 years, then write a Python function for it.",
        "response": 'I\'ll first calculate it, then write the function.\n\n<tool_call>{"name": "calculator", "arguments": {"expression": "250000 * (0.065/12 * (1 + 0.065/12)**360) / ((1 + 0.065/12)**360 - 1)"}}</tool_call>\n\nThe monthly payment is approximately $1,580.17.\n\nNow let me write a reusable function:\n\n<tool_call>{"name": "python", "arguments": {"code": "def monthly_payment(principal, annual_rate, years):\\n    r = annual_rate / 12\\n    n = years * 12\\n    payment = principal * (r * (1 + r)**n) / ((1 + r)**n - 1)\\n    return round(payment, 2)\\n\\nprint(f\\\"Monthly payment: ${monthly_payment(250000, 0.065, 30)}\\\")"}}</tool_call>\n\nThe function confirms: $1,580.17 per month for a $250,000 mortgage at 6.5% over 30 years.',
    },
]

# Replicate tool examples with variations
for ex in tool_use_examples:
    examples.append(ex)

print(f"Added {len(tool_use_examples)} tool-use examples")


# === 3. Generate agentic conversation examples ===
agentic_examples = [
    {
        "prompt": "Help me plan a study schedule for my biology exam next week.",
        "response": "I'll help you create a structured study plan. Let me break down the approach:\n\n**Day 1-2: Review & Identify Gaps**\n- Review your notes and textbook chapters\n- Identify topics you're weakest in\n- Create a topic list ranked by difficulty\n\n**Day 3-4: Deep Study**\n- Focus on your weakest topics first (spaced repetition)\n- Create flashcards for key terms\n- Practice drawing diagrams (cell structures, processes)\n\n**Day 5: Practice & Application**\n- Do practice questions from each chapter\n- Review past assignments and quizzes\n- Focus on application, not just memorization\n\n**Day 6: Review & Consolidate**\n- Quick review of all topics\n- Focus on connections between concepts\n- Teach the material to someone (or explain it aloud)\n\n**Day 7: Light Review**\n- Brief flashcard review\n- Relax and get good sleep\n\nWould you like me to customize this based on specific biology topics you're studying?",
    },
    {
        "prompt": "I keep making the same mistake in my code. I forget to handle null values. How can I prevent this?",
        "response": "This is one of the most common bugs in software development. Here are practical strategies:\n\n**1. Use strict type checking**\n- TypeScript: `strictNullChecks` in tsconfig\n- Python: type hints with `Optional[T]` and mypy\n- This catches null issues at compile/lint time\n\n**2. Defensive patterns**\n- Default values: `value = input ?? default`\n- Early returns: `if (x is None): return`\n- Optional chaining: `user?.address?.city`\n\n**3. Design patterns**\n- Use the Null Object pattern instead of actual nulls\n- Use Result/Either types for operations that might fail\n- Validate inputs at boundaries, trust internals\n\n**4. Process habits**\n- Add null-check to your personal code review checklist\n- Write tests that explicitly pass null/None inputs\n- Use linters that flag potential null dereferences\n\nThe real solution is making it impossible rather than remembering — strict typing catches what discipline misses.",
    },
    {
        "prompt": "Explain quantum entanglement like I'm 12 years old.",
        "response": "Imagine you have two magic coins. You put one in your pocket and mail the other to your friend in another city.\n\nHere's the weird part: when you flip your coin and get heads, your friend's coin ALWAYS lands on tails. And if yours lands tails, theirs is always heads. Every single time.\n\nThe really spooky thing? This happens instantly — even if your friend is on the other side of the planet. It's like the coins are secretly connected in a way we can't see.\n\nThat's basically quantum entanglement. Two tiny particles (way smaller than atoms) get \"linked\" together. When you measure one, you instantly know what the other one will be, no matter how far apart they are.\n\nEinstein thought this was so weird he called it \"spooky action at a distance.\" Scientists have now proven it's real — they've even done experiments with particles separated by over 1,000 kilometers.\n\nThe honest truth? Nobody fully understands WHY it works. We just know it does, and we're starting to use it to build super-secure communication systems and quantum computers.",
    },
]

for ex in agentic_examples:
    examples.append(ex)

print(f"Added {len(agentic_examples)} agentic conversation examples")


# === 4. Add reasoning chain examples ===
reasoning_examples = [
    {
        "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "response": "Let me think through this carefully — it's a classic trick question.\n\nLet's call the ball's price x.\nThe bat costs $1.00 more, so: bat = x + $1.00\n\nTotal: x + (x + $1.00) = $1.10\n2x + $1.00 = $1.10\n2x = $0.10\nx = $0.05\n\n**The ball costs $0.05 (5 cents).**\n\nVerification: Ball = $0.05, Bat = $1.05. Total = $1.10. Bat - Ball = $1.00. ✓\n\nThe intuitive (wrong) answer is $0.10, but that would make the bat $1.10 and the total $1.20.",
    },
    {
        "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "response": "This is another question where the intuitive answer (100 minutes) is wrong.\n\nLet's reason step by step:\n\n1. 5 machines make 5 widgets in 5 minutes\n2. That means each machine makes 1 widget in 5 minutes\n3. So 100 machines, each making 1 widget in 5 minutes...\n4. = 100 widgets in 5 minutes\n\n**It takes 5 minutes.**\n\nThe key insight: adding more machines increases throughput (widgets per minute), not the time each machine needs. Each machine independently produces 1 widget every 5 minutes, regardless of how many other machines are running.",
    },
    {
        "prompt": "Three friends split a $30 hotel bill evenly, paying $10 each. The receptionist realizes the bill should be $25 and gives $5 to the bellboy to return. The bellboy keeps $2 and gives back $1 to each friend. Now each friend paid $9 (total $27), plus the bellboy has $2 = $29. Where did the extra dollar go?",
        "response": "This is a famous riddle that uses a mathematical misdirection. There is no missing dollar.\n\nLet me untangle it:\n\n**What actually happened:**\n- Friends paid: $30 total\n- Hotel kept: $25\n- Bellboy kept: $2\n- Friends got back: $3\n- Total: $25 + $2 + $3 = $30 ✓\n\n**Where the trick is:**\nThe riddle adds $27 + $2 = $29, but this is WRONG accounting.\n\nThe $27 the friends effectively paid already INCLUDES the bellboy's $2:\n- $27 = $25 (hotel) + $2 (bellboy)\n\nYou should subtract the bellboy's $2, not add it:\n- $27 - $2 = $25 (hotel's actual bill) ✓\n\nOr equivalently: $27 (paid) + $3 (returned) = $30 (original) ✓\n\nThe riddle tricks you by adding two numbers that shouldn't be added together.",
    },
]

for ex in reasoning_examples:
    examples.append(ex)

print(f"Added {len(reasoning_examples)} reasoning examples")


# === 5. Add identity/personality examples ===
identity_examples = [
    {
        "prompt": "Who are you?",
        "response": "I'm AVA — an AI assistant built for thoughtful, step-by-step reasoning. I can use tools like a calculator and Python interpreter when needed, and I maintain memory across conversations to serve you better. I aim to be concise, accurate, and genuinely helpful.",
    },
    {
        "prompt": "What can you do?",
        "response": "I can help with:\n\n- **Reasoning & analysis**: Break down complex problems step by step\n- **Math & calculations**: Using my calculator tool for precise answers\n- **Code**: Write, explain, and debug code using my Python tool\n- **Research & knowledge**: Answer questions across science, history, technology, and more\n- **Memory**: Remember important details from our conversations\n- **Planning**: Help organize tasks, schedules, and approaches to problems\n\nI work best when you give me clear context about what you need. What would you like help with?",
    },
    {
        "prompt": "Tell me a joke.",
        "response": "Why do programmers prefer dark mode?\n\nBecause light attracts bugs.",
    },
]

for ex in identity_examples:
    examples.append(ex)

print(f"Added {len(identity_examples)} identity examples")


# === Shuffle and write ===
random.shuffle(examples)

output_file = OUTPUT_PATH / "ava_exp4_finetune_v1.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"\nTotal corpus: {len(examples)} examples")
print(f"Written to: {output_file}")
print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")

# Stats
avg_len = sum(len(ex.get("prompt", "") + ex.get("response", "")) for ex in examples) / len(examples)
max_len = max(len(ex.get("prompt", "") + ex.get("response", "")) for ex in examples)
print(f"Avg example length: {avg_len:.0f} chars")
print(f"Max example length: {max_len} chars")
