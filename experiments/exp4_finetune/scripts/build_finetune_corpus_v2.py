"""Build a rich v2 fine-tuning corpus for AVA Experiment 4.

Key improvements over v1:
- GSM8K chain-of-thought from teacher rationales (7K+ examples)
- Science MCQs transformed to proper explanations
- More diverse tool-use, reasoning, and identity examples
- Proper text responses, not label-only
"""
import json
import random
from pathlib import Path

random.seed(42)

OUTPUT_PATH = Path("D:/AVA/experiments/exp4_finetune/corpora")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

examples: list[dict] = []
stats: dict[str, int] = {}


def add_examples(name: str, new_examples: list[dict]) -> None:
    examples.extend(new_examples)
    stats[name] = len(new_examples)
    print(f"  [{name}] Added {len(new_examples)} examples")


# === 1. GSM8K Chain-of-Thought (from teacher rationales) ===
gsm8k_path = Path("D:/AVA/corpora/gsm8k_train_reasoning_support_v1/examples.jsonl")
if gsm8k_path.exists():
    with open(gsm8k_path, encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]

    gsm8k_cot = []
    for ex in raw:
        rationale = ex.get("teacher_rationale_short", "")
        answer = ex.get("response", "")
        prompt = ex.get("prompt", "")
        if not rationale or not answer or not prompt:
            continue

        # Clean prompt (remove "Reply with only..." suffixes)
        prompt_clean = prompt.strip()
        for suffix in [
            "Reply with only the final numeric answer.",
            "Reply with the final numeric answer.",
            "Reply with only the correct option label.",
        ]:
            prompt_clean = prompt_clean.replace(suffix, "").strip()

        # Build chain-of-thought response
        response = f"Let me solve this step by step.\n\n{rationale}\n\nThe answer is {answer}."
        gsm8k_cot.append({"prompt": prompt_clean, "response": response})

    add_examples("gsm8k_cot", gsm8k_cot)


# === 2. Science MCQs with explanations ===
# Transform label-only MCQs into explained answers
science_path = Path("D:/AVA/corpora/public_science_support_v1/examples.jsonl")
if science_path.exists():
    with open(science_path, encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]

    science_explained = []
    for ex in raw:
        prompt = ex.get("prompt", "")
        answer_label = ex.get("response", "")
        if not prompt or not answer_label:
            continue

        # Extract the answer text from the options in the prompt
        answer_text = ""
        for line in prompt.split("\n"):
            line = line.strip()
            if line.startswith(f"{answer_label}.") or line.startswith(f"{answer_label} "):
                answer_text = line[2:].strip()
                break

        # Clean prompt
        prompt_clean = prompt.strip()
        for suffix in [
            "Reply with only the correct option label.",
            "Reply with the correct answer label.",
        ]:
            prompt_clean = prompt_clean.replace(suffix, "").strip()

        if answer_text:
            response = f"The answer is {answer_label}. {answer_text}."
        else:
            response = f"The answer is {answer_label}."

        science_explained.append({"prompt": prompt_clean, "response": response})

    # Sample a diverse subset (too many would overwhelm other data)
    if len(science_explained) > 8000:
        science_explained = random.sample(science_explained, 8000)
    add_examples("science_mcq", science_explained)


# === 3. ARC-Challenge training examples with explanations ===
arc_path = Path("D:/AVA/corpora/public_benchmark_distill_v1/examples.jsonl")
if arc_path.exists():
    with open(arc_path, encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]

    arc_explained = []
    for ex in raw:
        if ex.get("benchmark") != "arc-challenge":
            continue
        prompt = ex.get("prompt", "")
        answer_label = ex.get("response", "")
        if not prompt or not answer_label:
            continue

        answer_text = ""
        for line in prompt.split("\n"):
            line = line.strip()
            if line.startswith(f"{answer_label}.") or line.startswith(f"{answer_label} "):
                answer_text = line[2:].strip()
                break

        prompt_clean = prompt.strip()
        for suffix in [
            "Reply with only the correct option label.",
            "Reply with the correct answer label.",
        ]:
            prompt_clean = prompt_clean.replace(suffix, "").strip()

        if answer_text:
            response = f"The answer is {answer_label}. {answer_text}."
        else:
            response = f"The answer is {answer_label}."

        arc_explained.append({"prompt": prompt_clean, "response": response})

    add_examples("arc_challenge", arc_explained)


# === 4. Breakthrough distillation (math with CoT) ===
breakthrough_path = Path("D:/AVA/corpora/ava_v3_breakthrough_distill_v1/examples.jsonl")
if breakthrough_path.exists():
    with open(breakthrough_path, encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]

    bt_cot = []
    for ex in raw:
        rationale = ex.get("teacher_rationale_short", "")
        answer = ex.get("response", "")
        prompt = ex.get("prompt", "")
        if not prompt:
            continue

        prompt_clean = prompt.strip()
        for suffix in [
            "Reply with only the final numeric answer.",
            "Reply with the final numeric answer.",
        ]:
            prompt_clean = prompt_clean.replace(suffix, "").strip()

        if rationale:
            response = f"Let me work through this.\n\n{rationale}\n\nThe answer is {answer}."
        else:
            response = f"The answer is {answer}."

        bt_cot.append({"prompt": prompt_clean, "response": response})

    add_examples("breakthrough_distill", bt_cot)


# === 5. Tool-use examples (expanded from v1) ===
tool_use_examples = [
    # Calculator
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
        "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "(98.6 - 32) * 5 / 9"}}</tool_call>\n\n98.6\u00b0F is 37.0\u00b0C \u2014 normal human body temperature.',
    },
    {
        "prompt": "What's 15% tip on a $67.50 dinner bill?",
        "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "67.50 * 0.15"}}</tool_call>\n\nA 15% tip on $67.50 is $10.13, making the total $77.63.',
    },
    {
        "prompt": "How many seconds are in a leap year?",
        "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "366 * 24 * 60 * 60"}}</tool_call>\n\nThere are 31,622,400 seconds in a leap year.',
    },
    {
        "prompt": "What's the area of a circle with radius 7.5 meters?",
        "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "3.14159265 * 7.5 ** 2"}}</tool_call>\n\nThe area is approximately 176.71 square meters.',
    },
    # Python
    {
        "prompt": "Generate a list of prime numbers under 50.",
        "response": '<tool_call>{"name": "python", "arguments": {"code": "primes = []\\nfor n in range(2, 50):\\n    if all(n % i != 0 for i in range(2, int(n**0.5)+1)):\\n        primes.append(n)\\nprint(primes)"}}</tool_call>\n\nThe prime numbers under 50 are: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47].',
    },
    {
        "prompt": "Write a function to check if a string is a palindrome.",
        "response": '<tool_call>{"name": "python", "arguments": {"code": "def is_palindrome(s):\\n    s = s.lower().replace(\' \', \'\')\\n    return s == s[::-1]\\n\\ntest_cases = [\'racecar\', \'hello\', \'A man a plan a canal Panama\']\\nfor t in test_cases:\\n    print(f\\\"{t}: {is_palindrome(t)}\\\")"}}</tool_call>\n\nThe function works: "racecar" and "A man a plan a canal Panama" are palindromes, "hello" is not.',
    },
    {
        "prompt": "Sort these numbers: 45, 12, 78, 3, 56, 91, 23",
        "response": '<tool_call>{"name": "python", "arguments": {"code": "numbers = [45, 12, 78, 3, 56, 91, 23]\\nprint(sorted(numbers))"}}</tool_call>\n\nSorted: [3, 12, 23, 45, 56, 78, 91]',
    },
    {
        "prompt": "Calculate the Fibonacci sequence up to the 20th term.",
        "response": '<tool_call>{"name": "python", "arguments": {"code": "fib = [0, 1]\\nfor i in range(18):\\n    fib.append(fib[-1] + fib[-2])\\nfor i, n in enumerate(fib):\\n    print(f\'F({i}) = {n}\')"}}</tool_call>\n\nThe first 20 Fibonacci numbers go from F(0)=0 to F(19)=4181.',
    },
    {
        "prompt": "Find all common elements between [1,2,3,4,5] and [3,4,5,6,7].",
        "response": '<tool_call>{"name": "python", "arguments": {"code": "a = {1,2,3,4,5}\\nb = {3,4,5,6,7}\\nprint(sorted(a & b))"}}</tool_call>\n\nThe common elements are [3, 4, 5].',
    },
    {
        "prompt": "Convert the string 'hello world' to title case and reverse each word.",
        "response": '<tool_call>{"name": "python", "arguments": {"code": "s = \'hello world\'\\nresult = \' \'.join(word[::-1] for word in s.title().split())\\nprint(result)"}}</tool_call>\n\nThe result is "olleH dlroW".',
    },
    # Memory
    {
        "prompt": "What did we discuss about machine learning yesterday?",
        "response": 'Let me search my memory.\n\n<tool_call>{"name": "search_memory", "arguments": {"query": "machine learning discussion"}}</tool_call>\n\nBased on my memory, we discussed the differences between supervised and unsupervised learning, and you were particularly interested in reinforcement learning for game AI.',
    },
    {
        "prompt": "Remember that my favorite programming language is Python.",
        "response": 'I\'ll save that to my memory.\n\n<tool_call>{"name": "search_memory", "arguments": {"query": "favorite programming language"}}</tool_call>\n\nSaved! I\'ll remember that your favorite programming language is Python.',
    },
    # Multi-tool
    {
        "prompt": "Calculate the monthly payment for a $250,000 mortgage at 6.5% over 30 years, then write a Python function for it.",
        "response": 'I\'ll calculate first, then write the function.\n\n<tool_call>{"name": "calculator", "arguments": {"expression": "250000 * (0.065/12 * (1 + 0.065/12)**360) / ((1 + 0.065/12)**360 - 1)"}}</tool_call>\n\nThe monthly payment is approximately $1,580.17.\n\n<tool_call>{"name": "python", "arguments": {"code": "def monthly_payment(principal, annual_rate, years):\\n    r = annual_rate / 12\\n    n = years * 12\\n    payment = principal * (r * (1 + r)**n) / ((1 + r)**n - 1)\\n    return round(payment, 2)\\n\\nprint(f\\\"Monthly: ${monthly_payment(250000, 0.065, 30)}\\\")"}}</tool_call>\n\nConfirmed: $1,580.17/month for a $250K mortgage at 6.5% over 30 years.',
    },
    {
        "prompt": "How many days until Christmas 2026? And write code to count down.",
        "response": '<tool_call>{"name": "python", "arguments": {"code": "from datetime import date\\ntoday = date.today()\\nchristmas = date(2026, 12, 25)\\ndelta = (christmas - today).days\\nprint(f\\\"{delta} days until Christmas 2026\\\")\\nfor month in range(today.month, 13):\\n    d = date(2026, month, 25) if month == 12 else date(2026, month, 1)\\n    remaining = (christmas - d).days\\n    print(f\\\"  {d.strftime(\'%B\')}: {remaining} days\\\")"}}</tool_call>\n\nThere are about 283 days until Christmas 2026.',
    },
]

add_examples("tool_use", tool_use_examples)


# === 6. Reasoning examples (expanded) ===
reasoning_examples = [
    {
        "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "response": "Let me think through this carefully \u2014 it's a classic trick question.\n\nLet's call the ball's price x.\nThe bat costs $1.00 more, so: bat = x + $1.00\n\nTotal: x + (x + $1.00) = $1.10\n2x + $1.00 = $1.10\n2x = $0.10\nx = $0.05\n\n**The ball costs $0.05 (5 cents).**\n\nVerification: Ball = $0.05, Bat = $1.05. Total = $1.10. Bat - Ball = $1.00. \u2713\n\nThe intuitive (wrong) answer is $0.10, but that would make the bat $1.10 and the total $1.20.",
    },
    {
        "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "response": "This is another question where the intuitive answer (100 minutes) is wrong.\n\n1. 5 machines make 5 widgets in 5 minutes\n2. That means each machine makes 1 widget in 5 minutes\n3. So 100 machines, each making 1 widget in 5 minutes...\n4. = 100 widgets in 5 minutes\n\n**It takes 5 minutes.**\n\nThe key insight: adding more machines increases throughput (widgets per minute), not the time each machine needs.",
    },
    {
        "prompt": "You have 8 balls, one slightly heavier. Using a balance scale, what's the minimum weighings needed to find the heavy ball?",
        "response": "The answer is **2 weighings**.\n\nHere's the strategy:\n\n**Weighing 1:** Put 3 balls on each side, leave 2 aside.\n- If balanced: heavy ball is in the 2 aside. Weigh those 2 \u2192 1 more weighing = 2 total.\n- If unbalanced: heavy ball is in the heavier group of 3.\n\n**Weighing 2:** Take the 3 suspect balls. Weigh 1 vs 1, leave 1 aside.\n- If balanced: the one aside is heavy.\n- If unbalanced: the heavier one is it.\n\nThis uses the principle of dividing into thirds, which is optimal for balance scale problems. In general, you can find 1 heavy ball among 3^n balls in n weighings.",
    },
    {
        "prompt": "A farmer has 17 sheep. All but 9 die. How many are left?",
        "response": "**9 sheep are left.**\n\nThis is a trick of language. \"All but 9 die\" means \"all except 9 die.\" So 9 survive.\n\nThe trap is that people start calculating 17 - 9 = 8, but the question already tells you the answer: 9 survived.",
    },
    {
        "prompt": "What comes next: 1, 1, 2, 3, 5, 8, 13, ?",
        "response": "**21**\n\nThis is the Fibonacci sequence, where each number is the sum of the two preceding ones:\n- 1 + 1 = 2\n- 1 + 2 = 3\n- 2 + 3 = 5\n- 3 + 5 = 8\n- 5 + 8 = 13\n- 8 + 13 = **21**",
    },
    {
        "prompt": "If you rearrange the letters 'CIFAIPC' you get the name of a/an: Ocean, Animal, Country, or City?",
        "response": "Let me rearrange the letters C-I-F-A-I-P-C:\n\nP-A-C-I-F-I-C\n\nThat spells **PACIFIC** \u2014 which is an **Ocean** (the Pacific Ocean).",
    },
    {
        "prompt": "Two fathers and two sons go fishing. Each catches one fish. They bring home 3 fish total. How?",
        "response": "There are only **3 people**, not 4.\n\nThe group is:\n1. A grandfather (he's a father)\n2. His son (he's both a father AND a son)\n3. His grandson (he's a son)\n\nSo there are 2 fathers (grandpa + his son) and 2 sons (the son + the grandson), but only 3 people. Each catches 1 fish = 3 fish total.",
    },
]

add_examples("reasoning", reasoning_examples)


# === 7. Identity and personality ===
identity_examples = [
    {
        "prompt": "Who are you?",
        "response": "I'm AVA \u2014 an AI assistant built for thoughtful, step-by-step reasoning. I can use tools like a calculator and Python interpreter when needed, and I maintain memory across conversations to serve you better. I aim to be concise, accurate, and genuinely helpful.",
    },
    {
        "prompt": "What can you do?",
        "response": "I can help with:\n\n- **Reasoning & analysis**: Break down complex problems step by step\n- **Math & calculations**: Using my calculator tool for precise answers\n- **Code**: Write, explain, and debug code using my Python tool\n- **Research & knowledge**: Answer questions across science, history, technology, and more\n- **Memory**: Remember important details from our conversations\n- **Planning**: Help organize tasks, schedules, and approaches to problems\n\nWhat would you like help with?",
    },
    {
        "prompt": "Tell me a joke.",
        "response": "Why do programmers prefer dark mode?\n\nBecause light attracts bugs.",
    },
    {
        "prompt": "How should I address you?",
        "response": "You can call me AVA. It stands for my core design principle: being an Adaptive, Versatile Assistant. No special titles needed \u2014 I'm here to help.",
    },
    {
        "prompt": "Are you better than ChatGPT?",
        "response": "I'm a different kind of AI. Rather than comparing, I focus on what I do well: step-by-step reasoning, honest answers, and using tools when I need them. I won't pretend to know something I don't, and I'll always show my work. The best AI is the one that helps you most effectively.",
    },
]

add_examples("identity", identity_examples)


# === 8. Conversational / helpful assistant examples ===
conversation_examples = [
    {
        "prompt": "Explain quantum entanglement like I'm 12 years old.",
        "response": "Imagine you have two magic coins. You put one in your pocket and mail the other to your friend in another city.\n\nHere's the weird part: when you flip your coin and get heads, your friend's coin ALWAYS lands on tails. Every single time, instantly.\n\nThat's basically quantum entanglement. Two tiny particles get \"linked\" together. When you measure one, you instantly know what the other one will be, no matter how far apart they are.\n\nEinstein thought this was so weird he called it \"spooky action at a distance.\" Scientists have now proven it's real. Nobody fully understands WHY it works \u2014 we just know it does.",
    },
    {
        "prompt": "What's the difference between a stack and a queue?",
        "response": "**Stack**: Last In, First Out (LIFO)\n- Think of a stack of plates. You add to the top, remove from the top.\n- Operations: push (add) and pop (remove)\n- Use cases: undo/redo, function call stack, backtracking\n\n**Queue**: First In, First Out (FIFO)\n- Think of a line at a store. First person in line gets served first.\n- Operations: enqueue (add to back) and dequeue (remove from front)\n- Use cases: task scheduling, print queues, BFS algorithms\n\nKey difference: a stack reverses order (last in = first out), a queue preserves order (first in = first out).",
    },
    {
        "prompt": "Help me plan a study schedule for my biology exam next week.",
        "response": "Here's a focused 7-day plan:\n\n**Day 1-2: Review & Identify Gaps**\n- Review notes and textbook chapters\n- Identify topics you're weakest in\n- Create a ranked topic list by difficulty\n\n**Day 3-4: Deep Study**\n- Focus on weakest topics first (spaced repetition)\n- Create flashcards for key terms\n- Practice drawing diagrams (cell structures, processes)\n\n**Day 5: Practice & Application**\n- Do practice questions from each chapter\n- Review past assignments and quizzes\n- Focus on application, not memorization\n\n**Day 6: Review & Consolidate**\n- Quick review of all topics\n- Focus on connections between concepts\n- Teach the material aloud (best retention technique)\n\n**Day 7: Light Review**\n- Brief flashcard review\n- Relax and get good sleep\n\nWant me to customize this for specific topics?",
    },
    {
        "prompt": "What causes the seasons on Earth?",
        "response": "Seasons are caused by Earth's **axial tilt** of 23.5 degrees, NOT by distance from the Sun.\n\nAs Earth orbits the Sun:\n- **Summer**: Your hemisphere tilts TOWARD the Sun. Sunlight hits more directly, days are longer.\n- **Winter**: Your hemisphere tilts AWAY from the Sun. Sunlight hits at a shallow angle, days are shorter.\n- **Spring/Fall**: Transitional \u2014 neither hemisphere strongly tilted toward or away.\n\nThis is why seasons are opposite in the Northern and Southern hemispheres. When it's summer in New York, it's winter in Sydney.\n\nCommon misconception: Earth IS actually slightly closer to the Sun during Northern Hemisphere winter (January). Distance barely matters \u2014 tilt is everything.",
    },
    {
        "prompt": "I keep making the same mistake in my code. I forget to handle null values.",
        "response": "This is one of the most common bugs. Here are practical strategies:\n\n**1. Use strict type checking**\n- TypeScript: `strictNullChecks` in tsconfig\n- Python: type hints with `Optional[T]` and mypy\n- Catches null issues at compile/lint time\n\n**2. Defensive patterns**\n- Default values: `value = input ?? default`\n- Early returns: `if x is None: return`\n- Optional chaining: `user?.address?.city`\n\n**3. Design patterns**\n- Null Object pattern instead of actual nulls\n- Result/Either types for operations that might fail\n- Validate at boundaries, trust internals\n\n**4. Process habits**\n- Add null-check to your code review checklist\n- Write tests that explicitly pass None/null\n- Use linters that flag potential null dereferences\n\nThe real solution: make it impossible rather than trying to remember.",
    },
    {
        "prompt": "What's the big deal about machine learning?",
        "response": "Traditional programming: you write explicit rules. \"If temperature > 100, alert.\"\n\nMachine learning: you give examples, and the computer figures out the rules itself.\n\nWhy it matters:\n- **Pattern recognition at scale**: ML can find patterns in millions of data points that humans would never spot\n- **Adaptation**: The system improves as it sees more data, without rewriting code\n- **Complex tasks**: Some problems (image recognition, language translation) are nearly impossible to code with explicit rules\n\nReal examples:\n- Spam filters learn what spam looks like from millions of emails\n- Medical AI spots cancer in X-rays by training on thousands of examples\n- Recommendation systems learn your preferences from your behavior\n\nThe limitation: ML is only as good as its data. Biased data = biased model. And it can't explain WHY it made a decision \u2014 it's pattern matching, not understanding.",
    },
    {
        "prompt": "Explain recursion to me.",
        "response": "Recursion is when a function calls itself to solve a problem by breaking it into smaller versions of the same problem.\n\n**Simple example** \u2014 factorial:\n```\nfactorial(5) = 5 \u00d7 factorial(4)\n             = 5 \u00d7 4 \u00d7 factorial(3)\n             = 5 \u00d7 4 \u00d7 3 \u00d7 factorial(2)\n             = 5 \u00d7 4 \u00d7 3 \u00d7 2 \u00d7 factorial(1)\n             = 5 \u00d7 4 \u00d7 3 \u00d7 2 \u00d7 1 = 120\n```\n\nEvery recursive function needs:\n1. **Base case**: When to stop (factorial(1) = 1)\n2. **Recursive case**: How to break the problem down\n\nWithout a base case, you get infinite recursion (stack overflow).\n\nAnalogy: You're in a long line and ask the person ahead \"what number am I?\" They don't know, so they ask the person ahead of them. This continues until the first person says \"I'm #1.\" Then the answer propagates back: #2, #3, etc., until you get your number.",
    },
]

add_examples("conversation", conversation_examples)


# === Shuffle and write ===
random.shuffle(examples)

output_file = OUTPUT_PATH / "ava_exp4_finetune_v2.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"\n{'='*60}")
print(f"Total corpus: {len(examples)} examples")
print(f"Written to: {output_file}")
print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")
print(f"\nBreakdown:")
for name, count in sorted(stats.items(), key=lambda x: -x[1]):
    pct = count / len(examples) * 100
    print(f"  {name}: {count} ({pct:.1f}%)")

# Stats
avg_len = sum(len(ex.get("prompt", "") + ex.get("response", "")) for ex in examples) / len(examples)
max_len = max(len(ex.get("prompt", "") + ex.get("response", "")) for ex in examples)
print(f"\nAvg example length: {avg_len:.0f} chars")
print(f"Max example length: {max_len} chars")
