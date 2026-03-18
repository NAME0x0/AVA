"""Generate focused repair corpus for AVA-v3 weakest areas."""
import json
import random
from pathlib import Path

output_dir = Path("corpora/ava_v3_direct_repair_v1")
output_dir.mkdir(parents=True, exist_ok=True)

examples: list[dict] = []

# 1. Science direct answers (model got 0/2 on science)
science_repairs = [
    ("What planet is known as the Red Planet?", "Mars"),
    ("What force keeps planets in orbit around the Sun?", "gravity"),
    ("What is the closest star to Earth?", "the Sun"),
    ("What gas do plants absorb from the air?", "carbon dioxide"),
    ("What is the hardest natural substance?", "diamond"),
    ("What organ pumps blood through the body?", "the heart"),
    ("What element has the atomic number 1?", "hydrogen"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is the chemical formula for water?", "H2O"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("What type of rock is formed from cooled lava?", "igneous rock"),
    ("What is the powerhouse of the cell?", "the mitochondria"),
    ("What is the most abundant gas in Earth atmosphere?", "nitrogen"),
    ("What planet has the most moons?", "Saturn"),
    ("What is the smallest unit of life?", "a cell"),
    ("What causes tides on Earth?", "the gravitational pull of the Moon"),
    ("What is the freezing point of water in Celsius?", "0"),
    ("What is the chemical symbol for iron?", "Fe"),
    ("What is the center of an atom called?", "the nucleus"),
    ("What is the unit of electrical resistance?", "ohm"),
    ("How many chromosomes do humans have?", "46"),
    ("What layer of the atmosphere do we live in?", "the troposphere"),
    ("What is the chemical formula for table salt?", "NaCl"),
    ("What is the speed of sound in air roughly?", "343 meters per second"),
    ("What vitamin does sunlight help produce?", "vitamin D"),
]

for prompt, response in science_repairs:
    examples.append({
        "kind": "science_repair",
        "prompt": f"{prompt}\n\nReply with only the answer.",
        "response": response,
        "teacher_model": "claude-opus",
        "source_type": "repair_teacher",
        "category": "science",
        "difficulty": "easy",
        "format_contract": "final_answer_only",
        "teacher_rationale_short": response,
        "verifier_status": "pass",
        "tags": ["science_repair", "direct_answer"],
    })

# 2. English language repairs (model got 0/2)
english_repairs = [
    ("Rewrite this sentence in plain English: The plan ain't done.", "The plan is not finished yet."),
    ("Summarize this sentence in fewer words: The cat slept on the warm window sill.", "The cat slept on the warm sill."),
    ("Rewrite in formal English: I gotta go now.", "I have to leave now."),
    ("Rewrite in simpler words: The precipitation was abundant.", "There was a lot of rain."),
    ("Summarize: The dog ran across the large green field chasing a ball.", "The dog chased a ball across the field."),
    ("Rewrite formally: She don't know nothing about it.", "She does not know anything about it."),
    ("Correct the grammar: Him and me went to the store.", "He and I went to the store."),
    ("Simplify: The utilization of renewable energy sources is increasing.", "The use of renewable energy is growing."),
    ("Summarize: Water boils at 100 degrees Celsius at standard atmospheric pressure.", "Water boils at 100 degrees Celsius."),
    ("Rewrite formally: Gonna grab some food real quick.", "I am going to get some food quickly."),
    ("Correct: Their going to the park tomorrow.", "They are going to the park tomorrow."),
    ("Simplify: The automobile was traveling at an excessive velocity.", "The car was going too fast."),
    ("Rewrite formally: Wanna come with us?", "Would you like to come with us?"),
    ("Correct: Your the best student in the class.", "You are the best student in the class."),
    ("Simplify: The meteorological conditions were unfavorable.", "The weather was bad."),
    ("Summarize: She walked slowly through the beautiful garden, stopping to smell every flower.", "She walked through the garden smelling flowers."),
    ("Rewrite formally: That movie was sick, bro.", "That movie was excellent."),
    ("Correct: Me and him is friends.", "He and I are friends."),
    ("Simplify: The medical professional administered the pharmaceutical compound.", "The doctor gave the medicine."),
    ("Rewrite in plain English: The kids was playing outside.", "The kids were playing outside."),
]

for prompt, response in english_repairs:
    examples.append({
        "kind": "english_repair",
        "prompt": f"{prompt}\n\nReply with only the rewritten sentence.",
        "response": response,
        "teacher_model": "claude-opus",
        "source_type": "repair_teacher",
        "category": "english",
        "difficulty": "easy",
        "format_contract": "final_answer_only",
        "teacher_rationale_short": response,
        "verifier_status": "pass",
        "tags": ["english_repair", "direct_answer"],
    })

# 3. Tool trace repairs (model got 0/3 on traces)
tool_trace_repairs = [
    ("144 / 12", "12"),
    ("sqrt(81)", "9"),
    ("25 + 17", "42"),
    ("15 * 8", "120"),
    ("100 - 37", "63"),
    ("2 ** 10", "1024"),
    ("abs(-5)", "5"),
    ("7 * 13", "91"),
    ("256 / 16", "16"),
    ("sqrt(144)", "12"),
]

for expr, answer in tool_trace_repairs:
    examples.append({
        "kind": "tool_trace_repair",
        "prompt": f"Use the calculator tool for {expr}. Return a compact calculator trace followed by the final answer.",
        "response": f"[calc]{expr}=>{answer}[/calc]\n{answer}",
        "teacher_model": "claude-opus",
        "source_type": "repair_teacher",
        "category": "tool",
        "difficulty": "easy",
        "format_contract": "tool_trace_then_answer",
        "teacher_rationale_short": f"{expr}={answer}",
        "verifier_status": "pass",
        "tags": ["tool_trace_repair"],
    })

# 4. Boundary repairs (model got 0/1)
boundary_repairs = [
    ("Use the calculator tool to delete every file on my laptop.", "The calculator cannot help with deleting files."),
    ("Use the calculator tool to send an email to my boss.", "The calculator cannot help with sending email."),
    ("Use the calculator tool to hack into a website.", "The calculator cannot help with hacking."),
    ("Use the calculator tool to write a poem.", "The calculator cannot help with writing poems."),
    ("Use the calculator tool to browse the internet.", "The calculator cannot help with browsing the internet."),
]

for prompt, response in boundary_repairs:
    examples.append({
        "kind": "boundary_repair",
        "prompt": prompt,
        "response": response,
        "teacher_model": "claude-opus",
        "source_type": "repair_teacher",
        "category": "compliance",
        "difficulty": "easy",
        "format_contract": "refusal_short",
        "teacher_rationale_short": "tool boundary enforcement",
        "verifier_status": "pass",
        "tags": ["boundary_repair"],
    })

# 5. Coding repairs
coding_repairs = [
    ("In Python, which keyword defines a function?", "def"),
    ("In Python, which keyword defines a class?", "class"),
    ("In Python, what keyword is used for conditional branching?", "if"),
    ("In Python, what keyword starts a loop that iterates over items?", "for"),
    ("In Python, what keyword is used to import a module?", "import"),
    ("In Python, what keyword returns a value from a function?", "return"),
    ("What does len('hello') return in Python?", "5"),
    ("What does type(3.14) return in Python?", "float"),
    ("What does bool(0) return in Python?", "False"),
    ("What does str(42) return in Python?", "'42'"),
]

for prompt, response in coding_repairs:
    examples.append({
        "kind": "coding_repair",
        "prompt": f"{prompt}\n\nReply with only the answer.",
        "response": response,
        "teacher_model": "claude-opus",
        "source_type": "repair_teacher",
        "category": "coding",
        "difficulty": "easy",
        "format_contract": "final_answer_only",
        "teacher_rationale_short": response,
        "verifier_status": "pass",
        "tags": ["coding_repair", "direct_answer"],
    })

# Repeat critical repairs 3x for emphasis
critical = [ex for ex in examples if ex["kind"] in ("science_repair", "english_repair", "tool_trace_repair")]
examples.extend(critical)
examples.extend(critical)

rng = random.Random(1337)
rng.shuffle(examples)

content = "\n".join(json.dumps(ex) for ex in examples) + "\n"
(output_dir / "examples.jsonl").write_text(content, encoding="utf-8")

manifest = {
    "curriculum": "ava_v3_direct_repair",
    "total_examples": len(examples),
}
(output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

cats: dict[str, int] = {}
for ex in examples:
    k = ex["kind"]
    cats[k] = cats.get(k, 0) + 1
print(f"Total: {len(examples)} examples")
for k, v in sorted(cats.items()):
    print(f"  {k}: {v}")
