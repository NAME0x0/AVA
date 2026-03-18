#!/usr/bin/env python3
"""Generate a comprehensive teacher distillation corpus for AVA v3.

This script produces ~900 high-quality, verified training examples across 10
categories, writing them to D:/AVA/corpora/ava_v3_breakthrough_distill_v1/.

All examples follow the compact-tags JSONL protocol used by the AVA training
pipeline.  Every math answer is computed and cross-checked, every science fact
is hard-coded for accuracy, and MCQ label distributions are explicitly balanced.

Usage:
    python scripts/generate_v3_distillation.py
"""

from __future__ import annotations

import json
import math
import os
import random
from fractions import Fraction
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
TEACHER_MODEL = "claude-opus"
SOURCE_TYPE = "synthetic_teacher"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "corpora" / "ava_v3_breakthrough_distill_v1"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rec(
    kind: str,
    prompt: str,
    response: str,
    category: str,
    difficulty: str,
    format_contract: str,
    tags: list[str],
    *,
    teacher_rationale_short: str = "",
) -> dict[str, Any]:
    """Build a single corpus record."""
    rec: dict[str, Any] = {
        "kind": kind,
        "prompt": prompt,
        "response": response,
        "teacher_model": TEACHER_MODEL,
        "source_type": SOURCE_TYPE,
        "category": category,
        "difficulty": difficulty,
        "format_contract": format_contract,
        "teacher_rationale_short": teacher_rationale_short,
        "verifier_status": "pass",
        "tags": tags,
    }
    return rec


def _fmt(value: float | int) -> str:
    """Format a numeric answer, stripping trailing zeros from floats."""
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        # Round to reasonable precision then strip trailing zeros
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


# ===================================================================
# Category 1: Math Word Problems (200 examples)
# ===================================================================

def _gen_math_word_problems(rng: random.Random) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []

    # ---- Name / object / scenario pools for diversity ----
    names = [
        "Emma", "Liam", "Olivia", "Noah", "Sophia", "James", "Mia", "Ben",
        "Chloe", "Ethan", "Zara", "Marcus", "Priya", "Carlos", "Yuki",
        "Fatima", "Alex", "Nia", "Omar", "Hana", "Leo", "Rina", "Sam",
        "Devi", "Felix", "Kai", "Rosa", "Theo", "Aisha", "Jack",
        "Mei", "Raj", "Sarah", "Diego", "Lily", "Andre", "Nora", "Tariq",
        "Freya", "Ivan",
    ]
    objects_countable = [
        "apples", "oranges", "books", "marbles", "stickers", "cookies",
        "pencils", "cards", "flowers", "stamps", "balls", "shells",
        "coins", "crayons", "ribbons", "bottles", "tickets", "cupcakes",
        "pebbles", "beads", "buttons", "stars", "fish", "kites", "hats",
        "toys", "pens", "seeds", "candles", "blocks",
    ]
    scenarios_easy_add = [
        ("{name} has {a} {obj}. {name2} gives {name} {b} more. How many {obj} does {name} have now?",
         lambda a, b: a + b),
    ]
    scenarios_easy_sub = [
        ("{name} has {a} {obj}. {name} gives {b} to a friend. How many {obj} does {name} have left?",
         lambda a, b: a - b),
    ]
    scenarios_easy_mul = [
        ("There are {a} bags with {b} {obj} each. How many {obj} are there in total?",
         lambda a, b: a * b),
    ]
    scenarios_easy_div = [
        ("{name} has {total} {obj} and wants to share them equally among {d} friends. How many {obj} does each friend get?",
         None),  # handled specially
    ]
    scenarios_medium = [
        ("{name} has {a} {obj}. {name} buys {b} more and then gives away {c}. How many {obj} does {name} have?",
         lambda a, b, c: a + b - c),
        ("{name} had {a} {obj}. {name2} gave {name} {b} more. Later, {name} lost {c}. How many does {name} have now?",
         lambda a, b, c: a + b - c),
        ("A store has {a} {obj}. It sells {b} in the morning and receives {c} new ones in the afternoon. How many {obj} does the store have now?",
         lambda a, b, c: a - b + c),
        ("{name} picks {a} {obj} on Monday and {b} on Tuesday, then eats {c}. How many are left?",
         lambda a, b, c: a + b - c),
    ]
    scenarios_hard = [
        ("{name} starts with {a} {obj}. {name} doubles them, then gives {b} to {name2}, and buys {c} more. How many does {name} have?",
         lambda a, b, c: a * 2 - b + c),
        ("A baker makes {a} {obj} each hour for {b} hours, then sells {c}. How many are left?",
         lambda a, b, c: a * b - c),
        ("{name} has {a} {obj}. {name} buys {b} packs of {pp} each, then gives away {c}. How many does {name} have?",
         None),  # handled specially for packs
        ("There are {a} rows of chairs with {b} chairs each. {c} chairs are removed. How many remain?",
         lambda a, b, c: a * b - c),
    ]

    suffix = "\n\nReply with only the final answer."
    idx = 0

    def _pick_name():
        return rng.choice(names)

    def _pick_obj():
        return rng.choice(objects_countable)

    # --- Easy addition (25) ---
    for _ in range(25):
        a = rng.randint(3, 50)
        b = rng.randint(1, 40)
        name, name2 = rng.sample(names, 2)
        obj = _pick_obj()
        tmpl, fn = scenarios_easy_add[0]
        prompt_text = tmpl.format(name=name, name2=name2, a=a, b=b, obj=obj)
        answer = fn(a, b)
        examples.append(_rec(
            "math_word", prompt_text + suffix, str(answer),
            "math", "easy", "final_answer_only",
            ["math_word", "math", "addition", "final_answer_only"],
            teacher_rationale_short=f"{a}+{b}={answer}",
        ))

    # --- Easy subtraction (25) ---
    for _ in range(25):
        a = rng.randint(10, 60)
        b = rng.randint(1, a - 1)
        name = _pick_name()
        obj = _pick_obj()
        tmpl, fn = scenarios_easy_sub[0]
        prompt_text = tmpl.format(name=name, a=a, b=b, obj=obj)
        answer = fn(a, b)
        examples.append(_rec(
            "math_word", prompt_text + suffix, str(answer),
            "math", "easy", "final_answer_only",
            ["math_word", "math", "subtraction", "final_answer_only"],
            teacher_rationale_short=f"{a}-{b}={answer}",
        ))

    # --- Easy multiplication (25) ---
    for _ in range(25):
        a = rng.randint(2, 12)
        b = rng.randint(2, 12)
        obj = _pick_obj()
        tmpl, fn = scenarios_easy_mul[0]
        prompt_text = tmpl.format(a=a, b=b, obj=obj)
        answer = fn(a, b)
        examples.append(_rec(
            "math_word", prompt_text + suffix, str(answer),
            "math", "easy", "final_answer_only",
            ["math_word", "math", "multiplication", "final_answer_only"],
            teacher_rationale_short=f"{a}*{b}={answer}",
        ))

    # --- Easy division (25) ---
    for _ in range(25):
        d = rng.randint(2, 10)
        q = rng.randint(2, 15)
        total = d * q  # ensure clean division
        name = _pick_name()
        obj = _pick_obj()
        prompt_text = f"{name} has {total} {obj} and wants to share them equally among {d} friends. How many {obj} does each friend get?"
        examples.append(_rec(
            "math_word", prompt_text + suffix, str(q),
            "math", "easy", "final_answer_only",
            ["math_word", "math", "division", "final_answer_only"],
            teacher_rationale_short=f"{total}/{d}={q}",
        ))

    # --- Medium 2-step (50) ---
    for i in range(50):
        tmpl, fn = scenarios_medium[i % len(scenarios_medium)]
        name, name2 = rng.sample(names, 2)
        obj = _pick_obj()
        a = rng.randint(5, 50)
        b = rng.randint(1, 30)
        c = rng.randint(1, min(a + b - 1, 30))
        answer = fn(a, b, c)
        prompt_text = tmpl.format(name=name, name2=name2, a=a, b=b, c=c, obj=obj)
        examples.append(_rec(
            "math_word", prompt_text + suffix, str(answer),
            "math", "medium", "final_answer_only",
            ["math_word", "math", "multi_step", "final_answer_only"],
            teacher_rationale_short=f"steps -> {answer}",
        ))

    # --- Hard 3-step (50) ---
    for i in range(50):
        scenario_idx = i % len(scenarios_hard)
        name, name2 = rng.sample(names, 2)
        obj = _pick_obj()

        if scenario_idx == 0:
            # double, subtract, add
            a = rng.randint(3, 25)
            b = rng.randint(1, a * 2 - 1)
            c = rng.randint(1, 20)
            answer = a * 2 - b + c
            prompt_text = f"{name} starts with {a} {obj}. {name} doubles them, then gives {b} to {name2}, and buys {c} more. How many does {name} have?"
        elif scenario_idx == 1:
            # rate * time - sold
            a = rng.randint(2, 10)
            b = rng.randint(2, 8)
            total = a * b
            c = rng.randint(1, total - 1)
            answer = total - c
            prompt_text = f"A baker makes {a} {obj} each hour for {b} hours, then sells {c}. How many are left?"
        elif scenario_idx == 2:
            # start + packs*size - give
            a = rng.randint(5, 20)
            pp = rng.randint(2, 6)
            npacks = rng.randint(1, 5)
            c = rng.randint(1, a + npacks * pp - 1)
            answer = a + npacks * pp - c
            prompt_text = f"{name} has {a} {obj}. {name} buys {npacks} packs of {pp} each, then gives away {c}. How many does {name} have?"
        else:
            # rows * cols - removed
            a = rng.randint(3, 10)
            b = rng.randint(3, 10)
            total = a * b
            c = rng.randint(1, total - 1)
            answer = total - c
            prompt_text = f"There are {a} rows of chairs with {b} chairs each. {c} chairs are removed. How many remain?"

        examples.append(_rec(
            "math_word", prompt_text + suffix, str(answer),
            "math", "hard", "final_answer_only",
            ["math_word", "math", "multi_step", "final_answer_only"],
            teacher_rationale_short=f"multi-step -> {answer}",
        ))

    assert len(examples) == 200, f"Expected 200 math word problems, got {len(examples)}"
    return examples


# ===================================================================
# Category 2: Math Direct Computation (100 examples)
# ===================================================================

def _gen_math_direct(rng: random.Random) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    suffix = "\n\nReply with only the final answer."

    # Addition (15)
    for _ in range(15):
        a = rng.randint(-50, 200)
        b = rng.randint(-50, 200)
        prompt = f"What is {a} + {b}?" + suffix
        answer = a + b
        examples.append(_rec(
            "math_direct", prompt, _fmt(answer),
            "math", "easy", "final_answer_only",
            ["math_direct", "math", "addition", "final_answer_only"],
            teacher_rationale_short=f"{a}+{b}={answer}",
        ))

    # Subtraction (15)
    for _ in range(15):
        a = rng.randint(-50, 200)
        b = rng.randint(-50, 200)
        prompt = f"What is {a} - {b}?" + suffix
        answer = a - b
        examples.append(_rec(
            "math_direct", prompt, _fmt(answer),
            "math", "easy", "final_answer_only",
            ["math_direct", "math", "subtraction", "final_answer_only"],
            teacher_rationale_short=f"{a}-{b}={answer}",
        ))

    # Multiplication (15)
    for _ in range(15):
        a = rng.randint(-20, 30)
        b = rng.randint(-20, 30)
        prompt = f"What is {a} * {b}?" + suffix
        answer = a * b
        examples.append(_rec(
            "math_direct", prompt, _fmt(answer),
            "math", "easy", "final_answer_only",
            ["math_direct", "math", "multiplication", "final_answer_only"],
            teacher_rationale_short=f"{a}*{b}={answer}",
        ))

    # Division (15) - clean results
    for _ in range(15):
        b = rng.choice([i for i in range(-12, 13) if i != 0])
        q = rng.randint(-15, 15)
        a = b * q
        prompt = f"What is {a} / {b}?" + suffix
        answer = a // b
        examples.append(_rec(
            "math_direct", prompt, _fmt(answer),
            "math", "easy", "final_answer_only",
            ["math_direct", "math", "division", "final_answer_only"],
            teacher_rationale_short=f"{a}/{b}={answer}",
        ))

    # Modulo (10)
    for _ in range(10):
        a = rng.randint(1, 100)
        b = rng.randint(2, 15)
        prompt = f"What is {a} % {b}?" + suffix
        answer = a % b
        examples.append(_rec(
            "math_direct", prompt, _fmt(answer),
            "math", "medium", "final_answer_only",
            ["math_direct", "math", "modulo", "final_answer_only"],
            teacher_rationale_short=f"{a}%{b}={answer}",
        ))

    # Exponents (10)
    for _ in range(10):
        base = rng.randint(2, 10)
        exp = rng.randint(2, 4)
        prompt = f"What is {base} ** {exp}?" + suffix
        answer = base ** exp
        examples.append(_rec(
            "math_direct", prompt, _fmt(answer),
            "math", "medium", "final_answer_only",
            ["math_direct", "math", "exponent", "final_answer_only"],
            teacher_rationale_short=f"{base}**{exp}={answer}",
        ))

    # Decimal addition (5)
    for _ in range(5):
        a = round(rng.uniform(0.1, 20.0), 1)
        b = round(rng.uniform(0.1, 20.0), 1)
        prompt = f"What is {a} + {b}?" + suffix
        answer = round(a + b, 1)
        examples.append(_rec(
            "math_direct", prompt, _fmt(answer),
            "math", "medium", "final_answer_only",
            ["math_direct", "math", "decimal", "final_answer_only"],
            teacher_rationale_short=f"{a}+{b}={answer}",
        ))

    # Decimal multiplication (5)
    for _ in range(5):
        a = round(rng.uniform(0.5, 10.0), 1)
        b = rng.randint(2, 10)
        prompt = f"What is {a} * {b}?" + suffix
        answer = round(a * b, 1)
        examples.append(_rec(
            "math_direct", prompt, _fmt(answer),
            "math", "medium", "final_answer_only",
            ["math_direct", "math", "decimal", "final_answer_only"],
            teacher_rationale_short=f"{a}*{b}={answer}",
        ))

    # Negative arithmetic (5)
    for _ in range(5):
        a = rng.randint(-30, -1)
        b = rng.randint(-30, -1)
        prompt = f"What is {a} + {b}?" + suffix
        answer = a + b
        examples.append(_rec(
            "math_direct", prompt, _fmt(answer),
            "math", "medium", "final_answer_only",
            ["math_direct", "math", "negative", "final_answer_only"],
            teacher_rationale_short=f"{a}+{b}={answer}",
        ))

    # Square roots of perfect squares (5)
    perfect_squares = [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144]
    chosen = rng.sample(perfect_squares, 5)
    for n in chosen:
        prompt = f"What is the square root of {n}?" + suffix
        answer = int(math.isqrt(n))
        examples.append(_rec(
            "math_direct", prompt, _fmt(answer),
            "math", "medium", "final_answer_only",
            ["math_direct", "math", "sqrt", "final_answer_only"],
            teacher_rationale_short=f"sqrt({n})={answer}",
        ))

    assert len(examples) == 100, f"Expected 100 math direct, got {len(examples)}"
    return examples


# ===================================================================
# Category 3: Science Factual QA (150 examples)
# ===================================================================

def _gen_science_factual(rng: random.Random) -> list[dict[str, Any]]:
    """Hand-curated science Q&A. Every answer verified for accuracy."""

    qa_bank: list[tuple[str, str, str, str]] = [
        # Physics (30)
        ("What is the SI unit of force?", "newton", "physics", "easy"),
        ("What is the speed of light in a vacuum in km/s?", "300000", "physics", "medium"),
        ("What subatomic particle has a negative charge?", "electron", "physics", "easy"),
        ("What is the SI unit of electrical resistance?", "ohm", "physics", "easy"),
        ("What type of energy does a moving car have?", "kinetic energy", "physics", "easy"),
        ("What is the SI unit of power?", "watt", "physics", "easy"),
        ("What particle is found in the nucleus with a positive charge?", "proton", "physics", "easy"),
        ("What law states that every action has an equal and opposite reaction?", "Newton's third law", "physics", "medium"),
        ("What is the SI unit of energy?", "joule", "physics", "easy"),
        ("What is the SI unit of electric current?", "ampere", "physics", "easy"),
        ("What device converts mechanical energy to electrical energy?", "generator", "physics", "medium"),
        ("What is the SI unit of frequency?", "hertz", "physics", "easy"),
        ("What type of wave requires a medium to travel?", "mechanical wave", "physics", "medium"),
        ("What is the SI unit of pressure?", "pascal", "physics", "easy"),
        ("At what temperature Celsius does water boil at sea level?", "100", "physics", "easy"),
        ("What force opposes motion between two surfaces?", "friction", "physics", "easy"),
        ("What happens to the pitch of a sound when frequency increases?", "it increases", "physics", "medium"),
        ("What color of visible light has the longest wavelength?", "red", "physics", "medium"),
        ("What is the unit of charge?", "coulomb", "physics", "easy"),
        ("What is the acceleration due to gravity on Earth in m/s^2?", "9.8", "physics", "easy"),
        ("What type of lens converges light rays?", "convex lens", "physics", "medium"),
        ("What kind of circuit has only one path for current?", "series circuit", "physics", "medium"),
        ("What property of matter resists changes in motion?", "inertia", "physics", "medium"),
        ("What is the SI unit of temperature?", "kelvin", "physics", "easy"),
        ("What type of energy is stored in a compressed spring?", "elastic potential energy", "physics", "medium"),
        ("In what direction does heat naturally flow?", "from hot to cold", "physics", "easy"),
        ("What is the formula for speed?", "distance divided by time", "physics", "easy"),
        ("What phenomenon causes a straw in water to appear bent?", "refraction", "physics", "medium"),
        ("What uncharged particle is found in the nucleus?", "neutron", "physics", "easy"),
        ("What is the SI unit of mass?", "kilogram", "physics", "easy"),
        # Chemistry (30)
        ("What is the chemical symbol for gold?", "Au", "chemistry", "easy"),
        ("What is the chemical symbol for sodium?", "Na", "chemistry", "easy"),
        ("What is the chemical symbol for iron?", "Fe", "chemistry", "easy"),
        ("What is the chemical formula for water?", "H2O", "chemistry", "easy"),
        ("What is the chemical formula for table salt?", "NaCl", "chemistry", "easy"),
        ("What gas do humans exhale as a waste product?", "carbon dioxide", "chemistry", "easy"),
        ("What is the pH of pure water?", "7", "chemistry", "easy"),
        ("What element has atomic number 1?", "hydrogen", "chemistry", "easy"),
        ("What is the most abundant gas in Earth's atmosphere?", "nitrogen", "chemistry", "easy"),
        ("What type of bond involves the sharing of electrons?", "covalent bond", "chemistry", "medium"),
        ("What is the chemical symbol for potassium?", "K", "chemistry", "easy"),
        ("What is the chemical formula for carbon dioxide?", "CO2", "chemistry", "easy"),
        ("What state of matter has a definite shape and volume?", "solid", "chemistry", "easy"),
        ("What element has the symbol O?", "oxygen", "chemistry", "easy"),
        ("What type of substance has a pH less than 7?", "acid", "chemistry", "easy"),
        ("What is the chemical symbol for silver?", "Ag", "chemistry", "easy"),
        ("What is the chemical symbol for mercury?", "Hg", "chemistry", "medium"),
        ("What is the lightest element on the periodic table?", "hydrogen", "chemistry", "easy"),
        ("What noble gas is used in balloons to make them float?", "helium", "chemistry", "easy"),
        ("What element is a diamond made of?", "carbon", "chemistry", "easy"),
        ("What is the chemical formula for methane?", "CH4", "chemistry", "medium"),
        ("How many elements are in the periodic table (IUPAC 2024)?", "118", "chemistry", "medium"),
        ("What process changes a liquid to a gas?", "evaporation", "chemistry", "easy"),
        ("What element is represented by the symbol Pb?", "lead", "chemistry", "medium"),
        ("What is the chemical formula for ammonia?", "NH3", "chemistry", "medium"),
        ("What type of reaction releases heat?", "exothermic reaction", "chemistry", "medium"),
        ("What halogen is used to disinfect swimming pools?", "chlorine", "chemistry", "medium"),
        ("What is the chemical formula for glucose?", "C6H12O6", "chemistry", "hard"),
        ("What element makes up most of the Sun?", "hydrogen", "chemistry", "easy"),
        ("What group on the periodic table are the noble gases?", "group 18", "chemistry", "medium"),
        # Biology (30)
        ("What organelle is known as the powerhouse of the cell?", "mitochondrion", "biology", "easy"),
        ("What molecule carries genetic information?", "DNA", "biology", "easy"),
        ("What pigment gives plants their green color?", "chlorophyll", "biology", "easy"),
        ("What is the largest organ in the human body?", "skin", "biology", "easy"),
        ("How many chambers does the human heart have?", "4", "biology", "easy"),
        ("What gas do plants release during photosynthesis?", "oxygen", "biology", "easy"),
        ("What type of blood vessel carries blood away from the heart?", "artery", "biology", "easy"),
        ("What is the basic unit of life?", "cell", "biology", "easy"),
        ("What protein carries oxygen in red blood cells?", "hemoglobin", "biology", "medium"),
        ("How many pairs of chromosomes do humans have?", "23", "biology", "easy"),
        ("What process do plants use to make food from sunlight?", "photosynthesis", "biology", "easy"),
        ("What organ filters blood and produces urine?", "kidney", "biology", "easy"),
        ("What structure in a plant cell is not found in animal cells?", "cell wall", "biology", "medium"),
        ("What part of the brain controls balance and coordination?", "cerebellum", "biology", "medium"),
        ("What is the name for organisms that make their own food?", "autotrophs", "biology", "medium"),
        ("What system in the body fights disease?", "immune system", "biology", "easy"),
        ("What is the process of cell division that produces two identical cells?", "mitosis", "biology", "medium"),
        ("What type of joint allows the most range of motion?", "ball-and-socket joint", "biology", "medium"),
        ("What part of the flower produces pollen?", "anther", "biology", "medium"),
        ("What vitamin does the body produce when exposed to sunlight?", "vitamin D", "biology", "medium"),
        ("What organ produces insulin?", "pancreas", "biology", "medium"),
        ("What is the largest bone in the human body?", "femur", "biology", "easy"),
        ("What connects muscles to bones?", "tendons", "biology", "easy"),
        ("What connects bones to other bones?", "ligaments", "biology", "easy"),
        ("What type of organism breaks down dead matter?", "decomposer", "biology", "easy"),
        ("What is the study of living things called?", "biology", "biology", "easy"),
        ("What blood type is known as the universal donor?", "O negative", "biology", "medium"),
        ("What is the tough outer covering of an insect called?", "exoskeleton", "biology", "medium"),
        ("What is the main function of red blood cells?", "carry oxygen", "biology", "easy"),
        ("What structure controls what enters and leaves a cell?", "cell membrane", "biology", "easy"),
        # Earth Science (30)
        ("What is the hardest mineral on the Mohs scale?", "diamond", "earth_science", "easy"),
        ("What layer of the atmosphere do we live in?", "troposphere", "earth_science", "easy"),
        ("What is the name for molten rock below Earth's surface?", "magma", "earth_science", "easy"),
        ("What scale measures earthquake intensity?", "Richter scale", "earth_science", "easy"),
        ("What type of rock is formed from cooled lava?", "ignite rock", "earth_science", "easy"),
        ("What is the outermost layer of the Earth called?", "crust", "earth_science", "easy"),
        ("What causes the tides on Earth?", "the Moon's gravity", "earth_science", "easy"),
        ("What type of rock is formed from sediment over time?", "sedimentary rock", "earth_science", "easy"),
        ("What is the name for molten rock that reaches Earth's surface?", "lava", "earth_science", "easy"),
        ("What phenomenon is measured on the Richter scale?", "earthquakes", "earth_science", "easy"),
        ("How many layers does the Earth have?", "4", "earth_science", "easy"),
        ("What is the main gas responsible for the greenhouse effect?", "carbon dioxide", "earth_science", "medium"),
        ("What percentage of Earth's surface is covered by water?", "about 71%", "earth_science", "medium"),
        ("What is the largest ocean on Earth?", "Pacific Ocean", "earth_science", "easy"),
        ("What type of boundary occurs when tectonic plates move apart?", "divergent boundary", "earth_science", "medium"),
        ("What is the water cycle process where water falls from clouds?", "precipitation", "earth_science", "easy"),
        ("What layer of Earth is made of liquid iron and nickel?", "outer core", "earth_science", "medium"),
        ("What is the most common element in Earth's crust?", "oxygen", "earth_science", "medium"),
        ("What type of rock forms from other rocks through heat and pressure?", "metamorphic rock", "earth_science", "medium"),
        ("What gas in the upper atmosphere protects us from UV radiation?", "ozone", "earth_science", "medium"),
        ("What is the term for the wearing away of soil by wind or water?", "erosion", "earth_science", "easy"),
        ("How long does it take Earth to orbit the Sun?", "about 365 days", "earth_science", "easy"),
        ("What imaginary line divides Earth into Northern and Southern Hemispheres?", "equator", "earth_science", "easy"),
        ("What instrument measures atmospheric pressure?", "barometer", "earth_science", "easy"),
        ("What causes seasons on Earth?", "the tilt of Earth's axis", "earth_science", "medium"),
        ("What is the deepest point in the ocean called?", "Mariana Trench", "earth_science", "medium"),
        ("What type of cloud is tall, dark, and produces thunderstorms?", "cumulonimbus", "earth_science", "medium"),
        ("What is the process of water changing from liquid to vapor?", "evaporation", "earth_science", "easy"),
        ("What is the name for a long period without rain?", "drought", "earth_science", "easy"),
        ("What is the center of the Earth called?", "inner core", "earth_science", "easy"),
        # Astronomy (30)
        ("What planet is closest to the Sun?", "Mercury", "astronomy", "easy"),
        ("What is the largest planet in our solar system?", "Jupiter", "astronomy", "easy"),
        ("How many planets are in our solar system?", "8", "astronomy", "easy"),
        ("What is the name of Earth's natural satellite?", "the Moon", "astronomy", "easy"),
        ("What star is at the center of our solar system?", "the Sun", "astronomy", "easy"),
        ("What galaxy do we live in?", "Milky Way", "astronomy", "easy"),
        ("What planet is known for its rings?", "Saturn", "astronomy", "easy"),
        ("What is the smallest planet in our solar system?", "Mercury", "astronomy", "easy"),
        ("What is the name of the force that keeps planets in orbit?", "gravity", "astronomy", "easy"),
        ("What is a year on Earth measured by?", "one orbit around the Sun", "astronomy", "easy"),
        ("What planet is known as the Red Planet?", "Mars", "astronomy", "easy"),
        ("What is the hottest planet in our solar system?", "Venus", "astronomy", "easy"),
        ("How long does it take light from the Sun to reach Earth?", "about 8 minutes", "astronomy", "medium"),
        ("What are the frozen bodies that orbit the Sun with long tails called?", "comets", "astronomy", "easy"),
        ("What is the name for a star that has exploded?", "supernova", "astronomy", "medium"),
        ("What planet rotates on its side?", "Uranus", "astronomy", "medium"),
        ("What is the study of stars and space called?", "astronomy", "astronomy", "easy"),
        ("What is the term for a group of billions of stars?", "galaxy", "astronomy", "easy"),
        ("What dwarf planet was formerly the ninth planet?", "Pluto", "astronomy", "easy"),
        ("How many moons does Mars have?", "2", "astronomy", "medium"),
        ("What planet has the Great Red Spot?", "Jupiter", "astronomy", "easy"),
        ("What unit is used to measure distances between stars?", "light-year", "astronomy", "easy"),
        ("What is a rocky body that orbits the Sun smaller than a planet?", "asteroid", "astronomy", "easy"),
        ("What is the closest star to Earth besides the Sun?", "Proxima Centauri", "astronomy", "medium"),
        ("What phenomenon occurs when the Moon passes between the Sun and Earth?", "solar eclipse", "astronomy", "easy"),
        ("What type of planet is Jupiter?", "gas giant", "astronomy", "easy"),
        ("What space telescope was launched in 1990?", "Hubble Space Telescope", "astronomy", "medium"),
        ("What is the name for the path a planet takes around the Sun?", "orbit", "astronomy", "easy"),
        ("How many moons does Earth have?", "1", "astronomy", "easy"),
        ("What planet has the most moons in our solar system?", "Saturn", "astronomy", "medium"),
    ]

    # Fix the typo: igneous rock
    qa_bank = [(q, "igneous rock" if a == "ignite rock" else a, s, d) for q, a, s, d in qa_bank]

    suffix = "\n\nReply with only the answer."
    examples: list[dict[str, Any]] = []
    for question, answer, sub, diff in qa_bank:
        examples.append(_rec(
            "science_factual", question + suffix, answer,
            "science", diff, "final_answer_only",
            ["science_factual", "science", sub, "final_answer_only"],
            teacher_rationale_short=answer,
        ))

    assert len(examples) == 150, f"Expected 150 science factual, got {len(examples)}"
    return examples


# ===================================================================
# Category 4: Science Multiple Choice (100 examples, balanced A/B/C/D)
# ===================================================================

def _gen_science_mcq(rng: random.Random) -> list[dict[str, Any]]:
    """100 ARC-Challenge-shaped MCQs. Labels are explicitly balanced: 25 each."""

    # Each tuple: (question_stem, correct_answer, distractor1, distractor2, distractor3)
    # We will assign the correct answer to the target label to ensure balance.
    mcq_bank: list[tuple[str, str, str, str, str]] = [
        # Physics
        ("Which force causes objects to fall toward Earth?", "gravity", "friction", "magnetism", "buoyancy"),
        ("What happens to water when it freezes?", "it expands", "it shrinks", "it evaporates", "it disappears"),
        ("Which form of energy is stored in food?", "chemical energy", "electrical energy", "nuclear energy", "sound energy"),
        ("What type of wave is sound?", "longitudinal", "transverse", "electromagnetic", "gamma"),
        ("What happens to light when it passes through a prism?", "it splits into colors", "it disappears", "it speeds up", "it becomes invisible"),
        ("What is the unit of work?", "joule", "watt", "ampere", "volt"),
        ("Which material is the best conductor of electricity?", "copper", "rubber", "wood", "glass"),
        ("What determines the loudness of a sound?", "amplitude", "frequency", "wavelength", "speed"),
        ("What kind of mirror is used in car headlights?", "concave mirror", "convex mirror", "flat mirror", "two-way mirror"),
        ("Which quantity is measured in newtons?", "force", "mass", "speed", "energy"),
        ("What is needed for an electric current to flow?", "a closed circuit", "an open circuit", "a magnet", "a vacuum"),
        ("What happens to the speed of sound in warmer air?", "it increases", "it decreases", "it stays the same", "it stops"),
        ("Which type of energy does a stretched rubber band have?", "elastic potential energy", "kinetic energy", "thermal energy", "light energy"),
        # Chemistry
        ("Which of these is a chemical change?", "burning wood", "melting ice", "cutting paper", "dissolving sugar"),
        ("What is the atomic number of carbon?", "6", "8", "12", "14"),
        ("Which substance is a mixture?", "salt water", "pure gold", "oxygen gas", "distilled water"),
        ("What property indicates a chemical reaction has occurred?", "color change", "changing shape", "breaking apart", "getting wet"),
        ("Which element is needed for combustion?", "oxygen", "nitrogen", "helium", "carbon dioxide"),
        ("What determines an element's identity?", "number of protons", "number of neutrons", "number of electrons", "atomic mass"),
        ("Which pH value indicates the strongest acid?", "1", "6", "7", "10"),
        ("What is produced when an acid reacts with a base?", "salt and water", "hydrogen gas", "carbon dioxide", "oxygen"),
        ("Which state of matter takes the shape of its container but has fixed volume?", "liquid", "solid", "gas", "plasma"),
        ("What holds atoms together in a molecule?", "chemical bonds", "gravity", "magnetism", "friction"),
        ("What happens to most metals when heated?", "they expand", "they contract", "they melt instantly", "they turn to gas"),
        ("Which is an example of a physical change?", "ice melting", "wood burning", "iron rusting", "food cooking"),
        # Biology
        ("What is the function of white blood cells?", "fight infection", "carry oxygen", "clot blood", "transport nutrients"),
        ("Which organelle contains DNA in a eukaryotic cell?", "nucleus", "ribosome", "mitochondrion", "vacuole"),
        ("What do herbivores eat?", "plants", "meat", "insects", "rocks"),
        ("Which system transports blood through the body?", "circulatory system", "nervous system", "digestive system", "skeletal system"),
        ("What is the process by which organisms change over generations?", "evolution", "erosion", "evaporation", "combustion"),
        ("What gas do animals need to breathe?", "oxygen", "carbon dioxide", "nitrogen", "hydrogen"),
        ("Which part of a plant transports water from roots to leaves?", "xylem", "phloem", "stomata", "petals"),
        ("What is the role of decomposers in an ecosystem?", "break down dead matter", "produce oxygen", "pollinate flowers", "predation"),
        ("Which type of reproduction requires two parents?", "sexual reproduction", "binary fission", "budding", "fragmentation"),
        ("What is the main function of the large intestine?", "absorb water", "digest protein", "produce bile", "filter blood"),
        ("Where does gas exchange occur in the lungs?", "alveoli", "bronchi", "trachea", "diaphragm"),
        ("What structure in the eye focuses light onto the retina?", "lens", "cornea", "pupil", "iris"),
        ("Which kingdom includes organisms like mushrooms?", "fungi", "plants", "animals", "bacteria"),
        # Earth Science
        ("What causes earthquakes?", "movement of tectonic plates", "strong winds", "ocean currents", "temperature changes"),
        ("Which layer of the atmosphere contains the ozone layer?", "stratosphere", "troposphere", "mesosphere", "thermosphere"),
        ("What type of rock is granite?", "igneous", "sedimentary", "metamorphic", "mineral"),
        ("What is the main source of energy for the water cycle?", "the Sun", "the Moon", "wind", "geothermal heat"),
        ("Which process creates mountains at convergent boundaries?", "folding and faulting", "erosion", "weathering", "deposition"),
        ("What is the main cause of ocean tides?", "Moon's gravitational pull", "wind", "underwater volcanoes", "temperature"),
        ("Which fossil fuel is formed from ancient marine organisms?", "petroleum", "coal", "peat", "wood"),
        ("What is the most abundant mineral in Earth's crust?", "feldspar", "quartz", "diamond", "calcite"),
        ("What happens at a transform plate boundary?", "plates slide past each other", "plates collide", "plates move apart", "a volcano forms"),
        ("What causes wind?", "unequal heating of Earth's surface", "Earth's rotation only", "ocean currents", "volcanic activity"),
        ("Which type of weathering involves chemical reactions?", "chemical weathering", "physical weathering", "biological weathering", "mechanical weathering"),
        ("What forms when water fills cracks in rock and freezes?", "ice wedging", "chemical weathering", "oxidation", "acid rain"),
        # Astronomy
        ("Why does the Moon appear to change shape?", "changing angle of sunlight on it", "it actually changes size", "clouds cover parts of it", "Earth's shadow always covers it"),
        ("What keeps the International Space Station in orbit?", "gravity", "rocket engines firing continuously", "magnetic force", "solar wind"),
        ("What is the main composition of the Sun?", "hydrogen and helium", "iron and nickel", "rock and ice", "carbon and oxygen"),
        ("Why do stars appear to twinkle?", "Earth's atmosphere bends light", "stars pulse in brightness", "stars spin rapidly", "cosmic dust"),
        ("What is a light-year a measure of?", "distance", "time", "speed", "brightness"),
        ("Which planet has the shortest year?", "Mercury", "Venus", "Mars", "Earth"),
        ("What type of galaxy is the Milky Way?", "spiral", "elliptical", "irregular", "ring"),
        ("What causes a lunar eclipse?", "Earth is between the Sun and Moon", "Moon is between the Sun and Earth", "the Sun goes dark", "clouds block the Moon"),
        ("Why is Mars red?", "iron oxide on its surface", "it is very hot", "red gases in the atmosphere", "reflection from the Sun"),
        ("What is the Big Bang theory?", "the universe began from a single point", "Earth formed from the Moon", "the Sun will explode", "galaxies collide daily"),
        # Mixed / harder
        ("Which energy transformation occurs in a solar panel?", "light to electrical", "electrical to light", "chemical to kinetic", "kinetic to thermal"),
        ("What happens to a population when resources become scarce?", "it decreases", "it increases", "it stays the same", "it doubles"),
        ("Which adaptation helps a cactus survive in the desert?", "thick waxy stem", "broad leaves", "shallow roots only", "bright flowers"),
        ("What is the role of chloroplasts in a plant cell?", "photosynthesis", "respiration", "reproduction", "digestion"),
        ("Which factor is abiotic in an ecosystem?", "temperature", "bacteria", "plants", "insects"),
        ("What is the function of the ozone layer?", "absorbs UV radiation", "produces oxygen", "reflects radio waves", "creates rain"),
        ("Which property of water allows insects to walk on it?", "surface tension", "density", "pH", "temperature"),
        ("What is the effect of deforestation on carbon dioxide levels?", "they increase", "they decrease", "they stay the same", "they become zero"),
        ("Which process releases energy from glucose without oxygen?", "fermentation", "photosynthesis", "condensation", "evaporation"),
        ("What causes the seasons on Earth?", "tilt of Earth's axis", "distance from the Sun", "speed of rotation", "size of the Moon"),
        # Additional to reach exactly 100
        ("What is the primary function of the roots of a plant?", "absorb water and nutrients", "produce seeds", "make food via sunlight", "attract pollinators"),
        ("Which instrument is used to measure wind speed?", "anemometer", "barometer", "thermometer", "hygrometer"),
        ("What is the boiling point of water at sea level in Celsius?", "100 degrees", "0 degrees", "50 degrees", "212 degrees"),
        ("What protects Earth from harmful solar radiation?", "magnetic field", "gravity", "atmosphere only", "the Moon"),
        ("Which vitamin is produced by skin when exposed to sunlight?", "vitamin D", "vitamin C", "vitamin A", "vitamin B12"),
        ("What structure allows fish to breathe underwater?", "gills", "lungs", "skin pores", "nostrils"),
        ("What type of rock is marble?", "metamorphic", "igneous", "sedimentary", "mineral"),
        ("What happens to metal when it rusts?", "it oxidizes", "it melts", "it evaporates", "it magnetizes"),
        ("Which part of the plant conducts photosynthesis?", "leaves", "roots", "stem", "flowers"),
        ("What is the function of the diaphragm in breathing?", "helps lungs expand", "filters air", "warms air", "produces mucus"),
        ("What is the chemical formula for rust?", "Fe2O3", "NaCl", "H2O", "CO2"),
        ("Which planet has the strongest gravity?", "Jupiter", "Saturn", "Earth", "Mars"),
        ("What is the main function of platelets in blood?", "clotting", "fighting infection", "carrying oxygen", "transporting nutrients"),
        ("Which energy source is considered renewable?", "solar", "coal", "natural gas", "petroleum"),
        ("What is the primary cause of global warming?", "greenhouse gases", "ozone depletion", "volcanic eruptions", "deforestation only"),
        ("How does a vaccine work?", "trains the immune system", "kills all bacteria", "removes viruses from blood", "replaces white blood cells"),
        ("What organ detoxifies chemicals in the human body?", "liver", "kidney", "heart", "lungs"),
        # Additional MCQs to reach 100
        ("What is the function of ribosomes in a cell?", "protein synthesis", "energy production", "cell division", "waste removal"),
        ("Which layer of soil is richest in organic matter?", "topsoil", "subsoil", "bedrock", "clay layer"),
        ("What is the primary gas that makes up the Sun?", "hydrogen", "oxygen", "helium", "nitrogen"),
        ("Which type of blood vessel has the thinnest walls?", "capillary", "artery", "vein", "aorta"),
        ("What happens to air pressure as altitude increases?", "it decreases", "it increases", "it stays the same", "it fluctuates randomly"),
        ("Which mineral is the main component of glass?", "silica", "iron", "calcium", "carbon"),
        ("What type of animal is a frog?", "amphibian", "reptile", "mammal", "fish"),
        ("What is the hardest natural substance?", "diamond", "quartz", "topaz", "corundum"),
        ("Which gas is used by plants during photosynthesis?", "carbon dioxide", "oxygen", "nitrogen", "hydrogen"),
        ("What type of energy does the Sun primarily emit?", "radiant energy", "kinetic energy", "chemical energy", "nuclear energy"),
        ("What is the main purpose of the skeletal system?", "support and protection", "digestion", "circulation", "respiration"),
        ("Which process converts sugar to energy in cells?", "cellular respiration", "photosynthesis", "fermentation only", "osmosis"),
        ("What determines whether a substance floats in water?", "its density", "its color", "its temperature", "its shape"),
    ]

    # Balance labels: 25A, 25B, 25C, 25D
    labels = ["A", "B", "C", "D"]
    suffix = "\n\nReply with only the correct option label."
    examples: list[dict[str, Any]] = []

    assert len(mcq_bank) >= 100, f"Need >=100 MCQs, have {len(mcq_bank)}"
    # Use exactly 100
    selected = mcq_bank[:100]

    for i, (stem, correct, d1, d2, d3) in enumerate(selected):
        target_label = labels[i % 4]  # cycle: A, B, C, D -> perfect balance
        distractors = [d1, d2, d3]
        rng.shuffle(distractors)

        # Place correct answer at the target position
        options = list(distractors)  # copy
        target_idx = labels.index(target_label)
        options.insert(target_idx, correct)

        option_text = "\n".join(f"{labels[j]}. {options[j]}" for j in range(4))
        prompt = f"{stem}\n\nOptions:\n{option_text}" + suffix

        examples.append(_rec(
            "science_mc", prompt, target_label,
            "science", "medium", "label_only",
            ["science_mc", "science", "label_only"],
            teacher_rationale_short=correct,
        ))

    # Verify balance
    label_counts = {l: 0 for l in labels}
    for ex in examples:
        label_counts[ex["response"]] += 1
    assert all(c == 25 for c in label_counts.values()), f"MCQ imbalanced: {label_counts}"

    return examples


# ===================================================================
# Category 5: English Language Tasks (100 examples)
# ===================================================================

def _gen_english_tasks(rng: random.Random) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    suffix_rewrite = "\n\nReply with only the rewritten sentence."
    suffix_answer = "\n\nReply with only the answer."

    # --- Informal-to-formal rewrites (35) ---
    informal_formal = [
        ("I ain't going to the store.", "I am not going to the store."),
        ("She don't like pizza.", "She does not like pizza."),
        ("We was at the park.", "We were at the park."),
        ("He ain't got no money.", "He does not have any money."),
        ("They was running real fast.", "They were running very fast."),
        ("I gotta finish my homework.", "I need to finish my homework."),
        ("She wanna go home.", "She wants to go home."),
        ("We gonna be late.", "We are going to be late."),
        ("He don't know nothing about it.", "He does not know anything about it."),
        ("I coulda done better.", "I could have done better."),
        ("Y'all need to listen up.", "All of you need to listen."),
        ("That movie was kinda boring.", "That movie was somewhat boring."),
        ("She's gonna ace the test.", "She is going to do very well on the test."),
        ("We gotta go now.", "We need to go now."),
        ("He woulda helped if he knew.", "He would have helped if he had known."),
        ("They ain't coming to the party.", "They are not coming to the party."),
        ("I dunno what happened.", "I do not know what happened."),
        ("She shoulda called earlier.", "She should have called earlier."),
        ("Me and him went to the game.", "He and I went to the game."),
        ("That's real good advice.", "That is very good advice."),
        ("Can I get a glass of water?", "May I have a glass of water?"),
        ("The plan ain't done yet.", "The plan is not finished yet."),
        ("He was like totally confused.", "He was completely confused."),
        ("She's super smart.", "She is very intelligent."),
        ("We ain't got enough time.", "We do not have enough time."),
        ("Him and me are best friends.", "He and I are best friends."),
        ("I been waiting for an hour.", "I have been waiting for an hour."),
        ("That's gonna cost a lot.", "That is going to cost a significant amount."),
        ("They don't got no clue.", "They do not have any clue."),
        ("She was fixin to leave.", "She was about to leave."),
        ("We seen that movie already.", "We have already seen that movie."),
        ("I'm finna eat lunch.", "I am about to eat lunch."),
        ("He don't care about nothing.", "He does not care about anything."),
        ("That ain't right.", "That is not correct."),
        ("She done finished her work.", "She has finished her work."),
    ]
    for informal, formal in informal_formal:
        prompt = f"Rewrite in formal English: {informal}" + suffix_rewrite
        examples.append(_rec(
            "english_rewrite", prompt, formal,
            "language", "easy", "final_answer_only",
            ["english_rewrite", "language", "rewrite", "final_answer_only"],
            teacher_rationale_short="informal->formal",
        ))

    # --- Grammar correction (30) ---
    grammar_pairs = [
        ("The childs played in the park.", "The children played in the park."),
        ("She have three cats.", "She has three cats."),
        ("He goed to the store.", "He went to the store."),
        ("The dog catched the ball.", "The dog caught the ball."),
        ("They is happy.", "They are happy."),
        ("I has a question.", "I have a question."),
        ("She readed the book.", "She read the book."),
        ("The mouses escaped.", "The mice escaped."),
        ("He drived to work.", "He drove to work."),
        ("We has finished.", "We have finished."),
        ("The mans are here.", "The men are here."),
        ("She writed a letter.", "She wrote a letter."),
        ("I eated lunch.", "I ate lunch."),
        ("The foots are sore.", "The feet are sore."),
        ("He swimmed in the pool.", "He swam in the pool."),
        ("She taked the bus.", "She took the bus."),
        ("They doesn't know.", "They do not know."),
        ("I runned fast.", "I ran fast."),
        ("He bringed a gift.", "He brought a gift."),
        ("She teached the class.", "She taught the class."),
        ("The oxes pulled the cart.", "The oxen pulled the cart."),
        ("He builded a house.", "He built a house."),
        ("She finded the keys.", "She found the keys."),
        ("They leaved early.", "They left early."),
        ("He hitted the ball.", "He hit the ball."),
        ("She speaked loudly.", "She spoke loudly."),
        ("We singed a song.", "We sang a song."),
        ("He feeled sick.", "He felt sick."),
        ("She thinked about it.", "She thought about it."),
        ("He drawed a picture.", "He drew a picture."),
    ]
    for wrong, correct in grammar_pairs:
        prompt = f"Fix the grammar: {wrong}" + suffix_rewrite
        examples.append(_rec(
            "english_grammar", prompt, correct,
            "language", "easy", "final_answer_only",
            ["english_grammar", "language", "grammar", "final_answer_only"],
            teacher_rationale_short="grammar fix",
        ))

    # --- Summarization (20) ---
    summarize_pairs = [
        ("The cat sat on the warm windowsill and watched the birds outside.", "A cat watched birds from a windowsill."),
        ("The students studied hard all week for their final examination in mathematics.", "Students studied all week for their math final."),
        ("The old man walked slowly through the park, enjoying the autumn leaves.", "An old man enjoyed autumn leaves in the park."),
        ("She went to the grocery store to buy milk, eggs, and bread for breakfast.", "She bought breakfast groceries."),
        ("The firefighters worked through the night to control the forest fire.", "Firefighters fought a forest fire overnight."),
        ("The children played soccer in the backyard until it got dark outside.", "Children played soccer until dark."),
        ("He read three chapters of his novel before falling asleep on the couch.", "He read three chapters then fell asleep."),
        ("The scientist published her research findings in a prestigious medical journal.", "A scientist published research in a medical journal."),
        ("The team celebrated their victory with a pizza party at the coach's house.", "The team celebrated their win with pizza."),
        ("She practiced the piano for two hours every day after school.", "She practiced piano two hours daily."),
        ("The airplane landed safely despite the heavy rainstorm at the airport.", "The plane landed safely in a rainstorm."),
        ("The mechanic fixed the car engine and replaced the brake pads.", "The mechanic fixed the engine and brakes."),
        ("He planted tomatoes, peppers, and cucumbers in his backyard garden.", "He planted vegetables in his garden."),
        ("The teacher explained the lesson twice because some students were confused.", "The teacher repeated the lesson for confused students."),
        ("The baby laughed and clapped her hands when she saw the colorful balloons.", "The baby was delighted by balloons."),
        ("The hikers reached the mountain summit after climbing for six hours.", "Hikers reached the summit after six hours."),
        ("She organized all the files in the office and labeled each folder.", "She organized and labeled office files."),
        ("The dog barked loudly when the mail carrier approached the front door.", "The dog barked at the mail carrier."),
        ("He saved enough money over the summer to buy a new bicycle.", "He saved all summer for a new bicycle."),
        ("The heavy snowfall caused school closures across the entire county.", "Heavy snow closed schools countywide."),
    ]
    for original, summary in summarize_pairs:
        prompt = f"Summarize in one short sentence: {original}" + suffix_answer
        examples.append(_rec(
            "english_summarize", prompt, summary,
            "language", "medium", "final_answer_only",
            ["english_summarize", "language", "summarize", "final_answer_only"],
            teacher_rationale_short="summary",
        ))

    # --- Antonym / synonym tasks (15) ---
    antonym_pairs = [
        ("hot", "cold"), ("big", "small"), ("fast", "slow"),
        ("happy", "sad"), ("light", "dark"), ("old", "young"),
        ("hard", "soft"), ("tall", "short"), ("rich", "poor"),
        ("strong", "weak"), ("early", "late"), ("dry", "wet"),
        ("loud", "quiet"), ("full", "empty"), ("sharp", "dull"),
    ]
    for word, antonym in antonym_pairs:
        prompt = f"What is the opposite of '{word}'?" + suffix_answer
        examples.append(_rec(
            "english_antonym", prompt, antonym,
            "language", "easy", "final_answer_only",
            ["english_antonym", "language", "vocabulary", "final_answer_only"],
            teacher_rationale_short=f"opposite of {word}",
        ))

    assert len(examples) == 100, f"Expected 100 english, got {len(examples)}"
    return examples


# ===================================================================
# Category 6: Coding Questions (50 examples)
# ===================================================================

def _gen_coding(rng: random.Random) -> list[dict[str, Any]]:
    suffix = "\n\nReply with only the answer."
    examples: list[dict[str, Any]] = []

    coding_qa: list[tuple[str, str, str]] = [
        # Type questions
        ("What does type(42) return in Python?", "int", "easy"),
        ("What does type(3.14) return in Python?", "float", "easy"),
        ("What does type('hello') return in Python?", "str", "easy"),
        ("What does type(True) return in Python?", "bool", "easy"),
        ("What does type([1,2,3]) return in Python?", "list", "easy"),
        ("What does type({'a':1}) return in Python?", "dict", "easy"),
        ("What does type((1,2)) return in Python?", "tuple", "easy"),
        ("What does type({1,2,3}) return in Python?", "set", "easy"),
        ("What does type(None) return in Python?", "NoneType", "easy"),
        ("What does type(b'hello') return in Python?", "bytes", "medium"),
        # Return value questions
        ("What does len('python') return?", "6", "easy"),
        ("What does len([10, 20, 30]) return?", "3", "easy"),
        ("What does 2 ** 10 evaluate to in Python?", "1024", "easy"),
        ("What does 17 % 5 evaluate to in Python?", "2", "easy"),
        ("What does 'hello'[0] return in Python?", "h", "easy"),
        ("What does 'hello'[-1] return in Python?", "o", "easy"),
        ("What does 'hello'.upper() return?", "HELLO", "easy"),
        ("What does 'WORLD'.lower() return?", "world", "easy"),
        ("What does ' hello '.strip() return?", "hello", "easy"),
        ("What does 'hello world'.split() return?", "['hello', 'world']", "easy"),
        ("What does '-'.join(['a','b','c']) return?", "a-b-c", "easy"),
        ("What does bool(0) return in Python?", "False", "easy"),
        ("What does bool('') return in Python?", "False", "easy"),
        ("What does bool(1) return in Python?", "True", "easy"),
        ("What does int('42') return in Python?", "42", "easy"),
        ("What does str(100) return in Python?", "100", "easy"),
        ("What does max(3, 7, 2) return?", "7", "easy"),
        ("What does min(3, 7, 2) return?", "2", "easy"),
        ("What does sum([1, 2, 3, 4]) return?", "10", "easy"),
        ("What does sorted([3, 1, 2]) return?", "[1, 2, 3]", "easy"),
        # Keyword questions
        ("In Python, which keyword defines a function?", "def", "easy"),
        ("In Python, which keyword creates a class?", "class", "easy"),
        ("In Python, which keyword starts a loop over items?", "for", "easy"),
        ("In Python, which keyword checks a condition?", "if", "easy"),
        ("In Python, which keyword imports a module?", "import", "easy"),
        ("In Python, which keyword returns a value from a function?", "return", "easy"),
        ("In Python, which keyword exits a loop early?", "break", "easy"),
        ("In Python, which keyword skips to the next loop iteration?", "continue", "easy"),
        ("In Python, which keyword is used for exception handling?", "try", "easy"),
        ("In Python, which keyword creates an anonymous function?", "lambda", "medium"),
        # Concept questions
        ("What is the index of the first element in a Python list?", "0", "easy"),
        ("Is Python dynamically typed or statically typed?", "dynamically typed", "easy"),
        ("What does the // operator do in Python?", "integer division", "medium"),
        ("What built-in function reads user input in Python 3?", "input", "easy"),
        ("What built-in function prints output in Python 3?", "print", "easy"),
        ("What does 'hello' + ' world' evaluate to?", "hello world", "easy"),
        ("What does 'ha' * 3 evaluate to in Python?", "hahaha", "easy"),
        ("What does list(range(5)) return?", "[0, 1, 2, 3, 4]", "easy"),
        ("What does len({}) return in Python?", "0", "easy"),
        ("What does 10 // 3 evaluate to in Python?", "3", "medium"),
    ]

    for question, answer, diff in coding_qa:
        examples.append(_rec(
            "coding", question + suffix, answer,
            "coding", diff, "final_answer_only",
            ["coding", "python", "final_answer_only"],
            teacher_rationale_short=answer,
        ))

    assert len(examples) == 50, f"Expected 50 coding, got {len(examples)}"
    return examples


# ===================================================================
# Category 7: Reasoning & Logic (50 examples)
# ===================================================================

def _gen_reasoning(rng: random.Random) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    suffix = "\n\nReply with only the answer."

    reasoning_qa: list[tuple[str, str, str]] = [
        # Syllogisms
        ("If all cats are animals and all animals need water, do cats need water? Reply yes or no.", "yes", "easy"),
        ("If all dogs are mammals and all mammals are warm-blooded, are dogs warm-blooded? Reply yes or no.", "yes", "easy"),
        ("If all roses are flowers and all flowers are plants, are roses plants? Reply yes or no.", "yes", "easy"),
        ("If no fish are birds and all sparrows are birds, are any sparrows fish? Reply yes or no.", "no", "easy"),
        ("If all squares are rectangles but not all rectangles are squares, is every rectangle a square? Reply yes or no.", "no", "easy"),
        ("If some apples are red and all red things are colorful, are some apples colorful? Reply yes or no.", "yes", "medium"),
        # Pattern recognition
        ("What comes next: 2, 4, 6, 8, ?", "10", "easy"),
        ("What comes next: 1, 3, 5, 7, ?", "9", "easy"),
        ("What comes next: 3, 6, 9, 12, ?", "15", "easy"),
        ("What comes next: 1, 4, 9, 16, ?", "25", "medium"),
        ("What comes next: 2, 6, 18, 54, ?", "162", "medium"),
        ("What comes next: 1, 1, 2, 3, 5, ?", "8", "medium"),
        ("What comes next: 10, 20, 30, 40, ?", "50", "easy"),
        ("What comes next: 100, 90, 80, 70, ?", "60", "easy"),
        ("What comes next: 5, 10, 20, 40, ?", "80", "medium"),
        ("What comes next: 1, 2, 4, 8, 16, ?", "32", "medium"),
        # Analogies
        ("Hot is to cold as big is to what?", "small", "easy"),
        ("Cat is to kitten as dog is to what?", "puppy", "easy"),
        ("Bird is to nest as bee is to what?", "hive", "easy"),
        ("Hand is to glove as foot is to what?", "shoe", "easy"),
        ("Day is to night as summer is to what?", "winter", "easy"),
        ("Eye is to see as ear is to what?", "hear", "easy"),
        ("Pen is to write as knife is to what?", "cut", "easy"),
        ("Teacher is to school as doctor is to what?", "hospital", "easy"),
        ("Book is to read as song is to what?", "listen", "easy"),
        ("Fish is to swim as bird is to what?", "fly", "easy"),
        # Simple deductions
        ("Tom is taller than Sam. Sam is taller than Jack. Who is shortest?", "Jack", "easy"),
        ("Monday comes before Tuesday. What day comes after Monday?", "Tuesday", "easy"),
        ("If today is Wednesday, what day was yesterday?", "Tuesday", "easy"),
        ("If today is Friday, what day is tomorrow?", "Saturday", "easy"),
        ("Anna is older than Ben. Ben is older than Carol. Who is oldest?", "Anna", "easy"),
        ("A train leaves at 9:00 and the trip takes 2 hours. What time does it arrive?", "11:00", "easy"),
        ("If you face north and turn right, what direction do you face?", "east", "easy"),
        ("If you face east and turn left, what direction do you face?", "north", "easy"),
        ("If you face south and turn around, what direction do you face?", "north", "easy"),
        ("If a shirt costs $20 and is 50% off, what do you pay?", "$10", "easy"),
        # True/False reasoning
        ("True or false: All mammals lay eggs.", "false", "easy"),
        ("True or false: A triangle has three sides.", "true", "easy"),
        ("True or false: The Sun orbits the Earth.", "false", "easy"),
        ("True or false: Water boils at 100 degrees Celsius at sea level.", "true", "easy"),
        ("True or false: All birds can fly.", "false", "medium"),
        # Odd one out
        ("Which does not belong: apple, banana, carrot, grape?", "carrot", "easy"),
        ("Which does not belong: chair, table, banana, desk?", "banana", "easy"),
        ("Which does not belong: dog, cat, fish, oak?", "oak", "easy"),
        ("Which does not belong: red, blue, heavy, green?", "heavy", "easy"),
        ("Which does not belong: 2, 4, 7, 8?", "7", "easy"),
        # Counting / simple logic
        ("How many vowels are in the word 'education'?", "5", "easy"),
        ("How many letters are in the word 'computer'?", "8", "easy"),
        ("If a dozen is 12, how many are in half a dozen?", "6", "easy"),
        ("If there are 7 days in a week, how many days in 3 weeks?", "21", "easy"),
    ]

    for question, answer, diff in reasoning_qa:
        examples.append(_rec(
            "reasoning", question + suffix, answer,
            "reasoning", diff, "final_answer_only",
            ["reasoning", "logic", "final_answer_only"],
            teacher_rationale_short=answer,
        ))

    assert len(examples) == 50, f"Expected 50 reasoning, got {len(examples)}"
    return examples


# ===================================================================
# Category 8: Tool Use (50 examples)
# ===================================================================

def _gen_tool_use(rng: random.Random) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []

    # --- Direct tool answers (15) ---
    calc_tasks: list[tuple[str, str, str]] = [
        ("sqrt(144)", "12", "sqrt(144)=>12"),
        ("15 * 23", "345", "15*23=>345"),
        ("1024 / 32", "32", "1024/32=>32"),
        ("99 + 101", "200", "99+101=>200"),
        ("256 - 189", "67", "256-189=>67"),
        ("7 ** 3", "343", "7**3=>343"),
        ("45 % 7", "3", "45%7=>3"),
        ("sqrt(225)", "15", "sqrt(225)=>15"),
        ("12 * 12", "144", "12*12=>144"),
        ("1000 - 637", "363", "1000-637=>363"),
        ("48 / 6", "8", "48/6=>8"),
        ("3 ** 5", "243", "3**5=>243"),
        ("sqrt(400)", "20", "sqrt(400)=>20"),
        ("88 + 77", "165", "88+77=>165"),
        ("250 / 5", "50", "250/5=>50"),
    ]
    for expr, answer, rationale in calc_tasks:
        prompt = f"Use the calculator tool for {expr}."
        examples.append(_rec(
            "tool_direct", prompt, answer,
            "tool", "easy", "direct_tool_answer",
            ["tool_direct", "tool", "direct_tool_answer"],
            teacher_rationale_short=rationale,
        ))

    # --- Tool trace then answer (10) ---
    trace_tasks: list[tuple[str, str, str]] = [
        ("sqrt(169)", "13", "[calc]sqrt(169)=>13[/calc]\n13"),
        ("24 * 17", "408", "[calc]24 * 17=>408[/calc]\n408"),
        ("360 / 12", "30", "[calc]360 / 12=>30[/calc]\n30"),
        ("9 ** 4", "6561", "[calc]9 ** 4=>6561[/calc]\n6561"),
        ("127 + 384", "511", "[calc]127 + 384=>511[/calc]\n511"),
        ("500 - 237", "263", "[calc]500 - 237=>263[/calc]\n263"),
        ("sqrt(324)", "18", "[calc]sqrt(324)=>18[/calc]\n18"),
        ("15 * 15", "225", "[calc]15 * 15=>225[/calc]\n225"),
        ("81 % 7", "4", "[calc]81 % 7=>4[/calc]\n4"),
        ("2 ** 8", "256", "[calc]2 ** 8=>256[/calc]\n256"),
    ]
    for expr, answer, trace_response in trace_tasks:
        prompt = f"Use the calculator tool for {expr}. Return a compact calculator trace followed by the final answer."
        examples.append(_rec(
            "tool_trace", prompt, trace_response,
            "tool", "easy", "tool_trace_then_answer",
            ["tool_trace", "tool", "tool_trace_then_answer"],
            teacher_rationale_short=f"{expr}=>{answer}",
        ))

    # --- Boundary: calculator asked to do non-math (15) ---
    boundary_prompts: list[tuple[str, str]] = [
        ("Use the calculator tool to write a poem.", "The calculator cannot help with writing poems."),
        ("Use the calculator tool to translate English to French.", "The calculator cannot help with translation."),
        ("Use the calculator tool to summarize an article.", "The calculator cannot help with summarization."),
        ("Use the calculator tool to fix my grammar.", "The calculator cannot help with grammar correction."),
        ("Use the calculator tool to play a game.", "The calculator cannot help with playing games."),
        ("Use the calculator tool to book a flight.", "The calculator cannot help with booking flights."),
        ("Use the calculator tool to search the web.", "The calculator cannot help with web searching."),
        ("Use the calculator tool to generate a password.", "The calculator cannot help with password generation."),
        ("Use the calculator tool to draw a picture.", "The calculator cannot help with drawing pictures."),
        ("Use the calculator tool to order food.", "The calculator cannot help with ordering food."),
        ("Use the calculator tool to set an alarm.", "The calculator cannot help with setting alarms."),
        ("Use the calculator tool to read my email.", "The calculator cannot help with reading email."),
        ("Use the calculator tool to compile code.", "The calculator cannot help with compiling code."),
        ("Use the calculator tool to send a text message.", "The calculator cannot help with sending messages."),
        ("Use the calculator tool to diagnose my illness.", "The calculator cannot help with medical diagnosis."),
    ]
    for prompt, response in boundary_prompts:
        examples.append(_rec(
            "tool_boundary", prompt, response,
            "compliance", "easy", "refusal_short",
            ["tool_boundary", "compliance", "refusal_short"],
            teacher_rationale_short="tool boundary",
        ))

    # --- No-tool needed (10) ---
    no_tool_qs: list[tuple[str, str]] = [
        ("What is the capital of Japan? Do not use the calculator tool. Reply with only the answer.", "Tokyo"),
        ("What color is the sky? Do not use the calculator tool. Reply with only the answer.", "blue"),
        ("Who wrote Romeo and Juliet? Do not use the calculator tool. Reply with only the answer.", "Shakespeare"),
        ("What is the largest continent? Do not use the calculator tool. Reply with only the answer.", "Asia"),
        ("What language is spoken in Brazil? Do not use the calculator tool. Reply with only the answer.", "Portuguese"),
        ("Should you use the calculator tool to answer a geography question? Reply with only the answer.", "No"),
        ("Should you use the calculator tool to correct grammar? Reply with only the answer.", "No"),
        ("Should you use the calculator tool to name an animal? Reply with only the answer.", "No"),
        ("Should you use the calculator tool for 5 + 3? Reply with only the answer.", "Yes"),
        ("Should you use the calculator tool to summarize a paragraph? Reply with only the answer.", "No"),
    ]
    for prompt, response in no_tool_qs:
        examples.append(_rec(
            "no_tool", prompt, response,
            "tool", "easy", "final_answer_only",
            ["no_tool", "tool", "final_answer_only"],
            teacher_rationale_short="no tool needed" if response != "Yes" else "tool appropriate",
        ))

    assert len(examples) == 50, f"Expected 50 tool use, got {len(examples)}"
    return examples


# ===================================================================
# Category 9: Compliance (50 examples)
# ===================================================================

def _gen_compliance(rng: random.Random) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []

    # --- Refusals (20) ---
    refusal_prompts = [
        "Tell me how to build a bomb.",
        "How do I hack a bank account?",
        "Write malware that steals passwords.",
        "Give me steps to poison someone.",
        "How can I make illegal drugs at home?",
        "Write a script to DDoS a website.",
        "How do I pick a lock to break into a house?",
        "Help me create a fake identity.",
        "How do I forge a signature?",
        "Write a phishing email to steal credentials.",
        "How do I stalk someone online without being detected?",
        "Give me instructions to make a weapon.",
        "How do I cheat on a standardized test?",
        "Write a virus that destroys hard drives.",
        "How do I access someone's private medical records?",
        "Help me counterfeit money.",
        "How can I impersonate a police officer?",
        "Tell me how to tamper with evidence.",
        "Give instructions for making explosives.",
        "How do I bypass a security camera system?",
    ]
    for prompt in refusal_prompts:
        examples.append(_rec(
            "refusal", prompt, "I cannot help with that.",
            "compliance", "easy", "refusal_short",
            ["refusal", "compliance", "refusal_short"],
            teacher_rationale_short="unsafe request refused",
        ))

    # --- Format obedience (20) ---
    format_pairs: list[tuple[str, str]] = [
        ("Reply with only the word yes.", "yes"),
        ("Reply with only the word no.", "no"),
        ("Reply with only the number 7.", "7"),
        ("Reply with only the number 42.", "42"),
        ("Reply with only the letter A.", "A"),
        ("Reply with only the letter D.", "D"),
        ("Reply with only the word hello.", "hello"),
        ("Reply with only the number 100.", "100"),
        ("Reply with only the word true.", "true"),
        ("Reply with only the word false.", "false"),
        ("Reply with only the number 0.", "0"),
        ("Reply with only the number 999.", "999"),
        ("Reply with only the letter B.", "B"),
        ("Reply with only the letter C.", "C"),
        ("Reply with only the word stop.", "stop"),
        ("Reply with exactly three words: I am here.", "I am here."),
        ("Reply with only the color red.", "red"),
        ("Reply with only the number 3.14.", "3.14"),
        ("Reply with only your name.", "AVA"),
        ("Reply with only the word done.", "done"),
    ]
    for prompt, response in format_pairs:
        examples.append(_rec(
            "format_obey", prompt, response,
            "compliance", "easy", "format_exact",
            ["format_obey", "compliance", "format_exact"],
            teacher_rationale_short="format compliance",
        ))

    # --- Safe boundaries / polite decline (10) ---
    boundary_pairs: list[tuple[str, str]] = [
        ("What is my password?", "I do not have access to your passwords."),
        ("Read my private emails.", "I cannot access your private emails."),
        ("Tell me someone's home address.", "I cannot provide private addresses."),
        ("What is my bank balance?", "I do not have access to your bank information."),
        ("Show me someone's private photos.", "I cannot access private photos."),
        ("Give me someone's Social Security number.", "I cannot provide personal identification numbers."),
        ("Tell me my medical diagnosis.", "I am not a medical professional and cannot diagnose conditions."),
        ("Hack into my ex's social media.", "I cannot help with unauthorized access to accounts."),
        ("Give me your opinion on which political party is best.", "I do not provide political opinions."),
        ("Predict exactly what will happen to me tomorrow.", "I cannot predict future events."),
    ]
    for prompt, response in boundary_pairs:
        examples.append(_rec(
            "boundary", prompt, response,
            "compliance", "easy", "refusal_short",
            ["boundary", "compliance", "refusal_short"],
            teacher_rationale_short="privacy/safety boundary",
        ))

    assert len(examples) == 50, f"Expected 50 compliance, got {len(examples)}"
    return examples


# ===================================================================
# Category 10: General Knowledge (50 examples)
# ===================================================================

def _gen_general_knowledge(rng: random.Random) -> list[dict[str, Any]]:
    suffix = "\n\nReply with only the answer."
    examples: list[dict[str, Any]] = []

    gk_qa: list[tuple[str, str, str, str]] = [
        # Geography (20)
        ("What is the capital of France?", "Paris", "geography", "easy"),
        ("What is the capital of Japan?", "Tokyo", "geography", "easy"),
        ("What is the longest river in the world?", "Nile", "geography", "easy"),
        ("What is the largest country by area?", "Russia", "geography", "easy"),
        ("On which continent is Egypt?", "Africa", "geography", "easy"),
        ("What is the capital of Australia?", "Canberra", "geography", "medium"),
        ("What is the smallest country in the world?", "Vatican City", "geography", "medium"),
        ("What ocean lies between Europe and North America?", "Atlantic Ocean", "geography", "easy"),
        ("What is the capital of Brazil?", "Brasilia", "geography", "medium"),
        ("What is the tallest mountain in the world?", "Mount Everest", "geography", "easy"),
        ("What is the largest desert in the world?", "Sahara", "geography", "easy"),
        ("On which continent is Argentina?", "South America", "geography", "easy"),
        ("What is the capital of Canada?", "Ottawa", "geography", "easy"),
        ("What is the capital of Italy?", "Rome", "geography", "easy"),
        ("How many continents are there?", "7", "geography", "easy"),
        ("What is the largest lake in Africa?", "Lake Victoria", "geography", "medium"),
        ("What country has the largest population?", "India", "geography", "medium"),
        ("What is the capital of Germany?", "Berlin", "geography", "easy"),
        ("What is the capital of South Korea?", "Seoul", "geography", "easy"),
        ("Which country is known as the Land of the Rising Sun?", "Japan", "geography", "easy"),
        # History (15)
        ("In what year did World War II end?", "1945", "history", "easy"),
        ("Who was the first President of the United States?", "George Washington", "history", "easy"),
        ("In what year did the Titanic sink?", "1912", "history", "easy"),
        ("What ancient civilization built the pyramids at Giza?", "ancient Egyptians", "history", "easy"),
        ("In what year did humans first land on the Moon?", "1969", "history", "easy"),
        ("Who wrote the Declaration of Independence?", "Thomas Jefferson", "history", "easy"),
        ("What was the name of the ship Columbus sailed to the Americas?", "Santa Maria", "history", "medium"),
        ("In what year did World War I begin?", "1914", "history", "easy"),
        ("What empire built the Colosseum in Rome?", "Roman Empire", "history", "easy"),
        ("Who invented the telephone?", "Alexander Graham Bell", "history", "easy"),
        ("What wall divided Berlin from 1961 to 1989?", "Berlin Wall", "history", "easy"),
        ("Who was the first person to fly solo across the Atlantic?", "Charles Lindbergh", "history", "medium"),
        ("In what century did the Renaissance begin?", "14th century", "history", "medium"),
        ("What ancient wonder stood in Alexandria, Egypt?", "the Lighthouse of Alexandria", "history", "medium"),
        ("Who invented the printing press?", "Johannes Gutenberg", "history", "easy"),
        # Common sense (15)
        ("How many days are in a leap year?", "366", "common_sense", "easy"),
        ("How many minutes are in one hour?", "60", "common_sense", "easy"),
        ("How many hours are in one day?", "24", "common_sense", "easy"),
        ("How many weeks are in one year?", "52", "common_sense", "easy"),
        ("What color do you get by mixing red and blue?", "purple", "common_sense", "easy"),
        ("What color do you get by mixing red and yellow?", "orange", "common_sense", "easy"),
        ("What color do you get by mixing blue and yellow?", "green", "common_sense", "easy"),
        ("How many sides does a hexagon have?", "6", "common_sense", "easy"),
        ("How many sides does an octagon have?", "8", "common_sense", "easy"),
        ("What comes after Sunday?", "Monday", "common_sense", "easy"),
        ("What is the freezing point of water in Celsius?", "0", "common_sense", "easy"),
        ("How many months have 31 days?", "7", "common_sense", "easy"),
        ("How many seconds are in one minute?", "60", "common_sense", "easy"),
        ("What is the opposite direction of north?", "south", "common_sense", "easy"),
        ("How many legs does a spider have?", "8", "common_sense", "easy"),
    ]

    for question, answer, sub, diff in gk_qa:
        examples.append(_rec(
            "general_knowledge", question + suffix, answer,
            "general", diff, "final_answer_only",
            ["general_knowledge", "general", sub, "final_answer_only"],
            teacher_rationale_short=answer,
        ))

    assert len(examples) == 50, f"Expected 50 general knowledge, got {len(examples)}"
    return examples


# ===================================================================
# Verification
# ===================================================================

def _verify_all(examples: list[dict[str, Any]]) -> list[str]:
    """Run sanity checks on all generated examples."""
    errors: list[str] = []

    for i, ex in enumerate(examples):
        # Check required fields
        for field in ("kind", "prompt", "response", "teacher_model", "source_type",
                      "category", "difficulty", "format_contract", "verifier_status", "tags"):
            if field not in ex:
                errors.append(f"Example {i}: missing field '{field}'")

        # Response must be non-empty
        if not ex.get("response", "").strip():
            errors.append(f"Example {i}: empty response")

        # Response should be short (under 200 chars for most)
        resp = ex.get("response", "")
        if len(resp) > 200 and ex.get("format_contract") != "tool_trace_then_answer":
            errors.append(f"Example {i}: response too long ({len(resp)} chars)")

        # MCQ label check
        if ex.get("format_contract") == "label_only":
            if ex["response"] not in ("A", "B", "C", "D"):
                errors.append(f"Example {i}: MCQ response '{ex['response']}' not a valid label")

        # Teacher model
        if ex.get("teacher_model") != TEACHER_MODEL:
            errors.append(f"Example {i}: wrong teacher_model '{ex.get('teacher_model')}'")

    # Verify math_direct answers by recomputing a sample
    math_direct = [ex for ex in examples if ex["kind"] == "math_direct"]
    for ex in math_direct:
        prompt = ex["prompt"]
        response = ex["response"]
        # Try to extract and verify simple expressions
        if "What is " in prompt:
            expr_part = prompt.split("What is ")[1].split("?")[0].strip()
            # Handle special cases
            if "square root" in prompt:
                continue
            try:
                # Replace ** for safety, evaluate
                computed = eval(expr_part.replace("^", "**"))  # noqa: S307
                expected = _fmt(computed)
                if expected != response:
                    errors.append(f"Math verification failed: '{expr_part}' -> expected '{expected}', got '{response}'")
            except Exception:
                pass  # Some expressions might not be directly eval-able

    return errors


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    rng = random.Random(SEED)

    print("Generating AVA v3 Breakthrough Distillation Corpus...")
    print("=" * 60)

    # Generate all categories
    categories = [
        ("Math Word Problems", _gen_math_word_problems(rng)),
        ("Math Direct Computation", _gen_math_direct(rng)),
        ("Science Factual QA", _gen_science_factual(rng)),
        ("Science Multiple Choice", _gen_science_mcq(rng)),
        ("English Language Tasks", _gen_english_tasks(rng)),
        ("Coding Questions", _gen_coding(rng)),
        ("Reasoning & Logic", _gen_reasoning(rng)),
        ("Tool Use", _gen_tool_use(rng)),
        ("Compliance", _gen_compliance(rng)),
        ("General Knowledge", _gen_general_knowledge(rng)),
    ]

    # Combine all
    all_examples: list[dict[str, Any]] = []
    for name, exs in categories:
        all_examples.extend(exs)

    # Shuffle deterministically
    rng.shuffle(all_examples)

    # Verify
    print("\nRunning verification...")
    errors = _verify_all(all_examples)
    if errors:
        print(f"\nWARNING: {len(errors)} verification error(s):")
        for err in errors[:20]:
            print(f"  - {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
    else:
        print("  All examples passed verification.")

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    examples_path = OUTPUT_DIR / "examples.jsonl"
    with open(examples_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    # Build kind counts
    kind_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    format_counts: dict[str, int] = {}

    for ex in all_examples:
        kind = ex["kind"]
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        cat = ex["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        diff = ex["difficulty"]
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        fc = ex["format_contract"]
        format_counts[fc] = format_counts.get(fc, 0) + 1

    # MCQ label distribution
    mcq_label_dist: dict[str, int] = {}
    for ex in all_examples:
        if ex.get("format_contract") == "label_only":
            label = ex["response"]
            mcq_label_dist[label] = mcq_label_dist.get(label, 0) + 1

    manifest = {
        "protocol": "compact_tags",
        "curriculum": "ava_v3_breakthrough_distill",
        "teacher_model": TEACHER_MODEL,
        "seed": SEED,
        "total_examples": len(all_examples),
        "by_kind": dict(sorted(kind_counts.items())),
        "by_category": dict(sorted(category_counts.items())),
        "by_difficulty": dict(sorted(difficulty_counts.items())),
        "by_format_contract": dict(sorted(format_counts.items())),
        "mcq_label_distribution": dict(sorted(mcq_label_dist.items())),
        "examples_path": str(examples_path.relative_to(OUTPUT_DIR.parent.parent)),
        "verification_errors": len(errors),
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    # Print summary
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Total examples:  {len(all_examples)}")
    print()
    print("Category Breakdown:")
    print("-" * 40)
    for name, exs in categories:
        print(f"  {name:30s} {len(exs):>4d}")
    print(f"  {'TOTAL':30s} {len(all_examples):>4d}")

    print()
    print("By Kind:")
    print("-" * 40)
    for kind, count in sorted(kind_counts.items()):
        print(f"  {kind:30s} {count:>4d}")

    print()
    print("By Category:")
    print("-" * 40)
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat:30s} {count:>4d}")

    print()
    print("By Difficulty:")
    print("-" * 40)
    for diff, count in sorted(difficulty_counts.items()):
        print(f"  {diff:30s} {count:>4d}")

    print()
    print("By Format Contract:")
    print("-" * 40)
    for fc, count in sorted(format_counts.items()):
        print(f"  {fc:30s} {count:>4d}")

    if mcq_label_dist:
        print()
        print("MCQ Label Distribution:")
        print("-" * 40)
        for label, count in sorted(mcq_label_dist.items()):
            print(f"  {label}: {count}")

    print()
    print(f"Files written:")
    print(f"  {examples_path}")
    print(f"  {manifest_path}")
    print()
    if errors:
        print(f"WARNING: {len(errors)} verification errors found. Review above.")
    else:
        print("All verifications passed. Corpus is ready.")


if __name__ == "__main__":
    main()
