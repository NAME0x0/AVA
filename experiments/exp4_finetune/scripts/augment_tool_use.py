"""Generate additional tool-use training examples.

Augments the v2 corpus with diverse calculator/python/memory tool-use examples.
Goal: boost tool-use from 18 → 100+ examples for better agentic learning.
"""
import json
import random
from pathlib import Path

random.seed(42)

OUTPUT = Path("D:/AVA/experiments/exp4_finetune/corpora/tool_use_augmented.jsonl")

examples: list[dict] = []


# === Calculator: Diverse math scenarios ===
calc_examples = [
    # Unit conversions
    {"prompt": "Convert 5 kilometers to miles.", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "5 * 0.621371"}}</tool_call>\n\n5 kilometers is approximately 3.11 miles.'},
    {"prompt": "How many centimeters are in 4.5 feet?", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "4.5 * 30.48"}}</tool_call>\n\n4.5 feet is 137.16 centimeters.'},
    {"prompt": "Convert 200 pounds to kilograms.", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "200 * 0.453592"}}</tool_call>\n\n200 pounds is approximately 90.72 kilograms.'},
    {"prompt": "How many liters are in 3 gallons?", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "3 * 3.78541"}}</tool_call>\n\n3 gallons is approximately 11.36 liters.'},
    {"prompt": "Convert 72 degrees Fahrenheit to Celsius.", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "(72 - 32) * 5 / 9"}}</tool_call>\n\n72°F is approximately 22.2°C.'},

    # Percentages and ratios
    {"prompt": "If a shirt costs $45 and is 30% off, what's the sale price?", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "45 * (1 - 0.30)"}}</tool_call>\n\nThe sale price is $31.50.'},
    {"prompt": "What percentage of 250 is 37?", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "37 / 250 * 100"}}</tool_call>\n\n37 is 14.8% of 250.'},
    {"prompt": "If I scored 84 out of 120 on a test, what's my percentage?", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "84 / 120 * 100"}}</tool_call>\n\nYou scored 70.0%.'},
    {"prompt": "A product's price increased from $80 to $96. What's the percentage increase?", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "(96 - 80) / 80 * 100"}}</tool_call>\n\nThe price increased by 20%.'},

    # Geometry
    {"prompt": "What's the volume of a cylinder with radius 5cm and height 12cm?", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "3.14159265 * 5**2 * 12"}}</tool_call>\n\nThe volume is approximately 942.48 cubic centimeters.'},
    {"prompt": "Find the hypotenuse of a right triangle with sides 7 and 24.", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "sqrt(7**2 + 24**2)"}}</tool_call>\n\nThe hypotenuse is 25.'},
    {"prompt": "What's the circumference of a circle with diameter 14 inches?", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "3.14159265 * 14"}}</tool_call>\n\nThe circumference is approximately 43.98 inches.'},
    {"prompt": "Calculate the area of a trapezoid with parallel sides 8 and 12, and height 5.", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "(8 + 12) / 2 * 5"}}</tool_call>\n\nThe area of the trapezoid is 50 square units.'},

    # Financial
    {"prompt": "If I invest $10,000 at 5% annual interest compounded monthly for 5 years, how much will I have?", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "10000 * (1 + 0.05/12) ** (12*5)"}}</tool_call>\n\nAfter 5 years, you\'ll have approximately $12,833.59.'},
    {"prompt": "What's 20% tip on a $87.50 restaurant bill?", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "87.50 * 0.20"}}</tool_call>\n\nA 20% tip is $17.50, making the total $105.00.'},
    {"prompt": "If I earn $52,000 per year, what's my monthly and weekly gross pay?", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "52000 / 12"}}</tool_call>\n\nMonthly: approximately $4,333.33.\n\n<tool_call>{"name": "calculator", "arguments": {"expression": "52000 / 52"}}</tool_call>\n\nWeekly: $1,000.00.'},
    {"prompt": "Split a $156 bill among 4 people with 18% tip.", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "156 * 1.18 / 4"}}</tool_call>\n\nEach person pays $46.02 (including 18% tip).'},

    # Speed, time, distance
    {"prompt": "A plane flies 3,500 miles in 7 hours. What's its average speed?", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "3500 / 7"}}</tool_call>\n\nThe average speed is 500 mph.'},
    {"prompt": "If light travels at 186,000 miles per second, how far does it travel in one minute?", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "186000 * 60"}}</tool_call>\n\nLight travels 11,160,000 miles in one minute.'},
    {"prompt": "A car accelerates from 0 to 60 mph in 8 seconds. What's the acceleration in mph/s?", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "60 / 8"}}</tool_call>\n\nThe acceleration is 7.5 mph per second.'},

    # Statistics
    {"prompt": "What's the average of 85, 92, 78, 95, 88, 90?", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "(85 + 92 + 78 + 95 + 88 + 90) / 6"}}</tool_call>\n\nThe average is 88.0.'},
    {"prompt": "Calculate the BMI for someone who is 5'10\" (178 cm) and weighs 165 lbs (75 kg).", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "75 / (1.78 ** 2)"}}</tool_call>\n\nThe BMI is approximately 23.67, which is in the normal weight range (18.5-24.9).'},

    # Scientific
    {"prompt": "How many atoms are in 2.5 moles of water?", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "2.5 * 6.022e23 * 3"}}</tool_call>\n\nThere are approximately 4.52 × 10²⁴ atoms in 2.5 moles of water (each water molecule has 3 atoms).'},
    {"prompt": "What's the kinetic energy of a 1500 kg car moving at 25 m/s?", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "0.5 * 1500 * 25**2"}}</tool_call>\n\nThe kinetic energy is 468,750 joules (about 469 kJ).'},
    {"prompt": "Calculate the wavelength of light with frequency 5.5 × 10^14 Hz.", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "3e8 / 5.5e14"}}</tool_call>\n\nThe wavelength is approximately 545 nanometers, which is green light.'},
]
examples.extend(calc_examples)


# === Python: Diverse programming tasks ===
python_examples = [
    # Data processing
    {"prompt": "Count the vowels in the sentence: 'The quick brown fox jumps over the lazy dog'", "response": '<tool_call>{"name": "python", "arguments": {"code": "s = \'The quick brown fox jumps over the lazy dog\'\\nvowels = sum(1 for c in s.lower() if c in \'aeiou\')\\nprint(f\'Vowels: {vowels}\')\\nfor v in \'aeiou\':\\n    print(f\'  {v}: {s.lower().count(v)}\')"}}</tool_call>\n\nThere are 11 vowels in that sentence: e(3), o(2), u(2), i(1), a(1), and more.'},
    {"prompt": "Find the most common word in this text: 'the cat sat on the mat and the cat ate the rat'", "response": '<tool_call>{"name": "python", "arguments": {"code": "from collections import Counter\\ntext = \'the cat sat on the mat and the cat ate the rat\'\\ncounts = Counter(text.split())\\nfor word, count in counts.most_common(5):\\n    print(f\'{word}: {count}\')"}}</tool_call>\n\n"the" appears 4 times, "cat" appears 2 times, and the rest appear once each.'},
    {"prompt": "Generate a random password with 16 characters including uppercase, lowercase, digits, and symbols.", "response": '<tool_call>{"name": "python", "arguments": {"code": "import random\\nimport string\\nchars = string.ascii_letters + string.digits + \'!@#$%^&*\'\\npassword = \'\'.join(random.choice(chars) for _ in range(16))\\nprint(f\'Password: {password}\')\\nprint(f\'Length: {len(password)}\')\\nprint(f\'Has upper: {any(c.isupper() for c in password)}\')\\nprint(f\'Has digit: {any(c.isdigit() for c in password)}\')"}}</tool_call>\n\nHere\'s a randomly generated 16-character password with mixed characters.'},

    # Algorithms
    {"prompt": "Implement binary search and find 42 in [1, 5, 12, 23, 35, 42, 58, 67, 89, 91].", "response": '<tool_call>{"name": "python", "arguments": {"code": "def binary_search(arr, target):\\n    low, high = 0, len(arr) - 1\\n    steps = 0\\n    while low <= high:\\n        steps += 1\\n        mid = (low + high) // 2\\n        print(f\'Step {steps}: checking index {mid} (value={arr[mid]})\')\\n        if arr[mid] == target:\\n            return mid, steps\\n        elif arr[mid] < target:\\n            low = mid + 1\\n        else:\\n            high = mid - 1\\n    return -1, steps\\n\\narr = [1, 5, 12, 23, 35, 42, 58, 67, 89, 91]\\nidx, steps = binary_search(arr, 42)\\nprint(f\'\\\\nFound 42 at index {idx} in {steps} steps\')"}}</tool_call>\n\nBinary search found 42 at index 5 in just a few steps — much faster than scanning the whole list.'},
    {"prompt": "Check if 97 is a prime number.", "response": '<tool_call>{"name": "python", "arguments": {"code": "def is_prime(n):\\n    if n < 2:\\n        return False\\n    for i in range(2, int(n**0.5) + 1):\\n        if n % i == 0:\\n            print(f\'{n} is divisible by {i}\')\\n            return False\\n    return True\\n\\nresult = is_prime(97)\\nprint(f\'97 is {\"\" if result else \"not \"}prime\')"}}</tool_call>\n\n97 is indeed a prime number — it has no divisors other than 1 and itself.'},
    {"prompt": "Calculate the factorial of 15.", "response": '<tool_call>{"name": "python", "arguments": {"code": "def factorial(n):\\n    result = 1\\n    for i in range(2, n + 1):\\n        result *= i\\n    return result\\n\\nprint(f\'15! = {factorial(15):,}\')"}}</tool_call>\n\n15! = 1,307,674,368,000.'},

    # Data analysis
    {"prompt": "Given test scores [88, 92, 75, 95, 87, 91, 78, 82, 96, 85], find the mean, median, and standard deviation.", "response": '<tool_call>{"name": "python", "arguments": {"code": "scores = [88, 92, 75, 95, 87, 91, 78, 82, 96, 85]\\nmean = sum(scores) / len(scores)\\nsorted_s = sorted(scores)\\nmid = len(sorted_s) // 2\\nmedian = (sorted_s[mid-1] + sorted_s[mid]) / 2\\nvariance = sum((x - mean)**2 for x in scores) / len(scores)\\nstd = variance ** 0.5\\nprint(f\'Mean: {mean:.1f}\')\\nprint(f\'Median: {median:.1f}\')\\nprint(f\'Std Dev: {std:.1f}\')\\nprint(f\'Min: {min(scores)}, Max: {max(scores)}\')"}}</tool_call>\n\nThe class scored well overall with a mean around 87, median around 86.5, and relatively tight standard deviation.'},
    {"prompt": "Create a frequency table for the letters in 'mississippi'.", "response": '<tool_call>{"name": "python", "arguments": {"code": "from collections import Counter\\nword = \'mississippi\'\\ncounts = Counter(word)\\nfor letter, count in sorted(counts.items(), key=lambda x: -x[1]):\\n    bar = \'█\' * count\\n    print(f\'  {letter}: {count} {bar}\')\\nprint(f\'\\\\nTotal: {len(word)} letters, {len(counts)} unique\')"}}</tool_call>\n\n"mississippi" has 11 letters with s(4), i(4), p(2), m(1) — heavily dominated by s and i.'},

    # String manipulation
    {"prompt": "Reverse every word in the sentence 'Hello World Python Programming' without reversing the word order.", "response": '<tool_call>{"name": "python", "arguments": {"code": "s = \'Hello World Python Programming\'\\nresult = \' \'.join(word[::-1] for word in s.split())\\nprint(f\'Original: {s}\')\\nprint(f\'Reversed: {result}\')"}}</tool_call>\n\nEach word is reversed individually: "olleH dlroW nohtyP gnimmargorP".'},
    {"prompt": "Check if two strings are anagrams: 'listen' and 'silent'.", "response": '<tool_call>{"name": "python", "arguments": {"code": "def are_anagrams(a, b):\\n    return sorted(a.lower()) == sorted(b.lower())\\n\\npairs = [(\'listen\', \'silent\'), (\'hello\', \'world\'), (\'cinema\', \'iceman\')]\\nfor a, b in pairs:\\n    result = are_anagrams(a, b)\\n    print(f\\\"{a} & {b}: {\'anagram\' if result else \'not anagram\'}\\\")"}}</tool_call>\n\n"listen" and "silent" are anagrams! They contain the exact same letters rearranged.'},
    {"prompt": "Convert the decimal number 255 to binary, octal, and hexadecimal.", "response": '<tool_call>{"name": "python", "arguments": {"code": "n = 255\\nprint(f\'Decimal: {n}\')\\nprint(f\'Binary:  {bin(n)}\')\\nprint(f\'Octal:   {oct(n)}\')\\nprint(f\'Hex:     {hex(n)}\')\\nprint(f\'\\\\nBinary breakdown: {n:08b}\')"}}</tool_call>\n\n255 in binary is 11111111 (all 8 bits set), in octal is 377, and in hex is ff.'},

    # Matrix/list operations
    {"prompt": "Find all duplicate numbers in [4, 2, 7, 3, 4, 8, 2, 9, 1, 7, 5, 2].", "response": '<tool_call>{"name": "python", "arguments": {"code": "from collections import Counter\\nnums = [4, 2, 7, 3, 4, 8, 2, 9, 1, 7, 5, 2]\\ncounts = Counter(nums)\\nduplicates = {n: c for n, c in counts.items() if c > 1}\\nprint(f\'Duplicates:\')\\nfor num, count in sorted(duplicates.items()):\\n    print(f\'  {num} appears {count} times\')"}}</tool_call>\n\nThe duplicates are: 2 (appears 3 times), 4 (twice), and 7 (twice).'},
    {"prompt": "Generate Pascal's triangle up to row 8.", "response": '<tool_call>{"name": "python", "arguments": {"code": "rows = 8\\ntriangle = [[1]]\\nfor i in range(1, rows):\\n    prev = triangle[-1]\\n    row = [1] + [prev[j] + prev[j+1] for j in range(len(prev)-1)] + [1]\\n    triangle.append(row)\\nfor i, row in enumerate(triangle):\\n    padding = \' \' * (rows - i) * 2\\n    nums = \'  \'.join(f\'{n:>3}\' for n in row)\\n    print(f\'{padding}{nums}\')"}}</tool_call>\n\nPascal\'s triangle shows the binomial coefficients. Each number is the sum of the two numbers above it.'},

    # Date/time
    {"prompt": "What day of the week was July 4, 1776?", "response": '<tool_call>{"name": "python", "arguments": {"code": "from datetime import date\\nd = date(1776, 7, 4)\\nprint(f\'July 4, 1776 was a {d.strftime(\\\"%A\\\")}\')"}}</tool_call>\n\nJuly 4, 1776 (American Independence Day) was a Thursday.'},
    {"prompt": "How many weekdays are there between March 1 and June 30, 2026?", "response": '<tool_call>{"name": "python", "arguments": {"code": "from datetime import date, timedelta\\nstart = date(2026, 3, 1)\\nend = date(2026, 6, 30)\\nweekdays = 0\\ncurrent = start\\nwhile current <= end:\\n    if current.weekday() < 5:\\n        weekdays += 1\\n    current += timedelta(days=1)\\nprint(f\'Weekdays from {start} to {end}: {weekdays}\')"}}</tool_call>\n\nThere are approximately 87 weekdays between March 1 and June 30, 2026.'},
]
examples.extend(python_examples)


# === Multi-tool examples ===
multi_tool_examples = [
    {"prompt": "A rectangular garden is 12.5 meters by 8.3 meters. Calculate the area, then generate a Python function to compute the cost of fencing it at $15 per meter.", "response": 'Let me calculate the area first.\n\n<tool_call>{"name": "calculator", "arguments": {"expression": "12.5 * 8.3"}}</tool_call>\n\nThe area is 103.75 square meters.\n\nNow let me calculate the perimeter and fencing cost:\n\n<tool_call>{"name": "python", "arguments": {"code": "def fencing_cost(length, width, price_per_meter):\\n    perimeter = 2 * (length + width)\\n    cost = perimeter * price_per_meter\\n    return perimeter, cost\\n\\np, c = fencing_cost(12.5, 8.3, 15)\\nprint(f\'Perimeter: {p} meters\')\\nprint(f\'Fencing cost: ${c:.2f}\')"}}</tool_call>\n\nThe perimeter is 41.6 meters, and fencing at $15/meter would cost $624.00.'},

    {"prompt": "Calculate the average speed of a runner who completed a 10K race in 48 minutes 35 seconds, and show the pace per kilometer.", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "10 / (48 + 35/60) * 60"}}</tool_call>\n\nThe average speed is about 12.35 km/h.\n\n<tool_call>{"name": "python", "arguments": {"code": "total_seconds = 48 * 60 + 35\\npace_seconds = total_seconds / 10\\npace_min = int(pace_seconds // 60)\\npace_sec = int(pace_seconds % 60)\\nspeed_kmh = 10 / (total_seconds / 3600)\\nprint(f\'Total time: {total_seconds} seconds\')\\nprint(f\'Pace: {pace_min}:{pace_sec:02d} per km\')\\nprint(f\'Speed: {speed_kmh:.2f} km/h\')"}}</tool_call>\n\nThe runner\'s pace was 4:51 per kilometer at 12.35 km/h. That\'s a solid 10K time!'},

    {"prompt": "I have a list of exam scores: 67, 82, 94, 71, 88, 95, 73, 86, 91, 79. Calculate the mean, then write code to determine how many students scored above average.", "response": '<tool_call>{"name": "calculator", "arguments": {"expression": "(67 + 82 + 94 + 71 + 88 + 95 + 73 + 86 + 91 + 79) / 10"}}</tool_call>\n\nThe mean is 82.6.\n\n<tool_call>{"name": "python", "arguments": {"code": "scores = [67, 82, 94, 71, 88, 95, 73, 86, 91, 79]\\nmean = sum(scores) / len(scores)\\nabove = [s for s in scores if s > mean]\\nbelow = [s for s in scores if s <= mean]\\nprint(f\'Mean: {mean}\')\\nprint(f\'Above average ({len(above)}): {sorted(above)}\')\\nprint(f\'At or below average ({len(below)}): {sorted(below)}\')"}}</tool_call>\n\n5 students scored above the average of 82.6: those with scores 86, 88, 91, 94, and 95.'},
]
examples.extend(multi_tool_examples)


# === Memory tool examples ===
memory_examples = [
    {"prompt": "Remember that my project deadline is March 28th.", "response": 'I\'ll note that.\n\n<tool_call>{"name": "search_memory", "arguments": {"query": "project deadline"}}</tool_call>\n\nSaved. Your project deadline is March 28th. I\'ll keep that in mind for future conversations.'},
    {"prompt": "What do you remember about my preferences?", "response": 'Let me check.\n\n<tool_call>{"name": "search_memory", "arguments": {"query": "user preferences settings"}}</tool_call>\n\nHere\'s what I recall about your preferences from our past conversations.'},
    {"prompt": "Have we talked about machine learning before?", "response": 'Let me search my memory.\n\n<tool_call>{"name": "search_memory", "arguments": {"query": "machine learning discussion topics"}}</tool_call>\n\nHere\'s what I found from our previous conversations about machine learning.'},
    {"prompt": "Remember: I prefer Python over JavaScript for backend work.", "response": 'Got it, I\'ll remember that.\n\n<tool_call>{"name": "search_memory", "arguments": {"query": "programming language preference backend"}}</tool_call>\n\nNoted! You prefer Python for backend development. I\'ll keep that in mind when making suggestions.'},
]
examples.extend(memory_examples)


# === "When NOT to use tools" examples (important for learning discrimination) ===
no_tool_examples = [
    {"prompt": "What is the capital of France?", "response": "The capital of France is Paris."},
    {"prompt": "Who wrote Romeo and Juliet?", "response": "William Shakespeare wrote Romeo and Juliet, believed to have been written between 1594 and 1596."},
    {"prompt": "What's the difference between HTTP and HTTPS?", "response": "HTTPS is the secure version of HTTP. The key difference is encryption:\n\n- **HTTP**: Data is sent in plain text, anyone intercepting can read it\n- **HTTPS**: Uses TLS/SSL encryption to protect data in transit\n\nHTTPS also verifies the server's identity through certificates, preventing man-in-the-middle attacks. Modern web standards require HTTPS for any site handling sensitive data."},
    {"prompt": "Can you explain what DNA is in simple terms?", "response": "DNA (deoxyribonucleic acid) is like a biological instruction manual inside every cell of your body.\n\nIt's a long, twisted molecule (the famous double helix) made up of just 4 chemical 'letters': A, T, C, and G. The order of these letters spells out instructions for building and maintaining your body — everything from eye color to how your cells divide.\n\nKey facts:\n- Every cell in your body has the same DNA\n- You get half from each parent\n- Human DNA is about 3 billion letters long\n- 99.9% of DNA is identical between all humans\n\nThat 0.1% difference is what makes each person unique."},
    {"prompt": "What is 2 + 2?", "response": "2 + 2 = 4."},
    {"prompt": "What year did World War II end?", "response": "World War II ended in 1945. Germany surrendered on May 8 (V-E Day) and Japan surrendered on August 15 (V-J Day), with the formal surrender signed on September 2, 1945."},
    {"prompt": "Explain the concept of Big O notation.", "response": "Big O notation describes how an algorithm's performance scales as input size grows. It's about the worst-case growth rate, ignoring constants.\n\nCommon complexities (fastest to slowest):\n- **O(1)** — Constant: array lookup by index\n- **O(log n)** — Logarithmic: binary search\n- **O(n)** — Linear: scanning a list\n- **O(n log n)** — Linearithmic: merge sort\n- **O(n²)** — Quadratic: nested loops\n- **O(2ⁿ)** — Exponential: brute-force subset problems\n\nWhy it matters: an O(n²) algorithm on 1 million items takes ~1 trillion operations. An O(n log n) algorithm takes ~20 million. That's the difference between hours and milliseconds."},
    {"prompt": "Thank you for your help!", "response": "You're welcome! Feel free to ask if you need anything else."},
]
examples.extend(no_tool_examples)


# Write output
random.shuffle(examples)
with open(OUTPUT, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Generated {len(examples)} tool-use augmentation examples")
print(f"  Calculator: {len(calc_examples)}")
print(f"  Python: {len(python_examples)}")
print(f"  Multi-tool: {len(multi_tool_examples)}")
print(f"  Memory: {len(memory_examples)}")
print(f"  No-tool (discrimination): {len(no_tool_examples)}")
print(f"Written to: {OUTPUT}")
