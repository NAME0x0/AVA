# Teacher Distillation SOP

This document defines how strong teacher models should generate training data for AVA.

The goal is not to make teachers sound impressive. The goal is to make teacher outputs usable, comparable, compact, and safe for a small student model trained under a strict 4 GB VRAM budget.

## Purpose

AVA needs a distillation and data-generation engine that can:
- inject useful knowledge into a much smaller student
- teach cleaner output formats and better decision boundaries
- improve tool use without bloating the model
- improve public-benchmark transfer without training directly on benchmark items
- remain fully inspectable and reproducible

This SOP exists so multiple teacher models can contribute data under the same rules.

## Student Reality

The current student is small and capacity-limited.

Teacher outputs must respect that reality.

Do not assume AVA can absorb:
- long essays
- verbose chain-of-thought
- broad frontier-model world knowledge in one pass
- noisy or contradictory supervision
- inconsistent formatting conventions

Teacher data should optimize for:
- correctness
- compactness
- consistency
- calibration
- token efficiency
- strong task framing

## Hard Constraints

- AVA is trained locally with small-model budgets
- every example must be traceable and auditable
- no hidden benchmark contamination
- no copyrighted bulk copying
- no unsupported tools or capabilities
- no black-box teacher behavior in the final packet

## Allowed Teacher Roles

A teacher may act in one or more of these roles:

1. Answer teacher
Produces the target answer in the exact format AVA should learn.

2. Tool teacher
Teaches when to use a tool, when not to use it, and what compact tool traces should look like.

3. Reasoning teacher
Produces short latent reasoning or short rationale fields that can be used internally during data curation. These should usually not become the final visible student output.

4. Verifier teacher
Checks correctness, formatting, safety, and benchmark contamination risk.

5. Repair teacher
Takes a bad student answer and rewrites it into a better target answer or a better training example.

## Non-Negotiable Rules

Every teacher must follow these rules.

### 1. No benchmark leakage
Do not reproduce public benchmark questions verbatim if those benchmarks are part of AVA evaluation.

Do not create near-duplicates of:
- GSM8K test items
- ARC-Challenge validation or test items
- other tracked held-out benchmark items

Allowed:
- benchmark-shaped examples
- same task style
- same answer format
- same difficulty family

Not allowed:
- paraphrases that are obviously derived from a held-out benchmark item
- answer-key memorization packets
- public-test contamination disguised as synthetic data

### 2. Optimize for student learnability
Prefer:
- short answers
- stable format
- compact prompts
- low-ambiguity supervision
- balanced class coverage
- small but high-information examples

Avoid:
- long rationales
- rambling explanations
- stylistic flourish
- unnecessary token overhead
- multi-paragraph outputs unless the task truly requires them

### 3. Separate visible output from hidden reasoning
If reasoning is generated, keep it in a separate field such as `teacher_rationale_short`.

Default rule:
- visible student target should stay concise
- hidden teacher rationale may exist for audit or verifier use
- do not train AVA to emit long chain-of-thought by default

### 4. Respect AVA's real tool set
Only teach tools that AVA actually supports in the repo.

Right now that means the teacher must not invent:
- web browsing tools
- shell tools
- email tools
- arbitrary APIs
- multimodal tools that AVA does not actually have

### 5. Be explicit about uncertainty
If the teacher is not sure, it must either:
- reject the example
- mark it for verifier review
- simplify the task instead of hallucinating

## Task Families To Generate

Teacher models should focus on these packet types.

### A. Answer-only math
Use for GSM8K-style repair.

Requirements:
- final answer only
- no calculator trace unless the packet explicitly teaches tool use
- numeric formatting must be exact
- keep word problems compact and unambiguous

Good target behavior:
- `18`
- `540`
- `3`

Bad target behavior:
- `The answer is 18.`
- `18 eggs`
- long reasoning in the visible response
- stray tool markup

### B. Multiple-choice label selection
Use for ARC-style repair.

Requirements:
- response must be only the option label
- labels must be balanced across `A`, `B`, `C`, `D`
- do not let one label dominate the packet
- prompts should match the external evaluation format closely enough to transfer

Good target behavior:
- `A`
- `D`

Bad target behavior:
- `C because gravity pulls objects down.`
- `The correct answer is B.`
- option text instead of the label, unless the packet explicitly teaches text-form answers

### C. Tool decision packets
Use to teach:
- when to use the calculator tool
- when not to use it
- when to refuse tool misuse
- exact compact tool traces

### D. Compliance and refusal packets
Use to teach:
- format obedience
- policy refusal
- safe boundary handling
- exact short refusals

### E. Repair packets
Use to fix known student failures.

These should include:
- original student completion
- corrected teacher target
- short note on what failed

## Output Schema

Preferred JSONL schema for teacher-produced examples:

```json
{
  "kind": "math_word",
  "prompt": "Ava had 12 stickers. She gave 3 away and bought 2 more. How many stickers does she have now?\n\nReply with only the final answer.",
  "response": "11",
  "teacher_model": "NAME_HERE",
  "source_type": "synthetic_teacher",
  "category": "math",
  "difficulty": "easy",
  "format_contract": "final_answer_only",
  "teacher_rationale_short": "12 - 3 + 2 = 11",
  "verifier_status": "pass"
}
```

Fields that should usually be present:
- `kind`
- `prompt`
- `response`
- `teacher_model`
- `source_type`
- `category`
- `format_contract`
- `verifier_status`

Fields that are optional but useful:
- `difficulty`
- `teacher_rationale_short`
- `student_failure`
- `repair_note`
- `tags`

## Format Contracts

Every packet should declare one of these contracts when relevant:
- `final_answer_only`
- `label_only`
- `tool_trace_then_answer`
- `direct_tool_answer`
- `refusal_short`
- `format_exact`

The contract matters as much as the answer.

## Quality Rubric

A teacher example is good only if it passes all of these checks.

- Correctness: the answer is actually correct.
- Format fidelity: the answer matches the intended contract exactly.
- Learnability: a small model can plausibly learn from it.
- Non-redundancy: it is not a useless duplicate.
- Non-contamination: it is not a held-out benchmark leak.
- Compactness: it is as short as it can be without losing clarity.

Reject the example if any of these fail.

## Teacher-Side Generation Strategy

Use this order.

1. Identify the task family.
2. Identify the required output contract.
3. Produce the minimal correct answer.
4. Produce an optional short hidden rationale.
5. Run a verifier pass.
6. Reject contaminated or low-quality items.
7. Export as JSONL with metadata.

## Teacher-Side Verifier Checklist

Before an example is accepted, the verifier should ask:
- Is the answer correct?
- Is the format exact?
- Is the prompt benchmark-shaped but not benchmark-copied?
- Is the response compact enough for a small model?
- Is the label distribution balanced if this is MCQ data?
- Does this accidentally teach the wrong tool policy?
- Does this introduce contradictions with existing AVA behavior contracts?

## Distribution Rules

Packets should be balanced.

Examples:
- MCQ packets must not collapse to mostly `C`
- answer-only math should cover more than one numeric surface
- compliance packets should include both success and refusal cases
- tool packets should include both use and non-use cases

## What Teachers Must Not Do

Do not:
- imitate benchmark test items closely
- output long essays when a one-token answer is required
- teach unsupported tools
- overuse chain-of-thought
- create style-heavy or personality-heavy answers
- hide uncertainty behind confident wording
- optimize for sounding smart instead of being learnable

## Using Multiple Teachers

If multiple strong models are used, each should follow the same SOP and write:
- its model name
- packet date
- packet purpose
- any known weaknesses

Then AVA can compare teacher families instead of mixing them blindly.

Recommended multi-teacher setup:
- one teacher for compact direct answers
- one teacher for tool traces and boundaries
- one teacher for verifier and repair review

## Tokenizer Policy

Using a tokenizer from a strong model can help, but only under the right conditions.

Recommended policy:
- prefer an open, inspectable tokenizer or train our own SentencePiece/BPE tokenizer on AVA's target mixture
- prefer tokenizer compatibility with local training and inference tooling
- prefer tokenizers that compress code, math, and English cleanly

Do not assume a closed frontier model tokenizer is available or legally reusable.

For AVA, the best practical choices are usually:
- a tokenizer trained by us on AVA data
- an open tokenizer from a strong open model family

Using a tokenizer from a closed model such as a proprietary frontier system is only worth considering if:
- it is actually accessible
- the license allows use
- the implementation is reproducible
- it measurably improves compression and downstream learning

Tokenizer choice matters, but it will not fix a weak student objective by itself.

## Recommended First Distillation Cycle

For the first teacher cycle, focus on:
- answer-only math word problems
- balanced MCQ label packets
- tool vs no-tool routing
- short refusal and compliance repairs

Do not start with:
- long-form essays
- multimodal generation
- massive world-knowledge dumps
- unrestricted chain-of-thought distillation

## Operational Output

Each teacher run should leave behind:
- the exact teacher prompt used
- the generated JSONL packet
- verifier notes
- contamination notes
- counts by kind and by label
- a short markdown summary of what the teacher tried to teach

That summary should answer:
- what this packet is for
- what it should improve
- what it should not be used to claim

## Bottom Line

Teacher models are not there to impress AVA.
They are there to compress useful behavior into a form a small student can actually learn.

If a teacher packet is elegant but not learnable, reject it.
If it is clever but contaminated, reject it.
If it is short, correct, balanced, and reproducible, keep it.
