# Data Plan

AVA is no longer limited to two disciplines.

The near-term text-first scope is:

- language
- math
- science
- coding

## Data Principles

- Prefer clean, compact, high-information text.
- Keep the first baseline small enough to iterate quickly.
- Treat math, science, and coding as first-class data, not as thin eval-only addons.
- Keep tool traces compact and explicit.
- Store raw corpora outside git.

## Suggested Mixture

- curated language prose and instruction text
- algebra, arithmetic, word problems, proofs, and worked examples
- science textbooks, explanations, factual QA, and worked reasoning examples
- high-quality code, short programming tasks, explanations, tests, and debugging traces
- tool traces for calculator calls
- compact reasoning traces

## Directory Layout

- `corpora/raw/`
  Untracked source files
- `corpora/processed/`
  Untracked packed training text
- `configs/`
  Tracked mixture and model configs
- `sessions/`
  Generated notes, sweep outputs, and the `activity/` audit ledger


