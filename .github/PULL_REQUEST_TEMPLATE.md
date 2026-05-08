## Summary

<!-- One or two sentences. What does this PR change? -->

## Why

<!-- Link the issue or describe the motivation. -->

## What changed

<!-- Bullet list of the concrete changes. -->

-

## Numbers (if applicable)

<!-- If this PR changes a benchmark, training config, or anything in docs/RESULTS.md, paste the run summary here and link the JSON. -->

## Tests

- [ ] `pytest -q` passes locally
- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `mypy src tests` passes
- [ ] If this changes training behavior, I ran a smoke run and confirmed loss converged

## Hardware tested on

<!-- Optional but very useful for AVA. Example: RTX A2000 Laptop, 4 GB VRAM, Windows 11, Python 3.13 -->

## Checklist

- [ ] My change is one focused concern (no drive-by refactors)
- [ ] I updated docs if behavior or interfaces changed
- [ ] I did not add experimental files into a published `experiments/expN_*/` directory
