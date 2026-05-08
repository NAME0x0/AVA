# Contributing to AVA

AVA is a personal research project, but contributions are welcome. The goal: keep AVA running on 4 GB VRAM, keep results honest, keep the repo small enough that one person can hold it in their head.

## Quick links

- [docs/INDEX.md](docs/INDEX.md) — start here if you want to read the project
- [docs/QUICKSTART.md](docs/QUICKSTART.md) — run AVA v2 in 5 minutes
- [docs/REPRODUCE.md](docs/REPRODUCE.md) — train AVA v2 from scratch
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) — what's been tried

## Ways to help

| Effort | Impact | Examples |
|---|---|---|
| Tiny | Big | Typo fixes, broken-link fixes, doc clarifications, README polish |
| Small | Medium | New benchmark integrations, new corpus recipes, eval harness fixes |
| Medium | Big | Reproduce AVA v2 on a different GPU and report numbers, port to ROCm/Metal |
| Large | Huge | Help build AVA v3 (ternary MoE student, MCP server, distillation pipeline) |

If you're not sure where to start, open an issue describing what you'd like to do.

## Development setup

```bash
git clone https://github.com/NAME0x0/AVA.git
cd AVA
pip install -e ".[dev,bench,train]"
pre-commit install
```

Run the test suite:

```bash
pytest -q
```

Run linters:

```bash
ruff check .
ruff format --check .
mypy src tests
```

## Pull request rules

- One concern per PR. Small, focused PRs merge fast.
- Add or update a test for behavior changes.
- If you change a number on the README or docs/RESULTS.md, link the run that produced it.
- Keep `experiments/` immutable once published. New experiments go in new subdirs (`experiments/expN_<topic>/`).
- Keep claims falsifiable. If you can't reproduce it, don't write it.

## Coding style

- `ruff` and `ruff format` are authoritative.
- Type-annotate new public functions.
- Prefer one short doc-comment over multi-paragraph docstrings.
- Don't add abstractions for hypothetical future requirements.

## Reporting bugs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml). Include:

1. The exact command you ran
2. The exact error (full traceback)
3. OS, Python version, GPU, VRAM, CUDA version
4. Whether the failure is reproducible

## Reporting evaluation issues

If you re-run a benchmark and get materially different numbers from those in [docs/RESULTS.md](docs/RESULTS.md):

1. Open an issue tagged `eval-discrepancy`
2. Include the exact GGUF file, llama-server version, sampling settings
3. Attach the per-task JSON if you have it

This is the most valuable kind of bug report. Reproducibility is the whole point of the project.

## Adding a benchmark

1. Add the benchmark to `src/ava/external_benchmarks.py` (registry).
2. Add a runner to `experiments/exp4_finetune/eval_v2/scripts/`.
3. Document the prompting / scoring protocol in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).
4. Run it on AVA v2, attach the JSON, and update [docs/RESULTS.md](docs/RESULTS.md).

## License

By contributing, you agree your contributions ship under the project's MIT license (code) and the Qwen license (released model weights).

## Code of conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
