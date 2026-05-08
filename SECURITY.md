# Security policy

## Reporting a vulnerability

If you discover a security issue in AVA — a model serving exploit, an unsafe tool path (e.g. python execution sandbox escape), credential leakage, or a supply-chain risk in the build/CI pipeline — please **do not open a public GitHub issue**.

Instead, email **lancearmour24200@gmail.com** with:

1. A clear description of the issue
2. Steps to reproduce
3. The affected component (training script, eval harness, MCP tool, GGUF build, etc.)
4. Optional: a proposed fix or mitigation

You should expect an acknowledgement within 7 days and a substantive response within 30 days.

## Scope

In scope:

- Code in this repository (`src/`, `scripts/`, `experiments/`, `site/`, CI workflows)
- Build pipelines (GitHub Actions, GGUF conversion path)
- The released LoRA adapter and GGUF artifacts on HuggingFace
- The MCP tool catalog when it ships in v3 (the python sandbox is the highest-priority target)

Out of scope:

- Vulnerabilities in upstream dependencies (Qwen base model, PyTorch, BitsAndBytes, llama.cpp, Triton, FLA, etc.) — please report those upstream
- Generic prompt-injection issues with public LLMs
- Issues only reproducible on private forks

## Disclosure

Once a fix is in place, the issue will be disclosed in the changelog with credit to the reporter (unless they prefer to remain anonymous). If a coordinated disclosure window is needed, we'll arrange one.

## Supply chain

AVA pins exact versions of training-critical dependencies in [docs/REPRODUCE.md](docs/REPRODUCE.md) and the CI workflow. The released adapter weights are reproducible from the listed corpus and config — the build is not signed but is bit-stable on the reference hardware. If you need stronger supply-chain guarantees, open a discussion.
