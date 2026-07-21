<div align="center">

<img src="AVA_logo.png" alt="AVA logo" width="160" />

# AVA

**Capable AI on a 4 GB laptop GPU. No cloud. No cluster. No budget.**

[![License](https://img.shields.io/badge/code-MIT-black?style=flat-square)](LICENSE)
[![Model](https://img.shields.io/badge/weights-Qwen_License-black?style=flat-square)](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/LICENSE)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-AVA--v2-yellow?style=flat-square)](https://huggingface.co/NAME0x0/AVA-v2)
[![GGUF](https://img.shields.io/badge/%F0%9F%A4%97-AVA--v2--GGUF-yellow?style=flat-square)](https://huggingface.co/NAME0x0/AVA-v2-GGUF)
[![CI](https://github.com/NAME0x0/AVA/actions/workflows/ci.yml/badge.svg)](https://github.com/NAME0x0/AVA/actions/workflows/ci.yml)
[![Site](https://img.shields.io/badge/site-name0x0.github.io%2FAVA-black?style=flat-square)](https://name0x0.github.io/AVA/)

**[Quickstart](docs/QUICKSTART.md)** · **[Results](docs/RESULTS.md)** · **[Compare](docs/COMPARE.md)** · **[Reproduce](docs/REPRODUCE.md)** · **[Experiments](docs/EXPERIMENTS.md)** · **[Roadmap](docs/ROADMAP.md)**

</div>

---

## TL;DR

AVA v2 is a **42 MB QLoRA adapter** for [Qwen 3.5 2B](https://huggingface.co/Qwen/Qwen3.5-2B), trained and evaluated entirely on a single **4 GB VRAM** laptop GPU.

```
   LAPTOP GPU              AVA v2
 ┌──────────────┐         ┌──────────────────┐
 │ RTX A2000    │  QLoRA  │ ARC-C   82.0%    │
 │ 4 GB VRAM    │ ──────▶ │ MMLU    59.2%    │
 │ Single card  │ 100 min │ GSM8K   44.0%*   │
 └──────────────┘         │ 42 MB adapter    │
                          └──────────────────┘
                              *k=5 self-cons
```

**What's special**: 82% ARC-Challenge on the full 1,172-question test set beats Llama 3.2 3B-Instruct (78.6%) and matches Phi-4-mini 3.8B (83.7%). Training peaks at **1.81 GB VRAM**. Full 17-benchmark, 52,027-instance eval — every number is reproducible end-to-end on the same laptop.

> New here? Read [docs/WHY.md](docs/WHY.md) for the one-paragraph version. Then [docs/QUICKSTART.md](docs/QUICKSTART.md) to actually run it.

---

## Paper

**AVA-v2: Reasoning on a 4 GB Laptop GPU** — a reproducible case study of QLoRA fine-tuning and full-set evaluation of a 2B model on a single 4 GB laptop GPU.

📄 **[Read the paper (PDF)](paper/AVA-v2.pdf)** · preprint (arXiv submission in progress)

It documents the training system, data provenance and a train/test contamination check, the evaluation methodology across 17 benchmarks with 95% Wilson confidence intervals, and the finding that small evaluation subsets substantially misstate compact-model performance.

---

## Try AVA v2 in 30 seconds

```bash
# Easiest — Ollama, no Python
# Download a GGUF from huggingface.co/NAME0x0/AVA-v2-GGUF
ollama create ava-v2 -f Modelfile
ollama run ava-v2
```

Other paths (Python adapter, HuggingFace API): [docs/QUICKSTART.md](docs/QUICKSTART.md).

---

## Headline numbers

Full eval: 17 benchmarks, 52,027 evaluation instances, Q8_0 GGUF, 95% Wilson CI. See [docs/RESULTS.md](docs/RESULTS.md) for the full table.

| Benchmark | n | AVA v2 |
|---|---:|---:|
| ARC-Challenge | 1,172 | **82.0%** |
| ARC-Easy | 2,376 | **92.0%** |
| MMLU 5-shot | 14,042 | **59.2%** |
| PIQA | 1,838 | **75.9%** |
| BoolQ | 3,270 | **75.0%** |
| GSM8K (greedy / k=5) | 1,319 | **35.3% / 44.0%** |

**vs other small models** (full table in [docs/COMPARE.md](docs/COMPARE.md)):

| Model | Params | ARC-C | MMLU | GSM8K |
|---|---:|---:|---:|---:|
| Llama 3.2 1B-Instruct | 1.0B | 59.4 | 49.3 | 44.4 |
| Qwen2.5 1.5B-Instruct | 1.5B | 54.7 | 60.9 | 68.5 |
| Gemma 2 2B-Instruct | 2.0B | 55.7 | 51.3 | 24.3 |
| **AVA v2 (this repo)** | **2.0B** | **82.0** | **59.2** | **35.3 / 44.0** |
| Llama 3.2 3B-Instruct | 3.0B | 78.6 | 63.4 | 77.7 |
| Phi-4-mini 3.8B-Instruct | 3.8B | 83.7 | 67.3 | 88.6 |
| Mistral 7B-Instruct v0.2 | 7.0B | 55.5 | 60.1 | 52.2 |

Where v2 wins: ARC reasoning at 2B beats every same-size and most 3B-class peers. Where v2 loses: math (GSM8K, MATH-500), narrative commonsense (HellaSwag), and tool routing — all targeted by [AVA v3](docs/ROADMAP.md).

---

## What's in this repo

```
AVA/
├── README.md              ← you are here
├── docs/                  ← all the deep docs (start at docs/INDEX.md)
├── site/                  ← marketing + research site (deployed to GitHub Pages)
├── src/ava/               ← core research library (installed as `ava` package)
├── scripts/               ← chat, GGUF conversion, corpus generation
├── experiments/
│   ├── exp4_finetune/     ← 📦 Qwen 3.5 QLoRA — released as AVA v2
│   ├── exp5_gemma4/       ← ⏸ Gemma 4 inference research — paused, files retained
│   └── exp6_v3/           ← 🚧 AVA v3 — ternary MoE student + MCP tools (active)
├── configs/               ← experiment YAML configs
├── corpora/               ← training corpora (JSONL, untracked)
├── tests/                 ← pytest suite
├── Modelfile              ← Ollama model definition
└── .github/               ← CI workflows, issue templates, FUNDING
```

Why three experiments? See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the full progression — what shipped (v2), what's paused and why (Exp 5), what's active (Exp 6 / v3).

---

## Documentation

The repo is documented progressively. Pick your depth:

| Want to... | Read |
|---|---|
| Run AVA v2 right now | [docs/QUICKSTART.md](docs/QUICKSTART.md) |
| See the full eval | [docs/RESULTS.md](docs/RESULTS.md) |
| Compare against other small models | [docs/COMPARE.md](docs/COMPARE.md) |
| Understand why this project exists | [docs/WHY.md](docs/WHY.md) |
| Reproduce the training run | [docs/REPRODUCE.md](docs/REPRODUCE.md) |
| Set up Triton/FLA on Windows | [public gist](https://gist.github.com/NAME0x0/8fe9084e606d3e7ae17d4f1da6a96667) ([repo copy](docs/WINDOWS_SETUP.md)) |
| See every experiment, what shipped, what failed | [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) |
| See where v3 is heading | [docs/ROADMAP.md](docs/ROADMAP.md) |
| Read the **AVA v3 full doc set** (HRM-Text + Mamba-3 + PrismML) | [docs/v3/INDEX.md](docs/v3/INDEX.md) |
| Browse the architecture | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Read the research roadmap (arXiv mapping) | [docs/RESEARCH_ROADMAP.md](docs/RESEARCH_ROADMAP.md) |
| Contribute | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Sponsor | [.github/FUNDING.yml](.github/FUNDING.yml) |

Full index at [docs/INDEX.md](docs/INDEX.md).

---

## Honest limits

AVA v2's weak spots, stated up front:

- **Math** — GSM8K 35% greedy / 44% with self-consistency; MATH-500 19%. Self-consistency is the cheapest reasoning lever before re-training.
- **Tool use** — corpus had ~55 tool examples vs 20K math examples. Model invokes tools in only 0.6% of agentic GSM8K. v3 fixes this with MCP-based routing.
- **Multilingual** — partial transfer: en 42.8% → es 32.0% → fr 28.4% on MGSM.
- **HellaSwag** — 56.8%, below most peers. Fine-tune leaned science + instructions, not narrative commonsense.
- **Sequence length** — max 384 training tokens. Long reasoning chains beyond that not seen during training.

Full caveats in [docs/RESULTS.md](docs/RESULTS.md#caveats).

---

## License

- **Code**: MIT (see [LICENSE](LICENSE))
- **AVA v2 weights**: same as the base model — [Qwen License](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/LICENSE)
- **Documentation and site content**: MIT

---

## Sponsor / Support

If AVA is useful to you, please consider sponsoring development through [GitHub Sponsors](https://github.com/sponsors/NAME0x0). See [.github/FUNDING.yml](.github/FUNDING.yml).

The project costs $0 to keep running, but funding accelerates AVA v3 (longer training runs, broader benchmarks, more tool integrations).

---

## Citation

```bibtex
@misc{mumtaz2026avav2,
  title  = {AVA-v2: Reasoning on a 4 GB Laptop GPU},
  author = {Muhammad Afsah Mumtaz},
  year   = {2026},
  url    = {https://github.com/NAME0x0/AVA}
}
```

Or via [CITATION.cff](CITATION.cff).

---

<div align="center">

Built local. No analytics. No cloud. No budget.

</div>
