# References

All citations behind the v3 documentation set. Every numerical claim or design decision in `docs/v3/` should be traceable to a row below.

---

> **Edge portfolio citations** (memory layers, Coconut, EAGLE-3, NSA, precision
> scaling laws, μP, Absolute Zero, Titans) live with their claims in
> [EDGES.md](EDGES.md) §References.

## Architecture & reasoning

- Wang et al., **Hierarchical Reasoning Model**, arXiv:[2506.21734](https://arxiv.org/abs/2506.21734). HRM-base, 27 M params, two-stack recurrence, ARC-AGI/Sudoku/maze. Foundation for the v3 H/L recurrence pattern.
- Sapient Intelligence, **HRM-Text launch**, PR Newswire, 18 May 2026 ([release](https://www.prnewswire.com/news-releases/sapient-intelligence-launches-hrm-text-challenging-the-llm-monopoly-with-a-brain-inspired-foundation-model-trained-on-up-to-1000x-fewer-tokens-302774638.html)). 1 B HRM scale-up. Independent April 2026 numbers: MMLU 60.7, ARC-C 81.9, MATH 56.2, DROP 82.2.
- Sapient Intelligence GitHub, [`sapientinc/HRM`](https://github.com/sapientinc/HRM). Reference implementation of HRM-base.
- Graves, **Adaptive Computation Time for Recurrent Neural Networks**, arXiv:[1603.08983](https://arxiv.org/abs/1603.08983). ACT ponder loss used in v3 P4 halting curriculum.
- ARC Prize, **The Hidden Drivers of HRM's Performance on ARC-AGI** ([blog](https://arcprize.org/blog/hrm-analysis)). Independent ablations: hierarchy ≈ ≤ 5 pp; outer refinement loop is the dominant lever (+13 pp from 1→2 loops; 16-loop training ≈ 2×). Basis for v3 design responses D1/D4 in [HRM_TEXT.md](HRM_TEXT.md) §1b.
- **Are Your Reasoning Models Reasoning or Guessing? A Mechanistic Analysis of Hierarchical Reasoning Models**, arXiv:[2601.10679](https://arxiv.org/abs/2601.10679). Fixed-point traps, grokking-style transitions; perturbation + bootstrapping lifts Sudoku-Extreme 54.5 % → 96.9 %. Basis for D2/D3 (convergence-aware halting, latent restart escape).

## Subquadratic attention

- Yang et al., **Gated Delta Networks: Improving Mamba2 with Delta Rule**, OpenReview [r8H7xhYPwz](https://openreview.net/forum?id=r8H7xhYPwz). Gated DeltaNet — v3 fallback subquadratic kernel.
- Mamba-3 authors, **Mamba-3: Improved Sequence Modeling using State Space Principles**, arXiv:[2603.15569](https://arxiv.org/abs/2603.15569) (ICLR 2026). Complex SSM update, MIMO formulation, half-state-size at equal perplexity. Primary v3 L-block sublayer.
- Peng et al., **A Systematic Analysis of Hybrid Linear Attention**, arXiv:[2507.06457](https://arxiv.org/abs/2507.06457). 2026 generation-3 taxonomy for linear attention.
- `fla-org/flash-linear-attention`, GitHub [link](https://github.com/fla-org/flash-linear-attention). Triton kernel library; `fla.layers.mamba3` and `fla.layers.gated_deltanet` both shipped (Mamba-3 merged April 2026, verified June 2026).
- Peng, **YaRN: Efficient Context Window Extension**, arXiv:[2309.00071](https://arxiv.org/abs/2309.00071). YaRN extension recipe used to stretch native 32 K → 256 K → 1 M.

## Quantization & deployment

- PrismML, **Bonsai 8B launch (1-bit)**, PR Newswire, 31 March 2026 ([release](https://www.prnewswire.com/news-releases/prismml-launches-worlds-first-1-bit-ai-model-to-redefine-intelligence-at-the-edge-302730568.html)). 1.125 bpw with group-128 FP16 scales.
- PrismML, **Ternary Bonsai 1.7 B / 4 B / 8 B (1.58-bit)**, Morningstar press, 16 April 2026 ([release](https://www.morningstar.com/news/pr-newswire/20260416sf36656/prismml-introduces-ternary-bonsai-model-family)).
- Miller, **PrismML exits stealth**, Machine Herald, April 2026 ([article](https://machineherald.io/article/2026-04/03-prismml-exits-stealth-with-first-commercially-viable-1-bit-large-language-models-fitting-an-8b-parameter-model-in-115-gb/)). Technical detail on group-128 FP16 scale.
- Du et al., **BitNet b1.58 2B4T Technical Report**, arXiv:[2504.12285](https://arxiv.org/abs/2504.12285). Ternary 1.58-bit baseline for the routed-expert path.
- Yan et al., **MoTE: Mixture of Ternary Experts**, NeurIPS 2025. MoTE pattern: ternary routed experts + BF16 shared expert.
- Du et al., **BitDistiller**, arXiv:[2402.10631](https://arxiv.org/abs/2402.10631). 3-stage QAT-via-distillation, the training scaffold v3 P3–P5 inherits.
- `ggml-org/llama.cpp`, GitHub [link](https://github.com/ggml-org/llama.cpp). Upstream hosts `TQ1_0` (1.6875 bpw) / `TQ2_0` (2.0625 bpw) ternary types with group-256 scales (2024, Compilade) — the formats v3 ships. PrismML's `BB1_0` is fork-only as of June 2026.
- `ggml-org/llama.cpp` discussion [#22019](https://github.com/ggml-org/llama.cpp/discussions/22019) — Bonsai group-128 vs upstream group-256 incompatibility; PrismML fork status. Trigger for the v3 group-256 pivot.

## Distillation, MoE, and base teachers

- Qwen Team, **Qwen3 Technical Report**, arXiv:[2505.09388](https://arxiv.org/abs/2505.09388). Open tokenizer, thinking budget; v3 tokenizer = Qwen 3.6 family.
- Qwen Team, **Qwen3.6-35B-A3B** model card on Hugging Face. 256 experts / top-8 + 1 shared, Gated DeltaNet 3:1, 256 K context. v3 teacher.
- Jiang et al., **Mixtral of Experts**, arXiv:[2401.04088](https://arxiv.org/abs/2401.04088). Sparse MoE baseline.
- Dai et al., **DeepSeekMoE**, arXiv:[2401.06066](https://arxiv.org/abs/2401.06066). Expert specialization techniques inherited by Qwen 3.6.
- Fedus, Zoph, Shazeer, **Switch Transformers**, arXiv:[2101.03961](https://arxiv.org/abs/2101.03961). Original sparse MoE design.

## Tool use

- Schick et al., **Toolformer**, arXiv:[2302.04761](https://arxiv.org/abs/2302.04761). Few-shot tool teaching with structured traces.
- ToolACE-R, arXiv:[2504.01400](https://arxiv.org/abs/2504.01400). Iterative tool-data refinement; informs v3 P8 "when NOT to call" set.
- ReTool, arXiv:[2504.11536](https://arxiv.org/abs/2504.11536). Tool-use abstention — v3 design hypothesis on tool discrimination.
- ToRL, arXiv:[2503.23383](https://arxiv.org/abs/2503.23383). Tool-integrated RL (out of v3 scope, in v3.1).
- FastMCP 3.0, project home. MCP server framework used in `experiments/exp6_v3/mcp/`.
- XGrammar / llguidance, constrained-decoding frameworks. v3 uses llguidance via llama-server.

## Datasets

- SYNTHETIC-1 (R1 distill) — math/code/STEM, 400–500 K, Apache 2.0.
- `allenai/tulu-3-sft-mixture` — 150 K, Apache 2.0.
- `NVIDIA/OpenMathInstruct-2` — Llama-3.1-405B math traces; sample 100 K.
- `NousResearch/hermes-function-calling-v1` + BFCL v4 train — 50 K, MIT/CC-BY-4.0.
- `bespokelabs/Bespoke-Stratos-17k` — 17 K, CC-BY-NC-4.0 (research-only branch in v3).
- LongAlpaca-12k + ProLong-train — 30 K, Apache 2.0.
- `lordx64/reasoning-distill-opus-4-7-max-sft` — 8 K Opus 4.7 traces, Apache 2.0.
- UltraFeedback binarized + Skywork-Reward — 60 K DPO pairs.
- SYNTHETIC-1-DPO — 11 K pairs, Apache 2.0.

## Evaluation

- BFCL v3 (Prompt-mode) — Berkeley Function Calling Leaderboard, v3 tool-routing gate.
- MCP-Bench — 28 servers / 250 tools, monthly regression gate.
- Tau-Bench / tau2-bench — task-success-based tool evaluation.
- RULER — long-context recall at 8 K – 256 K (v3 native target).
- LongBench v2 — long-context understanding.
- lm-evaluation-harness — host runner for ARC, MMLU, HellaSwag, etc.

## v2 baseline

- `docs/RESULTS.md` — AVA v2 full eval (17 benchmarks, 16 872 tasks, Q8_0 GGUF, 95 % Wilson CI). This is the line v3 must dominate.
- `docs/COMPARE.md` — cross-model comparison snapshot.

## Internal v3 documents (for circular linking)

- [INDEX.md](INDEX.md)
- [WHY_V3.md](WHY_V3.md)
- [ARCHITECTURE_V3.md](ARCHITECTURE_V3.md)
- [HRM_TEXT.md](HRM_TEXT.md)
- [SUBQUADRATIC.md](SUBQUADRATIC.md)
- [PRISMML.md](PRISMML.md)
- [V2_GAP_ANALYSIS.md](V2_GAP_ANALYSIS.md)
- [RECIPE.md](RECIPE.md)
- [PERF_TARGETS.md](PERF_TARGETS.md)
- [RISKS.md](RISKS.md)
- [`experiments/exp6_v3/DESIGN.md`](../../experiments/exp6_v3/DESIGN.md) — patched companion design doc.
- [`docs/ROADMAP.md`](../ROADMAP.md) — patched to reflect HRM / Mamba-3 / PrismML pillars.
