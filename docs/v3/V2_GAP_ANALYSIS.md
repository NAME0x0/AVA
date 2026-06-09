# v2 → v3 Gap Analysis

This document maps every measured AVA v2 weakness to the v3 mechanism that closes it. Source for v2 numbers: [`docs/RESULTS.md`](../RESULTS.md) (full eval, 17 benchmarks, 16 872 tasks, Q8_0 GGUF, 95 % Wilson CI). Source for v3 mechanisms: [`ARCHITECTURE_V3.md`](ARCHITECTURE_V3.md), [`HRM_TEXT.md`](HRM_TEXT.md), [`SUBQUADRATIC.md`](SUBQUADRATIC.md), [`PRISMML.md`](PRISMML.md).

The goal of this doc is to make the v3 claim "better than v2 in every regard" auditable. Each row below has a verifiable v2 number, a named v3 mechanism, and an expected delta.

---

## 1. Capability gaps

| v2 weakness | v2 score | Root cause | v3 mechanism | Expected v3 |
|---|---:|---|---|---:|
| **MATH-500** | 18.8 % | Single-pass dense forward; 384-token CoT cap; weak math corpus | HRM L-loop (latent multi-step reasoning, up to 6 iterations) + 100 K OpenMathInstruct-2 + Bespoke-Stratos | **≥ 50 %** |
| **HumanEval+** | 19.5 % | Tiny code share in v2 SFT; no code-distill from a strong teacher | SYNTHETIC-1 code subset + Tulu-3 + teacher KL on code completions | **≥ 50 %** |
| **MBPP+** | 35.7 % | Same as HumanEval+ root cause | Same fix as above | **≥ 50 %** |
| **GSM8K (greedy)** | 35.3 % | Math weakness + no self-consistency at decode | HRM recurrence does what self-consistency did externally; OpenMathInstruct-2 | **≥ 70 %** |
| **GSM8K (k=5 self-cons)** | 44.0 % | Ceiling on v2 base | n/a — v3 should beat v2's k=5 with k=1 greedy | **≥ 70 %** (k=1) |
| **MMLU 5-shot** | 59.2 % | Capacity ceiling at 2 B dense | Distill from Qwen 3.6 35B-A3B teacher into 3.26 B MoE student | **≥ 70 %** |
| **MMLU-Pro** | 30.9 % | Harder generalization; small model | Larger active capacity (~1.4 B routed + 0.4 B shared) + teacher KL | **≥ 50 %** |
| **IFEval (strict)** | 31.6 % | Tulu-3 underweighted in v2 mix; no DPO | Tulu-3 150 K + DPO 75 K pairs | **≥ 65 %** |
| **Agentic GSM8K (tool call rate)** | 0.6 % | 55 tool examples vs 20 K math examples (corpus bias) | MCP + XGrammar constrained decode + 300-example "when NOT to call" SFT | **call rate ≥ 25 %, accuracy ≥ 50 %** |
| **BFCL v3 Prompt-mode** | not measured | Tool routing not in v2 eval at all | MCP server + FastMCP 3.0 + JSON-schema constrained generation | **≥ 70 %** |
| **MGSM (fr)** | 28.4 % | Tokenizer + small base | Qwen 3.6 tokenizer (better non-English coverage) + larger active capacity | **≥ 50 %** |
| **MGSM (es)** | 32.0 % | Same | Same | **≥ 55 %** |
| **HellaSwag** | 56.8 % | Narrative completion under-represented in v2 SFT | Balanced Tulu-3 mix + more capacity | **≥ 70 %** |
| **TruthfulQA-MC1** | 47.5 % | Small base; alignment data thin | DPO on UltraFeedback + Skywork-Reward | **≥ 55 %** |
| **WinoGrande XL** | 56.4 % | Capacity ceiling | More capacity + distillation | **≥ 70 %** |
| **BoolQ** | 75.0 % | Already strong on v2 | Should stay or improve | **≥ 78 %** |
| **PIQA** | 75.9 % | Already strong | Should stay or improve | **≥ 80 %** |
| **ARC-Easy** | 92.0 % | Already excellent | Should stay; this is the don't-regress check | **≥ 92 %** |
| **ARC-Challenge** | 82.0 % | Already excellent | Larger model + recurrence | **≥ 85 %** |

Every v2 score above is a 95 % Wilson CI lower bound; expected v3 values are targets, not yet measured.

The non-obvious row is **GSM8K (greedy)**: the v3 target of 70 % at `k=1` decode is **higher than v2's k=5 self-consistency (44.0 %)**. The mechanism is HRM L-loop recurrence, which is "self-consistency in latent space" — it pays for the cost of k=5 inside one forward pass rather than across 5 forward passes. If that hypothesis fails, the v3 GSM8K target drops to 60 % k=1 / 70 % k=5.

---

## 2. Architectural gaps

| v2 limit | v2 value | Root cause | v3 mechanism |
|---|---|---|---|
| Max training context | 384 tokens | Memory budget at 4 GB with quadratic attention | Mamba-3 3 : 1 hybrid; 32 K native, YaRN to 256 K, target 1 M |
| Inference context (advertised) | 8 K | Qwen 3.5 base default | 32 K native, 256 K with YaRN |
| Tool-call protocol | none built in | No MCP path in v2 | Full MCP server + client, llama-server `--mcp-config` |
| Decode throughput | ~45 tok/s on RTX A2000 (Q8_0) | Quadratic attention dominates beyond 2 K | Mamba-3 linear decode + TQ2_0 2-bit-aligned kernels (PrismML demonstrated up to 8× FP16 at sub-2 bpw) |
| Quantization floor | Q8 (no further savings) | Single base; no QAT | 1.6875 bpw ternary experts (group-256 QAT) + Q8_0 embeddings; ~3.3 GB resident at 32 K |
| Reasoning budget knob | none | Static dense forward | `reasoning_budget` 1..6 L-steps |

Every architecture row has a single named mechanism in v3.

---

## 3. Capability deltas: line by line, comparable units

```
                          v2          v3 target    Δ
ARC-C                    82.0          ≥ 85        +3.0
ARC-E                    92.0          ≥ 92         0
MMLU                     59.2          ≥ 70       +10.8
MMLU-Pro                 30.9          ≥ 50       +19.1
GSM8K greedy             35.3          ≥ 70       +34.7
GSM8K k=5                44.0          ≥ 75       +31.0
MATH-500                 18.8          ≥ 50       +31.2
HumanEval+               19.5          ≥ 50       +30.5
MBPP+                    35.7          ≥ 50       +14.3
IFEval                   31.6          ≥ 65       +33.4
HellaSwag                56.8          ≥ 70       +13.2
PIQA                     75.9          ≥ 80        +4.1
BoolQ                    75.0          ≥ 78        +3.0
WinoGrande               56.4          ≥ 70       +13.6
TruthfulQA-MC1           47.5          ≥ 55        +7.5
MGSM en                  42.8          ≥ 65       +22.2
MGSM es                  32.0          ≥ 55       +23.0
MGSM fr                  28.4          ≥ 50       +21.6
BFCL v3 (tools)            —           ≥ 70         —
RULER 128 K              n/a           ≥ 90         —
```

If any single row regresses (v3 < v2), v3 does not ship. This is the **strict-dominance gate** for the v3 release. See [`PERF_TARGETS.md`](PERF_TARGETS.md) for the full pass/fail matrix.

---

## 4. What v3 explicitly does *not* attempt

Honest list of things v2 did poorly that v3 will also do poorly:

| v2 weak | v3 plan |
|---|---|
| Vision / audio (any multimodal task) | Out of scope; deferred to v3.1+ |
| Native non-Latin scripts beyond Qwen tokenizer coverage | Inherits Qwen 3.6 coverage; no additional tokenizer work |
| Real-time streaming voice | Out of scope |
| Function-calling formats other than MCP / JSON | Single tool protocol; no OpenAI-style functions adapter in v3 |

These are not v3 → v4 promises. They are explicit non-goals for v3.

---

## 5. References

- [`docs/RESULTS.md`](../RESULTS.md) for v2 numbers.
- [`ARCHITECTURE_V3.md`](ARCHITECTURE_V3.md), [`HRM_TEXT.md`](HRM_TEXT.md), [`SUBQUADRATIC.md`](SUBQUADRATIC.md), [`PRISMML.md`](PRISMML.md) for v3 mechanisms.
- [`PERF_TARGETS.md`](PERF_TARGETS.md) for the formal pass/fail gate.
