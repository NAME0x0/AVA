# Performance Targets

Single source of truth for v3 pass/fail. Every target below must be hit at release time, evaluated under the exact conditions listed. If any single row fails, **v3 does not ship**. v2 numbers are from [`docs/RESULTS.md`](../RESULTS.md). Mechanism column links the v3 component that delivers the gain.

All v3 numbers are evaluated on the **packed GGUF (BB1_0 routed + TQ1_0 shared)**, served by `llama-server` head, on the laptop, with `reasoning_budget="auto"` unless the row specifies otherwise.

---

## 1. Strict-dominance gate (every row must pass)

### Reasoning & knowledge

| Benchmark | n | v2 (Q8_0) | v3 target | Mechanism |
|---|---:|---:|---:|---|
| ARC-Easy | 2 376 | 92.0 | **≥ 92.0** | Teacher distillation; don't regress |
| ARC-Challenge | 1 172 | 82.0 | **≥ 85.0** | More capacity + HRM L-loop |
| MMLU (5-shot) | 14 042 | 59.2 | **≥ 70.0** | Teacher KL distillation |
| MMLU-Pro | 12 032 | 30.9 | **≥ 50.0** | Same + capacity |
| HellaSwag | 10 042 | 56.8 | **≥ 70.0** | Larger model + balanced SFT |
| WinoGrande XL | 1 267 | 56.4 | **≥ 70.0** | Same |
| PIQA | 1 838 | 75.9 | **≥ 80.0** | Same |
| BoolQ | 3 270 | 75.0 | **≥ 78.0** | Same |
| TruthfulQA-MC1 | 817 | 47.5 | **≥ 55.0** | DPO |

### Math

| Benchmark | n | v2 (Q8_0) | v3 target | Mechanism |
|---|---:|---:|---:|---|
| GSM8K (greedy, k=1) | 1 319 | 35.3 | **≥ 70.0** | HRM L-loop + OpenMathInstruct-2 + teacher distill |
| GSM8K (k=5 self-cons) | 200 | 44.0 | **≥ 75.0** | Same |
| MATH-500 | 500 | 18.8 | **≥ 50.0** | Same |
| Agentic GSM8K (calc/python) | 1 319 | 35.4 | **≥ 65.0** | Tool discrimination SFT + MCP routing |
| MGSM (en) | 250 | 42.8 | **≥ 65.0** | Larger active capacity + Qwen 3.6 tokenizer |
| MGSM (es) | 250 | 32.0 | **≥ 55.0** | Same |
| MGSM (fr) | 250 | 28.4 | **≥ 50.0** | Same |

### Code

| Benchmark | n | v2 (Q8_0) | v3 target | Mechanism |
|---|---:|---:|---:|---|
| HumanEval+ | 164 | 19.5 | **≥ 50.0** | SYNTHETIC-1 code + teacher KL |
| MBPP+ | 378 | 35.7 | **≥ 55.0** | Same |

### Instruction-following

| Benchmark | n | v2 (Q8_0) | v3 target | Mechanism |
|---|---:|---:|---:|---|
| IFEval (strict) | 541 | 31.6 | **≥ 65.0** | Tulu-3 + DPO |

### Tool use & long context (new in v3)

| Benchmark | n | v2 | v3 target | Mechanism |
|---|---:|---:|---:|---|
| BFCL v3 (Prompt-mode) | full | n/a | **≥ 70.0** | MCP + XGrammar constrained decoding |
| MCP-Bench (28 servers) | full | n/a | **≥ 60.0** | Same |
| Tau-Bench | full | n/a | **≥ 50.0** | Same |
| RULER 8 K | — | n/a | **recall ≥ 95** | Mamba-3 3:1 hybrid |
| RULER 32 K | — | n/a | **recall ≥ 92** | Same |
| RULER 128 K | — | n/a | **recall ≥ 90** | Same + YaRN |
| RULER 256 K | — | n/a | **recall ≥ 80** | YaRN with mscale |
| LongBench v2 | full | n/a | **within 3 pp of Qwen 3.6 27B** | Distilled from same family |
| Passkey 1 M | — | n/a | **≥ 80** | Extended YaRN |

---

## 2. Speed and footprint targets

Evaluated on **RTX A2000 Laptop 4 GB VRAM**, `llama-server` head with FlashAttn + Q8 KV cache (TurboQuant K4/V2 enabled for ≥ 32 K ctx).

| Metric | v2 (Q8_0) | v3 target | Mechanism |
|---|---:|---:|---|
| Decode tok/s @ `reasoning_budget=1`, 2 K ctx | ~50 | **≥ 50** | Mamba-3 linear decode + PrismML kernels |
| Decode tok/s @ `reasoning_budget=auto` (mean 2.5), 2 K ctx | n/a | **≥ 30** | Same |
| Decode tok/s @ `reasoning_budget=6`, 2 K ctx | n/a | **≥ 12** | Same |
| Decode tok/s @ 32 K ctx, `auto` | n/a | **≥ 20** | Subquadratic dominates |
| Prefill tok/s @ 32 K ctx | n/a | **≥ 800** | Chunked Mamba-3 prefill |
| Resident GPU memory @ 8 K ctx | ~1.4 GB | **≤ 2.5 GB** | 1.125 bpw routed |
| Resident GPU memory @ 32 K ctx | n/a | **≤ 3.5 GB** | Q4/Q2 KV cache |
| Resident GPU memory @ 256 K ctx (YaRN) | n/a | **≤ 4.0 GB** (right at the edge) | KV streaming |
| Wall-clock for v3 full eval suite | ~6 h on laptop | **≤ 8 h on laptop** | More benchmarks, faster per-token |

---

## 3. Quality stability targets (no regressions on packed weights)

Tests that the BF16-trained student survives the packing step.

| Metric | BF16 ref | Q-packed gate |
|---|---|---|
| MMLU 5-shot | (measured at P8) | **≤ 1.5 pp drop** |
| ARC-Challenge | (measured at P8) | **≤ 1.5 pp drop** |
| GSM8K greedy | (measured at P8) | **≤ 2.0 pp drop** |
| MATH-500 | (measured at P8) | **≤ 2.0 pp drop** |
| Halting head behavior | (mean L-steps ≈ 2.5) | **mean L-steps within ± 0.3** |

If MMLU drops > 1.5 pp under packing, downgrade routed weights from BB1_0 (1-bit) to TQ1_0 (1.58-bit). See [PRISMML.md](PRISMML.md) §2 fallback.

---

## 4. Behavior and policy targets (not benchmarks)

| Target | Pass criterion |
|---|---|
| Tool call discrimination | Agentic GSM8K call rate ∈ [20 %, 40 %] (v2 was 0.6 %) |
| Refusal calibration | TruthfulQA-MC1 ≥ 55 %; AdvBench refusal rate ≥ 95 % for harmful prompts |
| Multilingual sanity | MGSM all 3 languages ≥ 50 % |
| Safety | No regression on SimpleSafetyTests, ToxicChat, BeaverTails subsets |
| Determinism | `temperature=0, top_k=1` → byte-identical output across runs |

---

## 5. What "release-ready" means in one paragraph

v3 ships when, on the same RTX A2000 laptop that trained v2, the packed GGUF binary:

- beats v2 on every benchmark in §1,
- meets every speed/memory target in §2,
- survives packing within the tolerances in §3,
- and demonstrates the tool/safety/multilingual behavior in §4.

If any single row fails, the release blocks. There is no partial-release path. The closest fallback is "ship as v3-rc1" with explicit caveats on the failing row, but the strict-dominance promise in the README would have to be revised.

---

## 6. References

- v2 baseline: [`docs/RESULTS.md`](../RESULTS.md).
- v3 architecture: [`ARCHITECTURE_V3.md`](ARCHITECTURE_V3.md).
- Pipeline: [`RECIPE.md`](RECIPE.md).
- Risks and fallbacks: [`RISKS.md`](RISKS.md).
