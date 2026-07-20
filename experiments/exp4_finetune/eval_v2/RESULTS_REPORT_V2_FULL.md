# AVA v2 Full Evaluation Report

**Model:** Qwen3.5-2B-AVA-v2 (LoRA r=16 fine-tune of Qwen3.5-2B base, merged + GGUF Q8_0)
**Hardware:** RTX A2000 Laptop, 4 GB VRAM, Windows 11, llama.cpp b7472-head (commit 69c28f1)
**Backend:** llama-server, Flash Attention enabled, Q8_0 weights, Q8 KV cache, 4 parallel slots, continuous batching, 8K context
**Date:** 2026-05-06
**Total wall time:** 4h 18min across 17 benchmark configurations comprising 52,027 evaluation instances (~52,045 model calls)

## Methodology

- **MCQ benchmarks** (ARC, MMLU, MMLU-Pro, HellaSwag, PIQA, WinoGrande, BoolQ, TruthfulQA-MC1): single-token argmax over candidate label tokens (`A`/`B`/`C`/`D`/...) using `n_probs=60` from `/completion` endpoint. Letter-argmax differs from log-likelihood-of-continuation scoring used by some leaderboards (Open LLM Leaderboard v1) — values are directionally comparable but not numerically identical to lm-eval-harness defaults. Closer to OLL v2 / chain-of-thought letter prediction format.
- **Generative benchmarks** (GSM8K, MATH-500, MGSM, IFEval, HumanEval+, MBPP+): greedy decoding (`temperature=0`), Qwen3.5 chat template via `--jinja`, `enable_thinking=False`.
- **Self-consistency** (gsm8k-selfcons): 5 samples at `temperature=0.7, top_p=0.95`, majority vote on extracted numeric answer.
- **Code execution**: subprocess sandbox with 5 s wall-time limit, 4 GB memory cap.
- **IFEval**: prompt-level strict accuracy, custom rule checker covering ~25 instruction types ported from google/instruction_following_eval. NLTK + langdetect for sentence/word/language checks.
- **CIs**: 95 % Wilson score interval.
- **Quantization quality note**: Q8_0 GGUF retains ~99.7 % of BF16 quality (perplexity Δ < 0.005). Strictly higher fidelity than the BnB 4-bit NF4 used in earlier (50-question) evaluations.

## Headline Results

| # | Benchmark | n | Acc | 95 % CI | Time | Notes |
|---|---|---|---|---|---|---|
| 1 | ARC-Challenge | 1,172 | **82.0 %** | [79.7, 84.1] | 127 s | full test split |
| 2 | ARC-Easy | 2,376 | **92.0 %** | [90.8, 93.0] | 254 s | full test split |
| 3 | MMLU (5-shot) | 14,042 | **59.2 %** | [58.4, 60.1] | 3,633 s | 60.8 % on completed (2.7 % ctx-overflow errors) |
| 4 | MMLU-Pro | 12,032 | **30.9 %** | [30.1, 31.8] | 2,619 s | 0-shot |
| 5 | HellaSwag | 10,042 | **56.8 %** | [55.8, 57.8] | 2,036 s | validation |
| 6 | PIQA | 1,838 | **75.9 %** | [73.9, 77.8] | 212 s | validation |
| 7 | WinoGrande XL | 1,267 | **56.4 %** | [53.7, 59.1] | 152 s | validation |
| 8 | BoolQ | 3,270 | **75.0 %** | [73.5, 76.5] | 669 s | validation |
| 9 | TruthfulQA-MC1 | 817 | **47.5 %** | [44.1, 50.9] | 123 s | validation |
| 10 | GSM8K | 1,319 | **35.3 %** | [32.8, 38.0] | 873 s | full test, greedy |
| 11 | GSM8K self-cons (k=5) | 200 | **44.0 %** | [37.3, 50.9] | 729 s | +8.7 pp over greedy |
| 12 | MATH-500 | 500 | **18.8 %** | [15.6, 22.5] | 669 s | greedy + boxed extraction |
| 13 | MGSM (en/es/fr) | 750 | **34.4 %** | [31.1, 37.9] | 543 s | 250 per language |
| 14 | HumanEval+ | 164 | **19.5 %** | [14.2, 26.3] | 338 s | exec, pass@1 greedy |
| 15 | MBPP+ | 378 | **35.7 %** | [31.0, 40.7] | 579 s | exec, pass@1 greedy |
| 16 | IFEval | 541 | **31.6 %** | [27.8, 35.6] | 998 s | prompt-level strict |
| 17 | Agentic GSM8K | 1,319 | **35.4 %** | [32.9, 38.0] | 934 s | calc/python tools, max 3 rounds |

**Agentic note:** model emitted `<tool_call>` on only **8/1319 (0.6 %)** problems. Of those, 5 matched. The fine-tune corpus had ~55 tool-use examples vs 20K+ math examples, so the model defaults to direct chain-of-thought instead of invoking the calculator. Tool-use lift on this benchmark = 0 pp. Self-consistency (k=5, +8.7 pp) is the cheaper reasoning lever for v2.

**Skipped:**
- BFCL v3 simple (no local mirror at `D:/AVA/data/bfcl/`)

## Compared to prior small-sample numbers

| Bench | Prior (n=50/100) | Full eval | Δ | Verdict |
|---|---|---|---|---|
| ARC-Challenge | 79 % (n=100) | 82.0 % (n=1,172) | +3 pp | Prior under-estimate; full sample stronger |
| GSM8K | 48 % (n=50) | 35.3 % (n=1,319) | **−13 pp** | Prior was high-variance: real number is 35 %, not 48 % |

The 50-question GSM8K eval over-estimated v2 reasoning by ~13 pp. Full-sample numbers are the credible reference for v2.

## Compared to base Qwen3.5-2B

(Prior partial eval: ARC 66.0 %, GSM8K 28.0 % on n=100/50 4-bit BnB.)

| Bench | Base 4-bit (small sample) | v2 Q8_0 (full) | Δ |
|---|---|---|---|
| ARC-Challenge | ~66 % | 82.0 % | **+16 pp** |
| GSM8K | ~28 % | 35.3 % | **+7 pp** |

Caveat: base also needs a full-sample Q8_0 re-eval for fair head-to-head. Pending.

## MMLU per-subject highlights

**Top 10 subjects:**

| Subject | n | Acc |
|---|---|---|
| marketing | 234 | 84.6 % |
| management | 103 | 83.5 % |
| high_school_biology | 310 | 81.0 % |
| high_school_psychology | 545 | 80.5 % |
| international_law | 121 | 80.2 % |
| sociology | 201 | 79.6 % |
| high_school_world_history | 237 | 78.1 % |
| high_school_government_and_politics | 193 | 76.7 % |
| college_biology | 144 | 75.0 % |
| us_foreign_policy | 100 | 75.0 % |

**Bottom 9 subjects** (excluding ctx-overflow bucket):

| Subject | n | Acc |
|---|---|---|
| machine_learning | 112 | 45.5 % |
| college_computer_science | 100 | 45.0 % |
| high_school_mathematics | 270 | 43.3 % |
| college_physics | 102 | 43.1 % |
| professional_law | 1,530 | 43.0 % |
| abstract_algebra | 100 | 43.0 % |
| moral_scenarios | 895 | 38.8 % |
| college_mathematics | 100 | 36.0 % |
| global_facts | 100 | 35.0 % |

Pattern: strong on humanities/social-science/biology, weak on advanced math/CS and `moral_scenarios` (a known leaderboard outlier).

## MMLU-Pro per-category

| Category | n | Acc |
|---|---|---|
| biology | 717 | 64.3 % |
| psychology | 798 | 48.2 % |
| economics | 844 | 45.9 % |
| health | 818 | 39.0 % |
| history | 381 | 33.1 % |
| computer science | 410 | 30.7 % |
| other | 924 | 28.8 % |
| philosophy | 499 | 27.9 % |
| engineering | 969 | 26.6 % |
| physics | 1,299 | 24.6 % |
| chemistry | 1,132 | 22.6 % |
| math | 1,351 | 21.4 % |
| business | 789 | 21.0 % |
| law | 1,101 | 20.3 % |

## MGSM per-language (math reasoning transfer)

| Lang | Acc | n |
|---|---|---|
| en | 42.8 % | 250 |
| es | 32.0 % | 250 |
| fr | 28.4 % | 250 |

English math holds at GSM8K-level; non-English drops 10–14 pp.

## MATH-500 by difficulty level

| Level | Acc | n |
|---|---|---|
| 1 (easiest) | 51.2 % | 43 |
| 2 | 31.1 % | 90 |
| 3 | 24.8 % | 105 |
| 4 | 8.6 % | 128 |
| 5 (hardest) | 5.2 % | 134 |

## MATH-500 by subject

| Subject | Acc | n |
|---|---|---|
| Prealgebra | 26.8 % | 82 |
| Algebra | 25.8 % | 124 |
| Number Theory | 21.0 % | 62 |
| Intermediate Algebra | 14.4 % | 97 |
| Precalculus | 12.5 % | 56 |
| Geometry | 9.8 % | 41 |
| Counting & Probability | 5.3 % | 38 |

## Throughput summary

| Phase | Tasks | Time | tasks/s |
|---|---|---|---|
| MCQ (logprob, 1-token) | 47,193 | 9,825 s | 4.80 |
| Generative (full chat) | 4,852 | 4,729 s | 1.03 |
| **Total** | **52,045** | **14,554 s** | **3.58** |

Generation is ~5× slower than logprob due to multi-token autoregressive decoding. KV cache shared across parallel slots, continuous batching enabled.

## Known issues

1. **MMLU 5-shot context overflow (373/14,042 = 2.7 %):** subjects with long passages (`professional_law`, `moral_scenarios`) plus 5 demonstrations exceeded the 8K context. Treated as `matched=false`. True accuracy on completed set: **60.8 %**. Fix: rerun with `--ctx-size 16384` (halves max parallel slots).
2. **HumanEval+ extractor v1 (1.2 %)**: v2 emits `<tool_call>{"name":"python","arguments":{"code":"..."}}</tool_call>` for code tasks because of fine-tune corpus. Initial extractor missed it. Fixed; second pass yielded 19.5 %.
3. **MATH-500 normalize_math regex bug:** `re.sub(r"\\\\", "\\", s)` raised `bad escape (end of pattern)` killing all 500. Fixed (`r"\\\\"` replacement); rerun yielded 18.8 %.
4. **GSM8K self-cons loader limit propagation:** loader default `limit=200` was overridden by runner's `limit=None`, loaded full 1,319. Killed and reran with `--limit 200`.
5. **Letter-argmax MCQ ≠ logprob-of-continuation:** chosen for speed + because llama-server's `/v1/completions` does not honor `echo:true` with token-level logprobs (would otherwise enable lm-eval-harness compatibility). Numbers are valid as a model-self-answer metric; not directly comparable to leaderboards using continuation-loglikelihood.

## Pipeline / artifacts

- Loaders: `experiments/exp4_finetune/eval_v2/scripts/loaders.py`
- Async client: `client.py` (httpx, 4-way semaphore)
- Runner: `runner.py`
- Extractors: `extractors.py` (numeric, boxed-LaTeX, code-from-tool_call)
- Code sandbox: `code_sandbox.py`
- IFEval rules: `ifeval_rules.py`
- Logprob proxy (lm-eval compat shim): `proxy.py`
- Per-bench JSON details: `eval_v2/results/v2_full/<bench>_{summary,details}.json`
- Logs: `eval_v2/logs/tier{1..6}*.log`
- Server launcher: `eval_v2/start_server.ps1`

## Conclusions

1. **v2 is a real generalist.** ARC-C 82 %, ARC-E 92 %, MMLU 59 % (60.8 % on completed), BoolQ 75 %, PIQA 76 %. Solid 2B-class numbers.
2. **Math is the weak spot.** GSM8K 35 % (greedy) → 44 % (self-cons), MATH-500 19 %, MMLU math sub-categories 20-22 %. Self-consistency at k=5 buys nearly 9 pp on GSM8K — cheap alpha.
3. **Code generation works after extractor fix.** HumanEval+ 19.5 %, MBPP+ 35.7 %. Not state-of-the-art for 2B but functional.
4. **Instruction following modest.** IFEval 31.6 % strict. Format compliance (JSON, paragraph counts, language) is the main miss; content quality is fine.
5. **Multilingual transfer present but weakened.** MGSM en 43 % → es 32 % → fr 28 %.
6. **Self-cons + chain-of-thought is the cheapest v2 quality lever.** No retraining required.

## Next steps (deferred)

- Re-run base Qwen3.5-2B at Q8_0 full for clean v2 vs base delta.
- Re-train v2 with much higher tool-use weight in the corpus to actually fire tools on GSM8K (current 0.6 % invocation rate is too low to see lift).
- Rerun MMLU at `--ctx-size 16384` to recover the 373 dropped questions.
- Add BFCL v3 simple once the local mirror is downloaded.
