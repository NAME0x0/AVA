# Results

AVA v2 evaluated on 17 public benchmarks, 16,872 tasks total. Q8_0 GGUF served via llama-server. 4 h 18 min wall time on the same RTX A2000 4 GB laptop. 95% Wilson confidence intervals.

Full report: [`experiments/exp4_finetune/eval_v2/RESULTS_REPORT_V2_FULL.md`](../experiments/exp4_finetune/eval_v2/RESULTS_REPORT_V2_FULL.md). Per-benchmark JSON in `experiments/exp4_finetune/eval_v2/results/v2_full/`.

## Headline

| Benchmark | Score |
|---|---|
| ARC-Challenge | **82.0%** |
| ARC-Easy | **92.0%** |
| MMLU 5-shot | **59.2%** |
| PIQA | **75.9%** |
| BoolQ | **75.0%** |
| GSM8K (greedy / k=5 self-cons) | **35.3% / 44.0%** |

## Full table

| Benchmark | n | Accuracy | 95% CI |
|---|---:|---:|---|
| ARC-Easy | 2,376 | **92.0%** | [90.8, 93.0] |
| ARC-Challenge | 1,172 | **82.0%** | [79.7, 84.1] |
| PIQA | 1,838 | **75.9%** | [73.9, 77.8] |
| BoolQ | 3,270 | **75.0%** | [73.5, 76.5] |
| MMLU (5-shot) | 14,042 | **59.2%** | [58.4, 60.1] |
| HellaSwag | 10,042 | **56.8%** | [55.8, 57.8] |
| WinoGrande XL | 1,267 | **56.4%** | [53.7, 59.1] |
| TruthfulQA-MC1 | 817 | **47.5%** | [44.1, 50.9] |
| GSM8K self-cons (k=5) | 200 | **44.0%** | [37.3, 50.9] |
| MBPP+ | 378 | **35.7%** | [31.0, 40.7] |
| Agentic GSM8K (calc/python) | 1,319 | **35.4%** | [32.9, 38.0] |
| GSM8K (greedy) | 1,319 | **35.3%** | [32.8, 38.0] |
| MGSM (en/es/fr) | 750 | **34.4%** | [31.1, 37.9] |
| IFEval (strict) | 541 | **31.6%** | [27.8, 35.6] |
| MMLU-Pro | 12,032 | **30.9%** | [30.1, 31.8] |
| HumanEval+ | 164 | **19.5%** | [14.2, 26.3] |
| MATH-500 | 500 | **18.8%** | [15.6, 22.5] |

## Key findings

- Strong generalist on commonsense + knowledge: 75-92% range on ARC, BoolQ, PIQA; 59% MMLU.
- Math is the main weak spot: GSM8K 35.3% greedy, MATH-500 18.8%. k=5 self-consistency buys +8.7 pp on GSM8K.
- Tool use is mostly latent: only 0.6% of agentic GSM8K tasks invoked the calculator. Fine-tune corpus had ~55 tool examples vs 20K math examples — model defaults to direct chain-of-thought.
- Multilingual transfer is partial: en 42.8% → es 32.0% → fr 28.4%.

## Training stats

| Metric | AVA v1 | AVA v2 |
|---|---|---|
| Training corpus | 5,237 | 20,741 |
| Final train loss | 1.0185 | **0.4145** |
| Wall time | 251 min | 100.5 min |
| Trainable params | 10.9M (0.58% of 1.89B) | 10.9M (0.58% of 1.89B) |
| Peak VRAM | 1.81 GB | 1.81 GB |
| Steps/sec | 0.04 | 0.43 |
| Effective batch size | 8 | 8 |
| LR (cosine) | 2e-4 | 1.5e-4 |
| LoRA rank / alpha | 16 / 32 | 16 / 32 |
| Max seq length | 384 | 384 |
| Epochs | 1 | 1 |

AVA v2 trained **10.7× faster per step** than v1 thanks to Triton-compiled SDPA. The 4× larger corpus drove the +13 pp ARC jump (v1 showed zero ARC improvement over base).

## Loss trajectory

```
Step     Loss     LR
  20     1.118    1.47e-5
 100     1.072    5.85e-5
 300     1.046    1.09e-4
 500     1.030    1.39e-4
 700     1.057    1.49e-4
1000     1.002    1.43e-4
1500     0.954    1.12e-4
2000     0.942    6.50e-5
2260     0.937    3.68e-5  <- all-time low
2500     0.971    5.17e-7
2593     0.414    0.00e+0  <- final (epoch average)
```

## Caveats

- **MCQ scoring is letter-argmax** (1-token argmax over candidate label tokens via `/completion n_probs=60`). Differs slightly from lm-eval-harness's logprob-of-continuation. Numbers directionally comparable but not numerically identical to leaderboards.
- **MMLU 5-shot context overflow**: 2.7% of MMLU items errored on 8K context cap (long sub-categories like `professional_law`). Counted as failures. Accuracy on completed items: 60.8%.
- **Max training sequence length 384**. Long-form reasoning chains beyond that not seen during training.

See also: [COMPARE.md](COMPARE.md) for cross-model context, [BENCHMARKS.md](BENCHMARKS.md) for benchmark selection rationale.
