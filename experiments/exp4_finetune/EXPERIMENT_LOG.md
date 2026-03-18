# AVA Experiment 4: Fine-tuning Pre-trained Models

## Motivation
After 3 experiments training from scratch (14M params), we hit hard ceilings:
- ARC-Challenge: 24% baseline, 33% with retrieval (random chance = 25%)
- GSM8K: 0-2%
- The model lacks sufficient parameters to store factual knowledge

## Strategy
Fine-tune modern open-source models (Qwen3.5-2B) on our existing corpora,
then build an agentic harness with tool-use and memory.

## Hardware
- NVIDIA RTX A2000 Laptop GPU, 4.0 GB VRAM
- Windows 11, Python 3.13, PyTorch 2.10.0+cu130
- SDPA attention (Flash/Efficient SDP) enabled

---

## Phase 1: Base Model Evaluation

### Qwen3.5-2B (4-bit NF4 quantization)
- Architecture: 24 layers, hybrid linear/full attention, 2048 hidden, GQA
- VRAM: 1.73 GB allocated (3.54 GB reserved)
- Multimodal (text + vision), 262K context, 248K vocab

| Benchmark | Score | Comparison |
|-----------|-------|------------|
| ARC-Challenge (100) | **66.0%** | vs 24% scratch, 33% w/retrieval |
| GSM8K no-think (50) | 4.0% | vs 0-2% scratch |
| GSM8K with-think (50) | 4.0% | thinking didn't help at 2B |

Key insight: 66% ARC is nearly 3x our best scratch result. The model already
has the factual knowledge — no retrieval needed.

---

## Phase 2: QLoRA Fine-tuning

### Run 1: AVA-v1 (v1 corpus, crashed — no output saved)
- Base: Qwen3.5-2B, 4-bit quantization
- Corpus: 4125 examples (4106 rich posttrain + 19 agentic/tool-use/identity)
- LoRA: r=8, alpha=16, dropout=0.05
- Training: 2 epochs, batch=1, grad_accum=8, lr=2e-4, cosine schedule
- Max seq length: 384
- **Status: FAILED** — Process ran ~2 hours then died. No checkpoint saved.
  Output buffered so no error message captured.

### Run 2: AVA-v1 (v2 corpus, SDPA attention) — ACTIVE
- Base: Qwen3.5-2B, 4-bit NF4 + SDPA attention
- Corpus: **20,886 examples** (v2 corpus):
  - 7,473 GSM8K chain-of-thought (with teacher rationales)
  - 8,000 science MCQs (sampled from 33K)
  - 4,476 ARC-Challenge training examples
  - 900 breakthrough distillation (math CoT)
  - 37 tool-use, reasoning, identity, conversation examples
- LoRA: r=16, alpha=32, dropout=0.0
- Training: 1 epoch, batch=2, grad_accum=4, lr=2e-4, cosine schedule
- Max seq length: 512
- Trainable params: 10.9M / 1.89B (0.58%)
- GPU: 1.77 GB after LoRA setup
- Speed: ~10.3s/step, 2586 total steps, ~7.4 hours estimated
- Save steps: 200 (frequent checkpointing)
- **Status: TRAINING...**

---

## Phase 3: Agentic Harness (Built)

Components:
- `harness/engine.py`: Core AVAEngine with tool-use, memory, multi-turn chat
  - Supports LoRA adapter loading (auto-merge on load)
  - BitsAndBytes 4-bit quantization
  - Tool call parsing via `<tool_call>...</tool_call>` format
  - Agentic loop: generate → detect tool calls → execute → continue (max 5 rounds)
- `harness/cli.py`: Interactive CLI interface
- Tools: Calculator (safe eval), Python executor, Memory search
- Memory: Persistent JSONL store with keyword search
- All components tested (offline test suite passes)

---

## Optimization Notes

### What worked:
- BitsAndBytes 4-bit NF4 + double quantization: 1.73 GB VRAM for 2B model
- Manual parameter freezing instead of `prepare_model_for_kbit_training()` (avoids OOM)
- SDPA attention: Free speedup from PyTorch's built-in flash attention
- batch_size=2 (Unsloth couldn't load model at all due to OOM during loading)

### What didn't work:
- **Unsloth**: OOM during model loading — tries to materialize full model before quantization
- **flash-linear-attention**: Doesn't compile on Windows (causal-conv1d build failure)
- **prepare_model_for_kbit_training()**: Upcasts params to float32, OOM on 4GB VRAM
- **Buffered training output**: Process died with no logs captured (now using `-u` flag)

---

### Run 3: AVA-v1 Fast (fast corpus, SDPA + BnB) — ACTIVE
- Base: Qwen3.5-2B, 4-bit NF4 + SDPA attention
- Corpus: **5,437 examples** (fast corpus):
  - 3,000 GSM8K chain-of-thought
  - 1,500 ARC-Challenge
  - 900 breakthrough distillation
  - 37 tool-use, reasoning, identity
- LoRA: r=16, alpha=32, dropout=0.0
- Training: 1 epoch, batch=1, grad_accum=8, lr=2e-4, cosine schedule
- Max seq length: 384
- Trainable params: 10.9M / 1.89B (0.58%)
- Speed: ~22-24s/step, 655 total steps, ~4 hours estimated
- **Loss trajectory** (healthy convergence — new low at step 240):
  - Step 10: 1.886 → Step 50: 1.088 → Step 100: 1.118 → Step 150: 1.006
  - Step 160: 0.982 → Step 200: 0.989 → Step 230: 1.001 → Step 240: **0.954** ← new min
- Checkpoints saved: checkpoint-100, checkpoint-200
- Autonomous pipeline running (monitors training → runs benchmarks → auto-launches v2)
- **Eval fix discovered**: Original GSM8K baseline (4%) used max_tokens=128, truncating all responses.
  True base GSM8K will be re-evaluated with 768 tokens in the pipeline.
- **Status: TRAINING... (step ~240/655, 37%, as of 2026-03-17 16:17)**

### Prepared: AVA-v2 Full Corpus Training
- Script: `finetune_v2_full.py`
- Corpus: v2 augmented (20,941 examples — v2 base + 55 tool-use augmented)
  - Tool-use examples: 65 (0.3%, up from 18 in original v2)
  - Includes calculator, python, multi-tool, memory, and no-tool discrimination examples
- Config: LoRA r=16, alpha=32, lr=1.5e-4, warmup=80, seq=384, 1 epoch
- Estimated: ~2586 steps, ~16 hours at current speed
- Ready to launch after v1 fast results validate the approach

---

## Files
```
experiments/exp4_finetune/
├── corpora/
│   ├── ava_exp4_finetune_v1.jsonl       (4,125 examples)
│   ├── ava_exp4_finetune_v2.jsonl       (20,886 examples)
│   ├── ava_exp4_finetune_v2_augmented.jsonl (20,941 examples, v2 + tool-use)
│   ├── ava_exp4_finetune_fast.jsonl     (5,437 examples, curated fast run)
│   └── tool_use_augmented.jsonl         (55 extra tool-use examples)
├── harness/
│   ├── __init__.py
│   ├── engine.py     # Core engine (AVAEngine, tools, memory, LoRA support)
│   └── cli.py        # Interactive CLI (supports --base-model for LoRA)
├── models/
│   ├── Qwen3.5-2B/           # Base model (4.3 GB)
│   └── Qwen3.5-2B-AVA-v1/   # Fine-tuned output (training...)
├── results/
│   ├── arc_qwen35_2b_baseline_100.json
│   ├── gsm8k_qwen35_2b_baseline_50.json
│   └── gsm8k_qwen35_2b_thinking_50.json
└── scripts/
    ├── test_inference.py
    ├── benchmark_arc.py
    ├── benchmark_gsm8k.py
    ├── benchmark_finetuned.py
    ├── benchmark_full.py           # Comprehensive ARC + GSM8K eval
    ├── build_finetune_corpus.py    # v1 corpus builder
    ├── build_finetune_corpus_v2.py # v2 corpus builder (20K examples)
    ├── build_finetune_corpus_fast.py # Fast corpus builder (5K curated)
    ├── build_v2_augmented.py       # Augmented v2 builder (v2 + tool-use)
    ├── augment_tool_use.py         # Tool-use example generator
    ├── finetune_qlora.py           # Standard QLoRA training
    ├── finetune_qlora_v2.py        # V2 config for standard training
    ├── finetune_unsloth.py         # SDPA-accelerated training (active)
    ├── finetune_v2_full.py         # V2 full corpus training (ready)
    ├── quick_eval.py               # Quick quality check (5 ARC + 3 GSM8K)
    ├── auto_post_eval.py           # Auto-run benchmarks post-training
    ├── autonomous_pipeline.py      # Full pipeline: v1 eval → decision → v2 launch
    ├── benchmark_agentic.py        # GSM8K with tool use (calculator/python)
    ├── compare_checkpoints.py      # Compare all saved checkpoints
    ├── demo_showcase.py            # Visual demo comparing base vs tuned
    ├── launch_v2_training.sh       # Shell script to launch v2
    ├── monitor_and_eval.py         # Training monitor with auto-benchmark
    ├── test_harness.py
    └── test_harness_offline.py     # Offline component tests
```
