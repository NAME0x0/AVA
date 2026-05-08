# Comparison to other small models

All non-AVA scores from official model cards / technical reports. Evaluation protocols vary by source (shot count, prompting). AVA v2 numbers from the [full 17-benchmark eval](RESULTS.md). AVA v2 GSM8K shown as `greedy / k=5 self-cons`.

| Model | Params | ARC-C | MMLU | HellaSwag | GSM8K |
|---|---:|---:|---:|---:|---:|
| TinyLlama 1.1B-Chat | 1.1B | 30.1 | 25.3 | 60.3 | 2.0 |
| Llama 3.2 1B-Instruct | 1.0B | 59.4 | 49.3 | 60.8 | 44.4 |
| Qwen2.5 1.5B-Instruct | 1.5B | 54.7 | 60.9 | 67.9 | 68.5 |
| SmolLM2 1.7B-Instruct | 1.7B | 52.0 | 50.4 | 68.9 | 48.2 |
| Gemma 2 2B-Instruct | 2.0B | 55.7 | 51.3 | 73.0 | 24.3 |
| Qwen3.5 2B Base | 2.0B | 66.0 | — | — | 28.0 |
| **AVA v2 (this repo)** | **2.0B** | **82.0** | **59.2** | 56.8 | **35.3 / 44.0** |
| Qwen2.5 3B-Instruct | 3.0B | ~70 | 65.6 | 73.6 | 79.1 |
| Llama 3.2 3B-Instruct | 3.0B | 78.6 | 63.4 | 69.8 | 77.7 |
| Phi-4-mini 3.8B-Instruct | 3.8B | 83.7 | 67.3 | 76.2 | 88.6 |
| Phi-3.5-mini-Instruct | 3.8B | 84.6 | 69.0 | 69.4 | 86.2 |
| Mistral 7B-Instruct v0.2 | 7.0B | 55.5 | 60.1 | 81.3 | 52.2 |

## What this says

**ARC-Challenge — AVA v2 leads its size class.** 82.0% on the full 1,172-question test set is ahead of every 1-2B model surveyed and ahead of Llama 3.2 3B-Instruct (78.6%). Only Phi 3.5/4-mini 3.8B-class models beat it, by 1.7-2.6 pp. Achieved with a 42 MB LoRA on top of an open base — no cluster, no large pretraining run.

**MMLU is competitive at 59.2%.** Close to Mistral 7B (60.1%) and roughly +8 pp over Gemma 2 2B (51.3%). Below Qwen2.5 1.5B (60.9%) and the Phi/Llama 3B-class models (63-69%).

**GSM8K is the main weakness.** Greedy 35.3% trails Qwen2.5 1.5B (68.5%) and the 3B-class instruct models. With k=5 self-consistency v2 reaches 44.0%, near Llama 3.2 1B-Instruct (44.4%). Fine-tune corpus is reasoning-heavy but not math-heavy enough; AVA v3 targets this directly.

**HellaSwag (commonsense narrative) trails.** 56.8% — below most peers. Fine-tune leaned toward science and instructions, not narrative completion.

**Tool use is trained but mostly latent.** Agentic GSM8K invoked tools on only 0.6% of problems and matched plain greedy. AVA v3 fixes this with MCP-based tool routing instead of fine-tune-only tool teaching.

## Why this matters

The headline win is reasoning-on-tiny-hardware: 42 MB adapter, 4 GB VRAM, 100 minutes of training, 82% ARC-C. Most peers above used cluster-scale training compute.

That doesn't mean AVA is "better" than them. It means **the gap between consumer hardware and frontier-grade reasoning is smaller than it looks** — if you choose the right base model and curate the right data.
