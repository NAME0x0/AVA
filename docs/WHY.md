# Why AVA exists

Most AI progress in 2025-2026 happens on clusters with hundreds or thousands of GPUs. AVA asks a different question: **what can you build with a single laptop GPU and no budget?**

The answer turns out to be more than most people expect.

## The constraint

- **GPU**: NVIDIA RTX A2000 Laptop, 4 GB VRAM (a $400 mobile card)
- **Cloud spend**: $0
- **Cluster**: none
- **Author**: 1 person

Everything in this repo — training, evaluation, inference, GGUF export, the full 17-benchmark sweep — runs on that machine.

## What's been proven

- **82% ARC-Challenge** on the full 1,172-question set (AVA v2), in the range of 3B-class models like Llama 3.2 3B-Instruct (78.6%) and Phi-4-mini 3.8B (83.7%), each measured under its own protocol.
- **42 MB adapter**. The training run uses 1.81 GB peak VRAM and finishes in 100 minutes.
- **17 public benchmarks, 52,027 evaluation instances** evaluated end-to-end on the same laptop.
- Nothing requires special hardware, cloud access, or corporate resources.

## Why this matters

1. **Democratization is real, not theoretical.** Most "democratize AI" projects still require cloud GPUs. AVA trains and runs on hardware that students, researchers in developing countries, and hobbyists already own.

2. **Data quality dominates compute.** AVA v1 (5K examples) showed zero ARC improvement over base. AVA v2 (20K curated examples) jumped +13 pp. The difference was not more compute — it was better data. This validates the emerging consensus from Phi-4, LIMO, and DeepSeek that careful data curation can substitute for scale.

3. **QLoRA makes fine-tuning accessible.** Training 0.58% of parameters in 4-bit precision means a 2B model fits in under 2 GB of VRAM. Anyone can specialize a frontier-class base model for their domain without touching a cloud console.

4. **The research is reproducible.** Every script, corpus recipe, config file, and evaluation harness is in this repo. The model card has exact dependency versions. Run the scripts, get the numbers.

## What AVA is not claiming

AVA is not trying to compete with GPT-4 or Claude. It is proving that **meaningful AI capability — strong science reasoning, solid math, reliable tool use — can emerge from constraints that would have been considered impossible two years ago.**

Where AVA v2 falls short (math, multilingual, tool routing, narrative commonsense), it says so explicitly. See [RESULTS.md](RESULTS.md) and [ROADMAP.md](ROADMAP.md).
