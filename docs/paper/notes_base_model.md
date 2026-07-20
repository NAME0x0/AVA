# Base-model verification (Task 1) — verified 2026-07-20 via HF

**Source:** https://huggingface.co/Qwen/Qwen3.5-2B (Qwen org; 336 likes; ~2.1M downloads/month; accessed 2026-07-20 via gstack browse). Cross-checked against local `experiments/exp4_finetune/models/Qwen3.5-2B/config.json`.

## Identity
- **Canonical name:** `Qwen/Qwen3.5-2B` (post-trained). Pretrained base: `Qwen/Qwen3.5-2B-Base`.
- **Real & public:** yes, Apache-2.0.
- **Type:** "Causal Language Model with Vision Encoder" — multimodal, pipeline `image-text-to-text`. `model_type: qwen3_5`.
- **Parameters:** marketed "2B" ("Model size 2B params"); local config `total_params = 1,892,736,832` (1.89B actual). Paper uses "≈2B / 1.9B".
- **Architecture (hybrid):** Gated DeltaNet + gated attention + **sparse Mixture-of-Experts** FFN. Layout: `6 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`. Early-fusion unified vision-language foundation.
- **Context length:** 262,144 native.
- **Released:** February 2026 (Qwen Team, blog). No arXiv technical report linked — cite the `@misc`.

## Citation (verbatim from the model page)
```bibtex
@misc{qwen3.5,
  title  = {{Qwen3.5}: Towards Native Multimodal Agents},
  author = {{Qwen Team}},
  month  = {February},
  year   = {2026},
  url    = {https://qwen.ai/blog?id=qwen3.5}
}
```

## What AVA-v2 adapts (paper phrasing)
Only the **text transformer** projections (`q,k,v,o,gate,up,down_proj`, per `adapter_config.json`). The vision encoder is untouched and not exercised by any evaluation (all 17 evals are text-only).

Draft sentence: *"We adapt Qwen3.5-2B (Qwen Team, 2026), an approximately 2B-parameter hybrid vision-language model combining Gated DeltaNet layers, gated attention, and a sparse mixture-of-experts feed-forward network, released under Apache-2.0. We fine-tune only its text transformer with QLoRA; the vision encoder is left unchanged and is not exercised by any of our text-only evaluations."*

## Honesty caveat (load-bearing)
The base is a strong Feb-2026 model. Community-shared results on the base include **MMLU-Pro ≈ 55.3**, versus AVA-v2's 30.9. **Do not** imply AVA-v2 exceeds its base in general capability. AVA-v2's value = reproducible sub-2 GB pipeline + rigorous full-set evaluation + the subset-reliability finding, plus task-specific gains on ARC/GSM8K-style tasks via their train splits. This is exactly why §10 forbids superiority claims.
