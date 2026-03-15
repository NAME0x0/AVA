# Research Session: architecture-rag-tokenizer-2026-03-15

## Goal

Find stronger architecture, retrieval, tokenizer, and encoder directions for AVA under the single 4 GB GPU constraint, then validate the cheapest concrete branches locally.

## Research Findings

- Architecture: the current AVA core is still a plain GPT-2 style decoder. The best research-backed next architecture is not a giant MoE branch; it is a modern dense stack first, then a hybrid attention/SSM branch.
- Best architecture references:
  - Mamba-2 / SSD: https://arxiv.org/abs/2405.21060
  - Hymba: https://arxiv.org/abs/2411.13676
- Retrieval: the current exact/nearest support lookup is too brittle and too flat. The better long-term direction is LightRAG/HippoRAG2-style structured memory.
- Best retrieval references:
  - LightRAG: https://arxiv.org/abs/2410.05779
  - HippoRAG 2: https://arxiv.org/abs/2502.14802
  - FlashRAG toolkit: https://arxiv.org/abs/2405.13576
- Tokenizer: the repo now has real `tokenizers`-based BPE and Unigram branches, and `hf_bpe_512` is the compression winner. But late tokenizer migration still collapses the model, which lines up with FOCUS-style warnings.
- Best tokenizer/migration reference:
  - FOCUS: https://arxiv.org/abs/2305.14481
- Encoder: no dense encoder was downloaded in this loop. The first local candidate is `Alibaba-NLP/gte-modernbert-base`; the multilingual heavier candidate is `BAAI/bge-m3`.

## Experiments Run

### Tokenizer compression

- Full `corpora`:
  - byte: `11,827,907`
  - hf_bpe_512: `4,505,401` (`0.3809x`)
  - hf_unigram_512: `4,857,842` (`0.4107x`)
- `corpora/tool_sft`:
  - byte: `8,426`
  - hf_bpe_512: `2,792` (`0.3314x`)
  - hf_unigram_512: `2,966` (`0.3520x`)
- `corpora/public_benchmark_distill_v1`:
  - byte: `7,212,853`
  - hf_bpe_512: `2,823,770` (`0.3915x`)
  - hf_unigram_512: `3,047,019` (`0.4224x`)

Winner: `hf_bpe_512`

### Tokenizer migration from the best AVA checkpoint

- `hf-bpe-migrate-v1`: [notes](/D:/AVA/sessions/2026-03-15-130846-hf-bpe-migrate-v1/notes.md)
  - benchmark `2/10`
  - tool `0/6`
  - compliance `0/4`
- `hf-unigram-migrate-v1`: [notes](/D:/AVA/sessions/2026-03-15-130906-hf-unigram-migrate-v1/notes.md)
  - benchmark `2/10`
  - tool `0/6`
  - compliance `0/4`

Conclusion: tokenizer quality improved, but late-stage migration still fails. The mainline should not switch tokenizers midstream without a stronger migration method or earlier pretraining.

### Public benchmark retrieval

Using the same best checkpoint [ava-11m-failure-patch-v2.pt](/D:/AVA/sessions/2026-03-14-184859-failure-patch-v2-rerun-11m-96/checkpoints/ava-11m-failure-patch-v2.pt) and the ARC train support bank [arc_train_support_v1](/D:/AVA/corpora/arc_train_support_v1):

- plain `support_mc` first `100`: `30/100`
- new `hybrid_support_mc` first `100`: `31/100`
- plain `support_mc` full `299`: `84/299`
- new `hybrid_support_mc` full `299`: `87/299`

Tracked outputs:
- [support_mc 100](/D:/AVA/sessions/activity/arc-support-mc-100-kindscience.json)
- [hybrid_support_mc 100](/D:/AVA/sessions/activity/arc-hybrid-support-mc-100.json)
- [support_mc 299](/D:/AVA/sessions/activity/arc-support-mc-299-kindscience.json)
- [hybrid_support_mc 299](/D:/AVA/sessions/activity/arc-hybrid-support-mc-299.json)

Conclusion: the hybrid choice-aware retriever is the current best public ARC path in the repo.

### Broad support bank probe

- Using `corpora/public_benchmark_distill_v1` as the support bank for ARC hybrid retrieval timed out at `300s`.
- That means AVA now needs indexing or a real encoder-backed retriever before it can scale beyond the shaped ARC support bank.

## Decision

- Keep `hybrid_support_mc` as the best current public ARC mode.
- Keep `hf_bpe_512` as the best tokenizer artifact discovered in this loop.
- Do not switch the active student checkpoint to a new tokenizer midstream yet.
- Move AVA-v2 architecture toward a modern dense core first, not a speculative large MoE.
- If an external encoder branch is added next, start with `gte-modernbert-base` before heavier multilingual encoders.

## Next Step

1. Implement AVA-v2 model options for a modern dense decoder: RoPE, RMSNorm, SwiGLU, and GQA.
2. Add a pluggable encoder-backed retriever and benchmark it against `hybrid_support_mc` on ARC.
3. If a Hugging Face token is available, download `Alibaba-NLP/gte-modernbert-base` first. Keep `BAAI/bge-m3` as the multilingual branch after that.
