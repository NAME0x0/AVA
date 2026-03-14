# Session: repo-sanity-2026-03-14

## Parallel Sweeps

- model budget sweep
- tool protocol sweep

## Decision

- Recommended baseline: `ava-59m-byte`
- Recommended tool trace protocol: `compact_tags`
- Why: Start in the middle of the budget range: large enough to learn non-trivial language, math, science, and coding structure, small enough to keep pretraining on a 4 GB card realistic.

## Budget Sweep

- ava-11m-smoke-sft: 10,771,968 params, train 1.034 GB, infer 0.806 GB, fits_4gb=True
- ava-11m-smoke: 10,771,968 params, train 1.034 GB, infer 0.806 GB, fits_4gb=True
- ava-11m-tool-sft: 10,796,544 params, train 1.16 GB, infer 0.821 GB, fits_4gb=True
- ava-19m-smoke: 19,081,216 params, train 1.117 GB, infer 0.831 GB, fits_4gb=True
- ava-34m-byte: 25,484,288 params, train 1.09 GB, infer 0.804 GB, fits_4gb=True
- ava-59m-byte: 59,413,760 params, train 1.518 GB, infer 0.873 GB, fits_4gb=True
- ava-99m-byte: 99,628,032 params, train 2.008 GB, infer 0.954 GB, fits_4gb=True

## Protocol Sweep

- compact_tags: avg 24.25 tokens
- compact_json: avg 53.25 tokens
- compact_xml: avg 72.25 tokens

## Next Actions

- Pretrain the recommended byte-level baseline until the loss curve stabilizes.
- Replace the byte tokenizer with a compact SentencePiece/BPE tokenizer and rerun the protocol sweep.
- Distill calculator traces with the recommended compact protocol.
- Run an ablation with Titans-inspired memory on and off after the base model can answer short tasks.
