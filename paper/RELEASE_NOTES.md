# AVA-v2 Paper (paper-v1)

Preprint of **AVA-v2: Reasoning on a 4 GB Laptop GPU** — a reproducible case study of adapting and evaluating a 1.9B model entirely on a single 4 GB laptop GPU: 1.81 GB peak training VRAM, about 100 minutes, a 42 MB QLoRA adapter, and a full-set evaluation across 17 benchmarks with 95% Wilson confidence intervals.

## Highlights
- Trained and evaluated on one RTX A2000 laptop GPU (4 GB VRAM), no cloud or cluster.
- Strong on knowledge and science (ARC-Challenge 82.0%, ARC-Easy 92.0%, MMLU 59.2%); weak on competition mathematics and strict instruction following.
- A 50-question GSM8K probe (48%) overstated the full 1,319-question result (35.3%) by about 13 points — small subsets misstate compact-model performance.
- Self-consistency added 8.7 points on GSM8K, while the model's own tool-use pathway fired on only 0.6% of problems and changed nothing.
- Train/test contamination checked directly (GSM8K 0/1319, ARC-Challenge 8/1172, the latter inherent to ARC's own split).

## Artifacts
- Paper (PDF): attached to this release, and in the repo at `paper/AVA-v2.pdf`.
- Adapter: https://huggingface.co/NAME0x0/AVA-v2
- Merged weights: https://huggingface.co/NAME0x0/AVA-v2-merged
- GGUF ladder: https://huggingface.co/NAME0x0/AVA-v2-GGUF

## Citation
```bibtex
@misc{mumtaz2026avav2,
  title  = {AVA-v2: Reasoning on a 4 GB Laptop GPU},
  author = {Muhammad Afsah Mumtaz},
  year   = {2026},
  url    = {https://github.com/NAME0x0/AVA}
}
```

An arXiv submission is in progress; the arXiv ID and Hugging Face Paper page will be added once available.
