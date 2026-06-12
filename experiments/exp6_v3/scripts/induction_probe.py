"""Induction-head probe — evidence-based softmax-layer selection for T1.

The T1 linearization (CODE_PIVOT.md §3) keeps 9 of 36 donor layers as softmax
attention and converts the rest to Mamba-3. Hybrid-architecture ablations show
the kept layers must host the model's induction/retrieval heads — they carry
associative recall and copying (variable names, API echoes), which linear
mixers handle worst. Uniform 3:1 spacing is blind to where those heads live.

This probe measures per-head induction scores on the donor and emits a ranked,
spacing-constrained keep-set:

  score(head) = mean attention mass from position i (second half of a repeated
  sequence [A; A]) to position i - L + 1 — the token *after* the previous
  occurrence of the current token. High score = classic induction behaviour.

Usage:
  python induction_probe.py --model Qwen/Qwen3-4B --keep 9 --out probe.json
  python induction_probe.py --model <local dir> --load-4bit ...

The probe needs eager attention (output_attentions) and short sequences only —
it runs on the 4 GB laptop GPU in 4-bit, or on CPU for small models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def build_repeated_batch(
    vocab_size: int, batch: int, half_len: int, seed: int, device: torch.device
) -> torch.Tensor:
    """Sequences of the form [A; A] where A is uniform-random tokens."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    first = torch.randint(0, vocab_size, (batch, half_len), generator=gen)
    return torch.cat([first, first], dim=1).to(device)


@torch.no_grad()
def induction_scores(model, input_ids: torch.Tensor, half_len: int) -> torch.Tensor:
    """Return [num_layers, num_heads] mean induction scores.

    For query position i in the second half, the induction target is key
    position i - half_len + 1 (one past the previous occurrence). We average
    the attention probability mass on that exact offset.
    """
    out = model(input_ids=input_ids, output_attentions=True, use_cache=False)
    scores = []
    seq_len = input_ids.shape[1]
    # query positions: second half, excluding the very last token's wrap-around
    q_pos = torch.arange(half_len, seq_len - 1, device=input_ids.device)
    k_pos = q_pos - half_len + 1
    for attn in out.attentions:  # [batch, heads, q, k]
        layer = attn[:, :, q_pos, k_pos]  # [batch, heads, len(q_pos)]
        scores.append(layer.mean(dim=(0, 2)).float().cpu())
    return torch.stack(scores)  # [layers, heads]


def select_keep_set(
    layer_scores: torch.Tensor, keep: int, min_spacing: int = 2
) -> list[int]:
    """Greedy top-score selection with a minimum index spacing.

    Spacing keeps the softmax layers distributed through the depth (hybrid
    ablations show clustering hurts) while still following the evidence.
    """
    order = torch.argsort(layer_scores, descending=True).tolist()
    chosen: list[int] = []
    for idx in order:
        if len(chosen) >= keep:
            break
        if all(abs(idx - c) >= min_spacing for c in chosen):
            chosen.append(idx)
    # If spacing made the quota unreachable, relax it for the remainder.
    for idx in order:
        if len(chosen) >= keep:
            break
        if idx not in chosen:
            chosen.append(idx)
    return sorted(chosen)


def run_probe(
    model,
    vocab_size: int,
    keep: int,
    batch: int = 4,
    half_len: int = 64,
    seeds: tuple[int, ...] = (0, 1, 2),
    min_spacing: int = 2,
) -> dict:
    device = next(model.parameters()).device
    per_seed = []
    for seed in seeds:
        ids = build_repeated_batch(vocab_size, batch, half_len, seed, device)
        per_seed.append(induction_scores(model, ids, half_len))
    head_scores = torch.stack(per_seed).mean(0)  # [layers, heads]

    # A layer is as good as its best induction heads: mean of top-2 heads.
    k = min(2, head_scores.shape[1])
    layer_scores = head_scores.topk(k, dim=1).values.mean(1)
    keep_layers = select_keep_set(layer_scores, keep, min_spacing)

    return {
        "head_scores": head_scores.tolist(),
        "layer_scores": layer_scores.tolist(),
        "keep_softmax_layers": keep_layers,
        "convert_to_mamba3": [
            i for i in range(head_scores.shape[0]) if i not in keep_layers
        ],
        "config": {
            "batch": batch,
            "half_len": half_len,
            "seeds": list(seeds),
            "keep": keep,
            "min_spacing": min_spacing,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF id or local path")
    parser.add_argument("--keep", type=int, default=9)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--half-len", type=int, default=64)
    parser.add_argument("--min-spacing", type=int, default=2)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("induction_probe.json"))
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
        # eager is required for output_attentions
        "attn_implementation": "eager",
    }
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    result = run_probe(
        model,
        vocab_size=len(tokenizer),
        keep=args.keep,
        batch=args.batch,
        half_len=args.half_len,
        min_spacing=args.min_spacing,
    )
    result["model"] = args.model
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"keep softmax layers: {result['keep_softmax_layers']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
