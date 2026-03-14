from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ava.config import ExperimentConfig
from ava.model import TORCH_AVAILABLE, F, build_model, torch
from ava.tokenizer import ByteTokenizer


def _resolve_device(requested_device: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        warnings.append("CUDA was requested for inspection but is unavailable; inspection ran on CPU.")
        return "cpu", warnings
    return requested_device, warnings


def _single_token_text(tokenizer: ByteTokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id]).replace("\n", "\\n")


def _topk_entries(vector: Any, k: int) -> list[dict[str, float | int]]:
    if vector.ndim != 1:
        raise ValueError("top-k helper expects a 1D tensor")
    count = min(max(k, 0), int(vector.numel()))
    if count == 0:
        return []
    values, indices = torch.topk(vector, count)
    return [
        {"index": int(index), "value": round(float(value), 6)}
        for value, index in zip(values.tolist(), indices.tolist(), strict=True)
    ]


def _topk_logits(logits: Any, tokenizer: ByteTokenizer, k: int) -> list[dict[str, object]]:
    entries = _topk_entries(logits, k)
    for item in entries:
        item["token_id"] = item.pop("index")
        item["token_text"] = _single_token_text(tokenizer, int(item["token_id"]))
    return entries


def _attention_summary(
    weights: Any,
    token_ids: list[int],
    tokenizer: ByteTokenizer,
    *,
    top_k_attention: int,
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    last_query_weights = weights[0, :, -1, :]
    for head_index in range(last_query_weights.size(0)):
        head_weights = last_query_weights[head_index]
        top_positions = _topk_entries(head_weights, top_k_attention)
        positions: list[dict[str, object]] = []
        for item in top_positions:
            position = int(item["index"])
            positions.append(
                {
                    "position": position,
                    "token_id": int(token_ids[position]),
                    "token_text": _single_token_text(tokenizer, int(token_ids[position])),
                    "weight": item["value"],
                }
            )
        entropy = float(-(head_weights * head_weights.clamp_min(1e-9).log()).sum().item())
        summaries.append(
            {
                "head": head_index,
                "entropy": round(entropy, 6),
                "top_positions": positions,
            }
        )
    return summaries


def _layer_trace(
    *,
    layer_index: int,
    residual_before: Any,
    ln1: Any,
    attn_output: Any,
    attn_weights: Any,
    ln2: Any,
    mlp_hidden: Any,
    mlp_output: Any,
    token_ids: list[int],
    tokenizer: ByteTokenizer,
    top_k_neurons: int,
    top_k_attention: int,
) -> dict[str, object]:
    last_hidden = mlp_hidden[0, -1, :]
    positive_fraction = float((last_hidden > 0).float().mean().item())
    return {
        "layer": layer_index,
        "residual_norm_before": round(float(residual_before[0, -1, :].norm().item()), 6),
        "ln1_norm": round(float(ln1[0, -1, :].norm().item()), 6),
        "attention_output_norm": round(float(attn_output[0, -1, :].norm().item()), 6),
        "ln2_norm": round(float(ln2[0, -1, :].norm().item()), 6),
        "mlp_output_norm": round(float(mlp_output[0, -1, :].norm().item()), 6),
        "mlp_positive_fraction": round(positive_fraction, 6),
        "top_mlp_neurons": _topk_entries(last_hidden, top_k_neurons),
        "attention": _attention_summary(
            attn_weights,
            token_ids,
            tokenizer,
            top_k_attention=top_k_attention,
        ),
    }


@torch.no_grad()
def _forward_with_trace(
    model: Any,
    idx: Any,
    tokenizer: ByteTokenizer,
    *,
    top_k_neurons: int,
    top_k_logits_count: int,
    top_k_attention: int,
) -> tuple[Any, dict[str, object]]:
    _, sequence_length = idx.size()
    if sequence_length > model.config.block_size:
        raise ValueError("sequence length exceeds block size")

    positions = torch.arange(0, sequence_length, device=idx.device, dtype=torch.long)
    x = model.wte(idx) + model.wpe(positions)
    x = model.drop(x)
    layers: list[dict[str, object]] = []

    for layer_index, block in enumerate(model.blocks):
        residual_before = x
        ln1 = block.ln_1(x)
        batch_size, _, channels = ln1.size()
        qkv = block.attn.c_attn(ln1)
        query, key, value = qkv.split(channels, dim=2)
        query = query.view(batch_size, sequence_length, block.attn.n_head, block.attn.head_dim).transpose(1, 2)
        key = key.view(batch_size, sequence_length, block.attn.n_head, block.attn.head_dim).transpose(1, 2)
        value = value.view(batch_size, sequence_length, block.attn.n_head, block.attn.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(block.attn.head_dim)
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=idx.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        attn_output = block.attn.c_proj(attn_output)
        x = x + attn_output

        ln2 = block.ln_2(x)
        mlp_fc = block.mlp.c_fc(ln2)
        mlp_hidden = F.gelu(mlp_fc)
        mlp_output = block.mlp.dropout(block.mlp.c_proj(mlp_hidden))
        x = x + mlp_output

        layers.append(
            _layer_trace(
                layer_index=layer_index,
                residual_before=residual_before,
                ln1=ln1,
                attn_output=attn_output,
                attn_weights=attn_weights,
                ln2=ln2,
                mlp_hidden=mlp_hidden,
                mlp_output=mlp_output,
                token_ids=idx[0].tolist(),
                tokenizer=tokenizer,
                top_k_neurons=top_k_neurons,
                top_k_attention=top_k_attention,
            )
        )

    x = model.ln_f(x)
    logits = model.lm_head(x)
    step_trace = {
        "sequence_length": sequence_length,
        "input_token_ids": idx[0].tolist(),
        "input_token_text": [_single_token_text(tokenizer, token_id) for token_id in idx[0].tolist()],
        "final_residual_norm": round(float(x[0, -1, :].norm().item()), 6),
        "top_next_token_logits": _topk_logits(logits[0, -1, :], tokenizer, top_k_logits_count),
        "layers": layers,
    }
    return logits, step_trace


@torch.no_grad()
def trace_generation(
    model: Any,
    tokenizer: ByteTokenizer,
    prompt: str,
    *,
    requested_device: str = "cuda",
    max_new_tokens: int = 16,
    top_k_neurons: int = 8,
    top_k_logits: int = 8,
    top_k_attention: int = 4,
) -> dict[str, object]:
    device, warnings = _resolve_device(requested_device)
    model = model.to(device)
    model.eval()

    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_token_ids: list[int] = []
    generated_tokens: list[str] = []
    steps: list[dict[str, object]] = []
    stopped_on_eos = False

    for step_index in range(max_new_tokens):
        logits, step_trace = _forward_with_trace(
            model,
            idx,
            tokenizer,
            top_k_neurons=top_k_neurons,
            top_k_logits_count=top_k_logits,
            top_k_attention=top_k_attention,
        )
        next_token_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        next_token_text = _single_token_text(tokenizer, next_token_id)
        step_trace["step"] = step_index + 1
        step_trace["chosen_token_id"] = next_token_id
        step_trace["chosen_token_text"] = next_token_text
        steps.append(step_trace)
        generated_token_ids.append(next_token_id)
        generated_tokens.append(next_token_text)
        next_token = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        idx = torch.cat((idx, next_token), dim=1)
        if next_token_id == tokenizer.token_to_id["<eos>"]:
            stopped_on_eos = True
            break

    return {
        "requested_device": requested_device,
        "device_used": device,
        "warnings": warnings,
        "prompt": prompt,
        "prompt_token_ids": prompt_ids,
        "prompt_token_text": [_single_token_text(tokenizer, token_id) for token_id in prompt_ids],
        "max_new_tokens": max_new_tokens,
        "generated_token_ids": generated_token_ids,
        "generated_token_text": generated_tokens,
        "generated_text": tokenizer.decode(generated_token_ids).strip(),
        "stopped_on_eos": stopped_on_eos,
        "steps": steps,
    }


def trace_checkpoint(
    checkpoint_path: str | Path,
    prompt: str,
    *,
    requested_device: str = "cuda",
    max_new_tokens: int = 16,
    top_k_neurons: int = 8,
    top_k_logits: int = 8,
    top_k_attention: int = 4,
) -> dict[str, object]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for inspection.")

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    config = ExperimentConfig.from_dict(checkpoint["config"])
    tokenizer = ByteTokenizer()
    model = build_model(config.model, tokenizer.vocab_size)
    model.load_state_dict(checkpoint["model"])
    payload = trace_generation(
        model,
        tokenizer,
        prompt,
        requested_device=requested_device,
        max_new_tokens=max_new_tokens,
        top_k_neurons=top_k_neurons,
        top_k_logits=top_k_logits,
        top_k_attention=top_k_attention,
    )
    payload["checkpoint"] = str(checkpoint_path)
    payload["config_name"] = config.name
    return payload
