from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from random import Random, randint, seed
from typing import Any

from ava.config import ExperimentConfig, load_experiment_config
from ava.data import discover_corpus_files, encode_corpus, load_supervised_examples, load_text_corpus
from ava.experiments import estimate_budget
from ava.model import TORCH_AVAILABLE, build_model, torch
from ava.tokenizer import ByteBPETokenizer, SPECIAL_TOKEN_OFFSET, load_tokenizer, token_piece_bytes


def dry_run_summary(config: ExperimentConfig) -> dict[str, object]:
    budget = estimate_budget(config)
    return {
        "name": config.name,
        "parameters": budget.parameters,
        "train_vram_gb": budget.train_vram_gb,
        "infer_vram_gb": budget.infer_vram_gb,
        "tokens_per_optimizer_step": budget.tokens_per_optimizer_step,
        "fits_4gb": budget.fits_4gb,
    }


def summarize_corpus(corpus_root: str | Path, tokenizer_config: object | None = None) -> dict[str, object]:
    corpus_path = Path(corpus_root)
    files = discover_corpus_files(corpus_path)
    texts = load_text_corpus(corpus_path)
    supervised_examples = load_supervised_examples(corpus_path)
    tokenizer = load_tokenizer(tokenizer_config)
    token_count = sum(tokenizer.count_tokens(text, add_bos=True, add_eos=True) for text in texts)
    total_characters = sum(len(text) for text in texts)
    return {
        "corpus_root": str(corpus_path),
        "file_count": len(files),
        "text_count": len(texts),
        "character_count": total_characters,
        "token_count": token_count,
        "supervised_example_count": len(supervised_examples),
        "files": [str(path) for path in files],
    }


def _load_training_buffer(corpus_root: str | Path, tokenizer: Any) -> list[int]:
    texts = load_text_corpus(corpus_root)
    if not texts:
        raise RuntimeError(f"no training texts found under {corpus_root}")
    return encode_corpus(texts, tokenizer)


def _render_supervised_prompt(prompt: str) -> str:
    return f"Question: {prompt}\nAnswer: "


def _load_supervised_dataset(corpus_root: str | Path, block_size: int, tokenizer: Any) -> tuple[list[dict[str, list[int]]], int]:
    examples = load_supervised_examples(corpus_root)
    if not examples:
        raise RuntimeError(f"no supervised prompt/response pairs found under {corpus_root}")
    pad_id = tokenizer.token_to_id["<pad>"]
    samples: list[dict[str, list[int]]] = []
    total_tokens = 0
    for example in examples:
        prompt_ids = tokenizer.encode(_render_supervised_prompt(example["prompt"]), add_bos=True)
        response_ids = tokenizer.encode(example["response"], add_eos=True)
        full_ids = prompt_ids + response_ids
        if len(full_ids) < 2:
            continue
        input_ids = full_ids[:-1]
        next_token_ids = full_ids[1:]
        prompt_prefix = max(min(len(prompt_ids) - 1, len(next_token_ids)), 0)
        target_ids = ([-100] * prompt_prefix) + next_token_ids[prompt_prefix:]
        input_ids = input_ids[:block_size]
        target_ids = target_ids[:block_size]
        if not any(token_id != -100 for token_id in target_ids):
            continue
        if len(input_ids) < block_size:
            padding = block_size - len(input_ids)
            input_ids += [pad_id] * padding
            target_ids += [-100] * padding
        samples.append({"input_ids": input_ids, "target_ids": target_ids})
        total_tokens += len(full_ids)
    return samples, total_tokens


def summarize_supervised_dataset(corpus_root: str | Path, block_size: int, tokenizer: Any) -> dict[str, int | float]:
    examples = load_supervised_examples(corpus_root)
    if not examples:
        return {
            "total_examples": 0,
            "kept_examples": 0,
            "truncated_examples": 0,
            "skipped_no_target_examples": 0,
            "max_full_tokens": 0,
            "avg_full_tokens": 0.0,
        }
    total_examples = len(examples)
    truncated_examples = 0
    skipped_no_target_examples = 0
    full_lengths: list[int] = []
    kept_examples = 0
    for example in examples:
        prompt_ids = tokenizer.encode(_render_supervised_prompt(example["prompt"]), add_bos=True)
        response_ids = tokenizer.encode(example["response"], add_eos=True)
        full_ids = prompt_ids + response_ids
        full_lengths.append(len(full_ids))
        if len(full_ids) < 2:
            skipped_no_target_examples += 1
            continue
        input_ids = full_ids[:-1]
        next_token_ids = full_ids[1:]
        prompt_prefix = max(min(len(prompt_ids) - 1, len(next_token_ids)), 0)
        target_ids = ([-100] * prompt_prefix) + next_token_ids[prompt_prefix:]
        if len(input_ids) > block_size or len(target_ids) > block_size:
            truncated_examples += 1
        target_ids = target_ids[:block_size]
        if not any(token_id != -100 for token_id in target_ids):
            skipped_no_target_examples += 1
            continue
        kept_examples += 1
    return {
        "total_examples": total_examples,
        "kept_examples": kept_examples,
        "truncated_examples": truncated_examples,
        "skipped_no_target_examples": skipped_no_target_examples,
        "max_full_tokens": max(full_lengths),
        "avg_full_tokens": round(sum(full_lengths) / len(full_lengths), 2),
    }


def _sample_batch(buffer: list[int], block_size: int, batch_size: int, device: str) -> tuple[Any, Any]:
    if len(buffer) <= block_size + 1:
        raise RuntimeError("token buffer is too small for the configured block size")
    starts = [randint(0, len(buffer) - block_size - 2) for _ in range(batch_size)]
    x = torch.stack(
        [torch.tensor(buffer[start : start + block_size], dtype=torch.long) for start in starts]
    )
    y = torch.stack(
        [torch.tensor(buffer[start + 1 : start + block_size + 1], dtype=torch.long) for start in starts]
    )
    return x.to(device), y.to(device)


def _sample_supervised_batch(samples: list[dict[str, list[int]]], batch_size: int, device: str) -> tuple[Any, Any]:
    if not samples:
        raise RuntimeError("no supervised samples available")
    indices = [randint(0, len(samples) - 1) for _ in range(batch_size)]
    x = torch.stack(
        [torch.tensor(samples[index]["input_ids"], dtype=torch.long) for index in indices]
    )
    y = torch.stack(
        [torch.tensor(samples[index]["target_ids"], dtype=torch.long) for index in indices]
    )
    return x.to(device), y.to(device)


def _lr_for_step(config: ExperimentConfig, step: int) -> float:
    if step < config.training.warmup_steps:
        return config.training.learning_rate * ((step + 1) / max(config.training.warmup_steps, 1))
    return config.training.learning_rate


def _resolve_device(requested_device: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        warnings.append("CUDA was requested but is unavailable in the current PyTorch install; training ran on CPU.")
        return "cpu", warnings
    return requested_device, warnings


def _configure_trainable_parameters(model: Any, patterns: tuple[str, ...]) -> tuple[list[Any], list[str], int]:
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = True
        if patterns:
            parameter.requires_grad = any(pattern in name for pattern in patterns)
        if parameter.requires_grad:
            trainable_names.append(name)
    if patterns and not trainable_names:
        raise RuntimeError(f"no trainable parameters matched patterns: {patterns}")
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_parameter_count = sum(parameter.numel() for parameter in trainable_parameters)
    return trainable_parameters, trainable_names, trainable_parameter_count


def _expand_state_dict_for_tokenizer(state_dict: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    embedding = state_dict["wte.weight"]
    current_vocab, hidden_size = embedding.shape
    if tokenizer.vocab_size <= current_vocab:
        return state_dict
    if not isinstance(tokenizer, ByteBPETokenizer) or current_vocab < SPECIAL_TOKEN_OFFSET + 256:
        raise RuntimeError("init checkpoint vocab does not match current tokenizer and cannot be expanded safely")
    expanded = embedding.new_zeros((tokenizer.vocab_size, hidden_size))
    expanded[:current_vocab] = embedding
    fallback = embedding[:current_vocab].mean(dim=0)
    for token_id in range(current_vocab, tokenizer.vocab_size):
        piece = token_piece_bytes(tokenizer, token_id)
        if piece:
            byte_ids = [SPECIAL_TOKEN_OFFSET + value for value in piece]
            expanded[token_id] = embedding[byte_ids].mean(dim=0)
        else:
            expanded[token_id] = fallback
    updated = dict(state_dict)
    updated["wte.weight"] = expanded
    if "lm_head.weight" in updated:
        updated["lm_head.weight"] = expanded.clone()
    return updated


def _resize_state_dict_for_block_size(state_dict: dict[str, Any], block_size: int) -> dict[str, Any]:
    positional = state_dict.get("wpe.weight")
    if positional is None or positional.shape[0] == block_size:
        return state_dict
    source = positional.transpose(0, 1).unsqueeze(0)
    resized = torch.nn.functional.interpolate(source, size=block_size, mode="linear", align_corners=True)
    updated = dict(state_dict)
    updated["wpe.weight"] = resized.squeeze(0).transpose(0, 1).contiguous()
    return updated


def _load_init_checkpoint(model: Any, tokenizer: Any, init_checkpoint_path: Path) -> str | None:
    init_payload = torch.load(init_checkpoint_path, map_location="cpu")
    init_state = init_payload["model"]
    if model.wte.weight.shape != init_state["wte.weight"].shape:
        init_state = _expand_state_dict_for_tokenizer(init_state, tokenizer)
    if "wpe.weight" in init_state and model.wpe.weight.shape != init_state["wpe.weight"].shape:
        init_state = _resize_state_dict_for_block_size(init_state, model.config.block_size)
    model.load_state_dict(init_state)
    init_config = init_payload.get("config", {})
    tokenizer_config = init_config.get("tokenizer", {}) if isinstance(init_config, dict) else {}
    return str(tokenizer_config.get("kind", "byte"))


def _split_train_val_buffer(buffer: list[int], block_size: int) -> tuple[list[int], list[int]]:
    min_tokens_for_val = max((block_size + 2) * 4, 512)
    if len(buffer) < min_tokens_for_val:
        return buffer, []
    val_tokens = max(block_size + 2, len(buffer) // 10)
    train_buffer = buffer[:-val_tokens]
    val_buffer = buffer[-val_tokens:]
    if len(train_buffer) <= block_size + 1 or len(val_buffer) <= block_size + 1:
        return buffer, []
    return train_buffer, val_buffer


def _split_supervised_samples(samples: list[dict[str, list[int]]]) -> tuple[list[dict[str, list[int]]], list[dict[str, list[int]]]]:
    if len(samples) < 4:
        return samples, []
    val_count = max(1, len(samples) // 5)
    shuffled = list(samples)
    Random(1337).shuffle(shuffled)
    train_samples = shuffled[:-val_count]
    val_samples = shuffled[-val_count:]
    if not train_samples or not val_samples:
        return samples, []
    return train_samples, val_samples


@torch.no_grad()
def _estimate_loss_raw(
    model: Any,
    buffer: list[int],
    block_size: int,
    batch_size: int,
    device: str,
    *,
    batches: int = 4,
) -> float | None:
    if len(buffer) <= block_size + 1:
        return None
    was_training = model.training
    model.eval()
    losses: list[float] = []
    try:
        for _ in range(batches):
            x, y = _sample_batch(buffer, block_size, batch_size, device)
            _, loss = model(x, y)
            assert loss is not None
            losses.append(float(loss.item()))
    finally:
        if was_training:
            model.train()
    return round(sum(losses) / len(losses), 4) if losses else None


@torch.no_grad()
def _estimate_loss_supervised(
    model: Any,
    samples: list[dict[str, list[int]]],
    batch_size: int,
    device: str,
    *,
    batches: int = 4,
) -> float | None:
    if not samples:
        return None
    was_training = model.training
    model.eval()
    losses: list[float] = []
    try:
        for _ in range(batches):
            x, y = _sample_supervised_batch(samples, batch_size, device)
            _, loss = model(x, y)
            assert loss is not None
            losses.append(float(loss.item()))
    finally:
        if was_training:
            model.train()
    return round(sum(losses) / len(losses), 4) if losses else None


def run_training(
    config_path: str | Path,
    corpus_root: str | Path,
    *,
    max_steps: int | None = None,
    checkpoint_root: str | Path = "checkpoints",
) -> dict[str, object]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training. Install with `pip install -e .[train]`.")

    seed(1337)
    torch.manual_seed(1337)

    config = load_experiment_config(config_path)
    tokenizer = load_tokenizer(config.tokenizer)
    requested_device = config.training.device
    device, warnings = _resolve_device(requested_device)

    dataset_kind = config.training.loss_mode
    if dataset_kind == "supervised":
        supervised_stats = summarize_supervised_dataset(corpus_root, config.model.block_size, tokenizer)
        samples, corpus_tokens = _load_supervised_dataset(corpus_root, config.model.block_size, tokenizer)
        train_samples, val_samples = _split_supervised_samples(samples)
        train_buffer: list[int] = []
        val_buffer: list[int] = []
        if supervised_stats["truncated_examples"]:
            warnings.append(
                f"{supervised_stats['truncated_examples']} supervised examples exceeded block_size={config.model.block_size} and were truncated."
            )
    else:
        buffer = _load_training_buffer(corpus_root, tokenizer)
        train_buffer, val_buffer = _split_train_val_buffer(buffer, config.model.block_size)
        train_samples = []
        val_samples = []
        corpus_tokens = len(buffer)
        supervised_stats = None

    model = build_model(config.model, tokenizer.vocab_size).to(device)
    init_tokenizer_kind: str | None = None
    if config.training.init_checkpoint:
        init_checkpoint_path = Path(config.training.init_checkpoint)
        init_tokenizer_kind = _load_init_checkpoint(model, tokenizer, init_checkpoint_path)
    trainable_parameters, trainable_parameter_names, trainable_parameter_count = _configure_trainable_parameters(
        model,
        config.training.trainable_patterns,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    steps = max_steps or config.training.max_steps
    model.train()
    optimizer.zero_grad(set_to_none=True)

    use_amp = device.startswith("cuda") and config.training.dtype in {"float16", "bfloat16"}
    amp_dtype = getattr(torch, config.training.dtype, torch.float16)
    optimizer_steps = 0
    loss_history: list[dict[str, float | int]] = []
    start_time = time.perf_counter()

    for step in range(steps):
        lr = _lr_for_step(config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if dataset_kind == "supervised":
            x, y = _sample_supervised_batch(train_samples, config.training.micro_batch_size, device)
        else:
            x, y = _sample_batch(train_buffer, config.model.block_size, config.training.micro_batch_size, device)
        context_manager = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
        with context_manager:
            _, loss = model(x, y)
        assert loss is not None
        if not torch.isfinite(loss):
            warning = f"Non-finite loss at step {step + 1}; batch was skipped."
            if warning not in warnings:
                warnings.append(warning)
            optimizer.zero_grad(set_to_none=True)
            continue
        (loss / config.training.gradient_accumulation_steps).backward()

        if (step + 1) % config.training.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            loss_history.append({
                "step": step + 1,
                "loss": round(float(loss.item()), 4),
                "lr": round(lr, 8),
            })

    runtime_seconds = round(time.perf_counter() - start_time, 3)
    tokens_per_optimizer_step = (
        config.training.micro_batch_size
        * config.training.gradient_accumulation_steps
        * config.model.block_size
    )
    if dataset_kind == "supervised":
        train_eval_loss = _estimate_loss_supervised(
            model,
            train_samples,
            min(config.training.micro_batch_size, 4),
            device,
        )
        val_loss = _estimate_loss_supervised(
            model,
            val_samples,
            min(config.training.micro_batch_size, 4),
            device,
        )
    else:
        train_eval_loss = _estimate_loss_raw(
            model,
            train_buffer,
            config.model.block_size,
            min(config.training.micro_batch_size, 4),
            device,
        )
        val_loss = _estimate_loss_raw(
            model,
            val_buffer,
            config.model.block_size,
            min(config.training.micro_batch_size, 4),
            device,
        )

    checkpoint_dir = Path(checkpoint_root)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{config.name}.pt"
    torch.save({"model": model.state_dict(), "config": config.to_dict()}, checkpoint_path)

    final_loss = loss_history[-1]["loss"] if loss_history else None
    min_loss = min(item["loss"] for item in loss_history) if loss_history else None
    return {
        "name": config.name,
        "dataset_kind": dataset_kind,
        "steps": steps,
        "optimizer_steps": optimizer_steps,
        "final_loss": final_loss,
        "min_loss": min_loss,
        "train_eval_loss": train_eval_loss,
        "val_loss": val_loss,
        "runtime_seconds": runtime_seconds,
        "tokens_per_optimizer_step": tokens_per_optimizer_step,
        "tokens_seen": optimizer_steps * tokens_per_optimizer_step,
        "corpus_tokens": corpus_tokens,
        "supervised_stats": supervised_stats,
        "tokenizer_kind": config.tokenizer.kind,
        "tokenizer_path": config.tokenizer.path,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "trainable_patterns": list(config.training.trainable_patterns),
        "trainable_parameter_names": trainable_parameter_names,
        "trainable_parameter_count": trainable_parameter_count,
        "init_tokenizer_kind": init_tokenizer_kind,
        "init_checkpoint": config.training.init_checkpoint,
        "device_requested": requested_device,
        "device_used": device,
        "warnings": warnings,
        "checkpoint": str(checkpoint_path),
        "loss_history": loss_history,
    }


def load_config(path: str | Path) -> ExperimentConfig:
    return load_experiment_config(path)


