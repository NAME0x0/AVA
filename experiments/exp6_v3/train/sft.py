"""v3.0 resumable QLoRA SFT trainer — the C5 lane, platform-agnostic.

One entry point, `train_shard(cfg)`, run identically on Colab / Kaggle /
laptop. All state (adapters + optimizer + RNG + mixture cursor) round-trips
through CheckpointSync; preemption anywhere costs <= sync_minutes of work.

Everything configurable comes from a single YAML (configs/v30_sft.yaml) so
runs are reproducible from the config hash alone (RECIPE invariants).

`dry_run=True` swaps the donor for a tiny random model so the full loop —
mixture, masking, checkpoint round-trip, gate hook — is testable on CPU in
seconds. The GPU path differs only in model construction.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # exp6_v3 root

from scripts.checkpoint_sync import CheckpointSync  # noqa: E402

from .data import (  # noqa: E402
    DecontamFilter,
    MixtureStream,
    SourceSpec,
)


@dataclass
class SFTConfig:
    # identity
    phase: str = "C5"
    ckpt_repo: str = "NAME0x0/AVA-v3-checkpoints"
    donor: str = "Qwen/Qwen3.5-4B"
    seed: int = 1234
    # data
    seq_len: int = 1024
    sources: list[dict] = field(default_factory=list)   # [{name, weight, hf_id, split, fim_fraction}]
    decontam_against: list[str] = field(
        default_factory=lambda: ["evalplus/humanevalplus", "evalplus/mbppplus"]
    )
    # optimization
    lr: float = 1e-4
    micro_batch: int = 1
    grad_accum: int = 16
    total_steps: int = 20_000
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    # lora
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_dora: bool = True   # DoRA: decomposes weight into magnitude+direction,
                            # better quality than plain LoRA at same param count
                            # (PLAN_2026-07-15_v31 s4). peft>=0.12; ~small overhead.
    # schedule / fabric
    shard_minutes: float = 150.0
    sync_minutes: float = 30.0
    log_every: int = 20
    # switches
    dry_run: bool = False
    auto_hardware: bool = True
    load_4bit: bool = True     # profiles disable on cards that fit bf16 weights

    @classmethod
    def from_yaml(cls, path: str | Path) -> SFTConfig:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**raw)


# --------------------------------------------------------------------------- model builders


def _build_qlora_model(cfg: SFTConfig, compute_dtype: torch.dtype) -> tuple[Any, Any]:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(cfg.donor, trust_remote_code=True)
    # Field finding 2026-07-12: bnb 4-bit dequant is the throughput ceiling
    # (L4 == T4 at ~220 tok/s). Cards that fit bf16 weights (8 GB) skip
    # quantization entirely — native tensor-core GEMMs, no dequant tax.
    bnb = None
    if cfg.load_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    # Pin to one GPU when the 4-bit model fits (donor ~3.6 GB): device_map
    # "auto" shards across multi-GPU boxes (Kaggle 2xT4) into a sequential
    # pipeline where one GPU always idles + cross-device transfers. Only
    # fall back to auto-sharding on small-VRAM cards.
    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / 1e9
        if torch.cuda.is_available() else 0
    )
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    # Kaggle T4x2: shard the model across both cards (naive pipeline) so the
    # 248K-vocab CE at seq2048 has room to breathe. One card sits ~idle and
    # cross-device transfers cost throughput — the user chose headroom over
    # speed for the edit/agentic phase. A single big card (L4/A100, >=24 GB)
    # still pins to GPU0; sharding there would only add pipeline overhead.
    if os.environ.get("AVA_SINGLE_GPU") == "1":
        device_map = {"": 0}          # escape hatch: T4x2 sharding broke -> pin GPU0
    elif n_gpu > 1 and vram_gb < 24:
        device_map = "auto"
    elif vram_gb >= 12:
        device_map = {"": 0}
    else:
        device_map = "auto"
    print(f"[sft] device_map={device_map} (n_gpu={n_gpu}, vram0={vram_gb:.0f}GB)")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.donor,
        quantization_config=bnb,
        device_map=device_map,
        dtype=compute_dtype,  # torch_dtype deprecated in transformers >= 4.56
        trust_remote_code=True,
        # SDPA fuses the softmax H-block attention (6/32 layers); the
        # from_pretrained default (eager) has no fusion. Free throughput —
        # GDN layers are unaffected (they route through FLA, not attention).
        attn_implementation="sdpa",
    )
    if cfg.load_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
        print("[sft] unquantized bf16 base (no bnb) — dequant tax removed")
    # Target every linear projection generically: GDN hybrid layer names differ
    # from vanilla attention (q/k/v/o + gdn projections + mlp). all-linear is
    # the donor-agnostic choice and peft resolves it per-architecture.
    lcfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        use_dora=cfg.use_dora,
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    return model, tokenizer


def _build_dry_run_model() -> tuple[Any, Any]:
    """Tiny GPT2-ish stand-in so the loop is CPU-testable end to end."""
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=tok.vocab_size))
    return model, tok


# --------------------------------------------------------------------------- data wiring


def _hf_source_iter(
    hf_id: str,
    split: str,
    config: str | None = None,
    data_files: str | None = None,
    streaming: bool = True,
):
    """data_files: direct-file load for legacy script datasets (e.g.
    bigcode/commitpackft, whose loader script modern `datasets` refuses)."""
    def factory():
        from datasets import load_dataset

        if data_files:
            ds = load_dataset("json", data_files=data_files, split=split, streaming=streaming)
        elif config:
            ds = load_dataset(hf_id, config, split=split, streaming=streaming)
        else:
            ds = load_dataset(hf_id, split=split, streaming=streaming)
        return iter(ds)

    return factory


def _list_source_iter(rows: list[dict]):
    def factory():
        return iter(rows)

    return factory


def _openswe_source_iter(data_files: str, split: str = "train"):
    """Parquet loader + row filter for nvidia/Open-SWE-Traces: only
    execution-verified (resolved==1) traces on permissively-licensed repos reach
    the mixture. Filtering here (not just in the mapper) stops ~90% unresolved
    rows from wasting mixture draws."""
    from .data import _OSWE_PERMISSIVE

    def factory():
        from datasets import load_dataset

        ds = load_dataset("parquet", data_files=data_files, split=split, streaming=True)
        return (
            ex for ex in ds
            if ex.get("resolved") == 1 and ex.get("license") in _OSWE_PERMISSIVE
        )

    return factory


def build_stream(cfg: SFTConfig, test_rows: dict[str, list[dict]] | None = None) -> MixtureStream:
    """test_rows: {source_name: [raw rows]} overrides HF loading (dry runs/tests)."""
    decontam = None
    if cfg.decontam_against and test_rows is None:
        from datasets import load_dataset

        texts: list[str] = []
        for ds_id in cfg.decontam_against:
            for ex in load_dataset(ds_id, split="test"):
                texts.append(str(ex.get("prompt", "")) + "\n" + str(ex.get("canonical_solution", "")))
        decontam = DecontamFilter(texts)

    specs = []
    for s in cfg.sources:
        if test_rows is not None:
            factory = _list_source_iter(test_rows[s["name"]])
        elif s.get("loader") == "openswe":
            factory = _openswe_source_iter(s["data_files"], s.get("split", "train"))
        else:
            factory = _hf_source_iter(
                s["hf_id"], s.get("split", "train"),
                config=s.get("config"), data_files=s.get("data_files"),
            )
        specs.append(
            SourceSpec(
                name=s["name"],
                weight=float(s["weight"]),
                iterator_factory=factory,
                fim_fraction=float(s.get("fim_fraction", 0.0)),
            )
        )
    return MixtureStream(specs, seed=cfg.seed, decontam=decontam)


def _lr_lambda(warmup_steps: int, total_steps: int):
    """Linear warmup -> cosine decay to a 10% floor.

    Replaces the original warmup-then-constant schedule (2026-07-11): constant
    1e-4 over a 20K-step run leaves the tail stirring noise. Resume replays
    scheduler steps, so a mid-run schedule change lands as a one-time LR
    adjustment at the resumed step — acceptable.
    """
    import math

    def fn(st: int) -> float:
        if st < warmup_steps:
            return (st + 1) / max(warmup_steps, 1)
        span = max(total_steps - warmup_steps, 1)
        progress = min((st - warmup_steps) / span, 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


# --------------------------------------------------------------------------- train loop


def train_shard(cfg: SFTConfig, test_rows: dict[str, list[dict]] | None = None) -> dict:
    """Run one time-boxed shard. Returns summary dict (for tests + logs)."""
    import os as _os

    # big-vocab CE allocations fragment badly; expandable segments reclaims
    _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.manual_seed(cfg.seed)

    if cfg.dry_run:
        model, tokenizer = _build_dry_run_model()
    else:
        compute_dtype = torch.bfloat16
        if cfg.auto_hardware:
            from .hw_profile import apply_profile, detect_profile

            dtype_str = apply_profile(cfg, detect_profile())
            compute_dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
            }[dtype_str]
        model, tokenizer = _build_qlora_model(cfg, compute_dtype)

    from .data import tokenizer_has_fim

    if not tokenizer_has_fim(tokenizer):
        for s in cfg.sources:  # graceful FIM degrade — verified at C1, not assumed
            s["fim_fraction"] = 0.0
        print("[sft] tokenizer lacks FIM sentinels -> FIM disabled this run")

    stream = build_stream(cfg, test_rows)
    params = [p for p in model.parameters() if p.requires_grad]
    try:  # fused CUDA kernel when available; CPU/older stacks fall back
        opt = torch.optim.AdamW(params, lr=cfg.lr, fused=torch.cuda.is_available())
    except (RuntimeError, TypeError):
        opt = torch.optim.AdamW(params, lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _lr_lambda(cfg.warmup_steps, cfg.total_steps)
    )

    sync = CheckpointSync(
        cfg.ckpt_repo, phase=cfg.phase, every_minutes=cfg.sync_minutes, trainable_only=True
    ) if not cfg.dry_run else None

    start_step = 0
    if sync is not None:
        start_step = sync.resume(model, opt)
        if start_step > 0:
            # cursor rides in `extra`; restore exact data position
            import json as _json

            from huggingface_hub import hf_hub_download

            ptr = _json.loads(
                Path(hf_hub_download(cfg.ckpt_repo, f"checkpoints/{cfg.phase}/LATEST.json")).read_text()
            )
            payload = torch.load(
                hf_hub_download(cfg.ckpt_repo, ptr["path"]), map_location="cpu", weights_only=False
            )
            extra = payload.get("extra", {})
            # BUG HISTORY (2026-07-11): the end-of-shard save wrote the cursor
            # under "cursor" (summary key) while resume only read
            # "mixture_cursor" — the silent `if cur:` skip meant every session
            # re-trained on the head of the dataset. Read both; never be quiet
            # about a missing cursor again.
            cur = extra.get("mixture_cursor") or extra.get("cursor")
            if cur:
                stream.skip_to(cur)
                print(f"[sft] data cursor restored: {cur['consumed']}")
            else:
                print(
                    "[sft] *** WARNING: resumed step > 0 but checkpoint has no "
                    "data cursor — stream restarts from the dataset head. "
                    "Expect repeated samples this shard. ***"
                )
            for _ in range(start_step):
                sched.step()

    # -- throughput: length-bucketed micro-batches + background prefetch -----
    import queue
    import threading

    from .data import PaddedBatcher

    torch.backends.cuda.matmul.allow_tf32 = True  # free on Ampere+; no-op on T4
    torch.backends.cudnn.allow_tf32 = True

    batcher = PaddedBatcher(stream, tokenizer, cfg.seq_len, cfg.micro_batch)
    batch_q: queue.Queue = queue.Queue(maxsize=4)
    stop_evt = threading.Event()

    def _producer() -> None:  # tokenize batch N+1 while the GPU chews batch N
        try:
            while not stop_evt.is_set():
                batch_q.put(next(batcher))
        except Exception as err:  # surfaced by consumer via sentinel
            batch_q.put(("__error__", err))

    threading.Thread(target=_producer, daemon=True).start()

    def _next_batch() -> dict:
        item = batch_q.get()
        if isinstance(item, tuple) and item and item[0] == "__error__":
            raise item[1]
        return item

    deadline = time.monotonic() + cfg.shard_minutes * 60
    device = next(model.parameters()).device
    model.train()
    step, last_loss, ema_loss = start_step, float("nan"), None
    win_t0, win_tokens, win_steps = time.monotonic(), 0, 0

    while step < cfg.total_steps and time.monotonic() < deadline:
        opt.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum):
            b = _next_batch()
            out = model(
                input_ids=b["input_ids"].to(device),
                labels=b["labels"].to(device),
                attention_mask=b["attention_mask"].to(device),
            )
            (out.loss / cfg.grad_accum).backward()
            last_loss = float(out.loss.detach())
            win_tokens += int(b["attention_mask"].sum())
        torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        opt.step()
        sched.step()
        ema_loss = last_loss if ema_loss is None else 0.95 * ema_loss + 0.05 * last_loss
        win_steps += 1
        if step % cfg.log_every == 0:
            dt = max(time.monotonic() - win_t0, 1e-6)
            s_per_step = dt / max(win_steps, 1)
            eta_shard_m = max(deadline - time.monotonic(), 0) / 60
            eta_total_h = (cfg.total_steps - step) * s_per_step / 3600
            mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(
                f"[sft] step {step}/{cfg.total_steps} | loss {last_loss:.4f} "
                f"(ema {ema_loss:.4f}) | {win_tokens / dt:,.0f} tok/s | "
                f"{s_per_step:.1f} s/step | drops {batcher.drops} | "
                f"mem {mem:.1f}GB | shard ends {eta_shard_m:.0f}m | "
                f"ETA total {eta_total_h:.1f}h | lr {sched.get_last_lr()[0]:.2e}"
            )
            win_t0, win_tokens, win_steps = time.monotonic(), 0, 0
        if sync is not None:
            sync.maybe_save(
                step, model, opt,
                extra={"loss": last_loss, "mixture_cursor": stream.cursor()},
            )
        step += 1
    stop_evt.set()
    while not batch_q.empty():  # unblock the producer so the thread exits
        batch_q.get_nowait()
    encoded_drops = batcher.drops

    summary = {
        "start_step": start_step,
        "end_step": step,
        "last_loss": last_loss,
        "encoded_drops": encoded_drops,
        "cursor": stream.cursor(),
    }
    if sync is not None:
        sync.save(step - 1, model, opt, extra={**summary, "mixture_cursor": stream.cursor(), "shard_end": True})
    print(f"[sft] shard done: {summary}")
    return summary
