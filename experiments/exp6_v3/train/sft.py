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
    encode_completion_masked,
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
    # schedule / fabric
    shard_minutes: float = 150.0
    sync_minutes: float = 30.0
    log_every: int = 20
    # switches
    dry_run: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> SFTConfig:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**raw)


# --------------------------------------------------------------------------- model builders


def _build_qlora_model(cfg: SFTConfig) -> tuple[Any, Any]:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(cfg.donor, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.donor,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
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


def _hf_source_iter(hf_id: str, split: str, streaming: bool = True):
    def factory():
        from datasets import load_dataset

        ds = load_dataset(hf_id, split=split, streaming=streaming)
        return iter(ds)

    return factory


def _list_source_iter(rows: list[dict]):
    def factory():
        return iter(rows)

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
        factory = (
            _list_source_iter(test_rows[s["name"]])
            if test_rows is not None
            else _hf_source_iter(s["hf_id"], s.get("split", "train"))
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


# --------------------------------------------------------------------------- train loop


def train_shard(cfg: SFTConfig, test_rows: dict[str, list[dict]] | None = None) -> dict:
    """Run one time-boxed shard. Returns summary dict (for tests + logs)."""
    torch.manual_seed(cfg.seed)

    if cfg.dry_run:
        model, tokenizer = _build_dry_run_model()
    else:
        model, tokenizer = _build_qlora_model(cfg)

    from .data import tokenizer_has_fim

    if not tokenizer_has_fim(tokenizer):
        for s in cfg.sources:  # graceful FIM degrade — verified at C1, not assumed
            s["fim_fraction"] = 0.0
        print("[sft] tokenizer lacks FIM sentinels -> FIM disabled this run")

    stream = build_stream(cfg, test_rows)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: min(1.0, (st + 1) / max(cfg.warmup_steps, 1))
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
            cur = payload.get("extra", {}).get("mixture_cursor")
            if cur:
                stream.skip_to(cur)
            for _ in range(start_step):
                sched.step()

    deadline = time.monotonic() + cfg.shard_minutes * 60
    device = next(model.parameters()).device
    model.train()
    step, last_loss, encoded_drops = start_step, float("nan"), 0

    while step < cfg.total_steps and time.monotonic() < deadline:
        opt.zero_grad(set_to_none=True)
        micro_done = 0
        while micro_done < cfg.grad_accum:
            sample = next(stream)
            enc = encode_completion_masked(tokenizer, sample, cfg.seq_len)
            if enc is None:
                encoded_drops += 1
                continue
            ids = enc["input_ids"].unsqueeze(0).to(device)
            labels = enc["labels"].unsqueeze(0).to(device)
            out = model(input_ids=ids, labels=labels)
            (out.loss / cfg.grad_accum).backward()
            last_loss = float(out.loss)
            micro_done += 1
        torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        opt.step()
        sched.step()
        if step % cfg.log_every == 0:
            print(f"[sft] step {step} loss {last_loss:.4f} drops {encoded_drops}")
        if sync is not None:
            sync.maybe_save(
                step, model, opt,
                extra={"loss": last_loss, "mixture_cursor": stream.cursor()},
            )
        step += 1

    summary = {
        "start_step": start_step,
        "end_step": step,
        "last_loss": last_loss,
        "encoded_drops": encoded_drops,
        "cursor": stream.cursor(),
    }
    if sync is not None:
        sync.save(step - 1, model, opt, extra={**summary, "shard_end": True})
    print(f"[sft] shard done: {summary}")
    return summary
