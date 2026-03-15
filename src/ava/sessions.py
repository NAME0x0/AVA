from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from re import sub
from shlex import quote

from ava.benchmarks import serialize_benchmark_registry
from ava.config import load_experiment_config
from ava.eval import (
    benchmark_as_dicts,
    compliance_benchmark_as_dicts,
    evaluate_checkpoint,
    evaluate_compliance_checkpoint,
    evaluate_tool_use_checkpoint,
    tool_benchmark_as_dicts,
)
from ava.experiments import (
    choose_next_step,
    run_budget_sweep,
    run_memory_sweep,
    run_prompt_protocol_sweep,
    run_test_time_strategy_sweep,
    serialize_budget_sweep,
    serialize_memory_sweep,
    serialize_protocol_sweep,
    serialize_test_time_sweep,
    write_json,
)
from ava.activity import record_activity
from ava.inspect import trace_checkpoint
from ava.model import TORCH_AVAILABLE, torch
from ava.memory_transfer import (
    evaluate_transfer_suite_checkpoint,
    transfer_benchmark_as_dicts,
    transfer_compliance_benchmark_as_dicts,
    transfer_tool_benchmark_as_dicts,
)
from ava.research import (
    serialize_hypotheses,
    serialize_papers,
    serialize_recent_hf_hypotheses,
    serialize_recent_hf_papers,
)
from ava.tokenizer import ByteTokenizer
from ava.train import dry_run_summary, run_training, summarize_corpus


BASE_SOURCES = [
    "https://github.com/karpathy/build-nanogpt",
    "https://github.com/karpathy/nanoGPT",
    "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
    "https://arxiv.org/abs/2501.00663",
    "https://arxiv.org/abs/2302.04761",
    "https://arxiv.org/abs/2306.11644",
]

MOE_PAPER_KEYS = (
    "switch-transformer",
    "st-moe",
    "mixtral",
    "deepseek-moe",
    "jetmoe",
    "deepseek-v3",
    "qmoe",
    "squeezellm",
)

HF_RELEVANT_BENCHMARK_KEYS = (
    "bfcl",
    "deepplanning",
    "tau2-bench",
    "mathvista",
    "scienceqa",
    "docvqa",
)

MOE_FEASIBILITY_ESTIMATES = [
    {
        "model": "JetMoE-8B",
        "total_params_b": 8,
        "active_params_b": 2.2,
        "inferred_raw_4bit_weight_gb": 4.0,
        "note": "4-bit raw weights alone are already about 4 GB before metadata, embeddings, KV cache, and runtime overhead.",
    },
    {
        "model": "Mixtral-8x7B",
        "total_params_b": 47,
        "active_params_b": 13,
        "inferred_raw_4bit_weight_gb": 23.5,
        "note": "Active parameters are lower, but all experts still have to be stored or aggressively offloaded.",
    },
    {
        "model": "DeepSeek-V3",
        "total_params_b": 671,
        "active_params_b": 37,
        "inferred_raw_4bit_weight_gb": 335.5,
        "note": "Clearly outside a 4 GB single-GPU target.",
    },
    {
        "model": "QMoE compressed trillion-scale branch",
        "total_params_b": "trillion-scale",
        "active_params_b": "n/a",
        "compressed_weight_gb": ">>4",
        "note": "Sub-1-bit compression is still a server-scale story rather than a 4 GB local deployment story.",
    },
]


def slugify(text: str) -> str:
    lowered = text.strip().lower()
    lowered = sub(r"[^a-z0-9]+", "-", lowered)
    return lowered.strip("-") or "session"


def _sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_manifest(path: str | Path, *, root: str | Path | None = None) -> dict[str, object]:
    file_path = Path(path)
    payload: dict[str, object] = {
        "path": str(file_path),
        "size_bytes": file_path.stat().st_size,
        "sha256": _sha256_path(file_path),
    }
    if root is not None:
        payload["relative_path"] = str(file_path.relative_to(Path(root)))
    return payload


def _run_command(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _repo_manifest() -> dict[str, object]:
    head = _run_command(["git", "rev-parse", "HEAD"])
    status_output = _run_command(["git", "status", "--short", "--untracked-files=all"])
    dirty_paths = status_output.splitlines() if status_output else []
    return {
        "head": head,
        "dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
    }


def _nvidia_smi_manifest() -> list[dict[str, str]]:
    output = _run_command(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    )
    if not output:
        return []
    devices: list[dict[str, str]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            devices.append({"name": parts[0], "driver_version": parts[1], "memory_total": parts[2]})
    return devices


def _environment_manifest(requested_device: str) -> dict[str, object]:
    cuda_available = bool(TORCH_AVAILABLE and torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    gpu_devices: list[dict[str, object]] = []
    if cuda_available:
        for index in range(device_count):
            properties = torch.cuda.get_device_properties(index)
            gpu_devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(properties.total_memory),
                    "multi_processor_count": int(properties.multi_processor_count),
                }
            )
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_available": TORCH_AVAILABLE,
        "torch_version": getattr(torch, "__version__", None) if TORCH_AVAILABLE else None,
        "torch_cuda_version": getattr(torch.version, "cuda", None) if TORCH_AVAILABLE else None,
        "cuda_available": cuda_available,
        "cudnn_available": bool(TORCH_AVAILABLE and torch.backends.cudnn.is_available()),
        "device_requested": requested_device,
        "device_count": device_count,
        "gpu_devices": gpu_devices,
        "nvidia_smi": _nvidia_smi_manifest(),
        "repo": _repo_manifest(),
    }


def _corpus_manifest(corpus_root: str | Path, tokenizer_config: object | None = None) -> dict[str, object]:
    summary = summarize_corpus(corpus_root, tokenizer_config)
    files = [Path(path) for path in summary["files"]]
    summary["files"] = [_file_manifest(path, root=corpus_root) for path in files]
    return summary


def create_session(root: str | Path, name: str, *, sources: list[str] | None = None, kind: str = "baseline") -> Path:
    root_path = Path(root)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
    session_dir = root_path / f"{timestamp}-{slugify(name)}"
    (session_dir / "results").mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "slug": slugify(name),
        "kind": kind,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": sources or BASE_SOURCES,
    }
    write_json(session_dir / "session.json", payload)
    record_activity(
        root_path,
        "session",
        "started",
        event_id=session_dir.name,
        metadata={
            "session_dir": str(session_dir),
            "session_name": name,
            "session_kind": kind,
            "sources": payload["sources"],
        },
    )

    return session_dir


def _record_session_completion(
    root: str | Path,
    session_dir: Path,
    *,
    name: str,
    kind: str,
    metadata: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "session_dir": str(session_dir),
        "session_name": name,
        "session_kind": kind,
    }
    if metadata:
        payload.update(metadata)
    record_activity(
        root,
        "session",
        "completed",
        event_id=session_dir.name,
        metadata=payload,
    )


def render_notes(
    *,
    session_name: str,
    budget_results: list[dict[str, object]],
    protocol_results: list[dict[str, object]],
    recommendation: dict[str, object],
) -> str:
    lines = [
        f"# Session: {session_name}",
        "",
        "## Parallel Sweeps",
        "",
        "- model budget sweep",
        "- tool protocol sweep",
        "",
        "## Decision",
        "",
        f"- Recommended baseline: `{recommendation['recommended_model']}`",
        f"- Recommended tool trace protocol: `{recommendation['recommended_protocol']}`",
        f"- Why: {recommendation['reason']}",
        "",
        "## Budget Sweep",
        "",
    ]
    for item in budget_results:
        lines.append(
            "- "
            f"{item['name']}: {item['parameters']:,} params, "
            f"train {item['train_vram_gb']} GB, infer {item['infer_vram_gb']} GB, "
            f"fits_4gb={item['fits_4gb']}"
        )
    lines.extend(["", "## Protocol Sweep", ""])
    for item in protocol_results:
        lines.append(f"- {item['protocol']}: avg {item['average_tokens']} tokens")
    lines.extend(["", "## Next Actions", ""])
    for action in recommendation["next_actions"]:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def _best_memory_setting(memory_results: list[dict[str, object]]) -> dict[str, object]:
    return sorted(memory_results, key=lambda item: (-item["recall_at_1"], item["average_context_tokens"]))[0]


def _best_test_time_setting(test_time_results: list[dict[str, object]]) -> dict[str, object]:
    viable = [item for item in test_time_results if item["hard_task_coverage"] > 0]
    return sorted(viable or test_time_results, key=lambda item: (-item["hard_task_coverage"], item["average_extra_tokens"]))[0]


def _selected_benchmarks(keys: tuple[str, ...]) -> list[dict[str, object]]:
    registry = {item["key"]: item for item in serialize_benchmark_registry()}
    return [registry[key] for key in keys if key in registry]


def render_sota_notes(
    *,
    session_name: str,
    papers: list[dict[str, object]],
    hypotheses: list[dict[str, object]],
    budget_results: list[dict[str, object]],
    protocol_results: list[dict[str, object]],
    memory_results: list[dict[str, object]],
    test_time_results: list[dict[str, object]],
    recommendation: dict[str, object],
) -> str:
    lines = [
        f"# SOTA Session: {session_name}",
        "",
        "## Product Framing",
        "",
        "AVA is the product. This session is about improving the product via a research stack, not renaming the product into the lab.",
        "",
        "## Papers Driving This Session",
        "",
    ]
    for paper in papers:
        lines.append(f"- {paper['title']} ({paper['submitted']}) -> {paper['ava_takeaway']}")
    lines.extend(["", "## Local Sweeps", ""])
    for item in budget_results:
        lines.append(f"- Budget {item['name']}: train {item['train_vram_gb']} GB, fits_4gb={item['fits_4gb']}")
    for item in protocol_results:
        lines.append(f"- Tool protocol {item['protocol']}: avg {item['average_tokens']} tokens")
    for item in memory_results:
        lines.append(
            f"- Memory threshold {item['threshold']}: recall@1={item['recall_at_1']}, avg_context_tokens={item['average_context_tokens']}"
        )
    for item in test_time_results:
        lines.append(
            f"- Test-time strategy {item['strategy']}: avg_extra_tokens={item['average_extra_tokens']}, hard_task_coverage={item['hard_task_coverage']}"
        )
    lines.extend(["", "## Recommended AVA Direction", ""])
    lines.append(f"- Base model: `{recommendation['recommended_model']}`")
    lines.append(f"- Tool trace format: `{recommendation['recommended_protocol']}`")
    lines.append(f"- Memory threshold: `{recommendation['recommended_memory_threshold']}`")
    lines.append(f"- Test-time strategy: `{recommendation['recommended_test_time_strategy']}`")
    lines.extend(["", "## Next Experiments", ""])
    for hypothesis in hypotheses[:5]:
        lines.append(f"- {hypothesis['key']}: {hypothesis['name']}")
    lines.extend(["", "## Immediate Build Order", ""])
    for action in recommendation["next_actions"]:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def _moe_papers() -> list[dict[str, object]]:
    paper_map = {paper["key"]: paper for paper in serialize_papers()}
    return [paper_map[key] for key in MOE_PAPER_KEYS]


def render_hf_research_notes(
    *,
    session_name: str,
    papers: list[dict[str, object]],
    hypotheses: list[dict[str, object]],
    relevant_benchmarks: list[dict[str, object]],
    recommendation: dict[str, object],
) -> str:
    lines = [
        f"# HF Research Session: {session_name}",
        "",
        "## Focus",
        "",
        recommendation["focus"],
        "",
        "## Recent Papers Selected From The HF Feed",
        "",
    ]
    for paper in papers:
        lines.append(f"- {paper['title']} ({paper['submitted']}) -> {paper['ava_takeaway']}")
    lines.extend(["", "## What Changes Now", ""])
    for item in recommendation["adopt_now"]:
        lines.append(f"- {item}")
    lines.extend(["", "## What Stays Later", ""])
    for item in recommendation["later_only"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Relevant Benchmarks", ""])
    for item in relevant_benchmarks:
        lines.append(f"- {item['name']} ({item['capability']}, {item['stage']})")
    lines.extend(["", "## Queued Experiments", ""])
    for hypothesis in hypotheses:
        lines.append(f"- {hypothesis['key']}: {hypothesis['name']}")
    lines.extend(["", "## Next Actions", ""])
    for item in recommendation["next_actions"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def render_moe_notes(
    *,
    session_name: str,
    papers: list[dict[str, object]],
    feasibility: dict[str, object],
) -> str:
    lines = [
        f"# MoE Feasibility Session: {session_name}",
        "",
        "## Question",
        "",
        feasibility["question"],
        "",
        "## Verdict",
        "",
        feasibility["recommended_position"],
        "",
        "## Why Not Mainline",
        "",
    ]
    for reason in feasibility["reasons"]:
        lines.append(f"- {reason}")
    lines.extend(["", "## What The Papers Say", ""])
    for paper in papers:
        lines.append(f"- {paper['title']} ({paper['submitted']}) -> {paper['ava_takeaway']}")
    lines.extend(["", "## Practical Read", ""])
    lines.append("Sparse MoE can help AVA beat some larger dense models on selected tasks.")
    lines.append("Sparse MoE will not honestly support the target of beating frontier models everywhere on one 4 GB GPU.")
    lines.extend(["", "## Rough Size Checks", ""])
    for item in feasibility["estimates"]:
        total = item["total_params_b"]
        active = item.get("active_params_b", "n/a")
        weight = item.get("inferred_raw_4bit_weight_gb", item.get("compressed_weight_gb", "n/a"))
        total_label = f"{total}B" if isinstance(total, (int, float)) else str(total)
        active_label = f"{active}B" if isinstance(active, (int, float)) else str(active)
        lines.append(f"- {item['model']}: total={total_label}, active={active_label}, weight_budget={weight} GB, note={item['note']}")
    lines.extend(["", "## Next Experiments", ""])
    for action in feasibility["next_experiments"]:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def _training_next_actions(
    training: dict[str, object],
    evaluation: dict[str, object],
    tool_eval: dict[str, object],
    compliance: dict[str, object],
) -> list[str]:
    actions: list[str] = []
    if training.get("optimizer_steps", 0) == 0:
        actions.append("Lower gradient accumulation or raise the step count so the optimizer updates more than once.")
    if training.get("warnings"):
        actions.append("Resolve the recorded warnings before treating this run as a meaningful baseline.")
    final_loss = training.get("final_loss")
    min_loss = training.get("min_loss")
    if isinstance(final_loss, (int, float)) and isinstance(min_loss, (int, float)) and final_loss <= min_loss + 0.05:
        actions.append("The optimization loop looks stable enough for a larger language, math, science, and coding packet next.")
    else:
        actions.append("Tune learning rate or batch structure before scaling the model or corpus.")
    if evaluation.get("accuracy", 0.0) >= 0.66:
        actions.append("Retire the smoke corpus after this run; it is only for plumbing and overfit checks.")
    else:
        actions.append("Do not infer product quality from the smoke benchmark; move next to a broader curated corpus.")
    if tool_eval.get("accuracy", 0.0) >= 0.66:
        actions.append("Tool behavior is stable enough to start regenerating harder calculator cases instead of only hand-scripted traces.")
    else:
        actions.append("Expand the compact tool SFT packet; trace generation, no-tool abstention, and tool boundaries are not stable yet.")
    if compliance.get("accuracy", 0.0) >= 0.75:
        actions.append("Compliance behavior is strong enough to start measuring tradeoffs against English helpfulness and math exactness.")
    else:
        actions.append("Add compact supervised compliance data for refusals, terse formatting, and tool-boundary obedience before broader product claims.")
    return actions


def render_training_notes(
    *,
    session_name: str,
    command: str,
    config_path: str,
    corpus_root: str,
    budget: dict[str, object],
    environment: dict[str, object],
    corpus: dict[str, object],
    training: dict[str, object],
    evaluation: dict[str, object],
    tool_eval: dict[str, object],
    compliance: dict[str, object],
    next_actions: list[str],
) -> str:
    lines = [
        f"# Training Session: {session_name}",
        "",
        "## Command",
        "",
        f"`{command}`",
        "",
        "## Inputs",
        "",
        f"- Config: `{config_path}`",
        f"- Corpus root: `{corpus_root}`",
        f"- Requested device: `{environment['device_requested']}`",
        f"- Tokenizer kind: `{training.get('tokenizer_kind', 'byte')}`",
        f"- Tokenizer vocab size: `{training.get('tokenizer_vocab_size', 'unknown')}`",
        *( [f"- Tokenizer artifact: `{training['tokenizer_path']}`"] if training.get("tokenizer_path") else [] ),
        *( [f"- Init checkpoint: `{training['init_checkpoint']}`"] if training.get("init_checkpoint") else [] ),
        "",
        "## Environment",
        "",
        f"- Python: `{environment['python_version'].splitlines()[0]}`",
        f"- Torch: `{environment['torch_version']}`",
        f"- Torch CUDA: `{environment['torch_cuda_version']}`",
        f"- CUDA available: `{environment['cuda_available']}`",
        f"- GPU count: `{environment['device_count']}`",
    ]
    for gpu in environment["gpu_devices"]:
        lines.append(
            f"- GPU {gpu['index']}: `{gpu['name']}`, total_memory_bytes={gpu['total_memory_bytes']}"
        )
    if environment["repo"].get("head"):
        lines.append(f"- Git HEAD: `{environment['repo']['head']}`")
    lines.append(f"- Dirty worktree: `{environment['repo']['dirty']}`")
    lines.extend(["", "## Corpus", ""])
    lines.append(f"- Files: `{corpus['file_count']}`")
    lines.append(f"- Text records: `{corpus['text_count']}`")
    lines.append(f"- Characters: `{corpus['character_count']}`")
    lines.append(f"- Tokens: `{corpus['token_count']}`")
    lines.extend(["", "## Budget", ""])
    lines.append(f"- Parameters: `{budget['parameters']:,}`")
    lines.append(f"- Estimated train VRAM: `{budget['train_vram_gb']} GB`")
    lines.append(f"- Estimated infer VRAM: `{budget['infer_vram_gb']} GB`")
    lines.append(f"- Tokens per optimizer step: `{budget['tokens_per_optimizer_step']}`")
    lines.extend(["", "## Training Outcome", ""])
    lines.append(f"- Device used: `{training['device_used']}`")
    lines.append(f"- Steps: `{training['steps']}`")
    lines.append(f"- Optimizer steps: `{training['optimizer_steps']}`")
    lines.append(f"- Final loss: `{training['final_loss']}`")
    lines.append(f"- Minimum logged loss: `{training['min_loss']}`")
    lines.append(f"- Train eval loss: `{training['train_eval_loss']}`")
    lines.append(f"- Validation loss: `{training['val_loss']}`")
    lines.append(f"- Runtime seconds: `{training['runtime_seconds']}`")
    lines.append(f"- Tokens seen: `{training['tokens_seen']}`")
    if training.get("supervised_stats"):
        stats = training["supervised_stats"]
        lines.append(f"- Supervised examples kept: `{stats['kept_examples']}/{stats['total_examples']}`")
        lines.append(f"- Truncated supervised examples: `{stats['truncated_examples']}`")
        lines.append(f"- Max prompt+response tokens: `{stats['max_full_tokens']}`")
    lines.append(f"- Checkpoint: `{training['checkpoint']}`")
    if training.get("checkpoint_sha256"):
        lines.append(f"- Checkpoint sha256: `{training['checkpoint_sha256']}`")
    if training.get("warnings"):
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        for warning in training["warnings"]:
            lines.append(f"- {warning}")
    lines.extend(["", "## Benchmark Eval", ""])
    lines.append(f"- Accuracy: `{evaluation['correct']}/{evaluation['total']}` = `{evaluation['accuracy']}`")
    for category, payload in evaluation["by_category"].items():
        lines.append(
            f"- {category}: `{payload['correct']}/{payload['total']}` = `{payload['accuracy']}`"
        )
    lines.extend(["", "## Tool Eval", ""])
    lines.append(f"- Accuracy: `{tool_eval['correct']}/{tool_eval['total']}` = `{tool_eval['accuracy']}`")
    for category, payload in tool_eval["by_category"].items():
        lines.append(
            f"- {category}: `{payload['correct']}/{payload['total']}` = `{payload['accuracy']}`"
        )
    lines.extend(["", "## Compliance Eval", ""])
    lines.append(f"- Accuracy: `{compliance['correct']}/{compliance['total']}` = `{compliance['accuracy']}`")
    for category, payload in compliance["by_category"].items():
        lines.append(
            f"- {category}: `{payload['correct']}/{payload['total']}` = `{payload['accuracy']}`"
        )
    lines.extend(["", "## Per-Task Outputs", ""])
    for item in evaluation["results"]:
        lines.append(
            f"- [benchmark:{item['category']}] prompt=`{item['prompt']}` expected=`{item['expected']}` matched=`{item['matched']}` completion=`{item['completion']}`"
        )
    for item in tool_eval["results"]:
        lines.append(
            f"- [tool:{item['category']}] prompt=`{item['prompt']}` matched=`{item['matched']}` failed_checks=`{','.join(item['failed_checks']) or 'none'}` completion=`{item['completion']}`"
        )
    for item in compliance["results"]:
        lines.append(
            f"- [compliance:{item['category']}] prompt=`{item['prompt']}` matched=`{item['matched']}` failed_checks=`{','.join(item['failed_checks']) or 'none'}` completion=`{item['completion']}`"
        )
    if training.get("loss_history"):
        lines.extend(["", "## Loss Trace", ""])
        for item in training["loss_history"][:8]:
            lines.append(f"- step={item['step']}, loss={item['loss']}, lr={item['lr']}")
    lines.extend(["", "## Next Actions", ""])
    for action in next_actions:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def bootstrap_session(root: str | Path, name: str) -> Path:
    session_dir = create_session(root, name)
    config_paths = sorted(Path("configs/experiments").glob("*.yaml"))
    budget_results = run_budget_sweep(config_paths)
    protocol_results = run_prompt_protocol_sweep(ByteTokenizer())
    recommendation = choose_next_step(budget_results, protocol_results)

    serialized_budgets = serialize_budget_sweep(budget_results)
    serialized_protocols = serialize_protocol_sweep(protocol_results)

    write_json(session_dir / "results" / "budget_sweep.json", serialized_budgets)
    write_json(session_dir / "results" / "prompt_protocol_sweep.json", serialized_protocols)
    write_json(session_dir / "results" / "benchmark.json", benchmark_as_dicts())
    write_json(session_dir / "results" / "benchmark_registry.json", serialize_benchmark_registry())
    write_json(session_dir / "results" / "compliance_benchmark.json", compliance_benchmark_as_dicts())
    write_json(session_dir / "results" / "tool_benchmark.json", tool_benchmark_as_dicts())
    write_json(session_dir / "results" / "recommendation.json", recommendation)
    (session_dir / "notes.md").write_text(
        render_notes(
            session_name=name,
            budget_results=serialized_budgets,
            protocol_results=serialized_protocols,
            recommendation=recommendation,
        ),
        encoding="utf-8",
    )
    _record_session_completion(
        root,
        session_dir,
        name=name,
        kind="baseline",
        metadata={"recommendation_path": str(session_dir / "results" / "recommendation.json")},
    )

    return session_dir


def sota_session(root: str | Path, name: str) -> Path:
    papers = serialize_papers()
    hypotheses = serialize_hypotheses()
    session_dir = create_session(
        root,
        name,
        sources=[paper["arxiv_url"] for paper in papers],
        kind="sota",
    )

    config_paths = sorted(Path("configs/experiments").glob("*.yaml"))
    tokenizer = ByteTokenizer()
    budget_results = serialize_budget_sweep(run_budget_sweep(config_paths))
    protocol_results = serialize_protocol_sweep(run_prompt_protocol_sweep(tokenizer))
    memory_results = serialize_memory_sweep(run_memory_sweep(tokenizer))
    test_time_results = serialize_test_time_sweep(run_test_time_strategy_sweep())

    base_recommendation = choose_next_step(run_budget_sweep(config_paths), run_prompt_protocol_sweep(tokenizer))
    best_memory = _best_memory_setting(memory_results)
    best_test_time = _best_test_time_setting(test_time_results)

    recommendation = {
        **base_recommendation,
        "recommended_memory_threshold": best_memory["threshold"],
        "recommended_test_time_strategy": best_test_time["strategy"],
        "next_actions": [
            "Build a short-rationale language, math, science, and coding micro-dataset before scaling training tokens.",
            "Run domain-adaptive pretraining on math, science, and code-heavy text for the chosen dense baseline.",
            "Teach compact calculator traces first, then regenerate weak tool examples with the current checkpoint in the loop.",
            "Add verifiable RL only after AVA already solves easy arithmetic, short science QA, and simple coding tasks reliably.",
            "Enable extra test-time budget only on hard math prompts, not on ordinary English turns.",
            "Keep Titans-inspired memory external and sparse so AVA stays fast and small.",
        ],
    }

    write_json(session_dir / "results" / "papers.json", papers)
    write_json(session_dir / "results" / "hypotheses.json", hypotheses)
    write_json(session_dir / "results" / "budget_sweep.json", budget_results)
    write_json(session_dir / "results" / "prompt_protocol_sweep.json", protocol_results)
    write_json(session_dir / "results" / "memory_sweep.json", memory_results)
    write_json(session_dir / "results" / "test_time_sweep.json", test_time_results)
    write_json(session_dir / "results" / "benchmark.json", benchmark_as_dicts())
    write_json(session_dir / "results" / "benchmark_registry.json", serialize_benchmark_registry())
    write_json(session_dir / "results" / "compliance_benchmark.json", compliance_benchmark_as_dicts())
    write_json(session_dir / "results" / "tool_benchmark.json", tool_benchmark_as_dicts())
    write_json(session_dir / "results" / "recommendation.json", recommendation)
    (session_dir / "notes.md").write_text(
        render_sota_notes(
            session_name=name,
            papers=papers,
            hypotheses=hypotheses,
            budget_results=budget_results,
            protocol_results=protocol_results,
            memory_results=memory_results,
            test_time_results=test_time_results,
            recommendation=recommendation,
        ),
        encoding="utf-8",
    )
    _record_session_completion(
        root,
        session_dir,
        name=name,
        kind="sota",
        metadata={"recommendation_path": str(session_dir / "results" / "recommendation.json")},
    )

    return session_dir


def moe_feasibility_session(root: str | Path, name: str) -> Path:
    papers = _moe_papers()
    feasibility = {
        "question": "Can a sparse MoE let AVA beat frontier models everywhere on a 4 GB GPU?",
        "reasons": [
            "Sparse MoE reduces active compute more than total model storage.",
            "On a single 4 GB GPU, weight residency and memory bandwidth dominate before MoE advantages pay off.",
            "Published strong MoE systems still use much larger total parameter counts and far larger training budgets than AVA can assume.",
            "Compression helps, but current MoE compression work is still far above a 4 GB single-device budget for frontier-scale models.",
        ],
        "recommended_position": "No, not as the mainline plan. Keep dense AVA as the mainline and treat MoE as a later branch for teacher training, server-side inference, or a quantized/offloaded experiment.",
        "verdict": "no-as-mainline",
        "next_experiments": [
            "Stay dense for the 59M to sub-1B regime.",
            "Use data quality, tool-use distillation, verifiable RL, and selective test-time scaling first.",
            "If MoE is explored, try a tiny expert branch only after a strong dense teacher exists.",
            "If deployment memory becomes the main blocker, prioritize quantization and offload research over MoE-first training.",
        ],
        "estimates": MOE_FEASIBILITY_ESTIMATES,
    }

    session_dir = create_session(
        root,
        name,
        sources=[paper["arxiv_url"] for paper in papers],
        kind="moe-feasibility",
    )
    write_json(session_dir / "results" / "papers.json", papers)
    write_json(session_dir / "results" / "benchmark.json", benchmark_as_dicts())
    write_json(session_dir / "results" / "benchmark_registry.json", serialize_benchmark_registry())
    write_json(session_dir / "results" / "compliance_benchmark.json", compliance_benchmark_as_dicts())
    write_json(session_dir / "results" / "tool_benchmark.json", tool_benchmark_as_dicts())
    write_json(session_dir / "results" / "feasibility.json", feasibility)
    (session_dir / "notes.md").write_text(
        render_moe_notes(session_name=name, papers=papers, feasibility=feasibility),
        encoding="utf-8",
    )
    _record_session_completion(
        root,
        session_dir,
        name=name,
        kind="moe-feasibility",
        metadata={"feasibility_path": str(session_dir / "results" / "feasibility.json")},
    )

    return session_dir


def hf_research_session(root: str | Path, name: str) -> Path:
    papers = serialize_recent_hf_papers()
    hypotheses = serialize_recent_hf_hypotheses()
    relevant_benchmarks = _selected_benchmarks(HF_RELEVANT_BENCHMARK_KEYS)
    recommendation = {
        "focus": "Recent Hugging Face and arXiv papers that materially affect AVA's tool-use, planning, and multimodal roadmap.",
        "adopt_now": [
            "ICRL is useful, but only after compact supervised tool traces already work; it is not a replacement for AVA's first tool SFT pass.",
            "Add DeepPlanning to the benchmark contract now so long-horizon planning claims are measured before any agent push.",
            "Treat SkillNet as a runtime and evaluation reference for reusable skills, not as a base-model scaling recipe.",
            "Keep Penguin-VL as the leading compact multimodal reference for AVA's future vision branch.",
        ],
        "later_only": [
            "Use ToRL and ReTool as RL baselines after the tool RL environment and verifiers are stable.",
            "Use InternVL-U as a larger multimodal reference point, not a 4 GB mainline target.",
            "Use AReaL only if rollout throughput becomes the bottleneck after simpler RL loops already work.",
        ],
        "next_actions": [
            "Finish a stronger compact tool-supervision packet before opening the ICRL, ToRL, or ReTool branch.",
            "Keep DeepPlanning, BFCL, and tau2-bench visible together so tool use and planning are measured separately.",
            "Do not start a multimodal training branch until the text tokenizer, tool data, and compliance behavior are stable.",
            "When multimodal work starts, begin with a compact encoder path and MathVista or ScienceQA style targets rather than a broad unified model.",
        ],
    }

    session_dir = create_session(
        root,
        name,
        sources=[paper["arxiv_url"] for paper in papers] + ["https://huggingface.co/papers/trending"],
        kind="hf-research",
    )
    write_json(session_dir / "results" / "papers.json", papers)
    write_json(session_dir / "results" / "hypotheses.json", hypotheses)
    write_json(session_dir / "results" / "relevant_benchmarks.json", relevant_benchmarks)
    write_json(session_dir / "results" / "recommendation.json", recommendation)
    write_json(session_dir / "results" / "benchmark.json", benchmark_as_dicts())
    write_json(session_dir / "results" / "benchmark_registry.json", serialize_benchmark_registry())
    write_json(session_dir / "results" / "compliance_benchmark.json", compliance_benchmark_as_dicts())
    write_json(session_dir / "results" / "tool_benchmark.json", tool_benchmark_as_dicts())
    (session_dir / "notes.md").write_text(
        render_hf_research_notes(
            session_name=name,
            papers=papers,
            hypotheses=hypotheses,
            relevant_benchmarks=relevant_benchmarks,
            recommendation=recommendation,
        ),
        encoding="utf-8",
    )
    _record_session_completion(
        root,
        session_dir,
        name=name,
        kind="hf-research",
        metadata={"recommendation_path": str(session_dir / "results" / "recommendation.json")},
    )

    return session_dir


def training_session(
    root: str | Path,
    name: str,
    config_path: str | Path,
    corpus_root: str | Path,
    *,
    max_steps: int | None = None,
) -> Path:
    config = load_experiment_config(config_path)
    requested_steps = max_steps or config.training.max_steps
    command = " ".join(
        [
            "ava",
            "session",
            "train",
            quote(name),
            quote(str(config_path)),
            quote(str(corpus_root)),
            "--max-steps",
            str(requested_steps),
        ]
    )
    session_dir = create_session(
        root,
        name,
        sources=[str(Path(config_path)), str(Path(corpus_root))],
        kind="training",
    )
    artifacts_dir = session_dir / "artifacts"
    checkpoints_dir = session_dir / "checkpoints"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = artifacts_dir / Path(config_path).name
    config_snapshot_path.write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")
    (artifacts_dir / "command.txt").write_text(command + "\n", encoding="utf-8")

    environment = _environment_manifest(config.training.device)
    corpus = _corpus_manifest(corpus_root, config.tokenizer)
    budget = dry_run_summary(config)
    write_json(session_dir / "results" / "environment.json", environment)
    write_json(session_dir / "results" / "corpus.json", corpus)
    write_json(session_dir / "results" / "budget.json", budget)
    write_json(session_dir / "results" / "benchmark.json", benchmark_as_dicts())
    write_json(session_dir / "results" / "benchmark_registry.json", serialize_benchmark_registry())
    write_json(session_dir / "results" / "compliance_benchmark.json", compliance_benchmark_as_dicts())
    write_json(session_dir / "results" / "tool_benchmark.json", tool_benchmark_as_dicts())
    write_json(session_dir / "results" / "config.json", config.to_dict())

    training = run_training(
        config_path,
        corpus_root,
        max_steps=requested_steps,
        checkpoint_root=checkpoints_dir,
    )
    training["checkpoint_sha256"] = _sha256_path(training["checkpoint"])
    training["checkpoint_size_bytes"] = Path(training["checkpoint"]).stat().st_size
    evaluation = evaluate_checkpoint(
        training["checkpoint"],
        requested_device=str(training["device_used"]),
        max_new_tokens=48,
    )
    tool_eval = evaluate_tool_use_checkpoint(
        training["checkpoint"],
        requested_device=str(training["device_used"]),
        max_new_tokens=48,
    )
    compliance = evaluate_compliance_checkpoint(
        training["checkpoint"],
        requested_device=str(training["device_used"]),
        max_new_tokens=48,
    )
    next_actions = _training_next_actions(training, evaluation, tool_eval, compliance)

    write_json(session_dir / "results" / "training.json", training)
    write_json(session_dir / "results" / "evaluation.json", evaluation)
    write_json(session_dir / "results" / "tool_eval.json", tool_eval)
    write_json(session_dir / "results" / "compliance.json", compliance)
    write_json(session_dir / "results" / "next_actions.json", next_actions)
    (session_dir / "notes.md").write_text(
        render_training_notes(
            session_name=name,
            command=command,
            config_path=str(config_path),
            corpus_root=str(corpus_root),
            budget=budget,
            environment=environment,
            corpus=corpus,
            training=training,
            evaluation=evaluation,
            tool_eval=tool_eval,
            compliance=compliance,
            next_actions=next_actions,
        ),
        encoding="utf-8",
    )
    _record_session_completion(
        root,
        session_dir,
        name=name,
        kind="training",
        metadata={
            "checkpoint": str(training["checkpoint"]),
            "benchmark_accuracy": evaluation["accuracy"],
            "tool_accuracy": tool_eval["accuracy"],
            "compliance_accuracy": compliance["accuracy"],
        },
    )
    return session_dir









def render_inspection_notes(
    *,
    session_name: str,
    command: str,
    trace: dict[str, object],
) -> str:
    lines = [
        f"# Inspection Session: {session_name}",
        "",
        "## Command",
        "",
        f"`{command}`",
        "",
        "## Prompt",
        "",
        trace["prompt"],
        "",
        "## Summary",
        "",
        f"- Device used: `{trace['device_used']}`",
        f"- Generated text: `{trace['generated_text']}`",
        f"- Steps traced: `{len(trace['steps'])}`",
        f"- Stopped on eos: `{trace['stopped_on_eos']}`",
        "",
    ]
    if trace.get("warnings"):
        lines.extend(["## Warnings", ""])
        for warning in trace["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")
    if trace["steps"]:
        first_step = trace["steps"][0]
        lines.extend(["## First-Step Top Logits", ""])
        for item in first_step["top_next_token_logits"]:
            lines.append(
                f"- token_id={item['token_id']} token=`{item['token_text']}` logit={item['value']}"
            )
        lines.extend(["", "## First-Step Layer Summary", ""])
        for layer in first_step["layers"]:
            top_neurons = ", ".join(
                f"{item['index']}:{item['value']}" for item in layer["top_mlp_neurons"][:3]
            )
            head0 = layer["attention"][0]["top_positions"][0] if layer["attention"] and layer["attention"][0]["top_positions"] else None
            if head0 is not None:
                lines.append(
                    f"- layer {layer['layer']}: top_mlp_neurons={top_neurons}; head0_focus=pos {head0['position']} token `{head0['token_text']}` weight {head0['weight']}"
                )
            else:
                lines.append(f"- layer {layer['layer']}: top_mlp_neurons={top_neurons}")
    lines.extend(["", "## Artifact", "", "- Full trace: `results/trace.json`", ""])
    return "\n".join(lines) + "\n"


def inspection_session(
    root: str | Path,
    name: str,
    checkpoint_path: str | Path,
    prompt: str,
    *,
    requested_device: str = "cuda",
    max_new_tokens: int = 16,
    top_k_neurons: int = 8,
    top_k_logits: int = 8,
    top_k_attention: int = 4,
) -> Path:
    command = " ".join(
        [
            "ava",
            "session",
            "inspect",
            quote(name),
            quote(str(checkpoint_path)),
            "--prompt",
            quote(prompt),
            "--device",
            quote(requested_device),
            "--max-new-tokens",
            str(max_new_tokens),
            "--top-k-neurons",
            str(top_k_neurons),
            "--top-k-logits",
            str(top_k_logits),
            "--top-k-attention",
            str(top_k_attention),
        ]
    )
    session_dir = create_session(
        root,
        name,
        sources=[str(Path(checkpoint_path))],
        kind="inspection",
    )
    artifacts_dir = session_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (artifacts_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

    trace = trace_checkpoint(
        checkpoint_path,
        prompt,
        requested_device=requested_device,
        max_new_tokens=max_new_tokens,
        top_k_neurons=top_k_neurons,
        top_k_logits=top_k_logits,
        top_k_attention=top_k_attention,
    )
    write_json(session_dir / "results" / "trace.json", trace)
    (session_dir / "notes.md").write_text(
        render_inspection_notes(
            session_name=name,
            command=command,
            trace=trace,
        ),
        encoding="utf-8",
    )
    _record_session_completion(
        root,
        session_dir,
        name=name,
        kind="inspection",
        metadata={
            "trace_path": str(session_dir / "results" / "trace.json"),
            "generated_text": str(trace["generated_text"]),
        },
    )
    return session_dir





def _changed_results(
    baseline: dict[str, object],
    retrieval: dict[str, object],
) -> list[dict[str, object]]:
    changed: list[dict[str, object]] = []
    baseline_results = list(baseline.get("results", []))
    retrieval_results = list(retrieval.get("results", []))
    for before, after in zip(baseline_results, retrieval_results, strict=True):
        if before.get("matched") == after.get("matched") and before.get("completion") == after.get("completion"):
            continue
        changed.append(
            {
                "category": after.get("category"),
                "prompt": after.get("prompt"),
                "baseline_matched": before.get("matched"),
                "retrieval_matched": after.get("matched"),
                "baseline_completion": before.get("completion"),
                "retrieval_completion": after.get("completion"),
            }
        )
    return changed


def render_retrieval_notes(
    *,
    session_name: str,
    command: str,
    checkpoint_path: str,
    support_corpus: str,
    support_manifest: dict[str, object],
    retrieval_top_k: int,
    category_gated: bool,
    retrieval_mode: str,
    baseline_benchmark: dict[str, object],
    retrieval_benchmark: dict[str, object],
    baseline_tool: dict[str, object],
    retrieval_tool: dict[str, object],
    baseline_compliance: dict[str, object],
    retrieval_compliance: dict[str, object],
    changed_benchmark: list[dict[str, object]],
    changed_tool: list[dict[str, object]],
    changed_compliance: list[dict[str, object]],
    focus_prompt: str | None,
) -> str:
    lines = [
        f"# Retrieval Session: {session_name}",
        "",
        "## Command",
        "",
        f"`{command}`",
        "",
        "## Inputs",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Support corpus: `{support_corpus}`",
        f"- Support files: `{support_manifest['file_count']}`",
        f"- Support examples: `{support_manifest['supervised_example_count']}`",
        f"- Retrieval top_k: `{retrieval_top_k}`",
        f"- Retrieval mode: `{retrieval_mode}`",
        f"- Category gated: `{category_gated}`",
        "",
        "## Accuracy Delta",
        "",
        f"- Benchmark: `{baseline_benchmark['correct']}/{baseline_benchmark['total']}` -> `{retrieval_benchmark['correct']}/{retrieval_benchmark['total']}`",
        f"- Tool eval: `{baseline_tool['correct']}/{baseline_tool['total']}` -> `{retrieval_tool['correct']}/{retrieval_tool['total']}`",
        f"- Compliance: `{baseline_compliance['correct']}/{baseline_compliance['total']}` -> `{retrieval_compliance['correct']}/{retrieval_compliance['total']}`",
        "",
        "## Changed Benchmark Rows",
        "",
    ]
    if changed_benchmark:
        for item in changed_benchmark:
            lines.append(
                f"- [{item['category']}] prompt=`{item['prompt']}` baseline=`{item['baseline_completion']}` -> retrieval=`{item['retrieval_completion']}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Changed Tool Rows", ""])
    if changed_tool:
        for item in changed_tool:
            lines.append(
                f"- [{item['category']}] prompt=`{item['prompt']}` baseline=`{item['baseline_completion']}` -> retrieval=`{item['retrieval_completion']}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Changed Compliance Rows", ""])
    if changed_compliance:
        for item in changed_compliance:
            lines.append(
                f"- [{item['category']}] prompt=`{item['prompt']}` baseline=`{item['baseline_completion']}` -> retrieval=`{item['retrieval_completion']}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Focus Prompt", ""])
    if focus_prompt:
        lines.append(f"- `{focus_prompt}`")
        lines.append("- Baseline trace: `results/focus_trace_baseline.json`")
        lines.append("- Retrieval trace: `results/focus_trace_retrieval.json`")
    else:
        lines.append("- none")
    lines.extend(["", "## Artifacts", ""])
    lines.append("- Baseline benchmark/tool/compliance under `results/`")
    lines.append("- Retrieval benchmark/tool/compliance under `results/`")
    return "\n".join(lines) + "\n"


def retrieval_session(
    root: str | Path,
    name: str,
    checkpoint_path: str | Path,
    support_corpus: str | Path,
    *,
    requested_device: str = "cuda",
    max_new_tokens: int = 48,
    retrieval_top_k: int = 1,
    category_gated: bool = True,
    retrieval_mode: str = "prompt",
) -> Path:
    command_parts = [
        "ava",
        "session",
        "retrieval",
        quote(name),
        quote(str(checkpoint_path)),
        quote(str(support_corpus)),
        "--device",
        quote(requested_device),
        "--max-new-tokens",
        str(max_new_tokens),
        "--retrieval-top-k",
        str(retrieval_top_k),
        "--mode",
        quote(retrieval_mode),
    ]
    if not category_gated:
        command_parts.append("--no-category-gating")
    command = " ".join(command_parts)

    session_dir = create_session(
        root,
        name,
        sources=[str(Path(checkpoint_path)), str(Path(support_corpus))],
        kind="retrieval",
    )
    artifacts_dir = session_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "command.txt").write_text(command + "\n", encoding="utf-8")

    environment = _environment_manifest(requested_device)
    support_manifest = _corpus_manifest(support_corpus)
    write_json(session_dir / "results" / "environment.json", environment)
    write_json(session_dir / "results" / "support_corpus.json", support_manifest)

    baseline_benchmark = evaluate_checkpoint(
        checkpoint_path,
        requested_device=requested_device,
        max_new_tokens=max_new_tokens,
    )
    baseline_tool = evaluate_tool_use_checkpoint(
        checkpoint_path,
        requested_device=requested_device,
        max_new_tokens=max_new_tokens,
    )
    baseline_compliance = evaluate_compliance_checkpoint(
        checkpoint_path,
        requested_device=requested_device,
        max_new_tokens=max_new_tokens,
    )

    retrieval_benchmark = evaluate_checkpoint(
        checkpoint_path,
        requested_device=requested_device,
        max_new_tokens=max_new_tokens,
        support_corpus=support_corpus,
        retrieval_top_k=retrieval_top_k,
        category_gated=category_gated,
        retrieval_mode=retrieval_mode,
    )
    retrieval_tool = evaluate_tool_use_checkpoint(
        checkpoint_path,
        requested_device=requested_device,
        max_new_tokens=max_new_tokens,
        support_corpus=support_corpus,
        retrieval_top_k=retrieval_top_k,
        category_gated=category_gated,
        retrieval_mode=retrieval_mode,
    )
    retrieval_compliance = evaluate_compliance_checkpoint(
        checkpoint_path,
        requested_device=requested_device,
        max_new_tokens=max_new_tokens,
        support_corpus=support_corpus,
        retrieval_top_k=retrieval_top_k,
        category_gated=category_gated,
        retrieval_mode=retrieval_mode,
    )

    changed_benchmark = _changed_results(baseline_benchmark, retrieval_benchmark)
    changed_tool = _changed_results(baseline_tool, retrieval_tool)
    changed_compliance = _changed_results(baseline_compliance, retrieval_compliance)
    focus_prompt = None
    focus_category = None
    for item in changed_benchmark:
        if not item["baseline_matched"] and item["retrieval_matched"]:
            focus_prompt = str(item["prompt"])
            focus_category = str(item["category"])
            break
    if focus_prompt is None:
        for collection in (changed_tool, changed_compliance):
            for item in collection:
                if not item["baseline_matched"] and item["retrieval_matched"]:
                    focus_prompt = str(item["prompt"])
                    focus_category = str(item["category"])
                    break
            if focus_prompt is not None:
                break

    if focus_prompt is not None:
        baseline_trace = trace_checkpoint(
            checkpoint_path,
            focus_prompt,
            requested_device=requested_device,
            max_new_tokens=min(max_new_tokens, 16),
            top_k_neurons=8,
            top_k_logits=8,
            top_k_attention=4,
        )
        if retrieval_mode == "direct":
            retrieval_trace = {
                "mode": "direct",
                "prompt": focus_prompt,
                "generated_text": next(
                    item["retrieval_completion"]
                    for item in (changed_benchmark + changed_tool + changed_compliance)
                    if item["prompt"] == focus_prompt
                ),
                "steps": [],
                "retrieval": {
                    "enabled": True,
                    "mode": "direct",
                    "category_hint": focus_category,
                },
            }
        else:
            retrieval_trace = trace_checkpoint(
                checkpoint_path,
                focus_prompt,
                requested_device=requested_device,
                max_new_tokens=min(max_new_tokens, 16),
                top_k_neurons=8,
                top_k_logits=8,
                top_k_attention=4,
                support_corpus=support_corpus,
                retrieval_top_k=retrieval_top_k,
                category_hint=focus_category,
                category_gated=category_gated,
            )
        write_json(session_dir / "results" / "focus_trace_baseline.json", baseline_trace)
        write_json(session_dir / "results" / "focus_trace_retrieval.json", retrieval_trace)

    write_json(session_dir / "results" / "baseline_benchmark.json", baseline_benchmark)
    write_json(session_dir / "results" / "baseline_tool_eval.json", baseline_tool)
    write_json(session_dir / "results" / "baseline_compliance.json", baseline_compliance)
    write_json(session_dir / "results" / "retrieval_benchmark.json", retrieval_benchmark)
    write_json(session_dir / "results" / "retrieval_tool_eval.json", retrieval_tool)
    write_json(session_dir / "results" / "retrieval_compliance.json", retrieval_compliance)
    write_json(session_dir / "results" / "changed_benchmark.json", changed_benchmark)
    write_json(session_dir / "results" / "changed_tool.json", changed_tool)
    write_json(session_dir / "results" / "changed_compliance.json", changed_compliance)
    (session_dir / "notes.md").write_text(
        render_retrieval_notes(
            session_name=name,
            command=command,
            checkpoint_path=str(checkpoint_path),
            support_corpus=str(support_corpus),
            support_manifest=support_manifest,
            retrieval_top_k=retrieval_top_k,
            retrieval_mode=retrieval_mode,
            category_gated=category_gated,
            baseline_benchmark=baseline_benchmark,
            retrieval_benchmark=retrieval_benchmark,
            baseline_tool=baseline_tool,
            retrieval_tool=retrieval_tool,
            baseline_compliance=baseline_compliance,
            retrieval_compliance=retrieval_compliance,
            changed_benchmark=changed_benchmark,
            changed_tool=changed_tool,
            changed_compliance=changed_compliance,
            focus_prompt=focus_prompt,
        ),
        encoding="utf-8",
    )
    _record_session_completion(
        root,
        session_dir,
        name=name,
        kind="retrieval",
        metadata={
            "checkpoint": str(checkpoint_path),
            "support_corpus": str(support_corpus),
            "baseline_benchmark_accuracy": baseline_benchmark["accuracy"],
            "retrieval_benchmark_accuracy": retrieval_benchmark["accuracy"],
            "baseline_tool_accuracy": baseline_tool["accuracy"],
            "retrieval_tool_accuracy": retrieval_tool["accuracy"],
            "baseline_compliance_accuracy": baseline_compliance["accuracy"],
            "retrieval_compliance_accuracy": retrieval_compliance["accuracy"],
            "retrieval_mode": retrieval_mode,
        },
    )
    return session_dir

def _suite_total_score(suite: dict[str, object]) -> int:
    return int(suite["benchmark"]["correct"]) + int(suite["tool"]["correct"]) + int(suite["compliance"]["correct"])


def _suite_changes(before: dict[str, object], after: dict[str, object], key: str) -> list[dict[str, object]]:
    baseline_results = list(before[key]["results"])
    candidate_results = list(after[key]["results"])
    changed: list[dict[str, object]] = []
    for previous, current in zip(baseline_results, candidate_results, strict=True):
        if previous.get("matched") == current.get("matched") and previous.get("completion") == current.get("completion"):
            continue
        changed.append(
            {
                "category": current.get("category"),
                "prompt": current.get("prompt"),
                "baseline_matched": previous.get("matched"),
                "candidate_matched": current.get("matched"),
                "baseline_completion": previous.get("completion"),
                "candidate_completion": current.get("completion"),
                "retrieval": current.get("retrieval"),
            }
        )
    return changed


def render_memory_transfer_notes(
    *,
    session_name: str,
    command: str,
    checkpoint_path: str,
    support_corpus: str,
    support_manifest: dict[str, object],
    suite: str,
    nearest_threshold: float,
    nearest_margin: float,
    category_gated: bool,
    baseline: dict[str, object],
    direct: dict[str, object],
    nearest: dict[str, object],
    direct_benchmark_changes: list[dict[str, object]],
    direct_tool_changes: list[dict[str, object]],
    direct_compliance_changes: list[dict[str, object]],
    nearest_benchmark_changes: list[dict[str, object]],
    nearest_tool_changes: list[dict[str, object]],
    nearest_compliance_changes: list[dict[str, object]],
    winner: str,
) -> str:
    lines = [
        f"# Memory Transfer Session: {session_name}",
        "",
        "## Command",
        "",
        f"`{command}`",
        "",
        "## Inputs",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Support corpus: `{support_corpus}`",
        f"- Support files: `{support_manifest['file_count']}`",
        f"- Support examples: `{support_manifest['supervised_example_count']}`",
        f"- Transfer suite: `{suite}`",
        f"- Nearest threshold: `{nearest_threshold}`",
        f"- Nearest margin: `{nearest_margin}`",
        f"- Category gated: `{category_gated}`",
        "",
        "## Transfer Scores",
        "",
        f"- Baseline: benchmark `{baseline['benchmark']['correct']}/{baseline['benchmark']['total']}`, tool `{baseline['tool']['correct']}/{baseline['tool']['total']}`, compliance `{baseline['compliance']['correct']}/{baseline['compliance']['total']}`",
        f"- Exact direct: benchmark `{direct['benchmark']['correct']}/{direct['benchmark']['total']}`, tool `{direct['tool']['correct']}/{direct['tool']['total']}`, compliance `{direct['compliance']['correct']}/{direct['compliance']['total']}`",
        f"- Nearest direct: benchmark `{nearest['benchmark']['correct']}/{nearest['benchmark']['total']}`, tool `{nearest['tool']['correct']}/{nearest['tool']['total']}`, compliance `{nearest['compliance']['correct']}/{nearest['compliance']['total']}`",
        f"- Winner: `{winner}`",
        "",
        "## Exact Direct Changes",
        "",
    ]
    for heading, items in (("benchmark", direct_benchmark_changes), ("tool", direct_tool_changes), ("compliance", direct_compliance_changes)):
        lines.append(f"- {heading}: `{len(items)}` changed rows")
        for item in items[:6]:
            lines.append(
                f"- [{heading}:{item['category']}] prompt=`{item['prompt']}` baseline=`{item['baseline_completion']}` -> exact=`{item['candidate_completion']}`"
            )
    lines.extend(["", "## Nearest Direct Changes", ""])
    for heading, items in (("benchmark", nearest_benchmark_changes), ("tool", nearest_tool_changes), ("compliance", nearest_compliance_changes)):
        lines.append(f"- {heading}: `{len(items)}` changed rows")
        for item in items[:6]:
            lines.append(
                f"- [{heading}:{item['category']}] prompt=`{item['prompt']}` baseline=`{item['baseline_completion']}` -> nearest=`{item['candidate_completion']}`"
            )
    lines.extend(["", "## Artifacts", ""])
    lines.append("- Transfer benchmark definitions under `results/`")
    lines.append("- Per-mode transfer results under `results/`")
    lines.append("- Changed rows for exact and nearest memory under `results/`")
    return "\n".join(lines) + "\n"


def memory_transfer_session(
    root: str | Path,
    name: str,
    checkpoint_path: str | Path,
    support_corpus: str | Path,
    *,
    requested_device: str = "cuda",
    max_new_tokens: int = 48,
    nearest_threshold: float = 0.58,
    nearest_margin: float = 0.03,
    category_gated: bool = True,
    suite: str = "small",
) -> Path:
    command_parts = [
        "ava",
        "session",
        "memory-transfer",
        quote(name),
        quote(str(checkpoint_path)),
        quote(str(support_corpus)),
        "--device",
        quote(requested_device),
        "--max-new-tokens",
        str(max_new_tokens),
        "--nearest-threshold",
        str(nearest_threshold),
        "--nearest-margin",
        str(nearest_margin),
        "--suite",
        quote(suite),
    ]
    if not category_gated:
        command_parts.append("--no-category-gating")
    command = " ".join(command_parts)

    session_dir = create_session(
        root,
        name,
        sources=[str(Path(checkpoint_path)), str(Path(support_corpus))],
        kind="memory-transfer",
    )
    artifacts_dir = session_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "command.txt").write_text(command + "\n", encoding="utf-8")

    environment = _environment_manifest(requested_device)
    support_manifest = _corpus_manifest(support_corpus)
    write_json(session_dir / "results" / "environment.json", environment)
    write_json(session_dir / "results" / "support_corpus.json", support_manifest)
    write_json(session_dir / "results" / "transfer_benchmark.json", transfer_benchmark_as_dicts(suite))
    write_json(session_dir / "results" / "transfer_tool_benchmark.json", transfer_tool_benchmark_as_dicts(suite))
    write_json(session_dir / "results" / "transfer_compliance_benchmark.json", transfer_compliance_benchmark_as_dicts(suite))

    baseline = evaluate_transfer_suite_checkpoint(
        checkpoint_path,
        requested_device=requested_device,
        max_new_tokens=max_new_tokens,
        retrieval_mode="baseline",
        suite=suite,
    )
    direct = evaluate_transfer_suite_checkpoint(
        checkpoint_path,
        requested_device=requested_device,
        max_new_tokens=max_new_tokens,
        support_corpus=support_corpus,
        retrieval_mode="direct",
        category_gated=category_gated,
        nearest_threshold=nearest_threshold,
        nearest_margin=nearest_margin,
        suite=suite,
    )
    nearest = evaluate_transfer_suite_checkpoint(
        checkpoint_path,
        requested_device=requested_device,
        max_new_tokens=max_new_tokens,
        support_corpus=support_corpus,
        retrieval_mode="nearest",
        category_gated=category_gated,
        nearest_threshold=nearest_threshold,
        nearest_margin=nearest_margin,
        suite=suite,
    )

    direct_benchmark_changes = _suite_changes(baseline, direct, "benchmark")
    direct_tool_changes = _suite_changes(baseline, direct, "tool")
    direct_compliance_changes = _suite_changes(baseline, direct, "compliance")
    nearest_benchmark_changes = _suite_changes(baseline, nearest, "benchmark")
    nearest_tool_changes = _suite_changes(baseline, nearest, "tool")
    nearest_compliance_changes = _suite_changes(baseline, nearest, "compliance")

    scores = {
        "baseline": _suite_total_score(baseline),
        "direct": _suite_total_score(direct),
        "nearest": _suite_total_score(nearest),
    }
    winner = sorted(scores.items(), key=lambda item: item[1], reverse=True)[0][0]

    write_json(session_dir / "results" / "baseline_transfer.json", baseline)
    write_json(session_dir / "results" / "direct_transfer.json", direct)
    write_json(session_dir / "results" / "nearest_transfer.json", nearest)
    write_json(session_dir / "results" / "direct_benchmark_changes.json", direct_benchmark_changes)
    write_json(session_dir / "results" / "direct_tool_changes.json", direct_tool_changes)
    write_json(session_dir / "results" / "direct_compliance_changes.json", direct_compliance_changes)
    write_json(session_dir / "results" / "nearest_benchmark_changes.json", nearest_benchmark_changes)
    write_json(session_dir / "results" / "nearest_tool_changes.json", nearest_tool_changes)
    write_json(session_dir / "results" / "nearest_compliance_changes.json", nearest_compliance_changes)
    write_json(session_dir / "results" / "winner.json", {"winner": winner, "scores": scores})

    (session_dir / "notes.md").write_text(
        render_memory_transfer_notes(
            session_name=name,
            command=command,
            checkpoint_path=str(checkpoint_path),
            support_corpus=str(support_corpus),
            support_manifest=support_manifest,
            suite=suite,
            nearest_threshold=nearest_threshold,
            nearest_margin=nearest_margin,
            category_gated=category_gated,
            baseline=baseline,
            direct=direct,
            nearest=nearest,
            direct_benchmark_changes=direct_benchmark_changes,
            direct_tool_changes=direct_tool_changes,
            direct_compliance_changes=direct_compliance_changes,
            nearest_benchmark_changes=nearest_benchmark_changes,
            nearest_tool_changes=nearest_tool_changes,
            nearest_compliance_changes=nearest_compliance_changes,
            winner=winner,
        ),
        encoding="utf-8",
    )
    _record_session_completion(
        root,
        session_dir,
        name=name,
        kind="memory-transfer",
        metadata={
            "checkpoint": str(checkpoint_path),
            "support_corpus": str(support_corpus),
            "suite": suite,
            "winner": winner,
            "baseline_score": scores["baseline"],
            "direct_score": scores["direct"],
            "nearest_score": scores["nearest"],
        },
    )
    return session_dir
